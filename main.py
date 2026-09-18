"""Iron Man HUD — main loop.

    python main.py                # webcam + microphone ("Hey Jarvis, ...")
    python main.py --typed        # type commands in the terminal instead of speaking
    python main.py --echo         # also print everything the recogniser hears
    python main.py --camera 1 --mic 2
    python main.py --calibrate    # stand 60 cm away, press SPACE -> writes calibration.json

Loop per frame: grab -> poll command (voice queue or keyboard) -> dispatch ->
HUD overlay (or frozen one-shot result while holding) -> show.
"""

from __future__ import annotations

import argparse
import queue
import time
import traceback
from datetime import datetime

import cv2

from common import config, jpeg
from common.types import Command, CommandKind, Frame, HUDState
from effects import registry
from hud import distance, overlay
from hud.engine import HUDEngine
from voice import commands
from voice.interface import BaseVoiceInterface, TypedVoiceInterface, VoiceInterface
from voice.listener import ModelMissing

# Single-key fallbacks so every command works without a microphone.
KEYMAP: dict[str, str] = {
    "x": "xray",
    "s": "sketch",
    "c": "cartoon",
    "f": "spectrum",
    "l": "lowpass",
    "h": "highpass",
    "e": "edges",
    "g": "sobel",
    "b": "remove_bg",
    "k": "skin",
    "a": "enhance",
    "n": "night_vision",
    "t": "thermal",
    "r": "radar",
    "m": "motion_alert",
    "p": "snapshot",
    "u": "enrol",
    "w": "whoami",
    "0": "reset",
    "?": "help",
    "q": "quit",
    "i": "identity",
}
ESC = 27
NOTICE_SECONDS = 4.0
GREET_COOLDOWN_S = 120.0
HIDDEN_FROM_HELP = {"identity", "help"}


class App:
    def __init__(self, camera_index: int, calibrate: bool = False, mute: bool = False,
                 typed: bool = False, echo: bool = False, mic: int | None = None) -> None:
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {camera_index}")

        self.state = HUDState()
        self.hud = HUDEngine()
        self.voice: BaseVoiceInterface = (TypedVoiceInterface(mute=mute) if typed
                                          else VoiceInterface(mute=mute, echo=echo, device=mic))
        self.running = True
        self.calibrating = calibrate

        # (frozen result, name, start time, release time) while a one-shot effect is on screen
        self.held: tuple[Frame, str, float, float] | None = None
        # (text, expiry) — JARVIS's last reply, shown on the HUD so the audience can read it
        self.notice: tuple[str, float] | None = None
        self.greeted: dict[str, float] = {}          # name -> last greeting time
        self.enrol_prompted = False
        self.pending: queue.Queue = queue.Queue()     # callables handed over from other threads
        self.voice.wake_reply = lambda: self.say("Yes, sir?")
        config.CAPTURES_DIR.mkdir(exist_ok=True)
        if self.hud.recognition_enabled:
            print(f"[faces] recognition on; known: {', '.join(self.hud.known_names()) or 'nobody yet'}")
        else:
            print("[faces] recognition off (run tools/get_model.py faces to enable)")

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        try:
            self.voice.start()
        except ModelMissing as exc:
            print(f"[voice] {exc}\n[voice] falling back to typed commands")
            self.voice = TypedVoiceInterface(mute=not self.voice.tts.enabled)
            self.voice.start()
        cv2.namedWindow(config.WINDOW_NAME)
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    print("camera read failed")
                    break

                key = cv2.waitKey(1)
                cmd = self.voice.poll() or self._key_command(key)
                if cmd is not None:
                    self.dispatch(cmd, frame)
                self.state.listening = self.voice.listening
                while not self.pending.empty():      # work handed over from other threads
                    self.pending.get_nowait()()

                # keep the tracker / motion detector warm even while a result is frozen
                live = self.hud.process(frame, self.state)
                self._recognition_step()
                if self.calibrating:
                    self._calibration_step(live, key)

                now = time.monotonic()
                if self.held is not None and now < self.held[3]:
                    out = self._label_held(*self.held, now)
                else:
                    self.held = None
                    out = live
                self._draw_notice(out, now)
                cv2.imshow(config.WINDOW_NAME, out)
        finally:
            self.voice.stop()
            self.cap.release()
            cv2.destroyAllWindows()

    def _key_command(self, key: int) -> Command | None:
        if key == -1:
            return None
        if key == ESC:
            return commands.match("quit")
        name = KEYMAP.get(chr(key & 0xFF))
        if name is None:
            if self.held is not None:   # any other key dismisses a frozen result
                self.held = None
            return None
        return commands.match(name)

    @property
    def primary_name(self) -> str:
        return self.state.names[0] if self.state.names else ""

    def say(self, text: str) -> None:
        """Speak/print via the voice interface and show the line on the HUD for a moment.
        "sir" becomes the recognised person's name when JARVIS knows who is in front of him."""
        if not text:
            return
        if self.primary_name:
            text = text.replace("sir", self.primary_name)
        self.voice.say(text)
        self.notice = (text, time.monotonic() + NOTICE_SECONDS)

    # ----------------------------------------------------------- recognition

    def _recognition_step(self) -> None:
        """Greet a recognised person once in a while; drive the enrolment prompt."""
        name = self.primary_name
        now = time.monotonic()
        if name and self.state.enrolling < 0 and now - self.greeted.get(name, -1e9) > GREET_COOLDOWN_S:
            self.greeted[name] = now
            self.say(f"Welcome back, {name}.")

        if self.state.enrolling >= 0 and self.hud.enrol_ready() and not self.enrol_prompted:
            self.enrol_prompted = True
            self.say("Captured. What is your name? Type it in the terminal.")
            self.voice.ask_text("[faces] name: ", lambda answer: self.pending.put(lambda: self._finish_enrol(answer)))

    def _finish_enrol(self, name: str) -> None:
        self.enrol_prompted = False
        if not name:
            self.hud.cancel_enrol()
            self.say("Enrolment cancelled.")
            return
        n = self.hud.finish_enrol(name)
        self.greeted[name.title()] = time.monotonic()
        print(f"[faces] stored {n} samples for {name.title()}; known: {', '.join(self.hud.known_names())}")
        self.say(f"Pleasure to meet you, {name.title()}. I will remember you.")

    # -------------------------------------------------------------- dispatch

    def dispatch(self, cmd: Command, frame: Frame) -> None:
        if cmd.kind is CommandKind.TOGGLE:
            self._toggle(cmd.name)
        elif cmd.kind is CommandKind.ONESHOT:
            self._oneshot(cmd.name, frame)
        elif cmd.kind is CommandKind.SYSTEM:
            self._system(cmd.name, frame)

    def _toggle(self, name: str) -> None:
        new_value = not getattr(self.state, name)
        setattr(self.state, name, new_value)
        if new_value and name in ("night_vision", "thermal"):   # base modes are exclusive
            other = "thermal" if name == "night_vision" else "night_vision"
            setattr(self.state, other, False)
        self.say(f"{name.replace('_', ' ').capitalize()} {'on' if new_value else 'off'}, sir.")

    def _oneshot(self, name: str, frame: Frame) -> None:
        if name not in registry.EFFECTS:
            self.say(f"{name.replace('_', ' ')} is not installed yet.")
            return
        self.say(commands.reply_for(name))
        t0 = time.perf_counter()
        try:
            result = registry.apply(name, frame, self.state)
        except Exception:
            traceback.print_exc()
            self.say("That effect failed. Check the console.")
            return
        print(f"[effect] {name} took {(time.perf_counter() - t0) * 1000:.0f} ms")
        self._save(result, name)
        now = time.monotonic()
        self.held = (result, name, now, now + config.HOLD_SECONDS)

    def _system(self, name: str, frame: Frame) -> None:
        if name == "quit":
            self.say(commands.reply_for("quit"))
            self.running = False
        elif name == "reset":
            self.held = None
            self.state.night_vision = False
            self.state.thermal = False
            self.state.motion_alert = False
            self.say(commands.reply_for("reset"))
        elif name == "snapshot":
            shot = self.hud.process(frame.copy(), self.state)
            path = self._save(shot, "snapshot", ext=".jpg")
            stats = jpeg.summary(shot)
            print(f"[snapshot] {path.name}: {stats}")
            self.say("Captured.")
            self.notice = (f"SNAPSHOT  {stats}", time.monotonic() + NOTICE_SECONDS * 2)
        elif name == "enrol":
            if not self.hud.recognition_enabled:
                self.say("Face recognition models are not installed.")
            elif not self.state.faces:
                self.say("I need to see a face first.")
            elif self.state.enrolling >= 0:
                self.say("Already capturing. Hold still.")
            else:
                self.enrol_prompted = False
                self.hud.start_enrol()
                self.say(commands.reply_for("enrol"))
        elif name == "whoami":
            if not self.hud.recognition_enabled:
                self.say("Face recognition models are not installed.")
            elif not self.state.faces:
                self.say("I don't see anyone.")
            else:
                known = [n for n in self.state.names if n]
                if known:
                    self.say("I see " + " and ".join(dict.fromkeys(known)) + ".")
                else:
                    self.say("I don't recognise you. Say 'remember me' and I will.")
        elif name == "forget":
            self.hud.forget()
            self.greeted.clear()
            self.say(commands.reply_for("forget"))
        elif name == "help":
            names = [s.name for s in commands.VOCABULARY
                     if s.name not in HIDDEN_FROM_HELP
                     and (s.kind is not CommandKind.ONESHOT or s.name in registry.EFFECTS)]
            print("[help] say 'Hey Jarvis' then one of:")
            for s in commands.VOCABULARY:
                if s.name in names:
                    key = next((k for k, v in KEYMAP.items() if v == s.name), " ")
                    print(f"   [{key}] {s.phrases[0]:32s} ({s.kind.value})")
            self.say(f"I know {len(names)} commands. Try: find the edges, sketch this, "
                     "night vision, or remove the background. The full list is in the console.")

    # --------------------------------------------------------------- helpers

    def _calibration_step(self, out: Frame, key: int) -> None:
        """Overlay instructions; on SPACE, derive the focal length from the primary face box."""
        h, w = out.shape[:2]
        d = distance.CALIBRATION_DISTANCE_CM
        lines = [f"CALIBRATION: sit {d:.0f} cm from the camera, face the lens, press SPACE"]
        if self.state.faces:
            bw = self.state.faces[0][2]
            lines.append(f"face box width = {bw}px  ->  focal = {distance.focal_from_box(bw):.0f}px")
        else:
            lines.append("no face detected")
        for i, s in enumerate(lines):
            overlay.put_text(out, s, (16, h - 60 + 22 * i), config.HUD_ORANGE, 0.5)

        if key == 32 and self.state.faces:
            focal = distance.focal_from_box(self.state.faces[0][2])
            distance.save_focal_px(focal)
            self.hud.focal_px = focal
            self.calibrating = False
            self.say(f"Calibrated. Focal length {focal:.0f} pixels, saved.")

    @staticmethod
    def _label_held(frame: Frame, name: str, start: float, end: float, now: float) -> Frame:
        """Frozen result with a title and a draining progress bar (any key dismisses)."""
        out = frame.copy()
        h, w = out.shape[:2]
        overlay.put_text(out, f"HOLD  {name.upper().replace('_', ' ')}", (12, h - 16), config.HUD_ORANGE, 0.6, 2)
        remaining = max(0.0, (end - now) / max(end - start, 1e-6))
        bar_w = int(remaining * (w - 24))
        cv2.rectangle(out, (12, h - 8), (w - 12, h - 4), (60, 60, 60), -1)
        if bar_w:
            cv2.rectangle(out, (12, h - 8), (12 + bar_w, h - 4), config.HUD_ORANGE, -1)
        return out

    def _draw_notice(self, out: Frame, now: float) -> None:
        if self.notice is None:
            return
        text, expiry = self.notice
        if now >= expiry:
            self.notice = None
            return
        h, w = out.shape[:2]
        overlay.put_text(out, text, (16, h - 34 if self.held else h - 20), config.HUD_CYAN, 0.5)

    @staticmethod
    def _save(frame: Frame, name: str, ext: str = ".png"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = config.CAPTURES_DIR / f"{stamp}_{name}{ext}"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else [])
        print(f"[capture] {path.name}")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Iron Man HUD")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--calibrate", action="store_true",
                        help="distance calibration: sit 60 cm away and press SPACE")
    parser.add_argument("--mute", action="store_true", help="print JARVIS replies without speaking them")
    parser.add_argument("--typed", action="store_true", help="type commands instead of using the microphone")
    parser.add_argument("--echo", action="store_true", help="print every transcript the recogniser produces")
    parser.add_argument("--mic", type=int, default=None, help="sounddevice input device index")
    args = parser.parse_args()
    App(args.camera, calibrate=args.calibrate, mute=args.mute,
        typed=args.typed, echo=args.echo, mic=args.mic).run()


if __name__ == "__main__":
    main()
