"""Voice interfaces.

`VoiceInterface`      mic -> Vosk -> wake-word gate -> fuzzy match -> command queue
`TypedVoiceInterface` stdin -> fuzzy match -> command queue (development / no-mic fallback)

Both print JARVIS replies and (unless muted) speak them through the TTS thread.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

from common.types import Command
from voice import commands
from voice.listener import Listener, ModelMissing, model_path
from voice.tts import TTS
from voice.wake import WakeGate


class BaseVoiceInterface:
    def __init__(self, mute: bool = False) -> None:
        self._queue: queue.Queue[Command] = queue.Queue()
        self.tts = TTS(enabled=not mute)

    def start(self) -> None:
        self.tts.start()

    def stop(self) -> None:
        self.tts.stop()

    def poll(self) -> Command | None:
        """Non-blocking: return the next pending command, or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def listening(self) -> float:
        """Seconds left in which a bare command (no wake word) is accepted; 0 = idle."""
        return 0.0

    def say(self, text: str) -> None:
        """Non-blocking reply to the user: printed, and spoken if TTS is on."""
        if not text:
            return
        print(f"[JARVIS] {text}")
        self.tts.say(text)

    def submit_text(self, text: str) -> Command | None:
        """Match text and enqueue the command; shared by the typed and spoken paths."""
        cmd, score, phrase = commands.match_scored(text)
        if cmd is None:
            print(f"[voice] no match for {text.strip()!r}")
            self.say("I didn't catch that.")
            return None
        if score < 100:
            print(f"[voice] {text.strip()!r} ~ {phrase!r} ({score:.0f}%)")
        self._queue.put(cmd)
        return cmd


class TypedVoiceInterface(BaseVoiceInterface):
    """Reads one command per line from stdin on a background thread."""

    @property
    def listening(self) -> float:
        return float("inf")   # always accepting input

    def start(self) -> None:
        super().start()
        threading.Thread(target=self._read_loop, name="typed-voice", daemon=True).start()
        print("[voice] typed mode - enter a command (e.g. 'find the edges', 'night vision', 'quit'):")

    def _read_loop(self) -> None:
        for line in sys.stdin:
            if line.strip():
                self.submit_text(line)


GRAMMAR_EXCLUDE = {"cartoonify", "cartoonify this", "x-ray", "x-ray this", "xray this"}  # words absent from the small model


def grammar_phrases() -> list[str]:
    """Everything the recogniser needs to tell apart: commands, wake word, and [unk]."""
    phrases = {p for spec in commands.VOCABULARY for p in spec.phrases if spec.name != "identity"}
    phrases |= {"hey jarvis", "jarvis"}
    return sorted(phrases - GRAMMAR_EXCLUDE) + ["[unk]"]


class VoiceInterface(BaseVoiceInterface):
    """Real microphone path. Raises ModelMissing from start() if the Vosk model is absent."""

    ECHO_GUARD_S = 0.5    # ignore transcripts this long after JARVIS finished speaking
    MAX_SPEECH_S = 6.0    # never treat TTS as "speaking" longer than this (stuck-driver safety)

    def __init__(self, mute: bool = False, use_grammar: bool = True, device: int | None = None,
                 echo: bool = False) -> None:
        super().__init__(mute=mute)
        self.echo = echo   # print every transcript (debugging)
        self.gate = WakeGate(on_command=self._on_command, on_wake=lambda: self.say("Yes?"),
                             on_timeout=lambda: print("[voice] standing by - say 'Hey Jarvis' to continue"))
        self.listener = Listener(self._on_text, grammar=grammar_phrases() if use_grammar else None,
                                 device=device)

    def start(self) -> None:
        model_path()            # fail fast, before the TTS thread starts
        super().start()
        self.listener.start()
        print("[voice] listening - say 'Hey Jarvis' followed by a command")

    def stop(self) -> None:
        self.listener.stop()
        super().stop()

    def poll(self) -> Command | None:
        # the 5 s command window starts when JARVIS has finished saying "Yes?", not when he started
        if self.gate.armed and self._jarvis_talking():
            self.gate.armed_at = time.monotonic()
        self.gate.tick()
        return super().poll()

    @property
    def listening(self) -> float:
        return self.gate.remaining()

    def _jarvis_talking(self) -> bool:
        now = time.monotonic()
        if self.tts.speaking.is_set() and now - self.tts.speak_started_at < self.MAX_SPEECH_S:
            return True
        return now - self.tts.last_spoke_at < self.ECHO_GUARD_S

    # --------------------------------------------------------------- callbacks

    def _on_text(self, kind: str, text: str) -> None:
        if self._jarvis_talking():
            if kind == "final":
                self.listener.reset()
            return
        if self.echo:
            print(f"[stt:{kind}] {text}")
        if kind == "final":
            self.gate.feed_final(text)

    def _on_command(self, text: str) -> None:
        self.submit_text(text)
