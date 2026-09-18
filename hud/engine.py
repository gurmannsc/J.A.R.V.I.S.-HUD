"""HUDEngine — detection, tracking and overlay drawing.

Every DETECT_EVERY frames the Haar detector runs ("I" frame); in between, Lucas–Kanade
optical flow carries the boxes forward ("P" frame). If the tracker loses every target the
detector runs again on the next frame so lock-on stays quick.

Detection and motion always use the raw camera frame; night-vision / thermal only change
what is drawn underneath the overlays. All line art goes through one theme.Canvas per
frame and is composited in a single pass (see hud/theme.py).
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common import config
from common.types import Frame, HUDState
from hud import distance, nightvision, overlay, recognizer
from hud.detector import Box, FaceDetector
from hud.frame import StaticFrame
from hud.motion import MotionDetector
from hud.radar import draw_radar
from hud.theme import Canvas
from hud.tracker import FaceTracker

LOCK_FRAMES = 14   # frames for the lock-on animation to complete


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


class HUDEngine:
    def __init__(self, use_recognition: bool = True) -> None:
        self.recognizer: recognizer.FaceRecognizer | None = None
        if use_recognition and recognizer.models_available():
            self.detector = recognizer.YuNetDetector()
            self.recognizer = recognizer.FaceRecognizer()
        else:
            self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.motion = MotionDetector()
        self.focal_px = distance.load_focal_px()
        self.tick = 0
        self.lock_age = 0
        self._static: StaticFrame | None = None
        self._last_time = time.perf_counter()
        self._enrol: list | None = None       # embeddings being collected, or None

    # --------------------------------------------------------------- identity

    @property
    def recognition_enabled(self) -> bool:
        return self.recognizer is not None

    def known_names(self) -> list[str]:
        return self.recognizer.db.names() if self.recognizer else []

    def start_enrol(self) -> None:
        self._enrol = []

    def enrol_progress(self) -> int:
        return -1 if self._enrol is None else len(self._enrol)

    def enrol_ready(self) -> bool:
        return self._enrol is not None and len(self._enrol) >= recognizer.ENROL_SAMPLES

    def finish_enrol(self, name: str) -> int:
        """Store the collected embeddings under `name`; returns how many were saved."""
        embs, self._enrol = self._enrol or [], None
        if self.recognizer and embs and name.strip():
            self.recognizer.db.add(name, np.stack(embs))
            for tag in self.tracker.tags:      # re-vote so the label updates immediately
                if isinstance(tag, recognizer.IdentityVote):
                    tag.votes.clear()
        return len(embs)

    def cancel_enrol(self) -> None:
        self._enrol = None

    def forget(self, name: str | None = None) -> None:
        if self.recognizer:
            self.recognizer.db.forget(name)
            for tag in self.tracker.tags:
                if isinstance(tag, recognizer.IdentityVote):
                    tag.votes.clear()

    def _detect_and_identify(self, frame: Frame, gray: np.ndarray) -> tuple[list[Box], list[str]]:
        boxes = self.detector.detect(frame)
        # carry identity votes over from the boxes we were tracking (matched by overlap)
        old = list(zip(self.tracker.boxes, self.tracker.tags))
        tags = []
        for box in boxes:
            best = max(old, key=lambda ob: _iou(box, ob[0]), default=None)
            if best is not None and _iou(box, best[0]) > 0.3 and isinstance(best[1], recognizer.IdentityVote):
                tags.append(best[1])
                old.remove(best)
            else:
                tags.append(recognizer.IdentityVote())
        if self.recognizer is not None:
            for i, (row, tag) in enumerate(zip(self.detector.rows, tags)):
                name, score, emb = self.recognizer.identify(frame, row)
                tag.add(name, score)
                if i == 0 and self._enrol is not None and len(self._enrol) < recognizer.ENROL_SAMPLES:
                    self._enrol.append(emb)
        self.tracker.reset(gray, boxes, tags)
        return boxes, [t.name for t in tags]

    def process(self, frame: Frame, state: HUDState) -> Frame:
        """Return the frame with HUD graphics drawn on it.

        Updates state.faces, state.fps and state.frame_type in place.
        """
        self.tick += 1
        self._update_fps(state)
        h, w = frame.shape[:2]
        if self._static is None or self._static.masks["primary"].shape != (h, w):
            self._static = StaticFrame(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.tick % config.DETECT_EVERY == 0 or not self.tracker.active:
            boxes, names = self._detect_and_identify(frame, gray)
            state.frame_type = "I"
        else:
            boxes = self.tracker.update(gray)
            names = [t.name if isinstance(t, recognizer.IdentityVote) else "" for t in self.tracker.tags]
            state.frame_type = "P"
        state.faces = boxes
        state.names = names
        state.enrolling = self.enrol_progress()
        self.lock_age = self.lock_age + 1 if boxes else 0
        lock = min(1.0, self.lock_age / LOCK_FRAMES)

        if state.motion_alert:
            self.motion.update(gray)
        else:
            self.motion.reset()

        # base image
        if state.night_vision:
            base = nightvision.night_vision(frame)
        elif state.thermal:
            base = nightvision.thermal(frame)
        else:
            base = frame

        cv = Canvas(base, self._static.composite, self._static.vignette)

        az = el = dist_cm = None
        if boxes:
            x, y, bw, bh = boxes[0]
            cx, cy = x + bw / 2, y + bh / 2
            az = math.degrees(math.atan((cx - w / 2) / self.focal_px))
            el = math.degrees(math.atan((h / 2 - cy) / self.focal_px))
            dist_cm = distance.estimate_cm(bw, self.focal_px)

            overlay.draw_flow(cv, self.tracker.flow)
            for i, box in enumerate(boxes):
                overlay.draw_reticle(cv, box, self.tick, lock if i == 0 else 1.0, primary=(i == 0))
                overlay.draw_name_tag(cv, box, names[i] if i < len(names) else "",
                                      self.recognition_enabled, self.tick)
            primary_name = names[0] if names else ""
            mode = "TRACK" if state.frame_type == "P" else "DETECT"
            lines = [f"RNG {distance.format_distance(dist_cm)}   AZ {az:+.1f}  EL {el:+.1f}",
                     f"{mode}  {bw}x{bh}px"]
            if self.recognition_enabled:
                lines.insert(0, f"ID {primary_name.upper() if primary_name else 'UNKNOWN'}")
            overlay.draw_target_readout(cv, boxes[0], 0, lines, lock)
        else:
            overlay.draw_scanning(cv, self.tick)
        if state.enrolling >= 0:
            overlay.draw_enrolling(cv, state.enrolling, recognizer.ENROL_SAMPLES, self.tick)

        overlay.draw_azimuth(cv, az)
        overlay.draw_elevation(cv, el)
        overlay.draw_range_bar(cv, dist_cm)
        overlay.draw_reactor(cv, min(1.0, state.fps / 30.0), self.tick)
        overlay.draw_status(cv, state, self.tick)
        overlay.draw_listening(cv, state, self.tick)
        overlay.draw_mode_banner(cv, state)
        if state.radar:
            draw_radar(cv, boxes, self.tick)
        if state.motion_alert:
            self.motion.draw(cv, self.tick)
        return cv.render()

    def _update_fps(self, state: HUDState) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            inst = 1.0 / dt
            state.fps = inst if state.fps == 0 else 0.9 * state.fps + 0.1 * inst
