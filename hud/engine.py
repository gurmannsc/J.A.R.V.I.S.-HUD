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

from common import config
from common.types import Frame, HUDState
from hud import distance, nightvision, overlay
from hud.detector import FaceDetector
from hud.frame import StaticFrame
from hud.motion import MotionDetector
from hud.radar import draw_radar
from hud.theme import Canvas
from hud.tracker import FaceTracker

LOCK_FRAMES = 14   # frames for the lock-on animation to complete


class HUDEngine:
    def __init__(self) -> None:
        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.motion = MotionDetector()
        self.focal_px = distance.load_focal_px()
        self.tick = 0
        self.lock_age = 0
        self._static: StaticFrame | None = None
        self._last_time = time.perf_counter()

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
            boxes = self.detector.detect(frame)
            self.tracker.reset(gray, boxes)
            state.frame_type = "I"
        else:
            boxes = self.tracker.update(gray)
            state.frame_type = "P"
        state.faces = boxes
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
            overlay.draw_target_readout(cv, boxes[0], 0, [
                f"RNG {distance.format_distance(dist_cm)}",
                f"AZ {az:+.1f}  EL {el:+.1f}",
                f"{'TRACK' if state.frame_type == 'P' else 'DETECT'}  {bw}x{bh}px",
            ], lock)
        else:
            overlay.draw_scanning(cv, self.tick)

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
