"""Unit 8 — video fundamentals: motion detection by frame differencing.

|frame_t - frame_{t-1}| on blurred grayscale -> threshold -> dilate -> contours.
The fraction of changed pixels drives a level bar; above a threshold, "MOTION" flashes.
"""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from hud.theme import Canvas

DIFF_THRESHOLD = 25
MIN_CONTOUR_AREA = 400
ALERT_FRACTION = 0.02   # >2 % of pixels changed => alert
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))


class MotionDetector:
    def __init__(self) -> None:
        self.prev: np.ndarray | None = None
        self.level = 0.0
        self.regions: list[tuple[int, int, int, int]] = []

    def update(self, gray: np.ndarray) -> float:
        """Returns the fraction of changed pixels; also fills self.regions with moving boxes."""
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        if self.prev is None:
            self.prev = blurred
            return 0.0
        diff = cv2.absdiff(self.prev, blurred)
        self.prev = blurred
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, DILATE_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        self.level = float(np.count_nonzero(mask)) / mask.size
        return self.level

    def reset(self) -> None:
        self.prev = None
        self.level = 0.0
        self.regions = []

    def draw(self, cv: Canvas, tick: int) -> None:
        """Moving-region boxes, a level bar and the flashing alert, on the HUD canvas."""
        alert = self.level > ALERT_FRACTION
        layer = "alert" if alert else "accent"
        colour = config.HUD_RED if alert else config.HUD_ORANGE
        for (x, y, bw, bh) in self.regions:
            cv.bracket((x, y), (x + bw, y + bh), max(min(bw, bh) // 4, 4), layer, 1)

        bar_w, bar_h = 100, 6
        bx, by = 16, 146          # under the voice indicator, left of the EL scale
        cv.rect((bx, by), (bx + bar_w, by + bar_h), "dim", 1)
        fill = int(min(self.level / (ALERT_FRACTION * 4), 1.0) * bar_w)
        if fill:
            cv.fill_rect((bx, by), (bx + fill, by + bar_h), layer)
        cv.text(f"MOTION {self.level * 100:4.1f}%", (bx, by + bar_h + 15), colour, 0.38)
        if alert and (tick // 6) % 2 == 0:
            cv.text("! MOTION DETECTED !", (cv.w // 2 - 96, 76), config.HUD_RED, 0.6, 2)
