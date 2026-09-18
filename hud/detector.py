"""Haar-cascade face detection (Unit 6: detection as region-of-interest segmentation).

Runs on a downscaled grayscale copy for speed; boxes are scaled back to frame coordinates.
"""

from __future__ import annotations

import cv2

from common.types import Frame

Box = tuple[int, int, int, int]  # x, y, w, h in full-frame pixels


class FaceDetector:
    def __init__(self, scale: float = 0.5, min_size: int = 40) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {path}")
        self.scale = scale
        self.min_size = min_size

    def detect(self, frame: Frame) -> list[Box]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = cv2.equalizeHist(small)  # Unit 2: histogram equalisation helps Haar in low light
        min_px = max(int(self.min_size * self.scale), 20)
        found = self.cascade.detectMultiScale(
            small, scaleFactor=1.1, minNeighbors=5, minSize=(min_px, min_px)
        )
        inv = 1.0 / self.scale
        boxes = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in found]
        # largest first so the primary target is stable
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes
