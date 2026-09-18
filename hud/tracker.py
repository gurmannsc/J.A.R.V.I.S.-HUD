"""Unit 8 — motion estimation: Lucas–Kanade optical flow moves face boxes between detections.

Detection frames ("I") seed corner features inside each box; tracked frames ("P") shift each
box by the median displacement of its surviving features — the same predict-from-motion idea
that video codecs use between keyframes.
"""

from __future__ import annotations

import cv2
import numpy as np

from hud.detector import Box

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
FEATURE_PARAMS = dict(maxCorners=40, qualityLevel=0.01, minDistance=5, blockSize=5)


class FaceTracker:
    def __init__(self, min_points: int = 5, max_error: float = 25.0) -> None:
        self.min_points = min_points
        self.max_error = max_error
        self.prev_gray: np.ndarray | None = None
        self.boxes: list[Box] = []
        self.points: list[np.ndarray] = []  # per box: (N, 1, 2) float32
        self.flow: list[tuple[tuple[int, int], tuple[int, int]]] = []  # (from, to) for display

    @property
    def active(self) -> bool:
        return bool(self.boxes)

    def reset(self, gray: np.ndarray, boxes: list[Box]) -> None:
        """Seed features inside each detected box (inner 80 % so we avoid background corners)."""
        self.prev_gray = gray
        self.boxes, self.points, self.flow = [], [], []
        for (x, y, w, h) in boxes:
            mask = np.zeros_like(gray)
            mx, my = int(w * 0.1), int(h * 0.1)
            mask[y + my:y + h - my, x + mx:x + w - mx] = 255
            pts = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
            if pts is not None and len(pts) >= self.min_points:
                self.boxes.append((x, y, w, h))
                self.points.append(pts.astype(np.float32))

    def update(self, gray: np.ndarray) -> list[Box]:
        """Propagate every box one frame forward; boxes with too few survivors are dropped."""
        if self.prev_gray is None or not self.boxes:
            return []
        new_boxes, new_points, flow = [], [], []
        for box, pts in zip(self.boxes, self.points):
            nxt, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **LK_PARAMS)
            if nxt is None:
                continue
            ok = (status.ravel() == 1) & (err.ravel() < self.max_error)
            if ok.sum() < self.min_points:
                continue
            src, dst = pts[ok], nxt[ok]
            dx, dy = np.median(dst - src, axis=0).ravel()
            x, y, w, h = box
            new_boxes.append((int(round(x + dx)), int(round(y + dy)), w, h))
            new_points.append(dst)
            flow.extend(((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
                        for a, b in zip(src.reshape(-1, 2), dst.reshape(-1, 2)))
        self.prev_gray = gray
        self.boxes, self.points, self.flow = new_boxes, new_points, flow
        return list(self.boxes)
