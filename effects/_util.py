"""Small helpers shared by the effects."""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from common.types import Frame

FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_bgr(gray: np.ndarray) -> Frame:
    """Single-channel -> 3-channel BGR uint8 (the contract requires 3 channels)."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def label(frame: Frame, text: str, org: tuple[int, int] = (12, 24),
          colour: tuple[int, int, int] = config.HUD_CYAN, scale: float = 0.55) -> None:
    """In place: outlined caption."""
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, colour, 1, cv2.LINE_AA)


def tile(frame_shape: tuple[int, ...], panels: list[tuple[Frame, str]]) -> Frame:
    """Lay `panels` out on a dark canvas of the original frame size.

    Two panels go side by side; three or four use a 2x2 grid (bigger cells than one row).
    Each panel keeps its aspect ratio, so the result is a Frame with the input's shape.
    """
    h, w = frame_shape[:2]
    n = len(panels)
    cols = 2 if n in (3, 4) else n
    rows = -(-n // cols)
    cw, ch = w // cols, h // rows
    canvas = np.full((h, w, 3), 12, dtype=np.uint8)
    for i, (img, text) in enumerate(panels):
        ih, iw = img.shape[:2]
        s = min(cw / iw, ch / ih)
        rw, rh = int(iw * s), int(ih * s)
        r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
        x = (i % cols) * cw + (cw - rw) // 2
        y = (i // cols) * ch + (ch - rh) // 2
        canvas[y:y + rh, x:x + rw] = r
        cv2.rectangle(canvas, (x, y), (x + rw - 1, y + rh - 1), config.HUD_CYAN, 1)
        label(canvas, text, (x + 8, y + 20), scale=0.5)
    return canvas
