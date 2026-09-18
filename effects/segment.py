"""Unit 6 + 7 — colour segmentation: skin detection by thresholding in HSV.
Unit 4 — the raw mask is cleaned with an opening then a closing.

Skin sits in a narrow hue band (red-orange) with moderate saturation; HSV separates that
from brightness, so the same range works across lighting far better than RGB thresholds.
"""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from common.types import Frame, HUDState
from effects._util import tile, to_bgr

SKIN_LOWER = np.array([0, 50, 60], dtype=np.uint8)     # H, S, V
SKIN_UPPER = np.array([20, 255, 255], dtype=np.uint8)
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def skin_mask(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """(raw inRange mask, cleaned mask) as uint8 0/255."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, OPEN_KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return raw, clean


def skin(frame: Frame, state: HUDState) -> Frame:
    raw, clean = skin_mask(frame)
    dim = (frame * 0.3).astype(np.uint8)
    tint = np.full_like(frame, config.HUD_ORANGE)
    highlighted = cv2.addWeighted(frame, 0.6, tint, 0.4, 0)
    out = np.where(clean[:, :, None] > 0, highlighted, dim)
    pct = 100.0 * np.count_nonzero(clean) / clean.size
    lo, hi = SKIN_LOWER.tolist(), SKIN_UPPER.tolist()
    return tile(frame.shape, [(out, f"SKIN  {pct:.1f}% of frame"),
                              (to_bgr(raw), f"HSV inRange {lo}..{hi}"),
                              (to_bgr(clean), "open(5) + close(9)")])
