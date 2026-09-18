"""Unit 5 — edge detection: Canny with automatic thresholds, and Sobel gradient magnitude."""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from common.types import Frame, HUDState
from effects._util import label, to_bgr


def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """Thresholds bracketing the median intensity: lo = (1-σ)·m, hi = (1+σ)·m."""
    m = float(np.median(gray))
    return int(max(0, (1.0 - sigma) * m)), int(min(255, (1.0 + sigma) * m))


def canny(frame: Frame, state: HUDState) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # suppress noise before differentiating
    lo, hi = auto_canny_thresholds(gray)
    edges = cv2.Canny(gray, lo, hi)

    out = (frame * 0.2).astype(np.uint8)  # darkened original as context
    out[edges > 0] = config.HUD_CYAN
    label(out, f"CANNY  lo={lo} hi={hi}")
    return out


def sobel(frame: Frame, state: HUDState) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out = to_bgr(mag)
    label(out, "SOBEL  |G| = sqrt(Gx^2 + Gy^2)")
    return out
