"""Unit 2 — histogram equalisation, shown with before/after histograms.

Equalising only the Y (luma) channel of YCrCb stretches contrast without shifting colours.
"""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from common.types import Frame, HUDState
from effects._util import tile

STRIP_H = 90


def luma_histogram(frame: Frame) -> np.ndarray:
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return cv2.calcHist([y], [0], None, [256], [0, 256]).ravel()


def with_histogram_strip(frame: Frame, hist: np.ndarray, colour: tuple[int, int, int]) -> Frame:
    """Copy of frame with a dark strip along the bottom showing the 256-bin luma histogram."""
    out = frame.copy()
    h, w = out.shape[:2]
    strip = out[h - STRIP_H:h]
    strip[:] = (strip * 0.25).astype(np.uint8)
    peak = hist.max() or 1.0
    bar_w = w / 256.0
    for i, v in enumerate(hist):
        bh = int(v / peak * (STRIP_H - 12))
        x0, x1 = int(i * bar_w), max(int((i + 1) * bar_w) - 1, int(i * bar_w))
        cv2.rectangle(out, (x0, h - 4 - bh), (x1, h - 4), colour, -1)
    for q in (64, 128, 192):
        x = int(q * bar_w)
        cv2.line(out, (x, h - STRIP_H), (x, h), (90, 90, 90), 1)
    return out


def equalise(frame: Frame) -> Frame:
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def enhance(frame: Frame, state: HUDState) -> Frame:
    eq = equalise(frame)
    before = with_histogram_strip(frame, luma_histogram(frame), config.HUD_ORANGE)
    after = with_histogram_strip(eq, luma_histogram(eq), config.HUD_CYAN)
    return tile(frame.shape, [(before, "BEFORE  luma histogram"), (after, "AFTER  equalizeHist(Y)")])
