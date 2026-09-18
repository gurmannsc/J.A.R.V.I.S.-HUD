"""Unit 6 — segmentation with GrabCut, Unit 4 — morphological mask cleanup.

The seed rectangle is derived from the tracked face box (head + shoulders); with no face,
a centred rectangle is used. GrabCut runs at half resolution, the mask is cleaned with an
opening (removes speckle) and a closing (fills holes), then feathered and composited.
"""

from __future__ import annotations

import cv2
import numpy as np

from common import config
from common.types import Frame, HUDState
from effects._util import label

GRABCUT_ITERS = 3
SCALE = 0.5
BACKDROP = (45, 25, 8)   # dark HUD blue (BGR)
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def seed_rect(frame_shape: tuple[int, ...], faces: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Head-and-shoulders rectangle around the primary face, clamped to the frame."""
    h, w = frame_shape[:2]
    if faces:
        x, y, fw, fh = faces[0]
        rx, ry = int(x - 0.75 * fw), int(y - 0.5 * fh)
        rw, rh = int(2.5 * fw), int(3.5 * fh)
    else:
        rx, ry, rw, rh = int(0.25 * w), int(0.1 * h), int(0.5 * w), int(0.85 * h)
    rx, ry = max(rx, 1), max(ry, 1)
    rw, rh = min(rw, w - rx - 2), min(rh, h - ry - 2)
    return rx, ry, rw, rh


def grabcut_mask(frame: Frame, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Binary foreground mask (uint8 0/255) at full resolution."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    srect = tuple(int(v * SCALE) for v in rect)
    mask = np.zeros(small.shape[:2], dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(small, mask, srect, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, OPEN_KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)


def remove_bg(frame: Frame, state: HUDState) -> Frame:
    rect = seed_rect(frame.shape, state.faces if state else [])
    fg = grabcut_mask(frame, rect)
    alpha = cv2.GaussianBlur(fg, (11, 11), 0).astype(np.float32)[:, :, None] / 255.0
    backdrop = np.full_like(frame, BACKDROP)
    out = (frame * alpha + backdrop * (1.0 - alpha)).astype(np.uint8)
    x, y, w, h = rect
    cv2.rectangle(out, (x, y), (x + w, y + h), config.HUD_ORANGE, 1)
    label(out, f"GRABCUT {GRABCUT_ITERS} iters @ {SCALE:g}x  + open(5) / close(15)")
    return out
