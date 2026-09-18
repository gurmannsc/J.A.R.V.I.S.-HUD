"""Unit 2 — point processing (negative) + histogram processing (CLAHE); Unit 7 (LAB colour space).

negative -> CLAHE on the L channel in LAB -> blue-cyan tint
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types import Frame, HUDState
from effects._util import label

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
TINT = np.array([1.0, 0.85, 0.6], dtype=np.float32)  # BGR multipliers -> cold blue cast


def xray(frame: Frame, state: HUDState) -> Frame:
    negative = 255 - frame                                  # s = L - 1 - r
    lab = cv2.cvtColor(negative, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)                                      # local contrast on lightness only
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    out = np.clip(enhanced.astype(np.float32) * TINT, 0, 255).astype(np.uint8)
    label(out, "X-RAY  negative + CLAHE(L)")
    return out
