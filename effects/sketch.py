"""Unit 2 — spatial filtering: pencil sketch via Gaussian blur + colour dodge.

sketch = gray / (255 - blur(255 - gray))   (the "dodge" blend, scaled to 0..255)
Flat regions -> white; regions where the blur differs from the pixel (edges) -> dark strokes.
"""

from __future__ import annotations

import cv2

from common.types import Frame, HUDState
from effects._util import label, to_bgr

BLUR_KSIZE = (21, 21)


def sketch(frame: Frame, state: HUDState) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, BLUR_KSIZE, 0)
    dodged = cv2.divide(gray, 255 - blurred, scale=256.0)
    out = to_bgr(dodged)
    label(out, f"SKETCH  gaussian {BLUR_KSIZE[0]}x{BLUR_KSIZE[1]} + dodge", colour=(60, 60, 60))
    return out
