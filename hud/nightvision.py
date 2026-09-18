"""Night-vision and thermal base modes (applied to the whole frame before overlays).

Night vision — Unit 2 + 7: gray -> CLAHE (histogram) -> gamma LUT (point op) -> green tint
                            -> light Gaussian noise for the image-intensifier look.
Thermal      — Unit 7 + 2: gray -> CLAHE -> INFERNO colormap (pseudo-colour).
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types import Frame

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
GAMMA = 0.6
GAMMA_LUT = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)
GREEN = np.array([0.15, 1.0, 0.35], dtype=np.float32)  # BGR multipliers
_rng = np.random.default_rng(0)


def night_vision(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    gray = cv2.LUT(gray, GAMMA_LUT)                       # brighten shadows: s = 255*(r/255)^0.6
    noise = _rng.normal(0, 6, gray.shape).astype(np.float32)
    lit = np.clip(gray.astype(np.float32) + noise, 0, 255)
    out = lit[:, :, None] * GREEN                          # broadcast the tint over 3 channels
    return out.astype(np.uint8)


def thermal(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
