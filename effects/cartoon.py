"""Cartoon effect — Unit 2 (bilateral filter), Unit 7 (k-means colour quantisation),
Unit 5 (adaptive-threshold edge mask).

bilateral x3 (edge-preserving smoothing)  ->  k-means to K colours  ->  AND with edge mask
Heavy steps run at half resolution and are upscaled; the edge mask is computed at full res.
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types import Frame, HUDState
from effects._util import label

K_COLOURS = 8
BILATERAL_PASSES = 3
SATURATION_GAIN = 1.5


def boost_saturation(img: Frame, gain: float) -> Frame:
    """Scale S in HSV so the flat cartoon colours read as colours, not greys (Unit 7)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def quantise(img: Frame, k: int) -> Frame:
    """Replace every pixel by its nearest of k cluster centres in BGR space."""
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centres = cv2.kmeans(data, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    return centres.astype(np.uint8)[labels.ravel()].reshape(img.shape)


def edge_mask(frame: Frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)


def cartoon(frame: Frame, state: HUDState) -> Frame:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    for _ in range(BILATERAL_PASSES):
        small = cv2.bilateralFilter(small, d=9, sigmaColor=75, sigmaSpace=75)
    small = quantise(boost_saturation(small, SATURATION_GAIN), K_COLOURS)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.bitwise_and(smooth, smooth, mask=edge_mask(frame))
    label(out, f"CARTOON  bilateral x{BILATERAL_PASSES} + k-means k={K_COLOURS} + adaptive edges")
    return out
