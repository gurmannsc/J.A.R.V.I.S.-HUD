"""Unit 3 — DFT magnitude spectrum.  Unit 2 — frequency-domain filtering.

spectrum : gray -> float32 -> 2-D DFT -> shift DC to centre -> 20*log(1 + |F|) -> normalise
lowpass  : per channel, F(u,v) * H(u,v) with a Gaussian H centred on DC, inverse DFT
highpass : per channel, F(u,v) * (1 - H(u,v)); result added back to the image (sharpening)
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types import Frame, HUDState
from effects._util import tile, to_bgr

LOWPASS_SIGMA_FRAC = 0.05    # Gaussian sigma as a fraction of the shorter image side
HIGHPASS_SIGMA_FRAC = 0.03
SHARPEN_GAIN = 2.0


def dft_shifted(gray: np.ndarray) -> np.ndarray:
    """Centred complex spectrum as an (H, W, 2) float32 array [re, im]."""
    f = cv2.dft(gray.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(f, axes=(0, 1))


def magnitude_image(shifted: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum scaled to uint8 for display."""
    mag = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
    log_mag = 20.0 * np.log1p(mag)
    return cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def gaussian_mask(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """H(u,v) = exp(-D(u,v)^2 / 2 sigma^2), D = distance from the centre (DC)."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    d2 = (u - w / 2) ** 2 + (v - h / 2) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)


def filter_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply the centred spectrum by mask and transform back (float32 image)."""
    f = dft_shifted(channel) * mask[:, :, None]
    f = np.fft.ifftshift(f, axes=(0, 1))
    return cv2.idft(f, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


def spectrum(frame: Frame, state: HUDState) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mag = magnitude_image(dft_shifted(gray))
    spec = cv2.applyColorMap(mag, cv2.COLORMAP_OCEAN)
    return tile(frame.shape, [(frame, "SPATIAL  f(x,y)"), (spec, "DFT  20 log|F(u,v)|  centred")])


def lowpass(frame: Frame, state: HUDState) -> Frame:
    h, w = frame.shape[:2]
    mask = gaussian_mask((h, w), LOWPASS_SIGMA_FRAC * min(h, w))
    channels = [filter_channel(c, mask) for c in cv2.split(frame)]
    out = np.clip(cv2.merge(channels), 0, 255).astype(np.uint8)
    return tile(frame.shape, [(frame, "ORIGINAL  f(x,y)"),
                              (to_bgr((mask * 255).astype(np.uint8)), "H(u,v)  gaussian low-pass"),
                              (out, "IDFT[ F(u,v) H(u,v) ]  smooth")])


def highpass(frame: Frame, state: HUDState) -> Frame:
    h, w = frame.shape[:2]
    mask = 1.0 - gaussian_mask((h, w), HIGHPASS_SIGMA_FRAC * min(h, w))
    detail = [filter_channel(c, mask) for c in cv2.split(frame)]
    sharpened = [np.clip(c.astype(np.float32) + SHARPEN_GAIN * d, 0, 255)
                 for c, d in zip(cv2.split(frame), detail)]
    out = cv2.merge(sharpened).astype(np.uint8)
    return tile(frame.shape, [(frame, "ORIGINAL  f(x,y)"),
                              (to_bgr((mask * 255).astype(np.uint8)), "1 - H(u,v)  high-pass"),
                              (out, f"f + {SHARPEN_GAIN:g} x highpass  sharp")])
