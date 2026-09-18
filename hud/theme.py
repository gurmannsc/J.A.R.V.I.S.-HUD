"""Holographic compositor for the HUD.

All line art for a frame is drawn into a few single-channel masks (one per colour layer),
then composited onto the video in a single vectorised pass:

    alpha  = line_mask * a_line + blur(line_mask) * a_glow        (per layer)
    out    = frame * vignette * (1 - sum(alpha)) + sum(alpha * colour)

Unit 1: this is per-pixel alpha blending; the glow is a Gaussian-blurred copy of the
line mask, so every stroke gets the soft halo of a projected hologram.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common import config
from common.types import Frame

FONT = cv2.FONT_HERSHEY_SIMPLEX
Point = tuple[int, int]


@dataclass(frozen=True)
class Layer:
    colour: tuple[int, int, int]
    alpha: float        # opacity of the stroke itself
    glow: float         # opacity of the blurred halo (0 = no glow)
    glow_sigma: float = 3.0


LAYERS: dict[str, Layer] = {
    "dim": Layer(config.HUD_CYAN_DIM, 0.35, 0.0),
    "primary": Layer(config.HUD_CYAN, 0.92, 0.45),
    "bright": Layer(config.HUD_WHITE, 0.95, 0.35, 2.0),
    "accent": Layer(config.HUD_ORANGE, 0.95, 0.40),
    "alert": Layer(config.HUD_RED, 0.95, 0.45),
}


Composite = tuple[np.ndarray, list[np.ndarray]]   # (alpha HxW uint8, [b, g, r] HxW uint8 premultiplied)


def composite_layers(masks: dict[str, np.ndarray]) -> Composite:
    """Turn per-layer masks into a summed 8-bit alpha map and premultiplied colour planes.

    Everything is saturating uint8 arithmetic in OpenCV (SIMD), which is far cheaper than
    float32 passes and indistinguishable on screen.
    """
    first = next(iter(masks.values()))
    h, w = first.shape
    alpha = np.zeros((h, w), dtype=np.uint8)
    colour = [np.zeros((h, w), dtype=np.uint8) for _ in range(3)]
    for name, m in masks.items():
        if not m.any():
            continue
        spec = LAYERS[name]
        a = cv2.convertScaleAbs(m, alpha=spec.alpha)
        if spec.glow > 0:
            g = cv2.convertScaleAbs(cv2.GaussianBlur(m, (0, 0), spec.glow_sigma), alpha=spec.glow)
            a = cv2.max(a, g)
        alpha = cv2.add(alpha, a)
        for ch in range(3):
            colour[ch] = cv2.add(colour[ch], cv2.convertScaleAbs(a, alpha=spec.colour[ch] / 255.0))
    return alpha, colour


class Canvas:
    """Collects strokes per layer and text, then renders them onto a frame.

    `static` is a precomputed Composite (see composite_layers) of furniture that never
    changes, so its strokes and glow cost nothing per frame.
    """

    def __init__(self, frame: Frame, static: Composite | None = None,
                 vignette: np.ndarray | None = None) -> None:
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.masks: dict[str, np.ndarray] = {}
        self.texts: list[tuple[str, Point, tuple[int, int, int], float, int]] = []
        self.static = static
        self.vignette = vignette

    # ------------------------------------------------------------- drawing

    def mask(self, layer: str) -> np.ndarray:
        m = self.masks.get(layer)
        if m is None:
            m = self.masks[layer] = np.zeros((self.h, self.w), dtype=np.uint8)
        return m

    def line(self, p0: Point, p1: Point, layer: str = "primary", t: int = 1) -> None:
        cv2.line(self.mask(layer), p0, p1, 255, t, cv2.LINE_AA)

    def polyline(self, pts: list[Point], layer: str = "primary", t: int = 1, closed: bool = False) -> None:
        cv2.polylines(self.mask(layer), [np.array(pts, dtype=np.int32)], closed, 255, t, cv2.LINE_AA)

    def circle(self, c: Point, r: int, layer: str = "primary", t: int = 1) -> None:
        cv2.circle(self.mask(layer), c, max(r, 1), 255, t, cv2.LINE_AA)

    def dot(self, c: Point, r: int = 2, layer: str = "bright") -> None:
        cv2.circle(self.mask(layer), c, r, 255, -1, cv2.LINE_AA)

    def arc(self, c: Point, r: int, start: float, end: float, layer: str = "primary", t: int = 1,
            axes: tuple[int, int] | None = None, rotation: float = 0.0) -> None:
        ax = axes if axes is not None else (r, r)
        cv2.ellipse(self.mask(layer), c, ax, rotation, start, end, 255, t, cv2.LINE_AA)

    def rect(self, p0: Point, p1: Point, layer: str = "primary", t: int = 1) -> None:
        cv2.rectangle(self.mask(layer), p0, p1, 255, t, cv2.LINE_AA)

    def fill_rect(self, p0: Point, p1: Point, layer: str = "dim") -> None:
        cv2.rectangle(self.mask(layer), p0, p1, 255, -1)

    def bracket(self, p0: Point, p1: Point, arm: int, layer: str = "primary", t: int = 1) -> None:
        """Four L-shaped corners around the rectangle p0..p1."""
        (x0, y0), (x1, y1) = p0, p1
        for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.line((px, py), (px + dx * arm, py), layer, t)
            self.line((px, py), (px, py + dy * arm), layer, t)

    def text(self, s: str, org: Point, colour: tuple[int, int, int] = config.HUD_CYAN,
             scale: float = 0.42, t: int = 1) -> None:
        self.texts.append((s, org, colour, scale, t))

    # ------------------------------------------------------------- render

    def render(self) -> Frame:
        if self.masks:
            alpha, colour = composite_layers(self.masks)
            if self.static is not None:
                alpha = cv2.add(alpha, self.static[0])
                colour = [cv2.add(colour[ch], self.static[1][ch]) for ch in range(3)]
        elif self.static is not None:
            alpha, colour = self.static
        else:
            return self.frame.copy()

        # out = frame * vignette * (1 - alpha) + colour, channel by channel, all uint8
        keep = cv2.bitwise_not(alpha)                       # 255 - alpha
        if self.vignette is not None:
            keep = cv2.multiply(keep, self.vignette, scale=1.0 / 255.0)
        channels = [cv2.add(cv2.multiply(src, keep, scale=1.0 / 255.0), colour[ch])
                    for ch, src in enumerate(cv2.split(self.frame))]
        out = cv2.merge(channels)

        for s, org, col, scale, t in self.texts:
            put_text(out, s, org, col, scale, t)
        return out


def put_text(frame: Frame, text: str, org: Point, colour: tuple[int, int, int] = config.HUD_CYAN,
             scale: float = 0.42, thickness: int = 1) -> None:
    """In place: dark-outlined caption (readable over any background)."""
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float = 0.42, thickness: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]
