"""Static helmet-view furniture, built once per frame size.

Curved side brackets that frame the eye-slit view, a faint hexagonal grid, the top azimuth
strip and bottom rail, and a soft vignette. Everything here is a mask that the Canvas
copies at the start of every frame, so the cost is paid once.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from hud.theme import composite_layers

HEX_RADIUS = 26


def _hex_grid(mask: np.ndarray, r: int) -> None:
    h, w = mask.shape
    dx, dy = 1.5 * r, math.sqrt(3) * r
    row = 0
    y = -r
    while y < h + r:
        x = -r if row % 2 == 0 else -r + dx
        while x < w + r:
            pts = [(int(x + r * math.cos(math.radians(60 * k))), int(y + r * math.sin(math.radians(60 * k))))
                   for k in range(6)]
            cv2.polylines(mask, [np.array(pts, dtype=np.int32)], True, 255, 1, cv2.LINE_AA)
            x += 2 * dx
        y += dy / 2
        row += 1


def _side_arcs(primary: np.ndarray, dim: np.ndarray, w: int, h: int) -> None:
    """Large arcs hugging the left and right edges, with tick marks and end caps."""
    ax, ay = int(w * 0.50), int(h * 0.60)
    for side in (-1, 1):
        cx = int(w * 0.5 + side * w * 0.80)          # ellipse centre far off-screen
        centre = (cx, h // 2)
        base = 180 if side == 1 else 0                # which half of the ellipse faces inward
        # outer arc (primary) and inner arc (dim)
        cv2.ellipse(primary, centre, (ax, ay), 0, base - 38, base + 38, 255, 1, cv2.LINE_AA)
        cv2.ellipse(dim, centre, (int(ax * 0.955), int(ay * 0.955)), 0, base - 34, base + 34, 255, 1, cv2.LINE_AA)
        # ticks along the outer arc
        for deg in range(-36, 37, 4):
            ang = math.radians(base + deg)
            long = deg % 12 == 0
            r0, r1 = (1.0, 1.035) if long else (1.0, 1.018)
            p0 = (int(cx + ax * r0 * math.cos(ang)), int(h / 2 + ay * r0 * math.sin(ang)))
            p1 = (int(cx + ax * r1 * math.cos(ang)), int(h / 2 + ay * r1 * math.sin(ang)))
            cv2.line(primary if long else dim, p0, p1, 255, 1, cv2.LINE_AA)
        # end caps: short horizontal bars where the arc terminates
        for deg in (-38, 38):
            ang = math.radians(base + deg)
            p = (int(cx + ax * math.cos(ang)), int(h / 2 + ay * math.sin(ang)))
            cv2.line(primary, p, (p[0] - side * 14, p[1]), 255, 2, cv2.LINE_AA)


def _rails(primary: np.ndarray, dim: np.ndarray, w: int, h: int) -> None:
    """Top azimuth strip and bottom rail."""
    x0, x1, y = int(w * 0.30), int(w * 0.70), 26
    cv2.line(primary, (x0, y), (x1, y), 255, 1, cv2.LINE_AA)
    for x in range(x0, x1 + 1, 16):
        long = (x - x0) % 80 == 0
        cv2.line(primary if long else dim, (x, y), (x, y + (8 if long else 4)), 255, 1, cv2.LINE_AA)
    cv2.line(primary, (x0, y), (x0, y + 14), 255, 2, cv2.LINE_AA)
    cv2.line(primary, (x1, y), (x1, y + 14), 255, 2, cv2.LINE_AA)
    # centre index for the azimuth strip
    cv2.polylines(primary, [np.array([(w // 2 - 6, y - 10), (w // 2, y - 2), (w // 2 + 6, y - 10)], np.int32)],
                  False, 255, 1, cv2.LINE_AA)

    yb = h - 44
    cv2.line(dim, (int(w * 0.22), yb), (int(w * 0.78), yb), 255, 1, cv2.LINE_AA)
    cv2.line(primary, (int(w * 0.40), yb), (int(w * 0.60), yb), 255, 1, cv2.LINE_AA)
    cv2.line(primary, (int(w * 0.40), yb), (int(w * 0.40), yb - 8), 255, 1, cv2.LINE_AA)
    cv2.line(primary, (int(w * 0.60), yb), (int(w * 0.60), yb - 8), 255, 1, cv2.LINE_AA)


def _vignette(w: int, h: int) -> np.ndarray:
    """Elliptical falloff: 1.0 in the middle, ~0.45 in the corners (helmet eye-slit feel)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    d = np.sqrt(nx ** 2 * 1.0 + ny ** 2 * 1.3)
    v = 1.0 - 0.55 * np.clip((d - 0.55) / 0.75, 0.0, 1.0) ** 1.6
    return (v * 255).astype(np.uint8)      # 8-bit multiplier map for the compositor


class StaticFrame:
    def __init__(self, w: int, h: int) -> None:
        primary = np.zeros((h, w), dtype=np.uint8)
        dim = np.zeros((h, w), dtype=np.uint8)
        grid = np.zeros((h, w), dtype=np.uint8)
        _hex_grid(grid, HEX_RADIUS)
        # keep the grid faint and off the centre of view: fade it towards the middle
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        grid = (grid.astype(np.float32) * np.clip((d - 0.45) / 0.6, 0.0, 1.0) * 0.5).astype(np.uint8)
        _side_arcs(primary, dim, w, h)
        _rails(primary, dim, w, h)
        self.masks = {"primary": primary, "dim": np.maximum(dim, grid)}
        self.composite = composite_layers(self.masks)   # alpha + colour, glow included, once
        self.vignette = _vignette(w, h)
