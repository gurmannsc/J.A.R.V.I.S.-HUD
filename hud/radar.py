"""Radar widget (bottom-left): rings, rotating sweep with a fading wedge, target blips."""

from __future__ import annotations

import math

from common import config
from hud.detector import Box
from hud.theme import Canvas

RADIUS = 46
MARGIN = 22
SWEEP_DEG_PER_TICK = 5
WEDGE_DEG = 40


def draw_radar(cv: Canvas, boxes: list[Box], tick: int) -> None:
    c = (MARGIN + RADIUS, cv.h - MARGIN - RADIUS)
    cv.fill_rect((c[0] - RADIUS - 4, c[1] - RADIUS - 4), (c[0] + RADIUS + 4, c[1] + RADIUS + 4), "dim")
    cv.circle(c, RADIUS, "primary", 1)
    cv.circle(c, RADIUS * 2 // 3, "dim", 1)
    cv.circle(c, RADIUS // 3, "dim", 1)
    cv.line((c[0] - RADIUS, c[1]), (c[0] + RADIUS, c[1]), "dim", 1)
    cv.line((c[0], c[1] - RADIUS), (c[0], c[1] + RADIUS), "dim", 1)
    for k in range(12):
        a = math.radians(k * 30)
        p0 = (int(c[0] + RADIUS * math.cos(a)), int(c[1] + RADIUS * math.sin(a)))
        p1 = (int(c[0] + (RADIUS + 4) * math.cos(a)), int(c[1] + (RADIUS + 4) * math.sin(a)))
        cv.line(p0, p1, "primary", 1)

    angle = (tick * SWEEP_DEG_PER_TICK) % 360
    # fading wedge behind the sweep line: a few thin arcs of decreasing radius
    for i in range(1, 7):
        cv.arc(c, RADIUS - 2, angle - WEDGE_DEG * i / 6, angle - WEDGE_DEG * (i - 1) / 6, "dim", 1,
               axes=(RADIUS - 2 - i, RADIUS - 2 - i))
    tip = (int(c[0] + RADIUS * math.cos(math.radians(angle))), int(c[1] + RADIUS * math.sin(math.radians(angle))))
    cv.line(c, tip, "bright", 1)

    for i, (x, y, bw, bh) in enumerate(boxes):
        nx = (x + bw / 2) / cv.w * 2 - 1
        ny = (y + bh / 2) / cv.h * 2 - 1
        b = (int(c[0] + nx * RADIUS * 0.85), int(c[1] + ny * RADIUS * 0.85))
        pulse = 3 + (tick // 4) % 3
        cv.circle(b, pulse, "accent" if i == 0 else "primary", 1)
        cv.dot(b, 2, "accent" if i == 0 else "primary")

    cv.text("RADAR", (c[0] - 20, c[1] - RADIUS - 10), config.HUD_CYAN_DIM, 0.36)
