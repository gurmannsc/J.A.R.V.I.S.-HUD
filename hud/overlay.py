"""HUD widgets drawn onto a theme.Canvas: reticle, target readout, scales, gauges, status."""

from __future__ import annotations

import math

from common import config
from common.types import Frame, HUDState
from hud.detector import Box
from hud.theme import Canvas, Point, put_text, text_width  # noqa: F401  (put_text re-exported)


def _polar(c: Point, r: float, deg: float) -> Point:
    a = math.radians(deg)
    return int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a))


def reticle_radius(cv: Canvas, box: Box, lock: float) -> int:
    """Outer ring radius: scales with the face but is capped so it never swallows the HUD."""
    zoom = 1.0 + 0.6 * (1.0 - lock)
    return int(min(0.66 * max(box[2], box[3]), 0.30 * cv.h) * zoom)


# ------------------------------------------------------------------ reticle

def draw_reticle(cv: Canvas, box: Box, tick: int, lock: float, primary: bool = True) -> None:
    """Layered targeting reticle around a face box.

    lock: 0..1 lock-on progress. Rings start large and dim, shrink onto the target and
    brighten as the tracker confirms it (Unit 8: the box is carried by optical flow).
    """
    x, y, w, h = box
    c = (x + w // 2, y + h // 2)
    r_outer = reticle_radius(cv, box, lock)
    r_mid = int(r_outer * 0.86)
    r_inner = int(r_outer * 0.70)
    layer = "primary" if primary else "dim"
    seg_layer = "bright" if (primary and lock >= 1.0) else layer

    # outer ring: six rotating segments with small gaps
    a0 = (tick * 1.5) % 360
    for k in range(6):
        cv.arc(c, r_outer, a0 + k * 60, a0 + k * 60 + 42, layer, 2)
    # counter-rotating tick ring
    b0 = (-tick * 2.5) % 360
    for k in range(36):
        long = k % 9 == 0
        p0 = _polar(c, r_mid, b0 + k * 10)
        p1 = _polar(c, r_mid + (7 if long else 3), b0 + k * 10)
        cv.line(p0, p1, layer if long else "dim", 1)
    # inner ring with four gaps and cardinal notches
    for k in range(4):
        cv.arc(c, r_inner, k * 90 + 8, k * 90 + 82, seg_layer, 1)
        cv.line(_polar(c, r_inner - 6, k * 90), _polar(c, r_inner + 6, k * 90), seg_layer, 1)
    # crosshair with a gap
    gap, length = max(w // 9, 5), max(w // 5, 8)
    cv.line((c[0] - gap - length, c[1]), (c[0] - gap, c[1]), seg_layer, 1)
    cv.line((c[0] + gap, c[1]), (c[0] + gap + length, c[1]), seg_layer, 1)
    cv.line((c[0], c[1] - gap - length), (c[0], c[1] - gap), seg_layer, 1)
    cv.line((c[0], c[1] + gap), (c[0], c[1] + gap + length), seg_layer, 1)
    cv.dot(c, 2, "bright")
    # corner brackets on the face box itself
    pad = max(w // 12, 4)
    cv.bracket((x - pad, y - pad), (x + w + pad, y + h + pad), max(w // 5, 8), layer, 2 if primary else 1)

    if lock < 1.0:  # sweeping lock-on arc
        cv.arc(c, r_outer + 8, -90, -90 + 360 * lock, "accent", 2)


def draw_target_readout(cv: Canvas, box: Box, index: int, lines: list[str], lock: float) -> None:
    """Leader line from the box to a small data cluster (top-right of the target)."""
    x, y, w, h = box
    r = reticle_radius(cv, box, lock)
    c = (x + w // 2, y + h // 2)
    header = f"TGT {index + 1:02d}  " + ("LOCKED" if lock >= 1.0 else "LOCKING")
    block_w = max(text_width(s, 0.42) for s in [header, *lines]) + 10
    block_h = 15 * (len(lines) + 1) + 4

    # data cluster sits in a fixed clear spot (top-right, under the azimuth strip);
    # a leader line runs from the reticle rim to its lower-left corner
    bx, by = cv.w - block_w - 18, 66
    anchor = (bx - 6, by + block_h)
    p0 = _polar(c, r + 6, math.degrees(math.atan2(anchor[1] - c[1], anchor[0] - c[0])))
    elbow = (anchor[0] - 18, anchor[1])
    cv.line(p0, elbow, "primary", 1)
    cv.line(elbow, anchor, "primary", 1)
    cv.dot(p0, 2, "bright")
    cv.line((bx - 6, by - 4), (bx - 6, by + block_h), "primary", 1)
    cv.line((bx - 6, by - 4), (bx + 14, by - 4), "primary", 1)
    cv.text(header, (bx, by + 10), config.HUD_WHITE if lock >= 1.0 else config.HUD_ORANGE, 0.42)
    for i, s in enumerate(lines):
        cv.text(s, (bx, by + 10 + 15 * (i + 1)), config.HUD_CYAN, 0.38)


def draw_flow(cv: Canvas, flow: list[tuple[Point, Point]]) -> None:
    """Tracked feature points and their motion vectors (Unit 8)."""
    for (x0, y0), (x1, y1) in flow:
        cv.dot((x1, y1), 1, "bright")
        if abs(x1 - x0) + abs(y1 - y0) >= 2:
            cv.line((x0, y0), (x1, y1), "primary", 1)


def draw_scanning(cv: Canvas, tick: int) -> None:
    """Idle reticle in the centre while no target is locked."""
    c = (cv.w // 2, cv.h // 2)
    r = int(min(cv.w, cv.h) * 0.22)
    a0 = (tick * 1.2) % 360
    for k in range(3):
        cv.arc(c, r, a0 + k * 120, a0 + k * 120 + 70, "dim", 1)
    b0 = (-tick * 2) % 360
    for k in range(4):
        cv.arc(c, int(r * 0.8), b0 + k * 90, b0 + k * 90 + 30, "primary", 1)
    cv.line((c[0] - 14, c[1]), (c[0] + 14, c[1]), "dim", 1)
    cv.line((c[0], c[1] - 14), (c[0], c[1] + 14), "dim", 1)
    # horizontal sweep bar
    y = int(cv.h * 0.30 + (cv.h * 0.40) * (0.5 + 0.5 * math.sin(tick / 18.0)))
    cv.line((c[0] - r, y), (c[0] + r, y), "primary", 1)
    dots = "." * (1 + (tick // 10) % 3)
    cv.text(f"SCANNING{dots}", (c[0] - 42, c[1] + r + 24), config.HUD_ORANGE, 0.5)


# ------------------------------------------------------------------- scales

def draw_azimuth(cv: Canvas, az_deg: float | None) -> None:
    """Marker on the top strip: horizontal bearing of the primary target."""
    x0, x1, y = int(cv.w * 0.30), int(cv.w * 0.70), 26
    for i, deg in enumerate(range(-20, 21, 10)):
        x = x0 + (x1 - x0) * (i / 4)
        cv.text(f"{deg:+d}" if deg else "0", (int(x) - 8, y + 24), config.HUD_CYAN_DIM, 0.34)
    if az_deg is None:
        return
    x = int(cv.w / 2 + max(-1.0, min(1.0, az_deg / 20.0)) * (x1 - x0) / 2)
    cv.polyline([(x - 6, y + 14), (x, y + 6), (x + 6, y + 14)], "accent", 1)
    cv.text(f"AZ {az_deg:+.1f}", (x - 22, y + 38), config.HUD_ORANGE, 0.38)


def draw_elevation(cv: Canvas, el_deg: float | None) -> None:
    """Vertical scale on the left: elevation of the primary target."""
    x, y0, y1 = int(cv.w * 0.215), int(cv.h * 0.30), int(cv.h * 0.70)
    cv.line((x, y0), (x, y1), "primary", 1)
    for i in range(0, 9):
        y = y0 + (y1 - y0) * i / 8
        long = i % 2 == 0
        cv.line((x, int(y)), (x + (8 if long else 4), int(y)), "primary" if long else "dim", 1)
    cv.text("EL", (x - 4, y0 - 8), config.HUD_CYAN_DIM, 0.36)
    if el_deg is None:
        return
    y = int(cv.h / 2 - max(-1.0, min(1.0, el_deg / 15.0)) * (y1 - y0) / 2)
    cv.polyline([(x - 12, y - 5), (x - 4, y), (x - 12, y + 5)], "accent", 1)
    cv.text(f"{el_deg:+.1f}", (x - 46, y + 4), config.HUD_ORANGE, 0.36)


def draw_range_bar(cv: Canvas, dist_cm: float | None, max_cm: float = 200.0) -> None:
    """Vertical bar on the right: range to the primary target."""
    x, y0, y1 = int(cv.w * 0.785), int(cv.h * 0.30), int(cv.h * 0.70)
    cv.line((x, y0), (x, y1), "primary", 1)
    for i in range(0, 5):
        y = int(y0 + (y1 - y0) * i / 4)
        cv.line((x - 8, y), (x, y), "dim", 1)
        cv.text(f"{int(max_cm - max_cm * i / 4)}", (x + 6, y + 4), config.HUD_CYAN_DIM, 0.32)
    cv.text("RNG cm", (x - 14, y0 - 8), config.HUD_CYAN_DIM, 0.36)
    if dist_cm is None:
        return
    frac = max(0.0, min(1.0, dist_cm / max_cm))
    y = int(y1 - frac * (y1 - y0))
    cv.fill_rect((x - 5, y), (x - 1, y1), "primary")
    cv.polyline([(x + 12, y - 5), (x + 4, y), (x + 12, y + 5)], "accent", 1)


# ------------------------------------------------------------------- gauges

def draw_reactor(cv: Canvas, level: float, tick: int, label: str = "SYS") -> None:
    """Arc-reactor style gauge (bottom-right): segmented ring filled to `level` (0..1)."""
    c = (cv.w - 62, cv.h - 66)
    r = 34
    cv.circle(c, r + 8, "dim", 1)
    cv.circle(c, r - 12, "dim", 1)
    n = 24
    for k in range(n):
        a = -90 + k * (360 / n)
        lit = k / n < level
        cv.arc(c, r, a + 2, a + 360 / n - 4, "primary" if lit else "dim", 4 if lit else 1)
    # inner triangle (the reactor core) rotating slowly
    a0 = (tick * 0.6) % 360
    pts = [_polar(c, r - 18, a0 + k * 120) for k in range(3)]
    cv.polyline(pts, "bright", 1, closed=True)
    cv.dot(c, 3, "bright")
    cv.text(f"{label} {level * 100:3.0f}%", (c[0] - 26, c[1] + r + 24), config.HUD_CYAN, 0.4)


def draw_status(cv: Canvas, state: HUDState, tick: int) -> None:
    """Top-left status cluster with a bracket and a live heartbeat bar."""
    x, y = 14, 16
    modes = [m for m, on in (("NV", state.night_vision), ("THERMAL", state.thermal),
                             ("RADAR", state.radar), ("MOTION", state.motion_alert)) if on]
    lines = [
        "J.A.R.V.I.S.  ONLINE",
        f"FPS {state.fps:5.1f}   {cv.w}x{cv.h}",
        f"FRAME {state.frame_type}     TGT {len(state.faces)}",
        "MODE " + (" ".join(modes) if modes else "-"),
    ]
    width = max(text_width(s, 0.42) for s in lines) + 18
    height = 16 * len(lines) + 12
    cv.fill_rect((x, y), (x + width, y + height), "dim")
    cv.line((x, y), (x, y + height), "primary", 2)
    cv.line((x, y), (x + 18, y), "primary", 1)
    cv.line((x, y + height), (x + 18, y + height), "primary", 1)
    for i, s in enumerate(lines):
        cv.text(s, (x + 10, y + 16 * (i + 1)), config.HUD_WHITE if i == 0 else config.HUD_CYAN, 0.42)
    # heartbeat: a little animated bar graph under the panel
    for k in range(12):
        hgt = 2 + int(6 * (0.5 + 0.5 * math.sin(tick / 4.0 + k * 0.9)))
        cv.line((x + 8 + k * 7, y + height + 12), (x + 8 + k * 7, y + height + 12 - hgt), "primary", 2)


def draw_listening(cv: Canvas, state: HUDState, tick: int) -> None:
    """Voice status under the status panel: pulsing ring + countdown while JARVIS is armed."""
    x, y = 24, 118
    if state.listening > 0:
        pulse = 4 + int(3 * (0.5 + 0.5 * math.sin(tick / 3.0)))
        cv.circle((x, y), pulse, "accent", 1)
        cv.dot((x, y), 2, "accent")
        label = "TYPED INPUT" if math.isinf(state.listening) else f"LISTENING  {state.listening:.0f}s"
        cv.text(label, (x + 14, y + 5), config.HUD_ORANGE, 0.42)
    else:
        cv.circle((x, y), 4, "dim", 1)
        cv.text("SAY 'HEY JARVIS'", (x + 14, y + 5), config.HUD_CYAN_DIM, 0.38)


def draw_mode_banner(cv: Canvas, state: HUDState) -> None:
    label = "NIGHT VISION" if state.night_vision else "THERMAL" if state.thermal else None
    if label:
        cv.text(label, (cv.w // 2 - text_width(label, 0.5) // 2, cv.h - 52), config.HUD_ORANGE, 0.5)
