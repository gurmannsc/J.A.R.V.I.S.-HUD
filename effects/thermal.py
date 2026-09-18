"""Unit 7 — pseudo-colour: one-shot thermal view (same transform as the live HUD toggle)."""

from __future__ import annotations

from common.types import Frame, HUDState
from effects._util import label
from hud.nightvision import thermal as _thermal


def thermal(frame: Frame, state: HUDState) -> Frame:
    out = _thermal(frame)
    label(out, "THERMAL  CLAHE + COLORMAP_INFERNO")
    return out
