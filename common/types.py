"""Shared types — the interface contract between hud/, voice/ and effects/.

Everything else depends on this file; change it deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# A video frame: shape (H, W, 3), dtype uint8, BGR channel order. Always.
# Effects that work in grayscale must convert back to 3 channels before returning.
Frame = np.ndarray


class CommandKind(Enum):
    TOGGLE = "toggle"    # flips a HUDState field, persists until toggled again
    ONESHOT = "oneshot"  # applies an effect to the current frame, then holds the result
    SYSTEM = "system"    # snapshot, reset, help, quit


@dataclass
class Command:
    name: str        # canonical id: "edges", "night_vision", "quit", ...
    kind: CommandKind
    raw_text: str    # what the user actually said / typed


@dataclass
class HUDState:
    night_vision: bool = False
    thermal: bool = False
    radar: bool = True
    motion_alert: bool = False
    fps: float = 0.0
    frame_type: str = "I"  # "I" on detection frames, "P" on optical-flow-tracked frames
    listening: float = 0.0  # seconds left in the voice command window (0 = waiting for wake word)
    faces: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    names: list[str] = field(default_factory=list)   # recognised name per face ("" = unknown)
    enrolling: int = -1     # samples captured so far while enrolling a face; -1 = not enrolling
