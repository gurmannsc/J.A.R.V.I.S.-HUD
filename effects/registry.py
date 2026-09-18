"""Effect registry: maps a command name to a pure `(frame, state) -> frame` function.

Rules for every effect:
  * never mutate the input frame or the state
  * return a new Frame with the same shape and dtype (uint8, 3-channel BGR)
  * finish in under ~1 s on 640x480 (downscale internally if needed)

    python -m effects.registry tests/sample.jpg     # dump every effect to captures/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from common import config
from common.types import Frame, HUDState
from effects import background, cartoon, edges, enhance, segment, sketch, spectrum, thermal, xray

EffectFn = Callable[[Frame, HUDState], Frame]


def identity(frame: Frame, state: HUDState) -> Frame:
    return frame.copy()


EFFECTS: dict[str, EffectFn] = {
    "identity": identity,
    "edges": edges.canny,            # 5
    "sobel": edges.sobel,            # 5
    "sketch": sketch.sketch,         # 2 spatial
    "xray": xray.xray,               # 2 point + histogram, 7
    "spectrum": spectrum.spectrum,   # 3
    "lowpass": spectrum.lowpass,     # 2 frequency-domain
    "highpass": spectrum.highpass,   # 2 frequency-domain
    "cartoon": cartoon.cartoon,      # 2, 5, 7
    "remove_bg": background.remove_bg,  # 6, 4
    "skin": segment.skin,            # 6, 7, 4
    "thermal": thermal.thermal,      # 7
    "enhance": enhance.enhance,      # 2 histogram
}


def apply(name: str, frame: Frame, state: HUDState) -> Frame:
    """Run the named effect. Raises KeyError for an unknown name."""
    return EFFECTS[name](frame, state)


def names() -> list[str]:
    return sorted(EFFECTS)


def _dump_all(image_path: str) -> None:
    import cv2

    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"could not read {image_path}")
    config.CAPTURES_DIR.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    for name in names():
        out = apply(name, frame, HUDState())
        path = config.CAPTURES_DIR / f"{stem}_{name}.png"
        cv2.imwrite(str(path), out)
        print(f"{name:10s} -> {path.name}")


if __name__ == "__main__":
    _dump_all(sys.argv[1] if len(sys.argv) > 1 else "tests/sample.jpg")
