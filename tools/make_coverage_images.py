"""Render every effect and HUD mode to docs/img/ for the coverage document.

    python tools/make_coverage_images.py                 # uses tests/sample.jpg
    python tools/make_coverage_images.py captures/me.png  # use a real webcam frame

HUD modes are rendered by running the engine over a few synthetically panned copies of the
image, so the tracker produces a P-frame with flow vectors and the motion detector fires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.types import HUDState  # noqa: E402
from effects import registry  # noqa: E402
from hud.engine import HUDEngine  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
WIDTH = 560


def shift(img: np.ndarray, dx: int) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def save(name: str, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    small = cv2.resize(img, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  {path.relative_to(OUT_DIR.parent.parent)}")


def hud_mode(frame: np.ndarray, **flags) -> np.ndarray:
    eng = HUDEngine()
    st = HUDState()
    for k, v in flags.items():
        setattr(st, k, v)
    out = frame
    for i in range(8):                      # ticks 1..8: I at 1 and 5, P elsewhere
        out = eng.process(shift(frame, i * 3), st)
    return out


def main(image_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"could not read {image_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # a face box for the effects that use one; fall back to the centre if none is found
    eng = HUDEngine()
    st = HUDState()
    eng.process(frame, st)
    faces = st.faces or []

    print("effects:")
    for name in registry.names():
        if name == "identity":
            continue
        save(name, registry.apply(name, frame, HUDState(faces=list(faces))))

    print("hud modes:")
    save("hud_plain", hud_mode(frame))
    save("hud_night_vision", hud_mode(frame, night_vision=True))
    save("hud_thermal", hud_mode(frame, thermal=True))
    save("hud_motion", hud_mode(frame, motion_alert=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/sample.jpg")
