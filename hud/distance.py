"""Unit 1 — pinhole-camera distance estimate from the face box width.

    w_px = f_px * W_cm / d_cm    =>    d_cm = f_px * W_cm / w_px

f_px (focal length in pixels) is found once by calibration: hold a known distance D and
read the box width w, then f_px = w * D / W_cm.
"""

from __future__ import annotations

import json

from common import config

CALIBRATION_FILE = config.ROOT / "calibration.json"
CALIBRATION_DISTANCE_CM = 60.0


def load_focal_px() -> float:
    try:
        return float(json.loads(CALIBRATION_FILE.read_text())["focal_px"])
    except (OSError, ValueError, KeyError):
        return config.FOCAL_PX


def save_focal_px(focal_px: float) -> None:
    CALIBRATION_FILE.write_text(json.dumps({"focal_px": round(focal_px, 1),
                                            "face_width_cm": config.FACE_WIDTH_CM}, indent=2))


def focal_from_box(box_w_px: int, distance_cm: float = CALIBRATION_DISTANCE_CM) -> float:
    return box_w_px * distance_cm / config.FACE_WIDTH_CM


def estimate_cm(box_w_px: int, focal_px: float) -> float:
    return focal_px * config.FACE_WIDTH_CM / max(box_w_px, 1)


def format_distance(cm: float) -> str:
    return f"{cm / 100:.2f} m" if cm >= 100 else f"{cm:.0f} cm"
