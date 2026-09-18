"""Tracker follows a translating image; the engine alternates I and P frames."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from common import config
from common.types import HUDState
from hud.engine import HUDEngine
from hud.tracker import FaceTracker

SAMPLE = Path(__file__).with_name("sample.jpg")


def shifted(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    frame = cv2.imread(str(SAMPLE))
    assert frame is not None
    return frame


def test_tracker_follows_translation(sample: np.ndarray) -> None:
    gray0 = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    box = (80, 60, 180, 180)  # the textured rectangle in the synthetic sample
    tr = FaceTracker()
    tr.reset(gray0, [box])
    assert tr.active

    total = np.array([0, 0])
    for step in range(1, 6):
        dx, dy = 3, -2
        total += (dx, dy)
        gray = cv2.cvtColor(shifted(sample, int(total[0]), int(total[1])), cv2.COLOR_BGR2GRAY)
        boxes = tr.update(gray)
        assert len(boxes) == 1, f"lost the box at step {step}"
        x, y, w, h = boxes[0]
        assert abs(x - (box[0] + total[0])) <= 2 and abs(y - (box[1] + total[1])) <= 2
        assert (w, h) == (box[2], box[3])


def test_tracker_drops_box_without_features() -> None:
    flat = np.full((240, 320), 128, dtype=np.uint8)
    tr = FaceTracker()
    tr.reset(flat, [(50, 50, 100, 100)])
    assert not tr.active
    assert tr.update(flat) == []


def test_engine_alternates_i_and_p_frames(sample: np.ndarray, monkeypatch) -> None:
    eng = HUDEngine()
    box = (80, 60, 180, 180)
    monkeypatch.setattr(eng.detector, "detect", lambda frame: [box])
    st = HUDState()
    types = []
    for i in range(config.DETECT_EVERY * 2):
        eng.process(sample, st)
        types.append(st.frame_type)
        assert st.faces, "should always have a box while the detector returns one"
    n = config.DETECT_EVERY
    expected = ["I"] + ["P"] * (n - 2) + ["I"] + ["P"] * (n - 1) + ["I"]  # tick 1 forced, then every n
    assert types == expected
