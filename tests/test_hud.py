"""HUD widgets: distance maths, motion detector, base modes, and the engine in every mode."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from common import config
from common.types import HUDState
from hud import distance, nightvision
from hud.engine import HUDEngine
from hud.motion import MotionDetector

SAMPLE = Path(__file__).with_name("sample.jpg")


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    frame = cv2.imread(str(SAMPLE))
    assert frame is not None
    return frame


def test_distance_round_trip() -> None:
    focal = distance.focal_from_box(150, distance_cm=60)
    assert distance.estimate_cm(150, focal) == pytest.approx(60)
    assert distance.estimate_cm(75, focal) == pytest.approx(120)   # half the width -> twice as far
    assert distance.format_distance(45) == "45 cm"
    assert distance.format_distance(120) == "1.20 m"


def test_motion_detector_levels(sample: np.ndarray) -> None:
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    md = MotionDetector()
    assert md.update(gray) == 0.0          # first frame has nothing to compare against
    assert md.update(gray) == 0.0          # identical frame -> no motion
    moved = np.roll(gray, 15, axis=1)
    assert md.update(moved) > 0.01         # big shift -> plenty of changed pixels
    assert md.regions


@pytest.mark.parametrize("fn", [nightvision.night_vision, nightvision.thermal])
def test_base_modes_keep_shape(sample: np.ndarray, fn) -> None:
    out = fn(sample)
    assert out.shape == sample.shape and out.dtype == np.uint8
    assert not np.array_equal(out, sample)


def test_night_vision_is_green(sample: np.ndarray) -> None:
    b, g, r = cv2.split(nightvision.night_vision(sample))
    assert g.mean() > r.mean() and g.mean() > b.mean()


@pytest.mark.parametrize("mode", ["plain", "night_vision", "thermal", "motion_alert", "no_radar"])
def test_engine_every_mode(sample: np.ndarray, mode: str, monkeypatch) -> None:
    eng = HUDEngine()
    monkeypatch.setattr(eng.detector, "detect", lambda frame: [(80, 60, 180, 180)])
    st = HUDState()
    if mode == "no_radar":
        st.radar = False
    elif mode != "plain":
        setattr(st, mode, True)
    for _ in range(3):
        out = eng.process(sample, st)
    assert out.shape == sample.shape and out.dtype == np.uint8
    assert st.faces and st.fps > 0
