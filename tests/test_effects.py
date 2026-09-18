"""Contract tests for every registered effect: same shape, uint8, pure, fast."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from common.types import HUDState
from effects import registry

SAMPLE = Path(__file__).with_name("sample.jpg")
TIME_BUDGET_S = 1.0


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    frame = cv2.imread(str(SAMPLE))
    assert frame is not None, f"missing {SAMPLE} — run python tests/make_sample.py"
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    return frame


@pytest.mark.parametrize("name", registry.names())
def test_effect_contract(sample: np.ndarray, name: str) -> None:
    state = HUDState(faces=[(80, 60, 180, 180)])
    frame = sample.copy()
    before = frame.copy()

    t0 = time.perf_counter()
    out = registry.apply(name, frame, state)
    elapsed = time.perf_counter() - t0

    assert isinstance(out, np.ndarray)
    assert out.shape == frame.shape, f"{name}: shape {out.shape} != {frame.shape}"
    assert out.dtype == np.uint8, f"{name}: dtype {out.dtype}"
    assert np.array_equal(frame, before), f"{name}: mutated its input frame"
    assert elapsed < TIME_BUDGET_S, f"{name}: took {elapsed:.2f}s"
    if name != "identity":
        assert not np.array_equal(out, frame), f"{name}: returned the input unchanged"


def test_unknown_effect_raises(sample: np.ndarray) -> None:
    with pytest.raises(KeyError):
        registry.apply("does_not_exist", sample, HUDState())


def test_spectrum_has_peaks_for_stripes(sample: np.ndarray) -> None:
    """The stripe patch (period 16 px) must produce an off-centre horizontal-axis peak."""
    from effects.spectrum import dft_shifted, magnitude_image

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mag = magnitude_image(dft_shifted(gray))
    h, w = mag.shape
    row = mag[h // 2].astype(int)
    row[w // 2 - 5:w // 2 + 6] = 0  # ignore the DC lobe
    assert row.max() > 150, "expected a strong harmonic on the u-axis"


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_frequency_filters_change_sharpness(sample: np.ndarray) -> None:
    from effects.spectrum import filter_channel, gaussian_mask

    h, w = sample.shape[:2]
    lp = gaussian_mask((h, w), 0.05 * min(h, w))
    assert lp[h // 2, w // 2] == pytest.approx(1.0) and lp[0, 0] < 1e-3
    blurred = cv2.merge([np.clip(filter_channel(c, lp), 0, 255) for c in cv2.split(sample)]).astype(np.uint8)
    assert _sharpness(blurred) < 0.5 * _sharpness(sample)

    identity_mask = np.ones((h, w), dtype=np.float32)
    same = np.clip(filter_channel(sample[:, :, 0], identity_mask), 0, 255).astype(np.uint8)
    assert np.abs(same.astype(int) - sample[:, :, 0].astype(int)).max() <= 1  # DFT round trip


def test_skin_mask_finds_skin_patch() -> None:
    from effects.segment import skin_mask

    img = np.full((100, 100, 3), (200, 60, 20), dtype=np.uint8)         # blue-ish background
    img[20:80, 20:80] = (120, 160, 220)                                  # skin-toned BGR patch
    raw, clean = skin_mask(img)
    assert clean[50, 50] == 255 and clean[5, 5] == 0
    assert 0.3 < np.count_nonzero(clean) / clean.size < 0.4


def test_seed_rect_is_clamped_to_frame() -> None:
    from effects.background import seed_rect

    x, y, w, h = seed_rect((480, 640, 3), [(600, 400, 200, 200)])  # face near the corner
    assert x >= 0 and y >= 0 and x + w < 640 and y + h < 480
    assert seed_rect((480, 640, 3), []) == (160, 48, 320, 408)


def test_remove_bg_without_faces(sample: np.ndarray) -> None:
    out = registry.apply("remove_bg", sample, HUDState())
    assert out.shape == sample.shape


def test_equalise_flattens_histogram(sample: np.ndarray) -> None:
    from effects.enhance import equalise, luma_histogram

    dark = (sample * 0.3).astype(np.uint8)
    before, after = luma_histogram(dark), luma_histogram(equalise(dark))
    assert np.count_nonzero(after[128:]) > np.count_nonzero(before[128:])  # bright bins now used


def test_auto_canny_thresholds_bracket_median() -> None:
    from effects.edges import auto_canny_thresholds

    gray = np.full((50, 50), 100, dtype=np.uint8)
    lo, hi = auto_canny_thresholds(gray)
    assert lo < 100 < hi
