"""Face database + identity voting (pure NumPy), and the YuNet/SFace path when models exist."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from common.types import HUDState
from hud import recognizer

SAMPLE = Path(__file__).with_name("sample.jpg")


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def test_facedb_add_identify_forget(tmp_path: Path) -> None:
    db = recognizer.FaceDB(tmp_path / "faces.npz")
    a, b = unit(1), unit(2)
    db.add("gurmann", np.stack([a, a + 0.05 * unit(3)]))
    db.add("Alice", b)
    assert db.names() == ["Alice", "Gurmann"]

    name, score = db.identify(a + 0.1 * unit(4))       # noisy copy of a -> still Gurmann
    assert name == "Gurmann" and score > recognizer.THRESHOLD
    assert db.identify(b)[0] == "Alice"
    assert db.identify(unit(9))[0] == recognizer.UNKNOWN   # random vector matches nobody

    db2 = recognizer.FaceDB(tmp_path / "faces.npz")        # persisted
    assert db2.names() == ["Alice", "Gurmann"]
    db2.forget("Alice")
    assert db2.names() == ["Gurmann"]
    db2.forget()
    assert db2.names() == [] and not (tmp_path / "faces.npz").exists()


def test_identity_vote_majority() -> None:
    v = recognizer.IdentityVote(n=5)
    assert v.name == ""
    for n in ["Bob", "", "Bob", "Alice", "Bob"]:
        v.add(n, 0.5)
    assert v.name == "Bob"
    for n in ["Alice"] * 5:
        v.add(n, 0.6)
    assert v.name == "Alice"                               # window slides


@pytest.mark.skipif(not recognizer.models_available(), reason="YuNet/SFace models not downloaded")
def test_yunet_and_sface_run(tmp_path: Path) -> None:
    frame = cv2.imread(str(SAMPLE))
    det = recognizer.YuNetDetector()
    boxes = det.detect(frame)                              # synthetic image: no faces expected
    assert isinstance(boxes, list) and len(det.rows) == len(boxes)

    rec = recognizer.FaceRecognizer(recognizer.FaceDB(tmp_path / "faces.npz"))
    fake_row = np.array([200, 150, 200, 200, 260, 220, 340, 220, 300, 260, 270, 300, 330, 300, 0.9], np.float32)
    emb = rec.embed(frame, fake_row)
    assert emb.shape == (128,)
    rec.db.add("Test", emb)
    name, score, _ = rec.identify(frame, fake_row)
    assert name == "Test" and score > 0.99


@pytest.mark.skipif(not recognizer.models_available(), reason="YuNet/SFace models not downloaded")
def test_engine_enrol_flow(monkeypatch, tmp_path: Path) -> None:
    from hud.engine import HUDEngine

    eng = HUDEngine()
    eng.recognizer.db = recognizer.FaceDB(tmp_path / "faces.npz")
    frame = cv2.imread(str(SAMPLE))
    row = np.array([80, 60, 180, 180, 130, 120, 210, 120, 170, 160, 140, 200, 200, 200, 0.9], np.float32)

    def fake_detect(f):
        eng.detector.rows = [row]
        return [(80, 60, 180, 180)]

    monkeypatch.setattr(eng.detector, "detect", fake_detect)
    st = HUDState()
    eng.start_enrol()
    for _ in range(recognizer.ENROL_SAMPLES * 5 + 1):      # detection happens every 5th frame
        eng.process(frame, st)
    assert eng.enrol_ready() and st.enrolling == recognizer.ENROL_SAMPLES
    assert st.names == [""]                                 # nobody enrolled yet
    assert eng.finish_enrol("gurmann") == recognizer.ENROL_SAMPLES
    for _ in range(6):
        eng.process(frame, st)
    assert st.names == ["Gurmann"] and st.enrolling == -1
