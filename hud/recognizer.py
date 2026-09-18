"""Face detection + recognition with OpenCV's YuNet and SFace models.

YuNet  -> face boxes with 5 landmarks (eyes, nose, mouth corners)
SFace  -> aligns the crop on the landmarks and maps it to a 128-d embedding
match  -> cosine similarity against enrolled embeddings; above THRESHOLD = same person

Embeddings are stored in faces.npz (git-ignored). Both models are optional: when the
files are missing the engine keeps using the Haar cascade and no names are shown.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common import config
from common.types import Frame
from hud.detector import Box

YUNET = config.ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = config.ASSETS_DIR / "face_recognition_sface_2021dec.onnx"
DB_FILE = config.ROOT / "faces.npz"

DETECT_SCALE = 0.5          # YuNet runs on a half-size frame
SCORE_THRESHOLD = 0.75
THRESHOLD = 0.40            # cosine similarity for "same person" (OpenCV suggests 0.363)
ENROL_SAMPLES = 8
UNKNOWN = ""


def models_available() -> bool:
    return YUNET.exists() and SFACE.exists()


class YuNetDetector:
    """Drop-in for FaceDetector.detect() that also keeps the landmark rows for SFace."""

    def __init__(self) -> None:
        self.net = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 240), SCORE_THRESHOLD, 0.3, 50)
        self.size: tuple[int, int] | None = None
        self.rows: list[np.ndarray] = []   # (15,) float32 rows in full-frame coordinates, aligned with boxes

    def detect(self, frame: Frame) -> list[Box]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
        size = (small.shape[1], small.shape[0])
        if size != self.size:
            self.net.setInputSize(size)
            self.size = size
        _, faces = self.net.detect(small)
        self.rows = []
        if faces is None:
            return []
        inv = 1.0 / DETECT_SCALE
        rows = faces.astype(np.float32)
        rows[:, :14] *= inv                      # box + landmarks back to full-frame pixels
        order = np.argsort(-(rows[:, 2] * rows[:, 3]))   # largest first, like the Haar path
        boxes: list[Box] = []
        for r in rows[order]:
            x, y, bw, bh = (int(v) for v in r[:4])
            x, y = max(x, 0), max(y, 0)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))
                self.rows.append(r)
        return boxes


class FaceDB:
    def __init__(self, path: Path = DB_FILE) -> None:
        self.path = path
        self.people: dict[str, np.ndarray] = {}   # name -> (N, 128) float32
        self.load()

    def load(self) -> None:
        self.people = {}
        if self.path.exists():
            with np.load(self.path) as data:
                self.people = {k: data[k] for k in data.files}

    def save(self) -> None:
        if self.people:
            np.savez(self.path, **self.people)
        elif self.path.exists():
            self.path.unlink()

    def add(self, name: str, embeddings: np.ndarray) -> None:
        name = name.strip().title()
        embs = np.asarray(embeddings, dtype=np.float32).reshape(-1, 128)
        if name in self.people:
            embs = np.vstack([self.people[name], embs])
        self.people[name] = embs
        self.save()

    def forget(self, name: str | None = None) -> None:
        if name is None:
            self.people = {}
        else:
            self.people.pop(name.strip().title(), None)
        self.save()

    def names(self) -> list[str]:
        return sorted(self.people)

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        """Best (name, cosine) over all stored embeddings; ("", best) below the threshold."""
        e = embedding.reshape(-1).astype(np.float32)
        e /= np.linalg.norm(e) + 1e-9
        best_name, best = UNKNOWN, -1.0
        for name, embs in self.people.items():
            norms = np.linalg.norm(embs, axis=1) + 1e-9
            score = float(np.max(embs @ e / norms))
            if score > best:
                best_name, best = name, score
        return (best_name if best >= THRESHOLD else UNKNOWN), best


class FaceRecognizer:
    """SFace embeddings + the database, with per-target vote smoothing."""

    def __init__(self, db: FaceDB | None = None) -> None:
        self.net = cv2.FaceRecognizerSF.create(str(SFACE), "")
        self.db = db or FaceDB()

    def embed(self, frame: Frame, row: np.ndarray) -> np.ndarray:
        aligned = self.net.alignCrop(frame, row)
        return self.net.feature(aligned).reshape(-1).astype(np.float32)

    def identify(self, frame: Frame, row: np.ndarray) -> tuple[str, float, np.ndarray]:
        emb = self.embed(frame, row)
        name, score = self.db.identify(emb)
        return name, score, emb


class IdentityVote:
    """Majority vote over the last few recognitions for one tracked target."""

    def __init__(self, n: int = 5) -> None:
        self.votes: deque[str] = deque(maxlen=n)
        self.score = 0.0

    def add(self, name: str, score: float) -> None:
        self.votes.append(name)
        self.score = score

    @property
    def name(self) -> str:
        if not self.votes:
            return UNKNOWN
        return max(set(self.votes), key=self.votes.count)
