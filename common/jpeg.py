"""Unit 3 — lossy vs lossless compression readout for snapshots.

Encodes a frame as PNG (lossless) and JPEG at two qualities, reporting the byte size of
each and the PSNR of the JPEG reconstructions against the original.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common.types import Frame

JPEG_QUALITIES = (90, 20)


@dataclass(frozen=True)
class Encoding:
    label: str
    nbytes: int
    psnr_db: float | None   # None for lossless

    def __str__(self) -> str:
        kb = self.nbytes / 1024
        return f"{self.label} {kb:.0f} KB" + ("" if self.psnr_db is None else f" ({self.psnr_db:.1f} dB)")


def psnr(a: Frame, b: Frame) -> float:
    return float(cv2.PSNR(a, b))


def encode_stats(frame: Frame) -> list[Encoding]:
    ok, png = cv2.imencode(".png", frame)
    out = [Encoding("PNG", len(png), None)]
    for q in JPEG_QUALITIES:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(Encoding(f"JPEG q{q}", len(buf), psnr(frame, decoded)))
    return out


def summary(frame: Frame) -> str:
    stats = encode_stats(frame)
    ratio = stats[0].nbytes / max(stats[-1].nbytes, 1)
    return "  |  ".join(str(s) for s in stats) + f"  |  {ratio:.0f}x smaller"
