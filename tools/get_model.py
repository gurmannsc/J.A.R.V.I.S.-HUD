"""Download the models the app needs into assets/.

    python tools/get_model.py            # everything that is missing
    python tools/get_model.py vosk       # just the speech model
    python tools/get_model.py faces      # just the YuNet + SFace face models

Speech: vosk-model-small-en-us-0.15 (~41 MB, alphacephei.com). If that host is blocked on
your network, download the zip by hand and unzip it into assets/ so that
assets/vosk-model-small-en-us-0.15/am/final.mdl exists.
Faces:  YuNet detector (~230 KB) and SFace recogniser (~37 MB) from the OpenCV model zoo.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from voice.listener import MODEL_DIR, MODEL_URL  # noqa: E402

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
FACE_MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total > 0:
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)


def fetch(url: str, dst: Path) -> None:
    print(f"downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()
    tmp.replace(dst)


def get_vosk() -> None:
    if (MODEL_DIR / "am" / "final.mdl").exists():
        print(f"speech model present: {MODEL_DIR.name}")
        return
    zip_path = MODEL_DIR.parent / "vosk-model.zip"
    fetch(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODEL_DIR.parent)
    zip_path.unlink()
    print(f"unpacked to {MODEL_DIR}")


def get_faces() -> None:
    for name, url in FACE_MODELS.items():
        dst = config.ASSETS_DIR / name
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"face model present: {name}")
            continue
        fetch(url, dst)


def main(which: str) -> None:
    config.ASSETS_DIR.mkdir(exist_ok=True)
    if which in ("all", "vosk"):
        try:
            get_vosk()
        except Exception as exc:
            print(f"speech model download failed: {exc}")
    if which in ("all", "faces"):
        get_faces()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
