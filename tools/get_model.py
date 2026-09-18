"""Download and unpack the Vosk small English model into assets/.

    python tools/get_model.py

If the official host is unreachable from your network, download the zip by hand from
the URL below (any browser / another network) and unzip it into assets/ so that
assets/vosk-model-small-en-us-0.15/am/final.mdl exists.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.listener import MODEL_DIR, MODEL_URL  # noqa: E402


def main() -> None:
    if (MODEL_DIR / "am" / "final.mdl").exists():
        print(f"model already present at {MODEL_DIR}")
        return
    MODEL_DIR.parent.mkdir(exist_ok=True)
    zip_path = MODEL_DIR.parent / "vosk-model.zip"

    def progress(blocks: int, block_size: int, total: int) -> None:
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    print(f"downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, zip_path, progress)
    print()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODEL_DIR.parent)
    zip_path.unlink()
    print(f"unpacked to {MODEL_DIR}")


if __name__ == "__main__":
    main()
