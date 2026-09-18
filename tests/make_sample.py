"""Generate tests/sample.jpg — a synthetic 640x480 test image with a mix of content:
smooth gradient, sharp shapes, periodic stripes (a clean DFT peak), text and noise.

    python tests/make_sample.py
Replace sample.jpg with a real webcam photo if you prefer; tests only need a 3-channel image.
"""

from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480


def make() -> np.ndarray:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = (xx / W * 180 + 40).astype(np.uint8)      # blue ramp left->right
    img[:, :, 1] = (yy / H * 120 + 60).astype(np.uint8)      # green ramp top->bottom
    img[:, :, 2] = 90

    # vertical stripes region: period 16 px -> a sharp peak on the u-axis of the spectrum
    stripes = ((xx // 8) % 2 * 120 + 60).astype(np.uint8)
    img[300:460, 40:280] = np.dstack([stripes[300:460, 40:280]] * 3)

    cv2.circle(img, (440, 160), 90, (30, 200, 240), -1)
    cv2.circle(img, (440, 160), 90, (255, 255, 255), 3)
    cv2.rectangle(img, (80, 60), (260, 240), (200, 80, 40), -1)
    cv2.rectangle(img, (120, 100), (220, 200), (20, 20, 20), 6)
    cv2.putText(img, "IVP", (360, 400), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)

    noise = rng.normal(0, 8, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    out = Path(__file__).with_name("sample.jpg")
    cv2.imwrite(str(out), make(), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out}")
