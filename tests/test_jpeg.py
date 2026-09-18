from pathlib import Path

import cv2

from common import jpeg


def test_encode_stats_orders_sizes_and_quality() -> None:
    frame = cv2.imread(str(Path(__file__).with_name("sample.jpg")))
    png, q90, q20 = jpeg.encode_stats(frame)
    assert png.label == "PNG" and png.psnr_db is None
    assert png.nbytes > q90.nbytes > q20.nbytes          # lossless > high quality > low quality
    assert q90.psnr_db > q20.psnr_db > 20                # lower quality -> lower PSNR
    assert "x smaller" in jpeg.summary(frame)
