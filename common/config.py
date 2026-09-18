"""Tunable constants. Import as `from common import config`."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTURES_DIR = ROOT / "captures"
ASSETS_DIR = ROOT / "assets"

# --- camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_NAME = "IRON MAN HUD"

# --- one-shot effects ---
HOLD_SECONDS = 3.0  # how long a one-shot result stays frozen on screen

# --- hud ---
DETECT_EVERY = 5          # run the face detector every N frames; track in between
FACE_WIDTH_CM = 15.0      # average adult face width, for the distance readout
FOCAL_PX = 600.0          # set via `python main.py --calibrate` at 60 cm
HUD_ALPHA = 0.6

# --- colours (BGR) — holographic Mark III palette ---
HUD_CYAN = (255, 205, 110)     # primary line colour
HUD_CYAN_DIM = (190, 140, 60)  # grids, secondary ticks
HUD_ORANGE = (40, 165, 255)    # amber accent / warnings
HUD_RED = (40, 40, 255)
HUD_GREEN = (120, 255, 140)
HUD_WHITE = (245, 250, 255)
