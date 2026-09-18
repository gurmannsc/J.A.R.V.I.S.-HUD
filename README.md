# Iron Man HUD

A voice-controlled, real-time webcam heads-up display built for an Image and Video
Processing course. Python + OpenCV. Say "Hey Jarvis, find the edges" and the HUD freezes
the frame, runs the effect, and JARVIS talks back.

![HUD](docs/img/hud_plain.jpg)

- Live face detection with a targeting reticle, optical-flow tracking between detections,
  radar sweep, distance readout, night-vision and thermal modes, motion alert
- Wake word + offline speech recognition (Vosk), fuzzy command matching, spoken replies
- One-shot image-processing effects: x-ray, sketch, cartoon, DFT spectrum, frequency-domain
  smooth/sharpen, Canny/Sobel edges, GrabCut background removal, HSV skin segmentation,
  histogram equalisation, thermal
- Every feature maps to a syllabus unit — see [docs/coverage.md](docs/coverage.md)

## Setup

Python 3.11+ (developed on 3.14), Windows.

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

OpenCV is pinned to 4.x: OpenCV 5 removed `CascadeClassifier` and the bundled Haar files.

### Speech model (optional — the app falls back to typed commands without it)

```bash
.\.venv\Scripts\python.exe tools/get_model.py
```

This fetches `vosk-model-small-en-us-0.15` (~41 MB) from alphacephei.com into `assets/`.
If that host is blocked on your network, download the zip in a browser and unzip it so
that `assets/vosk-model-small-en-us-0.15/am/final.mdl` exists.

### Distance calibration (once)

```bash
.\.venv\Scripts\python.exe main.py --calibrate
```

Sit 60 cm from the camera and press SPACE. The focal length is saved to `calibration.json`.

## Run

```bash
.\.venv\Scripts\python.exe main.py
```

| Flag | Meaning |
|---|---|
| `--typed` | type commands in the terminal instead of speaking |
| `--echo` | print every transcript the recogniser produces (for tuning) |
| `--mute` | print JARVIS's replies without speaking them |
| `--camera N`, `--mic N` | choose devices (`python -c "import sounddevice as sd; print(sd.query_devices())"`) |
| `--calibrate` | distance calibration mode |

Say **"Hey Jarvis"** followed by a command in one breath, or say "Hey Jarvis", wait for
"Yes?", then the command. Every command also has a key:

| Command | Key | | Command | Key |
|---|---|---|---|---|
| x-ray this | `x` | | night vision | `n` |
| sketch this | `s` | | thermal vision | `t` |
| cartoonify this | `c` | | radar on / off | `r` |
| show me its frequency spectrum | `f` | | motion alert | `m` |
| smooth this / sharpen this | `l` / `h` | | take a snapshot | `p` |
| find the edges / sobel edges | `e` / `g` | | back to normal | `0` |
| remove the background | `b` | | what can you do | `?` |
| segment skin | `k` | | shut down | `q` / Esc |
| enhance this | `a` | | | |

One-shot effects freeze the frame for 3 s (any key dismisses early) and save the result to
`captures/`. Snapshots print PNG vs JPEG sizes and PSNR.

## Tests and tools

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

- `python -m effects.registry <image>` — dump every effect for an image into `captures/`
- `python tools/make_coverage_images.py <image>` — regenerate `docs/img/`
- `python tests/make_sample.py` — regenerate the synthetic test image

## Layout

```
main.py            loop: grab -> poll command -> dispatch -> HUD / frozen result -> show
common/            types.py (Frame, Command, HUDState — the contract), config.py, jpeg.py
hud/               detector (Haar), tracker (LK flow), overlay, radar, distance,
                   nightvision, motion, engine
voice/             listener (mic -> Vosk), wake (gate), commands (vocabulary + fuzzy match),
                   tts (pyttsx3 thread), interface
effects/           one pure (frame, state) -> frame function per effect, registry.py
docs/              coverage.md (syllabus mapping), demo.md (presentation script)
tests/             pytest suite (effects contract, tracker, HUD, commands, wake gate)
```

Design rule: every processing module takes a BGR `uint8` frame and returns one of the same
shape. The voice thread talks to the video loop only through a queue. See `PLAN.md` for the
full design.

## Troubleshooting

- **`cv2` has no `CascadeClassifier`** — you have OpenCV 5; `pip install "opencv-python<5"`.
- **Nothing is recognised** — run with `--echo`; if "jarvis" comes out as something else,
  add it to `WAKE_ALIASES` in `voice/wake.py`. Check the mic index with `--mic`.
- **JARVIS silent** — `--mute` not set? pyttsx3 uses Windows SAPI5; the console prints
  `[tts] disabled: …` if it could not initialise.
- **Slow** — the detector already runs at half resolution every 5th frame; lower
  `FRAME_WIDTH/HEIGHT` in `common/config.py` if the machine struggles.
