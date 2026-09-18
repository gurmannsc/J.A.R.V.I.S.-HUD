# Demo script (≈ 6 minutes, syllabus order)

Before the demo: `python main.py --calibrate` once at 60 cm, then start `python main.py`
(or `python main.py --typed` if the room is loud — every command has a key, shown in
brackets). Face the camera; wait for `TARGET LOCKED`.

| # | Say ("Hey Jarvis, …") | Key | What to point out | Unit |
|---|---|---|---|---|
| 1 | — (live feed) | | Frame is a 640×480×3 uint8 array; panel shows FPS; reticle, radar and panel are alpha-blended per pixel; distance from the pinhole model; the `I` / `P` tag flips every 5 frames and the dots are optical-flow features. If you enrolled beforehand ("remember me"), JARVIS greets you by name | 1, 8 |
| 2 | *night vision* | `n` | Grayscale → CLAHE (histogram) → gamma LUT (point op) → green tint (colour). Say *night vision* again to turn it off | 2, 7 |
| 3 | *enhance this* | `a` | Before/after luma histograms — equalisation spreads the histogram across 0–255 | 2 |
| 4 | *show me its frequency spectrum* | `f` | Centred log-magnitude DFT; straight edges in the room → streaks through the centre | 3 |
| 5 | *smooth this* then *sharpen this* | `l`, `h` | Gaussian `H(u,v)` multiplied into the DFT; the middle panel is the mask | 2 |
| 6 | *take a snapshot* | `p` | Console + HUD show PNG vs JPEG q90 vs q20 sizes and PSNR | 3 |
| 7 | *remove the background* | `b` | GrabCut seeded from the face box (orange rect), then opening + closing on the mask | 6, 4 |
| 8 | *segment skin* | `k` | HSV range threshold; raw mask vs mask after open/close | 6, 7, 4 |
| 9 | *find the edges* | `e` | Canny with thresholds from the median; optionally *sobel edges* (`g`) for gradient magnitude | 5 |
| 10 | *thermal vision* | `t` | INFERNO pseudo-colour on CLAHE luminance | 7 |
| 11 | *cartoonify this* | `c` | Bilateral ×3, HSV saturation boost, k-means k=8, adaptive-threshold outline | 2, 5, 7 |
| 12 | *motion alert*, then wave | `m` | Frame differencing → threshold → dilate → contours; level bar and alert | 8 |
| 13 | *back to normal* | `0` | Clears modes | |
| 14 | *shut down* | `q` | | |

Keep the wake word and command in one breath: "Hey Jarvis, find the edges." If you pause
after "Hey Jarvis", he answers "Yes?" and waits 5 s for the command.

Fallbacks if something misbehaves on the day:
- Speech not recognised → use the key in the third column; run with `--echo` to see what
  the recogniser heard.
- No face lock → move closer / face the lens; Haar needs a roughly frontal face.
- Record a backup run beforehand with any screen recorder and keep it on the laptop.
