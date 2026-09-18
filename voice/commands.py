"""Command vocabulary and text -> Command matching.

`match()` normalises the text, strips any wake-word residue ("hey jarvis ..."), tries an
exact hit on a canonical name or phrase, then falls back to rapidfuzz fuzzy matching over
every listed phrase with a score cutoff — so STT slips like "find the edge is" or
"sketch this please" still land on the right command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from common.types import Command, CommandKind

SCORE_CUTOFF = 75


@dataclass(frozen=True)
class Spec:
    name: str
    kind: CommandKind
    phrases: tuple[str, ...]
    reply: str


# The first phrase of each spec is the canonical one (shown in help). The rest are accepted
# variants; the recogniser's grammar is built from all of them, so the more natural
# phrasings listed here, the less exact the user has to be.
VOCABULARY: tuple[Spec, ...] = (
    # one-shot effects
    Spec("xray", CommandKind.ONESHOT,
         ("x-ray this", "xray this", "x ray this", "x-ray", "x ray", "x ray vision", "x ray mode", "show me the x ray"),
         "X-ray view, sir."),
    Spec("sketch", CommandKind.ONESHOT,
         ("sketch this", "sketch", "pencil sketch", "make a sketch", "draw this", "sketch mode"),
         "Rendering sketch."),
    Spec("cartoon", CommandKind.ONESHOT,
         ("cartoonify this", "cartoon this", "cartoonify", "cartoon", "cartoon mode", "make it a cartoon", "make me a cartoon"),
         "Cartoon mode."),
    Spec("spectrum", CommandKind.ONESHOT,
         ("show me its frequency spectrum", "show me the frequency spectrum", "frequency spectrum", "spectrum",
          "show the spectrum", "fourier transform", "show me the fourier transform", "frequency domain"),
         "Displaying Fourier spectrum."),
    Spec("lowpass", CommandKind.ONESHOT,
         ("smooth this", "smooth this out", "low pass filter", "smooth", "blur this", "smooth the image", "low pass"),
         "Applying low-pass filter."),
    Spec("highpass", CommandKind.ONESHOT,
         ("sharpen this", "high pass filter", "sharpen", "sharpen the image", "high pass", "make it sharper"),
         "Applying high-pass filter."),
    Spec("edges", CommandKind.ONESHOT,
         ("find the edges", "find edges", "edges", "edge detection", "show edges", "show me the edges", "detect edges",
          "detect the edges", "canny"),
         "Edges located."),
    Spec("sobel", CommandKind.ONESHOT, ("sobel edges", "gradient", "sobel", "show the gradient"), "Gradient magnitude."),
    Spec("remove_bg", CommandKind.ONESHOT,
         ("remove the background", "remove background", "isolate me", "background removal", "cut out the background",
          "hide the background", "remove my background"),
         "Isolating subject."),
    Spec("skin", CommandKind.ONESHOT,
         ("segment skin", "segment the skin", "find skin", "skin", "skin detection", "detect skin", "show skin",
          "find the skin"),
         "Segmenting skin tones."),
    Spec("enhance", CommandKind.ONESHOT,
         ("enhance this", "enhance", "equalize histogram", "enhance the image", "improve contrast", "histogram",
          "show the histogram"),
         "Enhancing."),
    # toggles
    Spec("night_vision", CommandKind.TOGGLE,
         ("night vision", "night vision on", "night vision off", "night mode", "night vision mode", "turn on night vision",
          "turn off night vision"),
         "Night vision toggled."),
    Spec("thermal", CommandKind.TOGGLE,
         ("thermal vision", "thermal on", "thermal off", "thermal", "thermal mode", "heat vision", "turn on thermal",
          "turn off thermal"),
         "Thermal toggled."),
    Spec("radar", CommandKind.TOGGLE, ("radar on", "radar off", "toggle radar", "radar", "hide radar", "show radar"),
         "Radar toggled."),
    Spec("motion_alert", CommandKind.TOGGLE,
         ("motion alert", "motion detection", "watch for motion", "detect motion", "motion", "motion alert on",
          "motion alert off"),
         "Monitoring motion."),
    # system
    Spec("snapshot", CommandKind.SYSTEM,
         ("take a snapshot", "take a picture", "snapshot", "capture", "take a photo", "screenshot", "capture this"),
         "Captured."),
    Spec("reset", CommandKind.SYSTEM,
         ("back to normal", "normal view", "reset", "go back", "back to live", "clear", "normal", "turn everything off"),
         "Returning to live feed."),
    Spec("help", CommandKind.SYSTEM, ("what can you do", "help", "list commands", "commands", "what can i say"), ""),
    Spec("quit", CommandKind.SYSTEM, ("shut down", "power down", "quit", "exit", "power off", "goodbye"),
         "Powering down."),
    # test-only
    Spec("identity", CommandKind.ONESHOT, ("identity",), "Identity."),
)

BY_NAME: dict[str, Spec] = {s.name: s for s in VOCABULARY}
_PHRASE_TO_SPEC: dict[str, Spec] = {p: s for s in VOCABULARY for p in s.phrases}
_PHRASES: list[str] = list(_PHRASE_TO_SPEC)

_WAKE_PREFIX = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+)?jarvis[,\s]*")


def normalise(text: str) -> str:
    text = re.sub(r"[^a-z0-9\- ]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return _WAKE_PREFIX.sub("", text).strip()


def match_scored(text: str) -> tuple[Command | None, float, str]:
    """(command or None, best score, best phrase) — handy for logging what STT produced."""
    norm = normalise(text)
    if not norm:
        return None, 0.0, ""
    if norm in BY_NAME:
        return Command(norm, BY_NAME[norm].kind, text), 100.0, norm
    if norm in _PHRASE_TO_SPEC:
        spec = _PHRASE_TO_SPEC[norm]
        return Command(spec.name, spec.kind, text), 100.0, norm
    # token_set_ratio ignores word order and extra words ("sketch this please") but, unlike
    # WRatio, does not let short phrases partial-match inside unrelated sentences.
    best = process.extractOne(norm, _PHRASES, scorer=fuzz.token_set_ratio, score_cutoff=SCORE_CUTOFF)
    if best is None:
        return None, 0.0, ""
    phrase, score, _ = best
    spec = _PHRASE_TO_SPEC[phrase]
    return Command(spec.name, spec.kind, text), float(score), phrase


def match(text: str) -> Command | None:
    return match_scored(text)[0]


def reply_for(name: str) -> str:
    return BY_NAME[name].reply if name in BY_NAME else ""
