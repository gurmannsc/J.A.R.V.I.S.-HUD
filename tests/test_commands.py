"""Phrase -> Command matching, including STT-style noise and wake-word residue."""

import pytest

from common.types import CommandKind
from voice import commands

POSITIVE = [
    # exact phrases and canonical names
    ("find the edges", "edges"),
    ("edges", "edges"),
    ("sketch this", "sketch"),
    ("x-ray this", "xray"),
    ("show me its frequency spectrum", "spectrum"),
    ("remove the background", "remove_bg"),
    ("night vision", "night_vision"),
    ("back to normal", "reset"),
    ("shut down", "quit"),
    ("what can you do", "help"),
    # punctuation / case / wake-word residue
    ("Hey Jarvis, find the edges!", "edges"),
    ("jarvis sketch this", "sketch"),
    ("OK Jarvis remove the background.", "remove_bg"),
    # STT-style variations
    ("x ray this", "xray"),
    ("find the edge is", "edges"),
    ("sketch this please", "sketch"),
    ("cartoonify", "cartoon"),
    ("show me the frequency spectrum", "spectrum"),
    ("smooth this out", "lowpass"),
    ("sharpen", "highpass"),
    ("remove background", "remove_bg"),
    ("night vision on", "night_vision"),
    ("thermal", "thermal"),
    ("take a picture", "snapshot"),
    ("power down", "quit"),
    ("segment the skin", "skin"),
]

NEGATIVE = ["", "   ", "hello there", "what time is it", "open the pod bay doors", "jarvis"]


@pytest.mark.parametrize("text,expected", POSITIVE)
def test_matches(text: str, expected: str) -> None:
    cmd = commands.match(text)
    assert cmd is not None, f"{text!r} matched nothing"
    assert cmd.name == expected, f"{text!r} -> {cmd.name}, expected {expected}"
    assert cmd.raw_text == text


@pytest.mark.parametrize("text", NEGATIVE)
def test_no_match(text: str) -> None:
    assert commands.match(text) is None


def test_kinds_and_replies() -> None:
    assert commands.match("edges").kind is CommandKind.ONESHOT
    assert commands.match("night vision").kind is CommandKind.TOGGLE
    assert commands.match("quit").kind is CommandKind.SYSTEM
    assert commands.reply_for("edges") == "Edges located, sir."
    assert commands.reply_for("nope") == ""


def test_every_registered_effect_has_a_command() -> None:
    from effects import registry

    for name in registry.names():
        spec = commands.BY_NAME.get(name)
        assert spec is not None, name
        # "thermal" is a live HUD toggle; its registry entry only serves the effect dump
        assert spec.kind is (CommandKind.TOGGLE if name == "thermal" else CommandKind.ONESHOT), name


def test_unknown_words_require_exact_phrase() -> None:
    assert commands.match("[unk] everyone") is None           # "good morning everyone"
    assert commands.match("turn off the [unk]") is None       # "turn up the volume"
    assert commands.match("hide the [unk]") is None
    assert commands.match("[unk] find the edges").name == "edges"   # noise before a real command
    assert commands.match("night vision [unk]").name == "night_vision"


def test_scored_reports_phrase() -> None:
    cmd, score, phrase = commands.match_scored("find the edge is")
    assert cmd.name == "edges" and score >= commands.SCORE_CUTOFF and phrase == "find the edges"
