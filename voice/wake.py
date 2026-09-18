"""Wake-word gate: nothing is treated as a command until the user says "Jarvis".

Works on *final* transcripts only (Vosk finalises ~0.5 s after speech stops), which is
simpler and more robust than acting on partial hypotheses:

    "hey jarvis find the edges"  -> command straight away
    "hey jarvis"                 -> ARMED, JARVIS says "Yes?", next utterance is the command
    after any command            -> stays ARMED for a follow-up window, so commands can be
                                    chained without repeating the wake word
    ARMED + silence past window  -> back to IDLE

The small Vosk model may not know the word "jarvis", so a few sound-alikes are accepted
and matched fuzzily.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from rapidfuzz import fuzz

WAKE_WORD = "jarvis"
WAKE_ALIASES = ("jarvis", "jarvas", "jervis", "travis", "jarves", "jar vis", "javis")
WAKE_SCORE = 80
ARMED_TIMEOUT_S = 5.0       # after a bare "hey jarvis"
FOLLOWUP_TIMEOUT_S = 8.0    # after a command: window for the next one without the wake word
_LEADIN = re.compile(r"^(?:hey|hi|ok|okay|a|hey a)\s+")


def split_wake(text: str) -> tuple[bool, str]:
    """(wake word heard?, remaining command text)."""
    words = text.lower().split()
    for i, w in enumerate(words):
        if fuzz.ratio(w, WAKE_WORD) >= WAKE_SCORE or w in WAKE_ALIASES:
            return True, " ".join(words[i + 1:]).strip()
        two = " ".join(words[i:i + 2])
        if two in WAKE_ALIASES:
            return True, " ".join(words[i + 2:]).strip()
    return False, text.strip()


class WakeGate:
    def __init__(self, on_command: Callable[[str], None], on_wake: Callable[[], None],
                 on_timeout: Callable[[], None] | None = None) -> None:
        self.on_command = on_command
        self.on_wake = on_wake
        self.on_timeout = on_timeout
        self.armed_at: float | None = None
        self.window_s = ARMED_TIMEOUT_S

    @property
    def armed(self) -> bool:
        return self.armed_at is not None

    def remaining(self, now: float | None = None) -> float:
        """Seconds left in the current listening window (0 when idle)."""
        if self.armed_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, self.window_s - (now - self.armed_at))

    def arm(self, now: float, window_s: float) -> None:
        self.armed_at = now
        self.window_s = window_s

    def tick(self, now: float | None = None) -> None:
        """Call periodically so an armed gate can time out."""
        now = time.monotonic() if now is None else now
        if self.armed_at is not None and now - self.armed_at > self.window_s:
            self.armed_at = None
            if self.on_timeout:
                self.on_timeout()

    def feed_final(self, text: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.tick(now)
        text = _LEADIN.sub("", text.strip().lower())
        woke, rest = split_wake(text)

        if woke:
            if rest:
                self._command(rest, now)
            else:
                self.arm(now, ARMED_TIMEOUT_S)
                self.on_wake()
            return

        if self.armed_at is not None and text:
            self._command(text, now)
        # not armed and no wake word: ignore (background speech)

    def _command(self, text: str, now: float) -> None:
        self.arm(now, FOLLOWUP_TIMEOUT_S)   # keep listening for a chained command
        self.on_command(text)
