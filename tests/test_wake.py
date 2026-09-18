"""Wake-word gate state machine and the grammar handed to Vosk."""

from voice import wake
from voice.interface import grammar_phrases


class Spy:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.wakes = 0
        self.timeouts = 0

    def gate(self) -> wake.WakeGate:
        return wake.WakeGate(on_command=self.commands.append,
                             on_wake=lambda: setattr(self, "wakes", self.wakes + 1),
                             on_timeout=lambda: setattr(self, "timeouts", self.timeouts + 1))


def test_split_wake_variants() -> None:
    assert wake.split_wake("hey jarvis find the edges") == (True, "find the edges")
    assert wake.split_wake("jarvis") == (True, "")
    assert wake.split_wake("travis sketch this") == (True, "sketch this")   # sound-alike
    assert wake.split_wake("jar vis night vision") == (True, "night vision")
    assert wake.split_wake("find the edges") == (False, "find the edges")


def test_one_shot_utterance() -> None:
    spy = Spy()
    g = spy.gate()
    g.feed_final("hey jarvis find the edges", now=0.0)
    assert spy.commands == ["find the edges"] and spy.wakes == 0 and g.armed


def test_two_step_utterance() -> None:
    spy = Spy()
    g = spy.gate()
    g.feed_final("hey jarvis", now=0.0)
    assert g.armed and spy.wakes == 1 and spy.commands == []
    g.feed_final("sketch this", now=1.0)
    assert spy.commands == ["sketch this"] and g.armed          # stays armed for a follow-up


def test_background_speech_ignored_when_idle() -> None:
    spy = Spy()
    g = spy.gate()
    g.feed_final("find the edges", now=0.0)
    g.feed_final("let's get lunch", now=1.0)
    assert spy.commands == [] and spy.wakes == 0


def test_followup_commands_without_wake_word() -> None:
    spy = Spy()
    g = spy.gate()
    g.feed_final("hey jarvis find the edges", now=0.0)
    g.feed_final("sketch this", now=3.0)                       # inside the follow-up window
    g.feed_final("night vision", now=6.0)                      # window was extended by the last command
    assert spy.commands == ["find the edges", "sketch this", "night vision"]
    assert 0 < g.remaining(now=6.0) <= wake.FOLLOWUP_TIMEOUT_S
    g.feed_final("thermal vision", now=6.0 + wake.FOLLOWUP_TIMEOUT_S + 0.5)   # too late
    assert spy.commands[-1] == "night vision" and not g.armed and spy.timeouts == 1


def test_armed_times_out() -> None:
    spy = Spy()
    g = spy.gate()
    g.feed_final("jarvis", now=0.0)
    g.tick(now=wake.ARMED_TIMEOUT_S + 0.1)
    assert not g.armed and spy.timeouts == 1
    g.feed_final("sketch this", now=wake.ARMED_TIMEOUT_S + 0.2)
    assert spy.commands == []                     # too late: needs the wake word again


def test_grammar_contains_vocabulary_and_unk() -> None:
    g = grammar_phrases()
    assert g[-1] == "[unk]"
    assert {"find the edges", "night vision", "hey jarvis", "shut down"} <= set(g)
    assert "identity" not in g
