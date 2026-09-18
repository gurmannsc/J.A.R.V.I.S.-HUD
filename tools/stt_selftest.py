"""Offline speech-pipeline self-test: TTS speaks each phrase into a WAV, Vosk transcribes it
(with the command grammar), the wake gate + fuzzy matcher turn it into a Command.

    python tools/stt_selftest.py            # all vocabulary phrases
    python tools/stt_selftest.py "hey jarvis find the edges" "cartoonify this"

Synthesised speech is cleaner than a real microphone, so this checks the model, grammar and
matching logic — not room acoustics.
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import commands  # noqa: E402
from voice.interface import grammar_phrases  # noqa: E402
from voice.listener import SAMPLE_RATE, model_path  # noqa: E402
from voice.wake import split_wake  # noqa: E402


def synthesise(text: str, path: Path) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.save_to_file(text, str(path))
    engine.runAndWait()
    engine.stop()


def load_16k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        audio = (audio - 128) * 256
    elif width == 4:
        audio = audio / 65536
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        t_src = np.arange(len(audio)) / rate
        t_dst = np.arange(int(len(audio) * SAMPLE_RATE / rate)) / SAMPLE_RATE
        audio = np.interp(t_dst, t_src, audio)
    pad = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)   # trailing silence so Vosk finalises
    return np.concatenate([audio, pad]).astype(np.int16).tobytes()


def transcribe(rec, pcm: bytes) -> str:
    rec.Reset()
    for i in range(0, len(pcm), 8000):
        rec.AcceptWaveform(pcm[i:i + 8000])
    return json.loads(rec.FinalResult()).get("text", "")


def main(phrases: list[str]) -> None:
    import vosk

    vosk.SetLogLevel(-1)
    model = vosk.Model(str(model_path()))
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, json.dumps(grammar_phrases()))

    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        for phrase in phrases:
            wav = Path(tmp) / "utt.wav"
            synthesise(phrase, wav)
            heard = transcribe(rec, load_16k_mono(wav))
            woke, rest = split_wake(heard)
            expected = commands.match(split_wake(phrase.lower())[1])
            cmd, score, _ = commands.match_scored(rest) if rest else (None, 0.0, "")
            good = (cmd is not None and expected is not None and cmd.name == expected.name) or (
                cmd is None and expected is None)
            ok += good
            print(f"{'OK ' if good else 'BAD'} said={phrase!r:38} heard={heard!r:38} "
                  f"wake={'Y' if woke else '-'} -> {cmd.name if cmd else None} ({score:.0f})")
    print(f"\n{ok}/{len(phrases)} phrases matched")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main([s.phrases[0] for s in commands.VOCABULARY if s.name != "identity"])
