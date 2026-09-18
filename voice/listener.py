"""Microphone -> Vosk speech recogniser, on a background thread.

sounddevice delivers 16 kHz mono int16 blocks through a callback; a worker thread feeds
them to a KaldiRecognizer and reports partial and final transcripts through `on_text`.

With a grammar (the command vocabulary + wake word + [unk]) the recogniser only has to
choose between a few dozen phrases, which makes it far more accurate than free dictation.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable

from common import config

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # samples per callback: 0.25 s at 16 kHz
MODEL_DIR = config.ASSETS_DIR / "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


class ModelMissing(RuntimeError):
    pass


def model_path() -> Path:
    if not (MODEL_DIR / "am" / "final.mdl").exists():
        raise ModelMissing(f"Vosk model not found at {MODEL_DIR}\n"
                           f"download {MODEL_URL} and unzip it into {config.ASSETS_DIR}")
    return MODEL_DIR


class Listener:
    """Streams the default microphone into Vosk; calls on_text(kind, text) with
    kind == "partial" (in-progress hypothesis) or "final" (utterance finished)."""

    def __init__(self, on_text: Callable[[str, str], None], grammar: list[str] | None = None,
                 device: int | None = None) -> None:
        self.on_text = on_text
        self.grammar = grammar
        self.device = device
        self._audio: queue.Queue[bytes] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reset = threading.Event()

    def start(self) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_path()))
        if self.grammar:
            self._rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, json.dumps(self.grammar))
        else:
            self._rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        self._thread = threading.Thread(target=self._run, name="listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset(self) -> None:
        """Drop the current hypothesis (e.g. after JARVIS spoke, so his voice isn't a command)."""
        self._reset.set()

    # ------------------------------------------------------------------ worker

    def _callback(self, indata, frames, time_info, status) -> None:
        self._audio.put(bytes(indata))

    def _run(self) -> None:
        import sounddevice as sd

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=self.device,
                               dtype="int16", channels=1, callback=self._callback):
            while not self._stop.is_set():
                try:
                    data = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._reset.is_set():
                    self._reset.clear()
                    self._rec.Reset()
                    continue
                if self._rec.AcceptWaveform(data):
                    text = json.loads(self._rec.Result()).get("text", "")
                    if text:
                        self.on_text("final", text)
                else:
                    text = json.loads(self._rec.PartialResult()).get("partial", "")
                    if text:
                        self.on_text("partial", text)
