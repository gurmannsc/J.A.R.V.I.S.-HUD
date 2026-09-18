"""JARVIS voice, on a dedicated worker thread.

On Windows this talks to SAPI directly through comtypes (pyttsx3's runAndWait() only
produces audio once per process on some Windows setups). It also reaches the newer
"OneCore" voices that SAPI's default enumeration hides, which is where the male voices
(George, David, Mark) usually live. Elsewhere it falls back to pyttsx3.

`say()` only enqueues; the video loop never waits on speech. `speaking` is set while audio
is playing so the listener can ignore JARVIS's own voice.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

RATE = 1                      # SAPI rate, -10..10
PITCH = -3                    # SAPI pitch offset, -10..10 (a little deeper)
PREFERRED_VOICES = ("george", "david", "mark", "james", "ryan", "richard")   # first match wins
ONECORE_KEY = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"
SVSF_IS_XML = 8


class TTS:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.ready = False
        self.voice_name = ""
        self.speaking = threading.Event()   # set while audio is playing
        self.speak_started_at = 0.0
        self.last_spoke_at = 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if self.enabled and text:
            self._queue.put(text)

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        try:
            speak = self._init_sapi() if sys.platform == "win32" else self._init_pyttsx3()
            self.ready = True
            print(f"[tts] voice: {self.voice_name}")
        except Exception as exc:  # no audio device / driver problems: degrade to silent
            print(f"[tts] disabled: {exc}")
            self.enabled = False
            return

        while True:
            text = self._queue.get()
            if text is None:
                break
            self.speak_started_at = time.monotonic()
            self.speaking.set()
            try:
                speak(text)
            except Exception as exc:
                print(f"[tts] error: {exc}")
            finally:
                self.last_spoke_at = time.monotonic()
                self.speaking.clear()

    def _init_sapi(self):
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        tokens = list(voice.GetVoices())
        try:  # OneCore voices (Windows 10/11 "Settings > Speech" voices)
            cat = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
            cat.SetId(ONECORE_KEY, False)
            tokens = list(cat.EnumerateTokens()) + tokens
        except Exception:
            pass
        chosen = None
        for key in PREFERRED_VOICES:
            chosen = next((t for t in tokens if key in t.GetDescription().lower()), None)
            if chosen is not None:
                break
        if chosen is None:  # any male voice, else whatever is first
            chosen = next((t for t in tokens if _attr(t, "Gender") == "Male"), tokens[0])
        voice.Voice = chosen
        voice.Rate = RATE
        self.voice_name = chosen.GetDescription()

        def speak(text: str) -> None:
            safe = text.replace("&", "and").replace("<", " ").replace(">", " ")
            voice.Speak(f'<pitch absmiddle="{PITCH}"/>{safe}', SVSF_IS_XML)

        return speak

    def _init_pyttsx3(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        voices = engine.getProperty("voices") or []
        for key in PREFERRED_VOICES:
            v = next((v for v in voices if key in (v.name or "").lower()), None)
            if v is not None:
                engine.setProperty("voice", v.id)
                self.voice_name = v.name
                break

        def speak(text: str) -> None:
            engine.say(text)
            engine.runAndWait()

        return speak


def _attr(token, name: str) -> str:
    try:
        return token.GetAttribute(name)
    except Exception:
        return ""
