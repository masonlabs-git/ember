"""Piper TTS: pronunciation shim, sentence streaming, language pick.

Sentence streaming is the perceived-latency fix: speak sentence one while
sentence two still generates.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import tempfile
import threading
from typing import Callable, Iterable, Iterator

from . import config

# Caleb-approved pronunciations (v2 by ear). Keys are case-insensitive.
PRONOUNCE = {
    "bugout": "bug-out",
    "hailo": "hy-loh",
    "fts5": "F T S five",
    "ics": "I C S",
}

_SENT_END = re.compile(r"([.!?])\s")
_ES_HINT = re.compile(
    r"\b(el|la|los|las|es|para|agua|de|una|con|usted|por)\b", re.I)
# Citation markers like [1], [2], [1, 2], [1,2,3] — shown on screen, not spoken.
_CITES = re.compile(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def strip_citations(text: str) -> str:
    """Remove [1]/[1, 2] markers before speech (they belong on the display)."""
    return _CITES.sub("", text).strip()


def apply_shim(text: str) -> str:
    text = strip_citations(text)
    for word, spoken in PRONOUNCE.items():
        text = re.sub(rf"\b{word}\b", spoken, text, flags=re.I)
    return text


def pick_voice(text: str) -> str:
    if _ES_HINT.search(text) and len(_ES_HINT.findall(text)) >= 2:
        return config.VOICE_ES
    return config.VOICE_EN


def sentences(fragments: Iterable[str]) -> Iterator[str]:
    """Re-assemble a token stream into complete sentences as early as
    possible; flush the remainder at end of stream."""
    buf = ""
    for frag in fragments:
        buf += frag
        while True:
            m = _SENT_END.search(buf)
            if not m:
                break
            sent, buf = buf[:m.end(1)], buf[m.end():]
            if sent.strip():
                yield sent.strip()
    if buf.strip():
        yield buf.strip()


def synth(text: str, voice: str = None) -> str:
    """Synthesize to a temp wav; returns the path."""
    voice = voice or pick_voice(text)
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        [config.PIPER_BIN, "--model", voice, "--output_file", wav],
        input=apply_shim(text).encode(), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav


def play(wav_path: str) -> None:
    subprocess.run(["aplay", "-q", "-D", config.AUDIO_DEVICE, wav_path],
                   check=False)


def speak(text: str, voice: str = None) -> None:
    play(synth(text, voice))


def speak_stream(fragments: Iterable[str],
                 on_event: Callable[[str], None] = None) -> str:
    """Speak a token stream sentence-by-sentence. Returns the full text.

    A dedicated player thread makes the overlap real: synthesis of
    sentence N+1 (and Gemma's generation of N+2) proceed WHILE sentence N
    is audible. The previous version played inline, which serialized
    playback against synthesis and stalled the token stream.

    `on_event` (optional) gets "sentence" when a sentence completes from
    the stream and "audio" when a wav starts playing — chain_test uses it
    to timestamp first-speech.
    """
    wavs: queue.Queue = queue.Queue()

    def player():
        while True:
            wav = wavs.get()
            if wav is None:
                return
            if on_event:
                on_event("audio")
            play(wav)
            os.unlink(wav)                    # synth() leaves temp files

    th = threading.Thread(target=player, daemon=True)
    th.start()
    spoken: list[str] = []
    for sent in sentences(fragments):
        if on_event:
            on_event("sentence")
        wavs.put(synth(sent, pick_voice(sent)))
        spoken.append(sent)
    wavs.put(None)
    th.join()
    return " ".join(spoken)
