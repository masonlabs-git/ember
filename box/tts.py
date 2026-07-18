"""Piper TTS: pronunciation shim, sentence streaming, language pick.

Sentence streaming is the perceived-latency fix: speak sentence one while
sentence two still generates.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import wave
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


# In-process piper voices, loaded once per process. Spawning the piper CLI
# per sentence reloaded onnxruntime + the voice every time: ~1s extra
# latency and a ~250MB transient RAM spike per sentence — the spike is what
# killed piper mid-answer next to the 6.8GB resident Gemma (swap full,
# malloc fail). Resident costs the same RAM once and holds it.
_voices: dict = {}


def _voice(model_path: str):
    v = _voices.get(model_path)
    if v is None:
        if config.PIPER_SITE not in sys.path:
            sys.path.insert(0, config.PIPER_SITE)
        from piper import PiperVoice
        v = _voices[model_path] = PiperVoice.load(model_path)
    return v


def speakable(text: str) -> bool:
    """False for fragments with nothing pronounceable — e.g. the ' [1].'
    tail left when a token cap cuts generation right after a citation
    strips to a bare '.'. Piper yields zero audio chunks for phoneme-less
    text and the wav write crashes ('# channels not specified')."""
    return bool(re.search(r"\w", strip_citations(text)))


def synth(text: str, voice: str = None) -> str:
    """Synthesize to a temp wav; returns the path."""
    voice = voice or pick_voice(text)
    text = apply_shim(text)
    if not re.search(r"\w", text):
        raise ValueError("nothing speakable after citation strip")
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        with wave.open(wav, "wb") as wf:
            _voice(voice).synthesize_wav(text, wf)
    except Exception:                     # in-process path broken → CLI
        with open("/tmp/piper.err", "ab") as err:
            subprocess.run(
                [config.PIPER_BIN, "--model", voice, "--output_file", wav],
                input=text.encode(), check=True,
                stdout=subprocess.DEVNULL, stderr=err)
    return wav


def play(wav_path: str) -> None:
    # -B 200ms buffer: absorbs scheduler stalls while Gemma saturates the
    # cores (underruns were audible as crackles mid-sentence without it).
    subprocess.run(["aplay", "-q", "-B", "200000",
                    "-D", config.AUDIO_DEVICE, wav_path], check=False)


def speak(text: str, voice: str = None) -> None:
    play(synth(text, voice))


# Pre-synthesized acknowledgment lines. Prefill of a fresh question costs
# seconds of silence on the Pi; playing a canned "checking my sources"
# the instant STT lands makes the box feel alive while Gemma works —
# and it costs nothing at answer time because the wavs are made at boot.
_ACKS: list[str] = []
_ack_i = 0

ACK_LINES = ("Checking my field manuals.",
             "One moment. Searching my sources.",
             "Let me check my references.")


def prepare_acks(lines: Iterable[str] = ACK_LINES) -> None:
    """Synthesize the acknowledgment lines once (boot time)."""
    for line in lines:
        _ACKS.append(synth(line, config.VOICE_EN))


def next_ack() -> str | None:
    """Rotating pre-synthesized ack wav, or None if not prepared."""
    global _ack_i
    if not _ACKS:
        return None
    wav = _ACKS[_ack_i % len(_ACKS)]
    _ack_i += 1
    return wav


def speak_stream(fragments: Iterable[str],
                 on_event: Callable[[str], None] = None,
                 preroll: str = None) -> str:
    """Speak a token stream sentence-by-sentence. Returns the full text.

    A dedicated player thread makes the overlap real: synthesis of
    sentence N+1 (and Gemma's generation of N+2) proceed WHILE sentence N
    is audible. `preroll` (a pre-synthesized wav, e.g. next_ack()) plays
    immediately — DURING prefill — and the queue guarantees it finishes
    before sentence one, so the streams never overlap.

    `on_event` (optional) gets "sentence" when a sentence completes from
    the stream and "audio" when a wav starts playing — chain_test uses it
    to timestamp first-speech.
    """
    wavs: queue.Queue = queue.Queue()

    def player():
        while True:
            item = wavs.get()
            if item is None:
                return
            wav, ephemeral = item
            if on_event:
                on_event("audio")
            play(wav)
            if ephemeral:
                os.unlink(wav)                # synth() leaves temp files

    th = threading.Thread(target=player, daemon=True)
    th.start()
    if preroll:
        wavs.put((preroll, False))            # reusable — never delete
    spoken: list[str] = []
    for sent in sentences(fragments):
        if on_event:
            on_event("sentence")
        spoken.append(sent)          # full text for the display/log …
        if speakable(sent):          # … but only real content is spoken
            wavs.put((synth(sent, pick_voice(sent)), True))
    wavs.put(None)
    th.join()
    return " ".join(spoken)
