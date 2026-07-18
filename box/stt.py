"""Speech-to-text on the Hailo NPU (Whisper base, hailo-apps).

Runs in the HAT's own memory, so STT costs the CPU/Gemma zero RAM — the
whole reason the split works on an 8GB Pi. Measured ~1.2s per 10s chunk.

We shell out to the hailo-apps speech_recognition module (its Whisper
pipeline owns the HailoRT device context) and parse the transcript it
prints between dashed rule lines.
"""
from __future__ import annotations

import os
import re
import subprocess

from . import config

_MODULE = "hailo_apps.python.standalone_apps.speech_recognition.speech_recognition"
_RULE = re.compile(r"^-{10,}$")


def transcribe_wav(wav_path: str) -> str:
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        ["python", "-m", _MODULE, "--audio", wav_path, "--arch", "hailo10h"],
        cwd=os.path.expanduser("~/hailo-apps"),
        capture_output=True, text=True, timeout=120, env=env)
    return _parse(proc.stdout)


def _parse(out: str) -> str:
    """The transcript sits on the line(s) between two dashed rules."""
    lines = out.splitlines()
    rule_idx = [i for i, ln in enumerate(lines) if _RULE.match(ln.strip())]
    if len(rule_idx) >= 2:
        body = lines[rule_idx[0] + 1:rule_idx[1]]
        return " ".join(l.strip() for l in body if l.strip()).strip()
    # fallback: last non-empty, non-status line
    for ln in reversed(lines):
        s = ln.strip()
        if s and not s.startswith(("✓", "Loading", "Transcrib", "Done",
                                   "Architecture", "Variant", "Encoder",
                                   "Decoder", "Initial", "(")):
            return s
    return ""
