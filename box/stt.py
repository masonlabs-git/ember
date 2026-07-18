"""Speech-to-text on the Hailo NPU via the resident whisper-service.

The service (box/whisper_service.py) loads Whisper once, keeps it resident,
and does mel-spectrograms in numpy — so STT runs on the NPU at ~1.2s/chunk
and costs the CPU almost no RAM (torch never loads). This client just hands
it a wav path over a unix socket. Falls back to the one-shot CLI if the
service isn't running.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess

SOCK = "/tmp/whisper.sock"
_MODULE = "hailo_apps.python.standalone_apps.speech_recognition.speech_recognition"
_RULE = re.compile(r"^-{10,}$")


def transcribe_wav(wav_path: str) -> str:
    if os.path.exists(SOCK):
        try:
            return _via_service(wav_path)
        except OSError:
            pass                      # service down — fall through to CLI
    return _via_cli(wav_path)


def _via_service(wav_path: str) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(120)
    s.connect(SOCK)
    s.sendall(wav_path.encode())
    s.shutdown(socket.SHUT_WR)
    chunks = []
    while True:
        b = s.recv(65536)
        if not b:
            break
        chunks.append(b)
    s.close()
    out = b"".join(chunks).decode().strip()
    if out.startswith("__ERR__"):
        raise OSError(out)
    return out


def _via_cli(wav_path: str) -> str:
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        ["python", "-m", _MODULE, "--audio", wav_path, "--arch", "hailo10h"],
        cwd=os.path.expanduser("~/hailo-apps"),
        capture_output=True, text=True, timeout=120, env=env)
    return _parse(proc.stdout)


def _parse(out: str) -> str:
    lines = out.splitlines()
    rule_idx = [i for i, ln in enumerate(lines) if _RULE.match(ln.strip())]
    if len(rule_idx) >= 2:
        body = lines[rule_idx[0] + 1:rule_idx[1]]
        return " ".join(l.strip() for l in body if l.strip()).strip()
    for ln in reversed(lines):
        s = ln.strip()
        if s and not s.startswith(("✓", "Loading", "Transcrib", "Done",
                                   "Architecture", "Variant", "Encoder",
                                   "Decoder", "Initial", "(")):
            return s
    return ""
