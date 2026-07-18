"""Central config — everything env-overridable, sane Pi defaults."""
import os
from pathlib import Path

VAULT = Path(os.environ.get("BOX_VAULT", "/media/caleb/Expansion/vault"))
INDEX_DB = Path(os.environ.get("BOX_INDEX_DB", str(VAULT / "index.db")))
SCRIBE_DB = Path(os.environ.get("BOX_SCRIBE_DB", str(VAULT / "scribe.db")))

OLLAMA_URL = os.environ.get("BOX_OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("BOX_MODEL", "gemma4:e2b")
NUM_CTX = int(os.environ.get("BOX_NUM_CTX", "2048"))
NUM_PREDICT = int(os.environ.get("BOX_NUM_PREDICT", "90"))

# STT runs on the Hailo NPU via hailo-apps (see box/stt.py). No URL needed.

PIPER_BIN = os.environ.get("BOX_PIPER_BIN",
                           str(Path.home() / "piper-venv/bin/piper"))
VOICE_EN = os.environ.get("BOX_VOICE_EN",
                          str(Path.home() / "piper-voices/en_US-lessac-medium.onnx"))
VOICE_ES = os.environ.get("BOX_VOICE_ES",
                          str(Path.home() / "piper-voices/es_MX-claude-high.onnx"))

AUDIO_DEVICE = os.environ.get("BOX_AUDIO_DEVICE", "plughw:2,0")
SAMPLE_RATE = 16000

RETRIEVAL_TOP_K = int(os.environ.get("BOX_TOP_K", "3"))
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200

DASHBOARD_PORT = int(os.environ.get("BOX_DASHBOARD_PORT", "8880"))

# Mute: text-only mode for development off the box (no piper/aplay).
MUTE = os.environ.get("BOX_MUTE", "") == "1"

