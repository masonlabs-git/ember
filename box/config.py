"""Central config — everything env-overridable, sane Pi defaults."""
import os
from pathlib import Path

VAULT = Path(os.environ.get("BOX_VAULT", "/media/caleb/Expansion/vault"))
INDEX_DB = Path(os.environ.get("BOX_INDEX_DB", str(VAULT / "index.db")))
SCRIBE_DB = Path(os.environ.get("BOX_SCRIBE_DB", str(VAULT / "scribe.db")))

OLLAMA_URL = os.environ.get("BOX_OLLAMA_URL", "http://localhost:11434")
# e2b-it-qat: Google's official quantization-aware-trained build of
# Gemma 4 E2B — ~half the RAM of the default q4_K_M blob at near-original
# quality. On an 8GB Pi the difference is "thrashes swap" vs "multi-GB
# headroom", and CPU tok/s scales with bytes touched per token.
MODEL = os.environ.get("BOX_MODEL", "gemma4:e2b-it-qat")
# 1536, not 2048: measured prompt is ~650 tok (persona + 1000-char context
# + question + 3-turn history head-room) and every unused ctx slot is KV
# RAM the 8GB box cannot spare next to the resident model.
NUM_CTX = int(os.environ.get("BOX_NUM_CTX", "1536"))
NUM_PREDICT = int(os.environ.get("BOX_NUM_PREDICT", "80"))

# STT runs on the Hailo NPU via hailo-apps (see box/stt.py). No URL needed.

PIPER_BIN = os.environ.get("BOX_PIPER_BIN",
                           str(Path.home() / "piper-venv/bin/piper"))
# piper's venv site-packages: tts.py imports PiperVoice from here to
# synthesize in-process (same CPython 3.13 as the system interpreter).
PIPER_SITE = os.environ.get(
    "BOX_PIPER_SITE",
    str(Path.home() / "piper-venv/lib/python3.13/site-packages"))
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

# Face-match models (YuNet + SFace ONNX) live in the repo's models/ dir.
MODELS_DIR = Path(os.environ.get("BOX_MODELS_DIR",
                                 str(Path(__file__).parent.parent / "models")))

# Offline places: POI index built from the OSM extract on the vault.
# BOX_LAT/LON = where the box is standing; set once at the venue.
POI_DB = Path(os.environ.get("BOX_POI_DB", str(VAULT / "pois.db")))
BOX_LAT = float(os.environ.get("BOX_LAT", "40.3916"))   # Lehi, UT
BOX_LON = float(os.environ.get("BOX_LON", "-111.8508"))

# Mute: text-only mode for development off the box (no piper/aplay).
MUTE = os.environ.get("BOX_MUTE", "") == "1"

