"""Central config — everything env-overridable, sane Pi defaults."""
import os
from pathlib import Path

# Vault: the USB drive when present, else the SD-card copy made by
# deploy/travel-vault.sh — so unplugging the drive (USB-power-budget
# demos, drive failure) degrades to a fully working box, not a dead one.
_HDD_VAULT = Path("/media/caleb/Expansion/vault")
_SD_VAULT = Path.home() / "vault"


def _vault_root() -> Path:
    # A stale mountpoint dir survives the drive being unplugged and
    # passes a bare exists() while every stat under it raises
    # PermissionError (field failure: both services dead at the venue).
    # Only a readable index proves the drive is really there.
    try:
        if (_HDD_VAULT / "index.db").is_file():
            return _HDD_VAULT
    except OSError:
        pass
    return _SD_VAULT


VAULT = Path(os.environ.get("BOX_VAULT", str(_vault_root())))
INDEX_DB = Path(os.environ.get("BOX_INDEX_DB", str(VAULT / "index.db")))
SCRIBE_DB = Path(os.environ.get("BOX_SCRIBE_DB", str(VAULT / "scribe.db")))

OLLAMA_URL = os.environ.get("BOX_OLLAMA_URL", "http://localhost:11434")
# e2b-it-qat: Google's official quantization-aware-trained build of
# Gemma 4 E2B — ~half the RAM of the default q4_K_M blob at near-original
# quality. On an 8GB Pi the difference is "thrashes swap" vs "multi-GB
# headroom", and CPU tok/s scales with bytes touched per token.
MODEL = os.environ.get("BOX_MODEL", "gemma4:e2b-it-qat")

# Hearth tier: an optional on-prem hub (Spark-class) running the same
# ollama API with a bigger Gemma. The box tries it first and falls back
# to local silently — the hub is an upgrade, never a dependency.
HUB_URL = os.environ.get("BOX_HUB_URL", "")
HUB_MODEL = os.environ.get("BOX_HUB_MODEL", "gemma4:12b-it-qat")
# 1536, not 2048: measured prompt is ~650 tok (persona + 1000-char context
# + question + 3-turn history head-room) and every unused ctx slot is KV
# RAM the 8GB box cannot spare next to the resident model.
NUM_CTX = int(os.environ.get("BOX_NUM_CTX", "1536"))
NUM_PREDICT = int(os.environ.get("BOX_NUM_PREDICT", "80"))
# 3 threads, not 4: the venue supply is an Apple brick (5V caps at 3A =
# 15W) and 4-thread Gemma bursts crest it — firmware-confirmed
# undervolt (0x50000) and two hard deaths. ~2s slower per answer, alive.
NUM_THREAD = int(os.environ.get("BOX_NUM_THREAD", "3"))

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

AUDIO_DEVICE = os.environ.get("BOX_AUDIO_DEVICE",
                              "plughw:CARD=Plus,DEV=0")
# CARD=Plus (the EMEET, by name) — USB re-enumeration at the
# venue must not be able to point the box at the wrong card
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

# Staff (FEMA-worker) mode PIN for the dashboard — unlocks destructive
# actions like removing a registry entry. Residents never need it.
STAFF_PIN = os.environ.get("BOX_STAFF_PIN", "3637")

