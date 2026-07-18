"""Intake camera: capture-and-attach (no inference). A photo bound to a
registry entry so relatives can identify by face, not just name."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from . import config

PHOTO_DIR = Path(config.VAULT) / "photos"


def capture(tag: str = "intake") -> str | None:
    """Snap a still with rpicam-still (Picamera stack). Returns path or None.
    walle-cam.service must be stopped first — it owns the sensor."""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    out = PHOTO_DIR / f"{tag}-{int(time.time())}.jpg"
    for cmd in (["rpicam-still", "-n", "-t", "600", "--width", "1024",
                 "--height", "768", "-o", str(out)],
                ["libcamera-still", "-n", "-t", "600", "--width", "1024",
                 "--height", "768", "-o", str(out)]):
        try:
            subprocess.run(cmd, check=True, timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if out.exists():
                return str(out)
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return None
