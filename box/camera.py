"""Intake camera: live MJPEG viewfinder + capture-from-stream.

One process owns the sensor (the dashboard runs Cam, an rpicam-vid MJPEG
reader); everyone else — including the brain's voice "do you recognize
me?" — takes the latest frame over HTTP. Stills are grabbed FROM the
stream, so preview and capture never fight over the camera. Falls back
to one-shot rpicam-still when no stream is running (tests, headless).
"""
from __future__ import annotations

import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import config

PHOTO_DIR = Path(config.VAULT) / "photos"

_SOI, _EOI = b"\xff\xd8", b"\xff\xd9"


def split_jpegs(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete JPEG frames out of an MJPEG byte stream; returns
    (frames, unconsumed_tail)."""
    frames = []
    while True:
        s = buf.find(_SOI)
        if s < 0:
            return frames, b""
        e = buf.find(_EOI, s + 2)
        if e < 0:
            return frames, buf[s:]
        frames.append(buf[s:e + 2])
        buf = buf[e + 2:]


class Cam:
    """Resident camera: rpicam-vid MJPEG at 10fps, latest frame in RAM."""

    def __init__(self, width=1024, height=768, fps=10):
        self.frame: bytes | None = None
        self.frame_at = 0.0
        self._size = (width, height, fps)
        self._lock = threading.Lock()
        self._proc = None
        self._spawn()
        threading.Thread(target=self._pump, daemon=True).start()

    def _spawn(self):
        w, h, fps = self._size
        self._proc = subprocess.Popen(
            ["rpicam-vid", "-n", "-t", "0", "--codec", "mjpeg",
             "--width", str(w), "--height", str(h),
             "--framerate", str(fps), "-o", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _pump(self):
        # self-healing: rpicam-vid can die on a CSI hiccup (field
        # failure: zombie stream -> viewfinder dead until restart).
        # Reap it and respawn forever.
        tail = b""
        while True:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                self._proc.wait()
                time.sleep(2)
                try:
                    self._spawn()
                    tail = b""
                except Exception:
                    time.sleep(5)
                continue
            frames, tail = split_jpegs(tail + chunk)
            if frames:
                with self._lock:
                    self.frame = frames[-1]
                    self.frame_at = time.time()

    def latest(self, max_age_s: float = 3.0) -> bytes | None:
        with self._lock:
            if self.frame and time.time() - self.frame_at <= max_age_s:
                return self.frame
        return None

    def snap(self, tag: str = "intake") -> str | None:
        f = self.latest()
        if f is None:
            return None
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        out = PHOTO_DIR / f"{tag}-{int(time.time())}.jpg"
        out.write_bytes(f)
        return str(out)


# The dashboard sets this to its Cam instance; same-process callers get
# frames directly, other processes go through /preview.jpg.
live_cam: Cam | None = None


def capture(tag: str = "intake") -> str | None:
    """Best available still: live stream (in-process, then HTTP), else a
    one-shot rpicam-still."""
    if live_cam is not None:
        got = live_cam.snap(tag)
        if got:
            return got
    try:
        req = urllib.request.urlopen(
            f"http://127.0.0.1:{config.DASHBOARD_PORT}/preview.jpg",
            timeout=3)
        data = req.read()
        if data.startswith(_SOI):
            PHOTO_DIR.mkdir(parents=True, exist_ok=True)
            out = PHOTO_DIR / f"{tag}-{int(time.time())}.jpg"
            out.write_bytes(data)
            return str(out)
    except OSError:
        pass
    out = PHOTO_DIR / f"{tag}-{int(time.time())}.jpg"
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    for cmd in (["rpicam-still", "-n", "-t", "600", "--width", "1024",
                 "--height", "768", "-o", str(out)],
                ["libcamera-still", "-n", "-t", "600", "--width", "1024",
                 "--height", "768", "-o", str(out)]):
        try:
            subprocess.run(cmd, check=True, timeout=15,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            if out.exists():
                return str(out)
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return None
