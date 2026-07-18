"""Resident, torch-free Whisper STT service on the Hailo NPU.

Two problems with calling the hailo-apps CLI per query:
  1. it re-imports torch (~1.5GB CPU RAM) every call, which evicts Gemma;
  2. it reloads the HEFs every call (~10s of cold start).

This service fixes both: it loads the WhisperPipeline ONCE and stays resident,
and it reimplements the only torch-dependent step (the mel-spectrogram) in
numpy — so torch never loads and STT costs the CPU almost nothing. Inference
still runs on the NPU. Clients send a wav path over a unix socket and get the
transcript back.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

import numpy as np

SOCK = "/tmp/whisper.sock"
APP_DIR = os.path.expanduser(
    "~/hailo-apps/hailo_apps/python/standalone_apps/speech_recognition")
MODELS = "/usr/local/hailo/resources/models/hailo10h"
ENCODER = f"{MODELS}/base-whisper-encoder-10s.hef"
DECODER = f"{MODELS}/base-whisper-decoder-10s-out-seq-64.hef"
NPY_DIR = "/usr/local/hailo/resources/npy"

# Whisper mel constants (universal for the base model)
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80

_WIN = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))  # periodic hann


def _load_audio(path: str) -> np.ndarray:
    """Decode any audio file to mono 16k float32 via ffmpeg (no torch)."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "0", "-i", path,
         "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _log_mel(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Numpy port of the app's torch log_mel_spectrogram (center=True STFT)."""
    pad = N_FFT // 2
    a = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(a) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = a[idx] * _WIN                                # (n_frames, N_FFT)
    spec = np.fft.rfft(frames, axis=1)                    # (n_frames, 201)
    mags = (np.abs(spec[:-1]) ** 2).T                     # (201, n_frames-1)
    mel = mel_filters @ mags                              # (80, T)
    log = np.log10(np.clip(mel, 1e-10, None))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


def _preprocess(audio: np.ndarray, mel_filters: np.ndarray,
                chunk_length: int) -> list[np.ndarray]:
    seg = chunk_length * SAMPLE_RATE
    mels = []
    for start in range(0, max(len(audio), 1), seg):
        chunk = audio[start:start + seg]
        if chunk.size == 0:
            break
        if len(chunk) < seg:
            chunk = np.pad(chunk, (0, seg - len(chunk)))
        mel = _log_mel(chunk, mel_filters)               # (80, T)
        mel = mel[None, :, None, :]                       # (1,80,1,T)
        mels.append(np.transpose(mel, [0, 2, 3, 1]))      # (1,1,T,80) NHWC
    return mels


def main() -> None:
    sys.path.insert(0, APP_DIR)
    from whisper_pipeline import WhisperPipeline          # no torch inside
    from postprocessing import clean_transcription

    mel_filters = np.load(os.path.join(APP_DIR, "assets", "mel_filters.npz")
                          )[f"mel_{N_MELS}"]
    pipeline = WhisperPipeline(ENCODER, DECODER, variant="base",
                               npy_dir=NPY_DIR, add_embed=False)
    chunk_length = pipeline.get_chunk_length()

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o666)
    srv.listen(4)
    print(f"whisper-service ready (chunk {chunk_length}s) on {SOCK}",
          flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            wav = conn.recv(4096).decode().strip()
            audio = _load_audio(wav)
            texts = []
            for mel in _preprocess(audio, mel_filters, chunk_length):
                pipeline.send_data(mel)
                texts.append(pipeline.get_transcription())
            conn.sendall(clean_transcription(" ".join(texts)).encode())
        except Exception as e:                            # keep serving
            conn.sendall(f"__ERR__ {e}".encode())
        finally:
            conn.close()


if __name__ == "__main__":
    main()
