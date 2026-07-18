"""Face-match reunification: find a registered person from a photo.

OpenCV YuNet (detector) + SFace (112x112 embedding, cosine match) — both
run on the Pi's CPU in ~100ms/image from ~40MB of ONNX. A relative shows
any photo of the person they're looking for; the box matches it against
consented intake photos. Nothing ever leaves the box.

Registry embeddings cache in the scribe DB (face_cache, keyed by photo
path) so repeat searches don't re-embed the whole board.
"""
from __future__ import annotations

import os

import numpy as np

from . import config

# SFace's published cosine-similarity threshold for "same person".
SAME_PERSON = 0.363

_det = None
_rec = None


def _engines():
    global _det, _rec
    if _det is None:
        import cv2
        _det = cv2.FaceDetectorYN.create(
            str(config.MODELS_DIR / "yunet.onnx"), "", (320, 320),
            score_threshold=0.6)
        _rec = cv2.FaceRecognizerSF.create(
            str(config.MODELS_DIR / "sface.onnx"), "")
    return _det, _rec


def embed(photo_path: str) -> np.ndarray | None:
    """Embedding of the LARGEST face in the image, or None if no face."""
    import cv2
    det, rec = _engines()
    img = cv2.imread(photo_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])       # widest x tallest
    aligned = rec.alignCrop(img, face)
    return rec.feature(aligned).flatten().astype(np.float32)


def _cached_embed(sconn, photo_path: str) -> np.ndarray | None:
    sconn.execute("CREATE TABLE IF NOT EXISTS face_cache "
                  "(photo TEXT PRIMARY KEY, emb BLOB)")
    row = sconn.execute("SELECT emb FROM face_cache WHERE photo=?",
                        (photo_path,)).fetchone()
    if row is not None:
        return (np.frombuffer(row[0], np.float32)
                if row[0] is not None else None)
    e = embed(photo_path)
    sconn.execute("INSERT OR REPLACE INTO face_cache VALUES (?,?)",
                  (photo_path, e.tobytes() if e is not None else None))
    sconn.commit()
    return e


def match(sconn, query_photo: str, households: list[dict]) -> list[dict]:
    """Rank registered households by face similarity to the query photo.

    Returns [{score, same_person, **household}] best-first; empty list if
    the query photo has no detectable face.
    """
    q = embed(query_photo)
    if q is None:
        return []
    qn = q / np.linalg.norm(q)
    out = []
    for hh in households:
        if not hh.get("photo") or not os.path.exists(hh["photo"]):
            continue
        e = _cached_embed(sconn, hh["photo"])
        if e is None:
            continue
        score = float(qn @ (e / np.linalg.norm(e)))
        out.append({"score": score, "same_person": score >= SAME_PERSON,
                    **hh})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
