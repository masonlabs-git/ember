"""Offline retrieval: SQLite FTS5 (BM25) over the survival corpus.

No vector models, no servers — measured RAM budget on an 8GB Pi leaves no
room for a resident embedder beside Gemma, and BM25 over terminology-rich
field manuals retrieves on par. One .db file on the vault drive.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import config

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    source,             -- short source name, e.g. 'FM 21-76'
    title,              -- article/section title when known
    text,
    tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Hit:
    source: str
    title: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.title}" if self.title else self.source


def connect(db_path: Path | str = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or config.INDEX_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


# ----------------------------------------------------------------- ingest

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str,
               size: int = config.CHUNK_CHARS,
               overlap: int = config.CHUNK_OVERLAP):
    """Paragraph-packing chunker: fill to ~size chars, break on paragraph
    boundaries when possible, carry a tail overlap for continuity."""
    text = _clean(text)
    if len(text) <= size:
        if text:
            yield text
        return
    paras = re.split(r"\n\s*\n", text)
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        while len(p) > size:          # pathological paragraph: hard split
            head, p = p[:size], p[max(0, size - overlap):]
            if buf:
                yield buf
                buf = ""
            yield head
        if len(buf) + len(p) + 2 > size and buf:
            yield buf
            buf = buf[-overlap:] + "\n\n" + p if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        yield buf.strip()


def ingest_txt(conn: sqlite3.Connection, path: Path, source: str) -> int:
    n = 0
    text = path.read_text(errors="ignore")
    rows = ((source, "", c) for c in chunk_text(text))
    cur = conn.executemany(
        "INSERT INTO chunks(source, title, text) VALUES (?,?,?)", rows)
    n = cur.rowcount
    conn.commit()
    return n


def ingest_jsonl(conn: sqlite3.Connection, path: Path, source: str,
                 max_chunk: int = 2400) -> int:
    """One JSON object per line: {title, text}. Used for the Wikipedia
    medicine extract — bigger chunks to cap row count."""
    n = 0
    batch = []
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = d.get("title", "")
            for c in chunk_text(d.get("text", ""), size=max_chunk):
                batch.append((source, title, c))
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO chunks(source, title, text) VALUES (?,?,?)",
                    batch)
                conn.commit()
                n += len(batch)
                batch = []
    if batch:
        conn.executemany(
            "INSERT INTO chunks(source, title, text) VALUES (?,?,?)", batch)
        conn.commit()
        n += len(batch)
    return n


# ----------------------------------------------------------------- search

_WORD = re.compile(r"[A-Za-z0-9']+")


def _fts_query(q: str) -> str:
    """User text -> safe FTS5 OR-query. OR keeps recall high; bm25 does the
    ranking. Quoting each term neutralizes FTS5 operator syntax."""
    words = _WORD.findall(q.lower())
    words = [w for w in words if len(w) > 1][:12]
    return " OR ".join(f'"{w}"' for w in words)


def search(conn: sqlite3.Connection, query: str,
           top_k: int = None) -> list[Hit]:
    fq = _fts_query(query)
    if not fq:
        return []
    rows = conn.execute(
        "SELECT source, title, text, bm25(chunks) AS score "
        "FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
        (fq, top_k or config.RETRIEVAL_TOP_K)).fetchall()
    return [Hit(source=r[0], title=r[1], text=r[2], score=r[3])
            for r in rows]


def context_block(hits: list[Hit], budget_chars: int = 3600) -> str:
    """Render hits into the prompt context, trimmed to a char budget so the
    2560-token window never overflows."""
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        take = h.text[:max(0, budget_chars - used)]
        if not take:
            break
        parts.append(f"[{i}] ({h.citation})\n{take}")
        used += len(take)
    return "\n\n".join(parts)
