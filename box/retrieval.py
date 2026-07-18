"""Offline retrieval: SQLite FTS5 (BM25) over the survival corpus.

No vector models, no servers — measured RAM budget on an 8GB Pi leaves no
room for a resident embedder beside Gemma, and BM25 over terminology-rich
field manuals retrieves on par. One .db file on the vault drive.
"""
from __future__ import annotations

import html
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
-- Small fast index of the hand-curated operational sources (~a few thousand
-- chunks). Queried FIRST so survival/shelter answers never touch the 540k
-- Wikipedia rows on the slow USB drive. Same schema, tiny table.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_authority USING fts5(
    source, title, text,
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
    text = html.unescape(html.unescape(text))   # double-escaped ZIM extracts
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

# High-frequency function words: excluded from queries — they bloat FTS
# posting-list intersections into multi-second scans on the vault HDD.
STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from had has have "
    "how i if in is it its me my of on or our should so that the their "
    "them then there these they this to us was we what when where which "
    "who will with would you your".split())

# Authority tier: hand-curated operational sources outrank the Wikipedia
# bulk layer — a shelter answer should cite FM 21-76 or Sphere before trivia.
BULK_SOURCES = ("Wikipedia Medicine",)

# Protocol pins: when a query names a formal protocol, its source is
# guaranteed a seat in the context regardless of BM25's vote — protocol
# coaching must be deterministic, not probabilistic.
PROTOCOL_PINS = {
    "triage": "START Triage Protocol",
    "ics": "NIMS ICS Forms",
}


# Query expansion: map colloquial words to the vocabulary the field manuals
# actually use, so "make water safe to drink" reaches "purify/boil/disinfect".
EXPAND = {
    "safe": ["purify", "disinfect"],
    "drink": ["potable", "drinking"],
    "clean": ["purify", "disinfect"],
    "water": ["water", "boil"],
    "cut": ["wound", "laceration"],
    "bleeding": ["hemorrhage", "wound"],
    "hurt": ["injury", "wound"],
    "food": ["ration", "edible"],
    "poop": ["latrine", "sanitation"],
    "bathroom": ["latrine", "sanitation"],
}


def _terms(q: str) -> list[str]:
    words = _WORD.findall(q.lower())
    content = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    expanded = list(content)
    for w in content:
        for extra in EXPAND.get(w, []):
            if extra not in expanded:
                expanded.append(extra)
    return expanded[:12]


def build_authority(conn: sqlite3.Connection) -> int:
    """(Re)populate the small authority table by copying every non-bulk row
    out of `chunks`. Fast (~a few thousand rows); makes survival queries
    never touch the 540k Wikipedia rows."""
    conn.execute("DELETE FROM chunks_authority")
    placeholders = ",".join("?" * len(BULK_SOURCES))
    conn.execute(
        f"INSERT INTO chunks_authority(source, title, text) "
        f"SELECT source, title, text FROM chunks "
        f"WHERE source NOT IN ({placeholders})", list(BULK_SOURCES))
    conn.commit()
    return conn.execute(
        "SELECT count(*) FROM chunks_authority").fetchone()[0]


def _match(conn: sqlite3.Connection, table: str, fq: str,
           limit: int) -> list[Hit]:
    """Query one FTS table by BM25. `table` is 'chunks_authority' (tiny/fast)
    or 'chunks' (the full 540k-row bulk index)."""
    rows = conn.execute(
        f"SELECT source, title, text, bm25({table}) AS score "
        f"FROM {table} WHERE {table} MATCH ? ORDER BY score LIMIT ?",
        (fq, limit)).fetchall()
    return [Hit(source=r[0], title=r[1], text=html.unescape(r[2]),
                score=r[3]) for r in rows]


def search(conn: sqlite3.Connection, query: str,
           top_k: int = None) -> list[Hit]:
    """Authority-first retrieval. The small field-manual table answers almost
    every emergency query in <50ms; the huge Wikipedia table is touched only
    when the manuals come up empty."""
    terms = _terms(query)
    if not terms:
        return []
    k = top_k or config.RETRIEVAL_TOP_K
    and_q = " ".join(f'"{w}"' for w in terms)      # implicit AND
    or_q = " OR ".join(f'"{w}"' for w in terms)

    hits: list[Hit] = []
    seen: set[tuple] = set()

    def take(new: list[Hit]) -> None:
        for h in new:
            key = (h.source, h.title, h.text[:80])
            if key not in seen and len(hits) < k:
                seen.add(key)
                hits.append(h)

    def safe(fn):
        try:
            return fn()
        except sqlite3.OperationalError:
            return []

    for term, source in PROTOCOL_PINS.items():     # deterministic protocols
        if term in terms:
            rows = conn.execute(
                "SELECT source, title, text, 0.0 FROM chunks_authority "
                "WHERE source = ? LIMIT 2", (source,)).fetchall()
            take([Hit(source=r[0], title=r[1],
                      text=html.unescape(r[2]), score=r[3]) for r in rows])

    # authority table first — tiny, so both AND and OR are instant
    take(safe(lambda: _match(conn, "chunks_authority", and_q, k)))
    if len(hits) < k:
        take(safe(lambda: _match(conn, "chunks_authority", or_q, k)))
    # bulk Wikipedia only as a genuine last resort (obscure medical terms)
    if not hits:
        take(safe(lambda: _match(conn, "chunks", and_q, k)))
    return hits


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
