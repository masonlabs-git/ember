#!/usr/bin/env python3
"""Build the FTS5 index from the vault corpus.

Usage:  BOX_VAULT=/path/to/vault python3 ingest.py [--fresh]
"""
import sys
import time
from pathlib import Path

from box import config
from box.retrieval import connect, ingest_jsonl, ingest_txt

# (filename, short source name) — text layers of the vault
TXT_SOURCES = [
    ("corpus-core/fm21-76-survival.txt", "FM 21-76 Survival"),
    ("corpus-core/fm4-25.11-first-aid.txt", "FM 4-25.11 First Aid"),
    ("corpus-core/where-there-is-no-doctor.txt", "Where There Is No Doctor"),
    ("corpus-core/fema-are-you-ready.txt", "FEMA Are You Ready"),
    ("corpus-core/fema-earthquake-checklist.txt", "FEMA Earthquake Checklist"),
    ("corpus-extended/sphere-handbook.txt", "Sphere Handbook 2018"),
    ("corpus-extended/start-triage.txt", "START Triage Protocol"),
    ("corpus-extended/nims-ics-forms-booklet.txt", "NIMS ICS Forms"),
]
JSONL_SOURCES = [
    ("corpus-extended/wikipedia-medicine.jsonl", "Wikipedia Medicine"),
]


def main() -> None:
    if "--fresh" in sys.argv and config.INDEX_DB.exists():
        config.INDEX_DB.unlink()
    conn = connect()
    t0 = time.time()
    total = 0
    for rel, name in TXT_SOURCES:
        p = config.VAULT / rel
        if not p.exists():
            print(f"SKIP (missing): {rel}")
            continue
        n = ingest_txt(conn, p, name)
        total += n
        print(f"{name}: {n} chunks")
    for rel, name in JSONL_SOURCES:
        p = config.VAULT / rel
        if not p.exists():
            print(f"SKIP (missing): {rel}")
            continue
        n = ingest_jsonl(conn, p, name)
        total += n
        print(f"{name}: {n} chunks")
    conn.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
    conn.commit()
    print(f"DONE: {total} chunks in {time.time()-t0:.0f}s -> {config.INDEX_DB}")


if __name__ == "__main__":
    main()
