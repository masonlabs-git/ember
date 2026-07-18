"""The disaster scribe: shelter registry, supply ledger, incident log.

Voice becomes structured, timestamped records — the shelter's memory and
its ICS-214-style paper trail. Plain SQLite on the vault drive.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS registry (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    names TEXT NOT NULL,          -- household members, comma-separated
    medical TEXT DEFAULT '',
    missing TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    photo TEXT DEFAULT ''         -- path to intake photo, if consented
);
CREATE TABLE IF NOT EXISTS supplies (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    item TEXT NOT NULL,
    delta REAL NOT NULL,          -- +received / -distributed
    unit TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    entry TEXT NOT NULL
);
"""


def connect(db_path: Path | str = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or config.SCRIBE_DB),
                           check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- registry

def register(conn, names: str, medical: str = "", missing: str = "",
             phone: str = "", photo: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO registry(ts,names,medical,missing,phone,photo) "
        "VALUES (?,?,?,?,?,?)",
        (time.time(), names.strip(), medical.strip(), missing.strip(),
         phone.strip(), photo))
    conn.commit()
    log(conn, f"Registered household: {names.strip()}")
    return cur.lastrowid


def find_person(conn, name: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, ts, names, medical, missing, phone, photo "
        "FROM registry WHERE names LIKE ? ORDER BY ts DESC",
        (f"%{name.strip()}%",)).fetchall()
    return [_reg_dict(r) for r in rows]


def households(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, ts, names, medical, missing, phone, photo "
        "FROM registry ORDER BY ts DESC").fetchall()
    return [_reg_dict(r) for r in rows]


def _reg_dict(r) -> dict:
    return {"id": r[0], "ts": r[1], "names": r[2], "medical": r[3],
            "missing": r[4], "phone": r[5], "photo": r[6]}


# ---------------------------------------------------------------- supplies

def supply(conn, item: str, delta: float, unit: str = "") -> None:
    conn.execute("INSERT INTO supplies(ts,item,delta,unit) VALUES (?,?,?,?)",
                 (time.time(), item.strip().lower(), delta, unit))
    conn.commit()
    verb = "Received" if delta >= 0 else "Distributed"
    log(conn, f"{verb} {abs(delta):g} {unit} {item.strip()}")


def stock(conn) -> dict[str, float]:
    rows = conn.execute(
        "SELECT item, SUM(delta) FROM supplies GROUP BY item").fetchall()
    return {r[0]: r[1] for r in rows}


def headcount(conn) -> int:
    rows = conn.execute("SELECT names FROM registry").fetchall()
    return sum(len([n for n in r[0].split(",") if n.strip()]) for r in rows)


def water_days_remaining(conn, litres_per_person_day: float = 15.0) -> float:
    """Sphere standard 2.1: 15 L/person/day for all needs (7.5 survival
    minimum). Returns days of water left at current headcount."""
    litres = stock(conn).get("water", 0.0)
    people = max(headcount(conn), 1)
    return litres / (litres_per_person_day * people)


# -------------------------------------------------------------------- log

def log(conn, entry: str) -> None:
    conn.execute("INSERT INTO log(ts,entry) VALUES (?,?)",
                 (time.time(), entry))
    conn.commit()


def recent_log(conn, hours: float = 8.0) -> list[tuple[float, str]]:
    since = time.time() - hours * 3600
    return conn.execute(
        "SELECT ts, entry FROM log WHERE ts > ? ORDER BY ts",
        (since,)).fetchall()


def shift_briefing_text(conn, hours: float = 8.0) -> str:
    """Raw material for the LLM shift-change summary."""
    lines = [f"- {time.strftime('%H:%M', time.localtime(ts))} {e}"
             for ts, e in recent_log(conn, hours)]
    s = stock(conn)
    inv = ", ".join(f"{v:g} {k}" for k, v in sorted(s.items())) or "none"
    return (f"ACTIVITY LOG (last {hours:g}h):\n" + "\n".join(lines or ["- none"])
            + f"\n\nCURRENT INVENTORY: {inv}"
            + f"\nREGISTERED PEOPLE: {headcount(conn)}")
