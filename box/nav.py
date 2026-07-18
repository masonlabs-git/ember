"""Offline places: OSM -> SQLite POI index; nearest-X answers by voice.

"Hey Ember, how far is the nearest hospital?" -> name, distance, compass
bearing, street — computed, not generated (no LLM in the loop, so the
numbers are exact) and fully offline from the Utah OSM extract on the
vault. The box's own location comes from BOX_LAT/BOX_LON (set it once at
the venue); turn-by-turn routing is deliberately out of scope — distance
plus bearing plus street name is what a stressed person can use.

Build the index once:  python3 -m box.nav /path/to/utah-latest.osm.pbf
(nodes-only streaming parse: no location index, so it runs in ~flat RAM
beside the resident Gemma; most OSM amenity POIs are nodes.)
"""
from __future__ import annotations

import math
import re
import sqlite3
import sys

from . import config

# Spoken alias -> (osm key, accepted values). Disaster-relevant kinds.
KINDS = {
    "hospital":      ("amenity", ("hospital", "clinic", "doctors")),
    "pharmacy":      ("amenity", ("pharmacy",)),
    "fire station":  ("amenity", ("fire_station",)),
    "police":        ("amenity", ("police",)),
    "shelter":       ("amenity", ("shelter", "community_centre")),
    "school":        ("amenity", ("school",)),
    "grocery store": ("shop",    ("supermarket", "convenience")),
    "gas station":   ("amenity", ("fuel",)),
    "water":         ("amenity", ("drinking_water", "water_point")),
    "church":        ("amenity", ("place_of_worship",)),
}
ALIASES = {
    "hospital": "hospital", "emergency room": "hospital", "er": "hospital",
    "clinic": "hospital", "doctor": "hospital",
    "pharmacy": "pharmacy", "drug store": "pharmacy",
    "drugstore": "pharmacy",
    "fire station": "fire station", "fire department": "fire station",
    "police station": "police", "police": "police",
    "shelter": "shelter", "community center": "shelter",
    "school": "school",
    "grocery store": "grocery store", "grocery": "grocery store",
    "supermarket": "grocery store", "food store": "grocery store",
    "gas station": "gas station", "gas": "gas station",
    "fuel": "gas station",
    "drinking water": "water", "water fountain": "water",
    "church": "church",
}
_INTENT = re.compile(
    r"\b(nearest|closest|how far|where is|where's|get to|distance to|"
    r"directions to)\b", re.I)

_COMPASS = ("north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest")


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def bearing_word(lat1, lon1, lat2, lon2) -> str:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    return _COMPASS[int((deg + 22.5) // 45) % 8]


def connect(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or config.POI_DB),
                           check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS pois ("
                 "kind TEXT, name TEXT, lat REAL, lon REAL, street TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS pois_kind ON pois(kind)")
    return conn


def nearest(conn, kind: str, lat: float, lon: float, n: int = 3):
    # bounding-box prefilter (~50 miles), exact haversine sort after
    dlat = 50 / 69.0
    dlon = 50 / (69.0 * max(math.cos(math.radians(lat)), 0.2))
    rows = conn.execute(
        "SELECT name, lat, lon, street FROM pois WHERE kind=? "
        "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (kind, lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
    scored = [(haversine_miles(lat, lon, r[1], r[2]), r) for r in rows]
    scored.sort(key=lambda t: t[0])
    return scored[:n]


def parse_kind(question: str) -> str | None:
    ql = question.lower()
    if not _INTENT.search(ql):
        return None
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"\b{alias}\b", ql):
            return ALIASES[alias]
    return None


def maybe_answer(question: str,
                 lat: float = None, lon: float = None) -> str | None:
    """Deterministic nearest-X answer, or None if not a places question
    (or no POI index has been built)."""
    kind = parse_kind(question)
    if kind is None:
        return None
    if not config.POI_DB.exists():
        return None
    lat = lat if lat is not None else config.BOX_LAT
    lon = lon if lon is not None else config.BOX_LON
    hits = nearest(connect(), kind, lat, lon)
    if not hits:
        return f"I have no {kind} in my offline map."
    parts = []
    for i, (miles, (name, plat, plon, street)) in enumerate(hits[:2]):
        where = bearing_word(lat, lon, plat, plon)
        on = f" on {street}" if street else ""
        label = name or f"an unnamed {kind}"
        lead = "The nearest" if i == 0 else "After that,"
        noun = f" {kind}" if i == 0 else ""
        parts.append(f"{lead}{noun} is {label}, "
                     f"{miles:.1f} miles {where}{on}.")
    return " ".join(parts)


# ------------------------------------------------------------- index build

def build(pbf_path: str, db_path=None) -> int:
    """Stream the PBF once (nodes only — flat RAM) into the POI table."""
    import osmium

    value_to_kind = {}
    for kind, (key, values) in KINDS.items():
        for v in values:
            value_to_kind[(key, v)] = kind

    conn = connect(db_path)
    conn.execute("DELETE FROM pois")
    rows = []

    class H(osmium.SimpleHandler):
        def node(self, n):
            for key in ("amenity", "shop"):
                kind = value_to_kind.get((key, n.tags.get(key, "")))
                if kind:
                    rows.append((kind, n.tags.get("name", ""),
                                 n.location.lat, n.location.lon,
                                 n.tags.get("addr:street", "")))
                    return

    H().apply_file(pbf_path)
    conn.executemany("INSERT INTO pois VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    count = build(sys.argv[1])
    print(f"indexed {count} POIs into {config.POI_DB}")
