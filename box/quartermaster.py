"""Voice supply ledger: "we gave out 40 liters of water" -> a deducted,
audit-logged ledger line and a spoken confirmation with what remains.

Deterministic parsing, no LLM — inventory numbers must be exact. The
scribe ledger and its activity log do the bookkeeping, so every spoken
transaction also lands in the shift brief automatically.
"""
from __future__ import annotations

import re

from . import scribe

_OUT = r"(?:gave|give|giving|gave\s+out|given|handed|distributed|" \
       r"dispensed|used|passed\s+out)"
_IN = r"(?:received|got|getting|delivered|donated|restocked|picked\s+up|" \
      r"brought)"

_UNITS = r"(?:liters?|litres?|gallons?|bottles?|cases?|boxes?|packs?|" \
         r"pairs?|rolls?)"

# "we gave out 40 liters of water" / "we received 200 meals"
_TXN = re.compile(
    rf"\b(?P<dir>{_OUT}|{_IN})\b[^0-9]*?"
    rf"(?P<qty>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNITS})?\s*(?:of\s+)?"
    rf"(?P<item>[a-z][a-z\s]{{2,40}}?)\s*[.!?]?$", re.I)

# "how much water do we have left" — left/have/remaining is what keeps
# "how much water do 85 people NEED" flowing to the RAG instead
_QUERY = re.compile(
    r"\bhow (?:much|many)\s+(?P<item>[a-z][a-z\s]{1,30}?)\s+"
    r"(?:do we have|is left|are left|left|remaining|in stock|on hand)",
    re.I)

# whisper usually writes digits, but insure the common spoken numbers
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "eighty": 80,
            "hundred": 100, "a hundred": 100, "two hundred": 200}


def _words_to_digits(text: str) -> str:
    for w in sorted(_WORDNUM, key=len, reverse=True):
        text = re.sub(rf"\b{w}\b", str(_WORDNUM[w]), text, flags=re.I)
    return text


def _canonical_item(raw: str, stock: dict) -> str:
    """Snap a spoken item onto an existing stock line when one matches,
    so 'meals' hits 'MRE meals' instead of opening a duplicate line."""
    raw = raw.strip().lower().rstrip(".!?")
    raw = re.sub(r"^(?:the|some|our|more)\s+", "", raw)
    for key in stock:
        kl = key.lower()
        if raw == kl or raw in kl or kl in raw:
            return key
        raw_words = set(raw.split())
        if raw_words & set(kl.split()):
            return key
    return raw


def parse(text: str) -> tuple[str, float, str, str] | None:
    """('out'|'in', qty, unit, item) for a supply transaction, else None."""
    t = _words_to_digits(text)
    m = _TXN.search(t)
    if not m:
        return None
    direction = "out" if re.fullmatch(_OUT, m["dir"], re.I) else "in"
    return (direction, float(m["qty"]), (m["unit"] or "").lower(),
            m["item"])


def _fmt_qty(v: float) -> str:
    return f"{v:g}"


# Things a shelter plausibly ledgers. A transaction whose item is not
# already in stock, has no unit, and isn't in this set falls through to
# the RAG — keeps "I gave him CPR 3 times" from opening a 'times' line.
COMMON = {"water", "blankets", "blanket", "meals", "food", "masks",
          "mask", "batteries", "diapers", "kits", "kit", "flashlights",
          "tarps", "cots", "formula", "medicine", "medications",
          "sanitizer", "soap", "gloves", "wipes", "socks", "toothbrushes"}


def maybe_answer(text: str, sconn=None) -> str | None:
    """Handle a supply transaction or stock query; None if neither."""
    sconn = sconn or scribe.connect()
    stock = scribe.stock(sconn)

    parsed = parse(text)
    if parsed:
        direction, qty, unit, raw_item = parsed
        item = _canonical_item(raw_item, stock)
        if not (item in stock or unit
                or set(item.split()) & COMMON):
            return None
        # water is ledgered in liters; convert spoken gallons
        if item == "water" and unit.startswith("gallon"):
            qty, unit = round(qty * 3.785, 1), "L"
        if item == "water" and unit in ("liter", "liters", "litre",
                                        "litres", ""):
            unit = "L"
        delta = qty if direction == "in" else -qty
        scribe.supply(sconn, item, delta, unit)
        remaining = scribe.stock(sconn).get(item, 0.0)
        verb = "Received" if direction == "in" else "Distributed"
        reply = (f"Logged. {verb} {_fmt_qty(qty)} {unit} {item}. "
                 f"{_fmt_qty(remaining)} {unit} remaining".strip() + ".")
        if item == "water":
            days = scribe.water_days_remaining(sconn)
            reply += (f" That is {days:.1f} days at Sphere rates for "
                      f"everyone registered.")
        return reply

    q = _QUERY.search(_words_to_digits(text))
    if q:
        item = _canonical_item(q["item"], stock)
        if item in stock:
            reply = f"You have {_fmt_qty(stock[item])} {item} on hand."
            if item == "water":
                days = scribe.water_days_remaining(sconn)
                reply = (f"You have {_fmt_qty(stock[item])} liters of "
                         f"water — {days:.1f} days at Sphere rates for "
                         f"everyone registered.")
            return reply
        return f"I have no {item} on the ledger."
    return None
