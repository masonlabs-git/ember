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

# Stock queries in natural phrasings: "how much water do we have left",
# "how much gallons of water we have in our storage" (verbatim venue
# miss). A have/left/storage keyword is required and 'need' vetoes —
# "how much water do 85 people NEED" must flow to the RAG.
_QUERY_LEAD = re.compile(r"\bhow (?:much|many)\s+(.*)$", re.I)
_QUERY_KEY = re.compile(r"\b(?:have|left|remaining|stock|storage|"
                        r"on hand|got)\b", re.I)
_QUERY_STOP = {"do", "we", "did", "is", "are", "have", "left",
               "remaining", "in", "on", "got"}


def _parse_query_item(text: str) -> str | None:
    if re.search(r"\bneed", text, re.I):
        return None
    m = _QUERY_LEAD.search(text)
    if not m or not _QUERY_KEY.search(m[1]):
        return None
    item_words = []
    for w in m[1].split():
        wl = w.lower().strip(",.?!")
        if wl in _QUERY_STOP:
            break
        item_words.append(wl)
    return " ".join(item_words) or None

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
    unit, item = (m["unit"] or "").lower(), m["item"]
    if not unit:
        # the unit hides inside the item when filler words sit between
        # the number and it — live failure: "10 MORE gallons of water"
        # parsed as 10 (liters) of "more gallons of water"
        um = re.match(rf"(?:[a-z]+\s+){{0,2}}({_UNITS})\s+(?:of\s+)?(.+)",
                      item, re.I)
        if um:
            unit, item = um[1].lower(), um[2]
    return direction, float(m["qty"]), unit, item


def _fmt_qty(v: float) -> str:
    return f"{v:g}"


# Things a shelter plausibly ledgers. A transaction whose item is not
# already in stock, has no unit, and isn't in this set falls through to
# the RAG — keeps "I gave him CPR 3 times" from opening a 'times' line.
COMMON = {"water", "blankets", "blanket", "meals", "food", "masks",
          "mask", "batteries", "diapers", "kits", "kit", "flashlights",
          "tarps", "cots", "formula", "medicine", "medications",
          "sanitizer", "soap", "gloves", "wipes", "socks", "toothbrushes"}


def apply_txn(sconn, direction: str, qty: float, unit: str,
              raw_item: str, gated: bool = False) -> str | None:
    """Deterministic ledger write + spoken confirmation. `gated` applies
    the COMMON-item sanity gate (regex path); router-sourced calls skip
    it (the model already judged intent)."""
    stock = scribe.stock(sconn)
    unit = (unit or "").lower()
    item = _canonical_item(raw_item, stock)
    if gated and not (item in stock or unit or set(item.split()) & COMMON):
        return None
    if unit and (unit == item or unit in item.split()):
        unit = ""                     # router echo: "15 blankets of blankets"
    spoken = f"{_fmt_qty(qty)} {unit}".strip() if unit else _fmt_qty(qty)
    converted = ""
    if item == "water" and unit.startswith("gallon"):
        qty = round(qty * 3.785, 1)
        converted = f" — {_fmt_qty(qty)} liters"
        unit = "L"
    if item == "water" and unit in ("liter", "liters", "litre",
                                    "litres", ""):
        unit = "L"
        spoken = f"{_fmt_qty(qty)} liters"
    delta = qty if direction == "in" else -qty
    scribe.supply(sconn, item, delta, unit)
    remaining = scribe.stock(sconn).get(item, 0.0)
    verb = "Received" if direction == "in" else "Distributed"
    reply = (f"Logged. {verb} {spoken} of {item}{converted}. "
             f"{_fmt_qty(remaining)} {unit} remaining".strip() + ".")
    if item == "water":
        days = scribe.water_days_remaining(sconn)
        reply += (f" That is {days:.1f} days at Sphere rates for "
                  f"everyone registered.")
    return reply


def stock_reply(sconn, raw_item: str, want_gallons: bool = False) -> str:
    """Deterministic stock report for an item named any which way."""
    stock = scribe.stock(sconn)
    item = _canonical_item(raw_item, stock)
    if item not in stock:
        return f"I have no {item} on the ledger."
    if item == "water":
        days = scribe.water_days_remaining(sconn)
        liters = stock[item]
        gal = (f" — about {liters / 3.785:.0f} gallons"
               if want_gallons else "")
        return (f"You have {_fmt_qty(liters)} liters of water{gal} — "
                f"{days:.1f} days at Sphere rates for everyone "
                "registered.")
    return f"You have {_fmt_qty(stock[item])} {item} on hand."


def maybe_answer(text: str, sconn=None) -> str | None:
    """Regex fast-path: supply transaction or stock query; None if
    neither pattern hits (the LLM router is the safety net behind us)."""
    sconn = sconn or scribe.connect()
    parsed = parse(text)
    if parsed:
        direction, qty, unit, raw_item = parsed
        return apply_txn(sconn, direction, qty, unit, raw_item,
                         gated=True)
    raw_q = _parse_query_item(text)
    if raw_q:
        return stock_reply(sconn, raw_q,
                           want_gallons="gallon" in text.lower())
    return None
