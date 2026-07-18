"""Shift-change briefing: summarize the last N hours of scribe activity for
an incoming volunteer. The feature that stops information dying at handoff."""
from __future__ import annotations

from . import llm, scribe

SYSTEM = (
    "You are Ember briefing an incoming shelter volunteer. From "
    "the activity log and inventory below, give a calm 4-6 sentence "
    "handoff: current headcount, notable arrivals or medical needs, supply "
    "levels and anything running low, and open concerns for the next "
    "shift. Be concrete. Do not invent anything not in the data. Keep "
    "every quantity in its listed unit — water is measured in LITERS."
)


def generate(sconn, hours: float = 8.0) -> str:
    material = scribe.shift_briefing_text(sconn, hours)
    return llm.generate(material, SYSTEM, num_predict=240)
