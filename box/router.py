"""Native tool-calling router: when the regex fast-paths miss, Gemma
picks a TOOL with typed arguments (ollama /api/chat function calling)
instead of a hand-rolled JSON classification.

The model chooses; deterministic code executes. No tool call means the
utterance is a normal question and the RAG handles it — always safe.
"""
from __future__ import annotations

import requests

from . import config

_SYSTEM = ("You are Ember, an offline emergency shelter assistant. "
           "If the request matches a tool, call it with exact "
           "arguments. If it is a general survival, first-aid, or "
           "how-to question, do NOT call a tool.")

TOOLS = [
    {"type": "function", "function": {
        "name": "check_stock",
        "description": "Report how much of a supply is CURRENTLY on "
                       "hand in the shelter ledger (not how much is "
                       "needed or required).",
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string",
                     "description": "the supply, e.g. water, blankets"}},
            "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "log_supply",
        "description": "Record supplies that were just handed out to "
                       "people or received into storage.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["in", "out"],
                          "description": "in = received, out = given"},
            "qty": {"type": "number"},
            "unit": {"type": "string",
                     "description": "liters, gallons, or empty"},
            "item": {"type": "string"}},
            "required": ["direction", "qty", "item"]}}},
    {"type": "function", "function": {
        "name": "nearest_place",
        "description": "Find the nearest hospital, pharmacy, fire "
                       "station, police, shelter, school, grocery "
                       "store, gas station, drinking water, or church, "
                       "with distance and direction.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string"}}, "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "water_planning",
        "description": "Compute how much water a group needs for a "
                       "number of days (Sphere standard).",
        "parameters": {"type": "object", "properties": {
            "people": {"type": "integer"},
            "days": {"type": "integer"}},
            "required": ["people", "days"]}}},
    {"type": "function", "function": {
        "name": "recognize_face",
        "description": "Look through the camera and identify the "
                       "person standing in front of the box.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_book",
        "description": "Read a storybook from the shelf aloud (Peter "
                       "Rabbit, Grimm, Andersen, Aesop, Wizard of Oz, "
                       "Winnie-the-Pooh).",
        "parameters": {"type": "object", "properties": {
            "book": {"type": "string"}}, "required": ["book"]}}},
    {"type": "function", "function": {
        "name": "tell_story",
        "description": "Make up and tell an original calming bedtime "
                       "story.",
        "parameters": {"type": "object", "properties": {}}}},
]


def route_tools(question: str) -> tuple[str, dict] | None:
    """(tool_name, arguments) from one temperature-0 chat turn, or None
    for plain questions / any failure."""
    body = {
        "model": config.MODEL,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": question}],
        "tools": TOOLS,
        "stream": False,
        "think": False,
        "options": {"num_ctx": config.NUM_CTX, "num_predict": 100,
                    "temperature": 0.0},
        "keep_alive": -1,
    }
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body,
                          timeout=(5, 60))
        r.raise_for_status()
        calls = (r.json().get("message") or {}).get("tool_calls") or []
        if not calls:
            return None
        fn = calls[0].get("function") or {}
        name, args = fn.get("name"), fn.get("arguments") or {}
        if not name:
            return None
        return name, (args if isinstance(args, dict) else {})
    except (requests.RequestException, ValueError, KeyError):
        return None
