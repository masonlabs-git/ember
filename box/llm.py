"""Ollama client for Gemma 4 — streaming, thinking OFF (measured: thinking
mode silently burns the whole token budget and the box says nothing)."""
from __future__ import annotations

import json
from typing import Iterator

import requests

from . import config


def generate_stream(prompt: str, system: str,
                    num_predict: int = None,
                    stats: dict = None) -> Iterator[str]:
    """Yield response text fragments as they generate.

    Pass a dict as `stats` to receive ollama's timing numbers from the
    final message (prompt_eval_* = prefill, eval_* = generation) — the
    split that tells you which side of inference is eating the latency.
    """
    body = {
        "model": config.MODEL,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {
            "num_ctx": config.NUM_CTX,
            "num_predict": num_predict or config.NUM_PREDICT,
            # model-card default is 1.0; an emergency assistant wants
            # consistent, instruction-tight answers, not creative ones
            "temperature": 0.7,
        },
        "keep_alive": -1,
    }
    with requests.post(f"{config.OLLAMA_URL}/api/generate", json=body,
                       stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get("response"):
                yield d["response"]
            if d.get("done"):
                if stats is not None:
                    for k in ("prompt_eval_count", "prompt_eval_duration",
                              "eval_count", "eval_duration",
                              "load_duration", "total_duration"):
                        if k in d:
                            stats[k] = d[k]
                return


def generate(prompt: str, system: str, num_predict: int = None) -> str:
    return "".join(generate_stream(prompt, system, num_predict))


def warmup(system: str = None) -> bool:
    """Pin the model resident (boot-time; cold load measured at ~2 min).

    Pass the persona the box will actually answer with: ollama caches the
    KV of the previous request's prefix, and the system prompt is the
    prefix — warming with the real persona means the first live query
    skips re-prefilling it.
    """
    try:
        generate("ready", system or "Reply with exactly: ready",
                 num_predict=4)
        return True
    except requests.RequestException:
        return False
