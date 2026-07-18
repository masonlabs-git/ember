"""Timed full-chain benchmark: wav -> NPU STT -> retrieval -> Gemma -> TTS.

Per-stage wall-clock marks plus ollama's own prefill/generation split, so
every optimization round has real numbers instead of vibes.

Usage: python3 -m box.chain_test <wav> [--mute]

Marks (seconds since wav handed to STT):
  stt          transcript back from the NPU whisper service
  retrieval    FTS hits + context block + prompt assembled
  first_token  first Gemma fragment arrives
  first_audio  first TTS wav starts playing (user hears the box)
  gen_done     Gemma finished generating
  speech_done  last TTS wav finished playing
"""
from __future__ import annotations

import sys
import time

from . import config, llm, persona, retrieval, stt, tts
from .brain import pick_persona


def main() -> None:
    import os
    os.nice(4)                    # mirror production: yield CPU to ollama
    wav = sys.argv[1]
    mute = "--mute" in sys.argv
    if not mute:
        tts.prepare_acks()        # boot-time cost in production

    marks: dict[str, float] = {}
    t0 = time.time()

    def mark(name: str) -> None:
        # printed the moment it happens: a crash mid-chain still leaves
        # every completed stage's number in the log
        if name not in marks:
            marks[name] = time.time() - t0
            print(f"[mark] {name:12s} {marks[name]:6.1f}s", flush=True)

    heard = stt.transcribe_wav(wav)
    mark("stt")
    print(f"[heard] {heard!r}")

    conn = retrieval.connect()
    hits = retrieval.search(conn, heard)
    context = retrieval.context_block(hits)
    prompt = persona.build_prompt(heard, context)
    mode, system = pick_persona(heard)
    mark("retrieval")
    print(f"[mode] {mode}")
    print(f"[sources] {[h.citation for h in hits]}")
    print(f"[prompt chars] system={len(system)} prompt={len(prompt)}")

    stats: dict = {}

    def timed(gen):
        for frag in gen:
            mark("first_token")
            yield frag
        mark("gen_done")

    audio_count = [0]

    def on_event(kind: str) -> None:
        if kind == "audio":
            audio_count[0] += 1
            mark("first_audio")           # the ack, if preroll is in use
            if audio_count[0] == 2:
                mark("first_answer_audio")

    cap = 36 if mode == "coach" else None       # mirror brain.answer
    stream = timed(llm.generate_stream(prompt, system, num_predict=cap,
                                       stats=stats))
    if mute:
        reply = "".join(stream).strip()
    else:
        reply = tts.speak_stream(stream, on_event=on_event,
                                 preroll=tts.next_ack())
    mark("speech_done")
    print(f"[reply] {reply}")

    print("\n=== chain breakdown (s since audio-in) ===")
    for k in ("stt", "retrieval", "first_token", "first_audio",
              "first_answer_audio", "gen_done", "speech_done"):
        if k in marks:
            print(f"  {k:12s} {marks[k]:6.1f}")
    if stats:
        pe, pd = stats.get("prompt_eval_count", 0), stats.get(
            "prompt_eval_duration", 0) / 1e9
        ec, ed = stats.get("eval_count", 0), stats.get(
            "eval_duration", 0) / 1e9
        load = stats.get("load_duration", 0) / 1e9
        print(f"\n=== ollama ===")
        print(f"  load     {load:6.1f}s  (>1s means the model was cold — "
              "measurement invalid, rerun)")
        if pd:
            print(f"  prefill  {pe} tok in {pd:5.1f}s  ({pe / pd:5.1f} tok/s)")
        if ed:
            print(f"  generate {ec} tok in {ed:5.1f}s  ({ec / ed:5.1f} tok/s)")


if __name__ == "__main__":
    main()
