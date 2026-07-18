"""The orchestrator: hear -> retrieve -> think -> speak, with an event log
the dashboard tails (the box's visible thought process)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config, llm, persona, retrieval, stt, tts

EVENTS = Path("/tmp/box-events.jsonl")


def emit(kind: str, **data) -> None:
    rec = {"t": round(time.time(), 2), "kind": kind, **data}
    with EVENTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# Utterance-level mode switches. A hands-busy emergency wants one-step-at-a-
# time coaching; intake wants the interview script; otherwise answer mode.
_COACH_TRIGGERS = ("bleeding", "not breathing", "choking", "burn", "cut",
                   "wound", "injured", "cpr", "help me", "triage",
                   "broken", "seizure", "unconscious")
_INTERVIEW_TRIGGERS = ("check in", "check us in", "register", "intake",
                       "we just arrived", "sign in")


def pick_persona(text: str) -> tuple[str, str]:
    """Returns (mode_name, system_prompt). Answer and coach share ONE
    prompt (see persona.MAIN) so the KV prefix stays cached across modes;
    the mode name is for the event log/dashboard only."""
    t = text.lower()
    if any(k in t for k in _INTERVIEW_TRIGGERS):
        return "interview", persona.INTERVIEW
    if any(k in t for k in _COACH_TRIGGERS):
        return "coach", persona.MAIN
    return "answer", persona.MAIN


class Brain:
    def __init__(self):
        self.conn = retrieval.connect()
        self.history: list[tuple[str, str]] = []   # (user, box) turns
        self.mode = persona.MAIN

    def answer(self, question: str, system: str = None) -> str:
        """One full turn: retrieve, generate (streamed to speech), log."""
        emit("heard", text=question)
        # sticky interview: stay in the intake flow until it resolves.
        # (coach/answer share one prompt — the model itself carries the
        # coaching flow via RECENT CONVERSATION, no stickiness needed)
        mode = "answer"
        if system is None:
            mode, system = pick_persona(question)
            if mode == "interview":
                self.mode = persona.INTERVIEW
            if self.mode is persona.INTERVIEW:
                mode, system = "interview", persona.INTERVIEW
        hits = retrieval.search(self.conn, question)
        emit("retrieved", citations=[h.citation for h in hits], mode=mode)
        context = retrieval.context_block(hits)
        prompt = persona.build_prompt(question, context)
        if self.history:
            recent = "\n".join(f"User: {u}\nBox: {b}"
                               for u, b in self.history[-3:])
            prompt = f"RECENT CONVERSATION:\n{recent}\n\n{prompt}"
        # coach = one step: enforced by token budget, not prompt hope —
        # the model was reliably ignoring the one-sentence instruction
        cap = 36 if mode == "coach" else None
        stream = llm.generate_stream(prompt, system, num_predict=cap)
        if config.MUTE:
            reply = "".join(stream).strip()
        else:
            # ack plays during prefill — the box responds ~2s after the
            # question even though the first real sentence is seconds out
            reply = tts.speak_stream(stream, preroll=tts.next_ack())
        emit("spoke", text=reply)
        self.history.append((question, reply))
        return reply

    def loop(self) -> None:
        from . import audio          # deferred: webrtcvad only on the Pi
        import os
        os.nice(4)      # yield CPU to ollama: the brain mostly waits on
        #                 it, and piper/mel bursts were costing ~40% of
        #                 generation speed in core contention
        emit("status", state="warming model")
        llm.warmup(persona.MAIN)
        tts.prepare_acks()
        emit("status", state="ready")
        tts.speak("Bug-out box ready.")
        while True:
            wav = audio.listen_for_utterance()
            if wav is None:
                emit("status", state="mic lost; retrying")
                time.sleep(2)
                continue
            try:
                heard = stt.transcribe_wav(wav)
            except Exception as e:      # STT hiccup: say so, keep living
                emit("error", stage="stt", detail=str(e))
                continue
            if len(heard.split()) < 2:
                continue                # breath, noise, fragment
            self.answer(heard)
