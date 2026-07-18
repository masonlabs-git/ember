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


def pick_persona(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _INTERVIEW_TRIGGERS):
        return persona.INTERVIEW
    if any(k in t for k in _COACH_TRIGGERS):
        return persona.COACH
    return persona.ANSWER


class Brain:
    def __init__(self):
        self.conn = retrieval.connect()
        self.history: list[tuple[str, str]] = []   # (user, box) turns
        self.mode = persona.ANSWER

    def answer(self, question: str, system: str = None) -> str:
        """One full turn: retrieve, generate (streamed to speech), log."""
        emit("heard", text=question)
        # sticky coaching/interview: stay in the flow until it resolves
        if system is None:
            picked = pick_persona(question)
            if picked is not persona.ANSWER:
                self.mode = picked
            system = self.mode
        hits = retrieval.search(self.conn, question)
        emit("retrieved", citations=[h.citation for h in hits],
             mode="coach" if system is persona.COACH else
                  "interview" if system is persona.INTERVIEW else "answer")
        context = retrieval.context_block(hits)
        prompt = persona.build_prompt(question, context)
        if self.history:
            recent = "\n".join(f"User: {u}\nBox: {b}"
                               for u, b in self.history[-3:])
            prompt = f"RECENT CONVERSATION:\n{recent}\n\n{prompt}"
        stream = llm.generate_stream(prompt, system)
        if config.MUTE:
            reply = "".join(stream).strip()
        else:
            reply = tts.speak_stream(stream)
        emit("spoke", text=reply)
        self.history.append((question, reply))
        return reply

    def loop(self) -> None:
        from . import audio          # deferred: webrtcvad only on the Pi
        emit("status", state="warming model")
        llm.warmup(persona.ANSWER)
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
