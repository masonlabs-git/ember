"""The orchestrator: hear -> retrieve -> think -> speak, with an event log
the dashboard tails (the box's visible thought process)."""
from __future__ import annotations

import json
import os
import re
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

# Wake word. Matched on the TRANSCRIPT (the NPU STT is fast enough to
# transcribe everything), not with a hotword model. "amber" is accepted
# because Whisper frequently hears "Ember" that way. \b keeps "remember"
# and "member" from waking the box.
WAKE = re.compile(
    r"\b(?:hey|hi|ok|okay)?[,\s]*\b(?:ember|amber)\b[,.!?]*\s*", re.I)
FOLLOWUP_S = 25          # after an answer: reply without re-waking
WAKE_WINDOW_S = 15       # after a bare "hey ember": time to ask

# Single words that count as a turn inside the follow-up window — the
# coach flow runs on these. Everything else one-word is whisper noise.
_ONE_WORD_TURNS = {"done", "next", "repeat", "yes", "no", "okay", "ok",
                   "ready", "stop", "continue", "help"}


def route(heard: str, awake: bool) -> tuple[str, str]:
    """Decide what to do with one transcript.

    Returns ("answer", question) | ("wake", "") | ("ignore", "").
    Pure function so the demo-critical routing is unit-testable.
    """
    m = WAKE.search(heard)
    if m:
        q = (heard[:m.start()] + " " + heard[m.end():]).strip(" ,.!?")
        if len(q.split()) >= 2:
            return "answer", q          # wake word + question in one breath
        return "wake", ""               # bare wake — acknowledge and listen
    if awake:
        words = heard.strip(" ,.!?").split()
        if len(words) >= 2:
            return "answer", heard.strip()      # follow-up: no wake needed
        if len(words) == 1 and words[0].lower() in _ONE_WORD_TURNS:
            return "answer", words[0].lower()   # coach: 'done' / 'next'
    return "ignore", ""                 # stray speech / whisper noise


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
        os.nice(4)      # yield CPU to ollama: the brain mostly waits on
        #                 it, and piper/mel bursts were costing ~40% of
        #                 generation speed in core contention
        emit("status", state="warming model")
        while not llm.warmup(persona.MAIN):   # boot race: ollama may still
            time.sleep(5)                     # be starting — keep trying
        tts.prepare_acks()
        emit("status", state="ready")
        tts.speak("This is Ember. Say, hey Ember, when you need me.")
        awake_until = 0.0
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
            finally:
                os.unlink(wav)          # capture wavs were leaking /tmp
            action, question = route(heard,
                                     awake=time.time() < awake_until)
            if action == "answer":
                self.answer(question)
                awake_until = time.time() + FOLLOWUP_S
            elif action == "wake":
                emit("wake", text=heard)
                tts.play_wake_ack()
                awake_until = time.time() + WAKE_WINDOW_S
            else:
                emit("ignored", text=heard)
