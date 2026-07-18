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


# Follow-up utterances must LOOK aimed at the box. Field-tested: during
# a kitchen conversation, one answered question opened the 25 s window
# and the box then "answered" fragments of two people talking to each
# other ("FimoWorker has a device, right?" -> a cited reply). Questions
# to the box start like questions; room chatter doesn't.
_DIRECTED_STARTS = {"how", "what", "where", "when", "why", "who", "which",
                    "can", "could", "do", "does", "did", "is", "are",
                    "should", "will", "would", "tell", "read", "show",
                    "find", "give", "we", "i", "my", "help", "ember"}
_MAX_FOLLOWUP_WORDS = 18


def _directed(heard: str) -> bool:
    words = heard.strip(" ,.!?").split()
    if not words or len(words) > _MAX_FOLLOWUP_WORDS:
        return False                    # long rambles are room talk
    first = words[0].lower().strip(",.!?'\"")
    if first in _DIRECTED_STARTS:
        return True
    return bool(re.search(r"\b(?:ember|amber)\b", heard, re.I))


def route(heard: str, awake: bool) -> tuple[str, str]:
    """Decide what to do with one transcript.

    Returns ("answer", question) | ("wake", "") | ("ignore", "").
    Pure function so the demo-critical routing is unit-testable.
    """
    if not re.search(r"[a-zA-Z]{2}", heard):
        return "ignore", ""             # whisper bracket-noise like '[ [ ['
    m = WAKE.search(heard)
    if m:
        q = (heard[:m.start()] + " " + heard[m.end():]).strip(" ,.!?")
        if len(q.split()) >= 2:
            return "answer", q          # wake word + question in one breath
        if q.lower() in _ONE_WORD_TURNS:
            return "answer", q.lower()  # "hey ember, next" mid-story
        return "wake", ""               # bare wake — acknowledge and listen
    if awake:
        words = heard.strip(" ,.!?").split()
        if len(words) >= 2 and _directed(heard):
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


# Turns that continue a flow rather than start one. Measured failure they
# fix: on "next" the box re-retrieved for the literal word "next" and
# re-stated step one forever instead of advancing the protocol.
_CONTINUATIONS = {"next", "done", "okay", "ok", "yes", "ready", "continue"}

# "Do you recognize me?" — camera + face match by voice.
_RECOGNIZE = ("recognize me", "who am i", "who am I", "do you know me",
              "do you see me", "look at me", "recognize my face")

# Story/comfort requests must NOT go through the RAG — the grounded
# 3-sentence persona once answered "read a bedtime story" with a cited
# sentence from Where There Is No Doctor.
_STORY = re.compile(
    r"\b(?:tell|read|make\s+up|give|share)\b.{0,40}\bstor(?:y|ies)\b"
    r"|\bbedtime\b", re.I)


def is_story(text: str) -> bool:
    return bool(_STORY.search(text))


_ES_WORDS = re.compile(r"\b(el|la|los|las|es|agua|para|de|una|con|"
                       r"hierve|filtre|cocine|purifica\w*)\b", re.I)
_EN_WORDS = re.compile(r"\b(the|and|water|boil|if|you|from|it)\b")


def _clean_spanish(text: str) -> bool:
    """Spanish-dominant with no meaningful English — code-switched
    replies ('Use bottled water. Cocine el agua...') must fail this."""
    return (len(_ES_WORDS.findall(text)) >= 2
            and len(_EN_WORDS.findall(text)) < 2)


class Brain:
    def __init__(self):
        self.conn = retrieval.connect()
        self.history: list[tuple[str, str]] = []   # (user, box) turns
        self.mode = persona.MAIN
        self.last_mode = "answer"
        self.topic = ""            # last real question — continuations
        #                            retrieve against THIS, not "next"
        self.reading: tuple[str, str, int] | None = None
        #   (title, filename, next_passage) while a storybook is open

    def _read_next(self, n: int = 2) -> str:
        """Speak the next passages of the open storybook."""
        from . import stories
        title, filename, idx = self.reading
        chunk = stories.passages(filename)[idx:idx + n]
        if not chunk:
            self.reading = None
            return f"And that is the end of {title}. Sweet dreams."
        self.reading = (title, filename, idx + n)
        text = " ".join(chunk)
        tail = "" if self.reading is None else \
            " ... Say next when you're ready."
        emit("reading", title=title, passage=idx)
        if not config.MUTE:
            tts.speak_stream(iter([text + tail]))
        self.last_mode = "reading"
        return text

    def _say(self, question: str, reply: str, mode: str) -> str:
        emit("spoke", text=reply, mode=mode)
        if not config.MUTE:
            tts.speak(reply)
        self.history.append((question, reply))
        return reply

    def _recognize(self) -> str:
        """Camera + face match: 'Hey Ember, do you recognize me?'"""
        from . import camera, faces, scribe
        if not config.MUTE:
            tts.speak("Hold still. Let me look.")
        shot = camera.capture("recognize")
        if not shot:
            return "My camera is not responding."
        sconn = scribe.connect()
        results = faces.match(sconn, shot, scribe.households(sconn))
        emit("face_recognize", matches=len(results))
        if not results:
            return ("I can't see a face clearly. Step in front of my "
                    "camera and ask again.")
        best = results[0]
        if best["same_person"]:
            when = time.strftime("%I:%M %p",
                                 time.localtime(best["ts"])).lstrip("0")
            return (f"Yes, I recognize you. You checked in as "
                    f"{best['names']} at {when}.")
        return ("I see you, but I don't recognize you from check-in. "
                "You can register at the intake desk.")

    def answer(self, question: str, system: str = None) -> str:
        """One full turn: retrieve, generate (streamed to speech), log."""
        emit("heard", text=question)
        ql = question.lower()
        # open storybook owns the turn: next/stop control the reading
        if self.reading and system is None:
            w = ql.strip(" .!?")
            if w in ("stop", "stop reading", "that's enough", "the end"):
                self.reading = None
                return self._say(question,
                                 "Closing the book. Sweet dreams.",
                                 "reading")
            if w in _CONTINUATIONS:
                reply = self._read_next()
                self.history.append((question, "[read more of the book]"))
                return reply
            self.reading = None      # a real question interrupts the book
        # storybook shelf: "read peter rabbit" -> the actual text
        from . import stories
        book = stories.match(question) if system is None else None
        if book:
            title, filename = book
            self.reading = (title, filename, 0)
            if not config.MUTE:
                tts.speak(f"{title}. Here we go.")
            reply = self._read_next()
            self.history.append((question, f"[opened {title}]"))
            return reply
        # recognition fast-path: camera + face match, no LLM
        if any(k in ql for k in map(str.lower, _RECOGNIZE)):
            return self._say(question, self._recognize(), "recognize")
        # comfort fast-path: stories skip retrieval, drop the citation
        # rules, and get a real token budget
        if system is None and is_story(question):
            emit("retrieved", citations=[], mode="story")
            stream = llm.generate_stream(
                f"REQUEST: {question}\nTell the story now.",
                persona.STORY, num_predict=300)
            if config.MUTE:
                reply = "".join(stream).strip()
            else:
                reply = tts.speak_stream(stream, preroll=tts.next_ack())
            emit("spoke", text=reply, mode="story")
            self.history.append((question, reply))
            self.last_mode = "story"
            return reply
        # quartermaster fast-path: spoken supply transactions and stock
        # queries hit the ledger directly — inventory must be exact
        try:
            from . import quartermaster
            q = quartermaster.maybe_answer(question)
        except Exception:
            q = None
        if q:
            return self._say(question, q, "supplies")
        # places fast-path: "nearest hospital" answers are computed from
        # the offline OSM index, not generated — exact and instant
        try:
            from . import nav
            n = nav.maybe_answer(question)
        except Exception:
            n = None
        if n:
            return self._say(question, n, "places")
        cont = question.lower().strip(" .!?") in _CONTINUATIONS
        # sticky interview: stay in the intake flow until it resolves.
        mode = "answer"
        if system is None:
            mode, system = pick_persona(question)
            if mode == "interview":
                self.mode = persona.INTERVIEW
            if self.mode is persona.INTERVIEW:
                mode, system = "interview", persona.INTERVIEW
            elif cont and self.last_mode == "coach":
                mode = "coach"          # sticky: 'next' keeps coaching
        if not cont or not self.topic:
            self.topic = question
        hits = retrieval.search(self.conn, self.topic)
        emit("retrieved", citations=[h.citation for h in hits], mode=mode)
        if not hits and mode == "answer":
            # no sources -> no answer. Observed failure: garbage input
            # retrieved nothing and the model still invented advice with
            # a fabricated [1]. Grounded-with-receipts is the product.
            return self._say(question,
                             "I could not find that in my field manuals. "
                             "Try asking a different way.", "answer")
        context = retrieval.context_block(hits)
        if cont and mode == "coach":
            done_steps = "; ".join(
                tts.strip_citations(b) for _, b in self.history[-3:])
            ask = (f"Emergency: {self.topic}\nThe user has ALREADY "
                   f"COMPLETED these steps: {done_steps}\nThey said "
                   f"'{question}'. Give the ONE next, different step. Do "
                   "not repeat a completed step.")
        elif question.lower().strip(" .!?") == "repeat":
            ask = (f"Repeat your previous instruction for '{self.topic}' "
                   "in different words.")
        else:
            ask = question
        # explicit language requests must not depend on sampling luck —
        # measured: prompt instructions alone still code-switched 1 run
        # in 3, so forced-Spanish replies are generated-then-verified
        # (retry once at temp 0) before any audio plays
        es_forced = bool(re.search(r"\bin spanish\b|\ben español\b", ql))
        if es_forced:
            ask = ("EN ESPAÑOL — responde únicamente en español. "
                   + ask)
        prompt = persona.build_prompt(ask, context)
        if es_forced:
            reply = ""
            for temp, hard in ((0.3, ""),
                               (0.0, "RESPONDE SOLO EN ESPAÑOL, SIN "
                                     "NINGUNA PALABRA EN INGLÉS.\n")):
                reply = "".join(llm.generate_stream(
                    hard + prompt, system, temperature=temp)).strip()
                if _clean_spanish(reply):
                    break
            emit("spoke", text=reply, mode="answer-es")
            if not config.MUTE:
                tts.speak_stream(iter([reply]), preroll=tts.next_ack())
            self.history.append((question, reply))
            self.last_mode = "answer"
            return reply
        if self.history:
            recent = "\n".join(f"User: {u}\nBox: {b}"
                               for u, b in self.history[-3:])
            prompt = f"RECENT CONVERSATION:\n{recent}\n\n{prompt}"
        # coach = one step: enforced by token budget, not prompt hope —
        # the model was reliably ignoring the one-sentence instruction
        cap = 36 if mode == "coach" else None
        self.last_mode = mode
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
