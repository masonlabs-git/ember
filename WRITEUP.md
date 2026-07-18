# bug-out box — Kaggle Writeup (draft)

_Build with Gemma: JustBuild — Track 1: On-Device AI with Gemma 4_

## The one-liner

An offline emergency-preparedness and shelter-operations assistant in a box.
Raspberry Pi 5 + Hailo NPU, Gemma 4 E2B on-device, voice in / voice out,
grounded in a curated survival corpus with citations. **No cloud. No
subscription. The internet being gone is the use case.** Built as a bug-out
box; deploys as a shelter-ops assistant — the corpus is the cartridge.

## Why on-device is the product, not a constraint

Every other on-device entry must argue why local matters. Ours demonstrates
it: when the disaster comes, the cloud is the first thing to die. The
signature demo beat is unplugging the router and watching it keep working —
the thing that ruins other demos is our applause line. (Positioning note:
the Gemma community has validated offline-first crisis AI as a category; we
push it into dedicated appliance hardware and shelter operations — the layer
where paper chaos actually happens.)

## Architecture — heterogeneous edge, every subsystem on the silicon it can afford

- **Ears — Whisper on the Hailo NPU.** STT runs in the HAT's own 8 GB, off
  the system RAM that Gemma needs.
- **Brain — Gemma 4 E2B on the CPU** (`think:false`; a measured trap —
  thinking mode silently spends the whole token budget and the box says
  nothing). Pinned resident at boot (cold load ~137 s, so never mid-demo).
- **Memory — SQLite FTS5/BM25 on an SSD.** No vector model: measured RAM
  budget leaves no room for a resident embedder beside Gemma. BM25 over
  terminology-rich field manuals retrieves on par, at zero RAM. Tiered
  ranking puts hand-curated operational sources ahead of the Wikipedia bulk
  layer; formal protocols (START triage, ICS) are pinned deterministically.
- **Mouth — Piper TTS**, streamed sentence-by-sentence so speech starts
  before generation finishes; auto-switches to a Spanish voice on Spanish
  replies.
- **Every phone is the screen.** The box runs its own Wi-Fi AP; a Flask
  dashboard shows the live thought stream (with citations), registry photo
  board, supply ledger, and a chat box — no per-seat hardware, and the
  OFFLINE state is self-evident because you're on the box's network.

## Shelter-ops features (one interaction engine, three faces)

Voice becomes structured, timestamped records — the shelter's memory and its
ICS-214-style paper trail:
- **Intake interview** — multilingual registration (names, medical needs,
  missing persons); intake photo attached for face-based reunification.
- **Registry + reunification** — "has anyone named Rodriguez checked in?"
- **Supply ledger + burn-rate** — water-days-remaining vs headcount, cited to
  the Sphere Handbook's 15 L/person/day standard.
- **Hands-busy coach** — one step at a time for wound care / triage, driven
  by the START protocol; waits for "done" between steps.
- **Shift briefing** — summarizes the last N hours for an incoming volunteer.

## Grounding: hallucination defense as a feature

Every answer cites its source on screen; the persona is safety-biased ("if
uncertain about food, medication, or deep wounds, say so and advise
professional care"). Grounded-with-receipts is the credibility story.

## Measured numbers (real hardware, this build)

- Gemma 4 E2B on Pi 5 CPU: **~5.5 tok/s** generation, ~34 tok/s prompt eval.
- Whisper STT round-trip: **~2.7 s** for a 10 s utterance.
- Piper TTS: **~0.5× real-time** (2× faster than speech).
- Retrieval over **542,648 chunks** (8 field manuals + 71,529 Wikipedia
  medicine articles): **~140 ms** on the SSD after stopword tuning.
- Full voice loop: box responds in **~2 s**, first substantive answer
  sentence at **~11–13 s** (fresh question) — see full-chain section.

## Corpus (prepared as data before the event; indexed by our code after)

FM 21-76 Survival, FM 4-25.11 First Aid, Where There Is No Doctor, Where
There Is No Dentist, The Ship's Medicine Chest, FEMA Are You Ready + quake
checklist, START triage, NIMS ICS forms, Sphere Handbook 2018, Wikipedia
medicine (71.5k articles). Kiwix-served layer for browsing: EN + ES medicine,
iFixit. All public-domain or open-licensed; documented in DATA.md.

## Cost at scale

Prototype ~$410 off-the-shelf; modeled BOM ~$252 @1k units, ~$211 @10k;
retail $499–599 with **zero recurring cost** (vs. Garmin inReach's hardware +
monthly subscription). Software marginal cost is zero (open weights).

## 3-minute demo script

1. Unplug the router. "This box has no internet. On purpose."
2. "How do I make creek water safe?" — spoken answer + citation on screen.
3. Multilingual intake: register a Spanish-speaking arrival by voice.
4. "How much water do 85 people need for three days?" — Sphere-cited math.
5. "Twenty injured — walk me through triage." — START protocol, one step at
   a time.
6. Close on the power bank and the price tag.

## What's next

Meshtastic LoRa (an off-grid oracle for a whole neighborhood), ggwave
acoustic box-to-box data (proven), face-match reunification, full offline
Wikipedia (49 GB, on the drive).

---
_Development was AI-assisted (Claude Code). All application code was written
during the event; the repo was created public at kickoff with no prior code._

## Measured on the actual box (Pi 5 + Hailo, home wifi, 2026-07-18)

- Retrieval (authority-split index, 2,871 field-manual chunks): **37–110 ms**.
- Gemma 4 E2B warm, trimmed context: **1.4 s to first token, 11.9 s full**
  for a ~90-token cited answer. Sentence-streamed TTS starts speaking at the
  first sentence, so perceived latency ≈ a couple seconds.
- Hailo NPU genai zoo has no Gemma (Qwen/Llama/DeepSeek only), so Gemma runs
  on CPU by design; the NPU is reserved for Whisper STT. NPU = ears, CPU =
  brain, SSD = memory.

## Full-chain status (2026-07-18, on the box — LATENCY TARGET MET)

END-TO-END WORKS, OFFLINE, FAST. Measured full chain (wav in → spoken
cited answer), repeated runs on the real hardware:

- **Box audibly responds in 1.3–2.0 s** (pre-synthesized "checking my
  field manuals" plays the instant STT lands, while Gemma prefills).
- **First substantive answer audio: 11.3–12.7 s** on a never-seen
  question; **4.4–5.4 s** on repeated/cached prompts.
- STT on the Hailo NPU: **1.3–2.0 s**. Retrieval: **<100 ms**.
- Generation: **~5.5 tok/s** alongside live TTS; whole compute chain
  (gen_done) typically 13–16 s fresh, 7–9 s cached; the tail beyond that
  is just the reply's own spoken length.
- 46/46 unit tests pass on the box. RAM: 2.3+ GB headroom at all times.

How it got there (the engineering story, all measured on-device):
1. Resident torch-free Whisper service (numpy mel, unix socket) — STT
   10.9 s → 1.3 s and no more torch evicting the LLM.
2. `gemma4:e2b-it-qat` (Google's official QAT build, 4.3 GB) instead of
   the 7.2 GB q4_K_M — the 8 GB Pi went from swap-thrash (1 tok/s) to
   2.5 GB headroom.
3. Flash-attention OFF on CPU: 7.3 vs 4.0 tok/s solo.
4. Gemma's sliding-window attention caps ollama's KV prefix reuse at a
   ~93-token checkpoint, so every fresh question re-prefills ~490 tok at
   ~60 tok/s — persona and context are trimmed hard because of it.
5. In-process Piper (one onnxruntime load), player-thread TTS overlap,
   `nice(4)` on the brain so piper bursts stop stealing ollama's cores.
6. Coach mode: one-step-at-a-time enforced by a 36-token cap in code.

Remaining: (1) live-mic VAD front-end (loop code exists in box/audio.py +
brain.py, just needs wiring to the resident STT), (2) rename bug-out box
-> EMBER (repo, code, TTS intro, AP SSID), (3) print lid today (tub
printed overnight).
