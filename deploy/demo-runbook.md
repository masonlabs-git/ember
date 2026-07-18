# Ember demo runbook

## Power-on
1. Power the Pi (bank must do 5V/5A or USB-PD 27W — or a MacBook USB-C
   port IF the Seagate is unplugged and the travel vault is synced).
2. ~60 s later Ember speaks: "This is Ember. Say, hey Ember, when you
   need me." It is now listening. Nothing else to start — everything is
   systemd (`ember`, `ember-whisper`, `ember-dash`).
3. Any failure mid-demo: power-cycle. It self-heals to listening in
   ~60 s. (Faster: `sudo systemctl restart ember`.)

## Before leaving the house
- `bash deploy/travel-vault.sh`   (fresh SD mirror, ~1 min)
- Register yourself at /intake WITH photo (reunification beat needs a
  real face on the board).
- Decide the voice (lessac vs hfc) — must match the film.

## At the venue
- Own network: `bash deploy/hotspot.sh`  → Wi-Fi **EMBER** /
  pass **shelter72** → dashboard http://10.42.0.1:8880
  (SSH from the Mac on that network: `ssh caleb@10.42.0.1`.)
- Hotspot reverts on reboot — power-cycle brings back normal Wi-Fi.
- Staff PIN: **3637** (dashboard → staff).
- Noisy hall + missed wake words: raise VAD aggressiveness
  (box/audio.py `listen_for_utterance(aggressiveness=3)`), restart ember.
- Camera: needs a sightline at standing head height; check
  `vcgencmd measure_temp` after 30 min inside WALL-E (<80°C is fine).

## Beats and exact phrasings (all rehearsed)
| Beat | Say | Expect |
|---|---|---|
| Wake | "Hey Ember." | "Yes? I'm listening." (kokoro voice) |
| RAG | "Hey Ember, how do I make creek water safe to drink?" | ack ~2 s, cited answer ~12 s |
| Coach | "My friend has a deep cut on his arm and it will not stop bleeding. Help me." then "next" (×2) | one step at a time, protocol advances |
| Places | "How far is the nearest hospital?" | computed distance + bearing |
| Ledger | "We just gave out 40 liters of water." | deduction + days-of-water |
| Stock | "How much water do we have left?" | exact liters + days |
| Recognize | "Do you recognize me?" | "Hold still." → greets by name |
| Spanish | "Tell me in Spanish how to purify water." | Spanish answer, Spanish voice |
| Story | "Can you read Peter Rabbit?" then "next" / "stop" | the actual book |
| Brief | dashboard → brief | LLM shift handoff from real ledger |

Follow-up window: ~25 s after any answer — no wake word needed ("next",
"done", questions). After that, say "Hey Ember" again.

## Things that look like bugs but aren't
- Box says "I could not find that in my field manuals" → honest
  no-source refusal (this is a feature; say so to judges).
- Whisper hears "Amber" → handled, still wakes.
- First question after a persona switch is a few seconds slower →
  KV prefix re-prefill (SWA checkpoint limit), by design.
