"""Every phone is the box's screen: Flask app served over the box's own
Wi-Fi AP. Thought stream, registry photo board, supplies, chat, RAM gauge.
Plain HTML + meta-refresh — no JS frameworks to break offline."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, request

from . import briefing, config, extract, llm, persona, retrieval, scribe
from .brain import EVENTS, emit

app = Flask(__name__)
_rconn = None
_sconn = None


def conns():
    global _rconn, _sconn
    if _rconn is None:
        _rconn = retrieval.connect()
    if _sconn is None:
        _sconn = scribe.connect()
    return _rconn, _sconn


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=6>
<title>bug-out box</title>
<style>
body{{font-family:system-ui;margin:0;background:#141a14;color:#dfe8df}}
header{{padding:10px 16px;background:#1e2a1e;display:flex;gap:12px;
       align-items:center;flex-wrap:wrap}}
h1{{font-size:18px;margin:0}} .badge{{padding:2px 10px;border-radius:10px;
background:#8a2f2f;font-weight:600;font-size:12px}}
.ok{{background:#2f6a2f}}
nav a{{color:#9fd49f;margin-right:12px;text-decoration:none}}
main{{padding:12px 16px}} .card{{background:#1c241c;border-radius:8px;
padding:10px 14px;margin-bottom:10px}}
small{{color:#8fa58f}} img{{max-width:96px;border-radius:6px}}
input,button{{font-size:16px;padding:8px;border-radius:6px;border:none}}
button{{background:#2f6a2f;color:#fff}}
</style>
<header><h1>&#128230; bug-out box</h1>
<span class=badge>OFFLINE &#10003;</span>
<span class=badge ok>RAM {ram}</span>
<nav><a href=/>mind</a><a href=/board>board</a><a href=/intake>intake</a>
<a href=/supplies>supplies</a><a href=/brief>brief</a><a href=/chat>ask</a>
</nav></header><main>{body}</main>"""


def ram() -> str:
    try:
        line = Path("/proc/meminfo").read_text().splitlines()
        avail = int([x for x in line if "MemAvailable" in x][0].split()[1])
        return f"{avail // 1024}MB free"
    except Exception:
        return "n/a"


def page(body: str) -> str:
    return PAGE.format(ram=ram(), body=body)


@app.get("/")
def mind():
    rows = []
    if EVENTS.exists():
        for ln in EVENTS.read_text().splitlines()[-25:][::-1]:
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            t = time.strftime("%H:%M:%S", time.localtime(e["t"]))
            detail = {k: v for k, v in e.items() if k not in ("t", "kind")}
            rows.append(f"<div class=card><small>{t} — {e['kind']}</small>"
                        f"<div>{html.escape(json.dumps(detail, ensure_ascii=False)[:400])}"
                        f"</div></div>")
    return page("".join(rows) or "<div class=card>No thoughts yet.</div>")


@app.get("/board")
def board():
    _, s = conns()
    cards = []
    for h in scribe.households(s):
        t = time.strftime("%b %d %H:%M", time.localtime(h["ts"]))
        img = f"<img src=/photo/{h['id']}>" if h["photo"] else ""
        med = f"<br><small>medical: {html.escape(h['medical'])}</small>" if h["medical"] else ""
        mis = f"<br><small>&#9888; missing: {html.escape(h['missing'])}</small>" if h["missing"] else ""
        cards.append(f"<div class=card>{img}<b>{html.escape(h['names'])}</b>"
                     f"<br><small>checked in {t}</small>{med}{mis}</div>")
    return page(f"<div class=card><b>{scribe.headcount(s)} people "
                f"registered</b></div>" + "".join(cards))


@app.get("/photo/<int:rid>")
def photo(rid: int):
    _, s = conns()
    row = s.execute("SELECT photo FROM registry WHERE id=?",
                    (rid,)).fetchone()
    if row and row[0] and Path(row[0]).exists():
        from flask import send_file
        return send_file(row[0])
    return "", 404


@app.get("/supplies")
def supplies():
    _, s = conns()
    stock = scribe.stock(s)
    days = scribe.water_days_remaining(s)
    rows = "".join(f"<div class=card><b>{html.escape(k)}</b>: {v:g}</div>"
                   for k, v in sorted(stock.items()))
    return page(f"<div class=card>&#128167; water on hand lasts "
                f"<b>{days:.1f} days</b> at Sphere 15 L/person/day</div>"
                + (rows or "<div class=card>No supplies logged.</div>"))


@app.get("/chat")
def chat_form():
    return page("""<form method=post action=/chat class=card>
      <input name=q style="width:70%" placeholder="ask the box anything">
      <button>ask</button></form>""")


@app.post("/chat")
def chat():
    q = request.form.get("q", "").strip()
    if not q:
        return redirect("/chat")
    r, _ = conns()
    hits = retrieval.search(r, q)
    emit("phone_asked", text=q)
    reply = llm.generate(persona.build_prompt(q, retrieval.context_block(hits)),
                         persona.ANSWER)
    emit("phone_answered", text=reply)
    cites = "".join(f"<div class=card><small>{html.escape(h.citation)}"
                    f"</small></div>" for h in hits)
    return page(f"<div class=card><b>Q:</b> {html.escape(q)}</div>"
                f"<div class=card>{html.escape(reply)}</div>{cites}"
                f"<a href=/chat>ask another</a>")


@app.get("/intake")
def intake_form():
    return page("""<form method=post action=/intake class=card>
      <input name=names style="width:90%" placeholder="household names (comma-separated)"><br><br>
      <input name=medical style="width:90%" placeholder="medical needs / allergies"><br><br>
      <input name=missing style="width:90%" placeholder="anyone unaccounted for"><br><br>
      <input name=phone style="width:60%" placeholder="phone (optional)"><br><br>
      <button>register</button></form>
      <div class=card><small>Consent: photos are taken only if the arrival
      agrees, used solely for family reunification, and never leave this box.
      </small></div>""")


@app.post("/intake")
def intake_submit():
    names = request.form.get("names", "").strip()
    if not names:
        return redirect("/intake")
    _, s = conns()
    rid = scribe.register(s, names, request.form.get("medical", ""),
                          request.form.get("missing", ""),
                          request.form.get("phone", ""))
    emit("registered", id=rid, names=names)
    return redirect("/board")


@app.get("/brief")
def brief():
    _, s = conns()
    text = briefing.generate(s) if scribe.recent_log(s) else \
        "No activity logged yet this shift."
    return page(f"<div class=card><b>Shift briefing</b><br><br>"
                f"{html.escape(text)}</div>")


@app.post("/supply")
def supply_add():
    """Voice/text supply logging: 'received 40 blankets'."""
    text = request.form.get("text", "")
    parsed = extract.parse_supply(text)
    if parsed:
        _, s = conns()
        scribe.supply(s, parsed["item"], parsed["delta"], parsed["unit"])
    return redirect("/supplies")


@app.get("/api/state")
def api_state():
    _, s = conns()
    return jsonify({"people": scribe.headcount(s),
                    "stock": scribe.stock(s),
                    "water_days": scribe.water_days_remaining(s)})


def main():
    app.run(host="0.0.0.0", port=config.DASHBOARD_PORT)


if __name__ == "__main__":
    main()
