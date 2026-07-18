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
{refresh}
<title>EMBER</title>
<style>
body{{font-family:system-ui;margin:0;background:#141a14;color:#dfe8df}}
header{{padding:10px 16px;background:#1e2a1e;display:flex;gap:12px;
       align-items:center;flex-wrap:wrap}}
h1{{font-size:18px;margin:0}} .badge{{padding:2px 10px;border-radius:10px;
background:#8a2f2f;font-weight:600;font-size:12px}}
.ok{{background:#2f6a2f}} .staff{{background:#6a5a2f}}
.danger{{background:#8a2f2f;width:auto;padding:6px 12px;font-size:14px}}
nav a{{color:#9fd49f;margin-right:12px;text-decoration:none}}
main{{padding:12px 16px}} .card{{background:#1c241c;border-radius:8px;
padding:10px 14px;margin-bottom:10px}}
small{{color:#8fa58f}} img{{max-width:96px;border-radius:6px}}
input,button{{font-size:17px;padding:12px;border-radius:6px;border:none;
box-sizing:border-box}}
input{{width:100%;margin-bottom:10px;background:#0f140f;color:#dfe8df}}
button{{background:#2f6a2f;color:#fff;width:100%}}
.viewfinder{{width:100%;max-width:480px;border-radius:10px;display:block;
margin:0 auto 10px}}
</style>
<header><h1>&#128293; EMBER</h1>
<span class=badge>OFFLINE &#10003;</span>
<span class=badge ok>RAM {ram}</span>
<nav><a href=/>mind</a><a href=/board>board</a><a href=/intake>intake</a>
<a href=/find>find</a>
<a href=/supplies>supplies</a><a href=/brief>brief</a><a href=/chat>ask</a>
<a href=/staff>{staff_label}</a>
</nav></header><main>{body}</main>"""


def is_staff() -> bool:
    return request.cookies.get("staff") == config.STAFF_PIN


def ram() -> str:
    try:
        line = Path("/proc/meminfo").read_text().splitlines()
        avail = int([x for x in line if "MemAvailable" in x][0].split()[1])
        return f"{avail // 1024}MB free"
    except Exception:
        return "n/a"


def page(body: str, live: bool = False) -> str:
    """live=True auto-reloads every 6s (thought stream, board). Form
    pages must NOT refresh — it wipes whatever the user is typing."""
    r = "<meta http-equiv=refresh content=6>" if live else ""
    label = "&#128737; staff" if is_staff() else "staff"
    return PAGE.format(refresh=r, ram=ram(), body=body, staff_label=label)


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
    return page("".join(rows) or "<div class=card>No thoughts yet.</div>", live=True)


@app.get("/board")
def board():
    _, s = conns()
    cards = []
    staff = is_staff()
    for h in scribe.households(s):
        t = time.strftime("%b %d %H:%M", time.localtime(h["ts"]))
        img = f"<img src=/photo/{h['id']}>" if h["photo"] else ""
        med = f"<br><small>medical: {html.escape(h['medical'])}</small>" if h["medical"] else ""
        mis = f"<br><small>&#9888; missing: {html.escape(h['missing'])}</small>" if h["missing"] else ""
        rm = (f"<form method=post action=/remove/{h['id']} "
              f"style='margin-top:6px'><button class=danger "
              f"onclick=\"return confirm('Remove {html.escape(h['names'])}?')\">"
              f"remove</button></form>") if staff else ""
        cards.append(f"<div class=card>{img}<b>{html.escape(h['names'])}</b>"
                     f"<br><small>checked in {t}</small>{med}{mis}{rm}</div>")
    return page(f"<div class=card><b>{scribe.headcount(s)} people "
                f"registered</b></div>" + "".join(cards), live=True)


def _cam():
    from . import camera
    if camera.live_cam is None:
        try:
            camera.live_cam = camera.Cam()
            time.sleep(0.8)               # first frames arrive
        except Exception:
            pass
    return camera.live_cam


@app.get("/preview.jpg")
def preview():
    cam = _cam()
    f = cam.latest() if cam else None
    if not f:
        return "", 503
    from flask import Response
    return Response(f, mimetype="image/jpeg")


@app.get("/stream.mjpg")
def stream():
    cam = _cam()
    if cam is None:
        return "", 503
    from flask import Response

    def gen():
        while True:
            f = cam.latest()
            if f:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(f)).encode()
                       + b"\r\n\r\n" + f + b"\r\n")
            time.sleep(0.12)

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


_VIEWFINDER = ("<div class=card><img class=viewfinder src=/stream.mjpg "
               "alt='camera warming up...'>"
               "<small>live view from Ember's camera</small></div>")


@app.get("/photo/<int:rid>")
def photo(rid: int):
    _, s = conns()
    row = s.execute("SELECT photo FROM registry WHERE id=?",
                    (rid,)).fetchone()
    if row and row[0] and Path(row[0]).exists():
        from flask import send_file
        return send_file(row[0])
    return "", 404


@app.get("/find")
def find_form():
    return page("""<div class=card><b>Find a missing person by photo</b>
      <br><small>Upload any photo of who you're looking for — the box
      matches it against consented check-in photos. Photos never leave
      this box.</small></div>
      <form method=post action=/find enctype=multipart/form-data class=card>
      <input type=file name=photo accept=image/* capture>
      <button>search</button></form>""" + _VIEWFINDER + """
      <form method=post action=/find-camera class=card>
      <button>&#128247; stand in front of Ember and use its camera</button>
      </form>""")


@app.post("/find-camera")
def find_camera():
    from . import camera
    shot = camera.capture("find")
    if not shot:
        return page("<div class=card>Camera did not respond.</div>"
                    "<a href=/find>back</a>")
    return _find_results(shot)


@app.post("/find")
def find():
    f = request.files.get("photo")
    if not f or not f.filename:
        return redirect("/find")
    from pathlib import Path as _P
    qdir = _P(config.VAULT) / "photos"
    qdir.mkdir(parents=True, exist_ok=True)
    qpath = str(qdir / f"query-{int(time.time())}.jpg")
    f.save(qpath)
    return _find_results(qpath)


def _find_results(qpath: str):
    from . import faces
    _, s = conns()
    results = faces.match(s, qpath, scribe.households(s))
    emit("face_search", matches=len(results))
    if not results:
        return page("<div class=card>No face found in that photo — "
                    "try a clearer, front-facing shot.</div>"
                    "<a href=/find>try again</a>")
    cards = []
    for r in results[:3]:
        t = time.strftime("%b %d %H:%M", time.localtime(r["ts"]))
        badge = ("&#9989; likely match" if r["same_person"]
                 else "possible match")
        cards.append(
            f"<div class=card><img src=/photo/{r['id']}>"
            f"<b>{html.escape(r['names'])}</b> — {badge} "
            f"({r['score']:.0%})<br><small>checked in {t}</small></div>")
    return page("".join(cards) + "<a href=/find>search again</a>")


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
    return page(_VIEWFINDER + """<form method=post action=/intake class=card>
      <input name=names placeholder="household names (comma-separated)">
      <input name=medical placeholder="medical needs / allergies">
      <input name=missing placeholder="anyone unaccounted for">
      <input name=phone placeholder="phone (optional)">
      <label><input type=checkbox name=take_photo value=1 checked
      style="width:auto;margin-right:8px">
      take a check-in photo with the box camera (consented, for
      reunification only)</label><br><br>
      <button>&#128247; register &amp; capture</button></form>
      <div class=card><small>Consent: photos are taken only if the arrival
      agrees, used solely for family reunification, and never leave this box.
      </small></div>""")


@app.post("/intake")
def intake_submit():
    names = request.form.get("names", "").strip()
    if not names:
        return redirect("/intake")
    photo = ""
    if request.form.get("take_photo"):
        from . import camera
        photo = camera.capture("intake") or ""
    _, s = conns()
    rid = scribe.register(s, names, request.form.get("medical", ""),
                          request.form.get("missing", ""),
                          request.form.get("phone", ""), photo=photo)
    emit("registered", id=rid, names=names, photo=bool(photo))
    return redirect("/board")


@app.get("/staff")
def staff_form():
    if is_staff():
        return page("<div class=card>&#128737; <b>Staff mode is ON</b> — "
                    "the board now shows remove buttons for bad entries. "
                    "Removals are audit-logged.</div>"
                    "<form method=post action=/staff-off class=card>"
                    "<button>switch back to resident mode</button></form>")
    return page("""<div class=card><b>Staff mode</b><br><small>For shelter
      workers: unlocks removing registry entries with bad photos or data.
      Every removal is written to the activity log.</small></div>
      <form method=post action=/staff class=card>
      <input name=pin type=password placeholder="staff PIN">
      <button>enter staff mode</button></form>""")


@app.post("/staff")
def staff_login():
    from flask import make_response
    if request.form.get("pin", "") != config.STAFF_PIN:
        return page("<div class=card>Wrong PIN.</div>"
                    "<a href=/staff>try again</a>")
    resp = make_response(redirect("/board"))
    resp.set_cookie("staff", config.STAFF_PIN, max_age=8 * 3600,
                    samesite="Lax")
    return resp


@app.post("/staff-off")
def staff_off():
    from flask import make_response
    resp = make_response(redirect("/board"))
    resp.set_cookie("staff", "", max_age=0)
    return resp


@app.post("/remove/<int:rid>")
def remove(rid: int):
    if not is_staff():
        return redirect("/staff")
    _, s = conns()
    if scribe.remove(s, rid):
        emit("registry_removed", id=rid)
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
    _cam()          # sensor takes ~2s to wake — start it at boot, not on
    #                 the first viewfinder request (which 503'd)
    app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, threaded=True)


if __name__ == "__main__":
    main()
