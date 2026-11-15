"""Mission control: a live view of the shared record (read-only, plus launch/stop)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from ..config import ROOT, RUNS_DIR
from ..record import Record

TASK_COLS = "id,parent_id,stage,kind,title,status,origin,depth,verdict,revisions,created,started,ended,triage_note,meta"


def create_app(db: str | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    record = Record(db or RUNS_DIR / "synaptic.db")
    from .detective import bp as detective_bp

    app.register_blueprint(detective_bp)  # the challenge app: /trend

    def cdir(cid: str) -> Path:
        c = record.one("SELECT workdir FROM campaigns WHERE id=?", (cid,))
        if not c:
            abort(404)
        return Path(c["workdir"])

    def web_path(cid: str, p: str | None) -> str | None:
        if not p:
            return None
        try:
            rel = Path(p).resolve().relative_to(cdir(cid).resolve())
            return f"/files/{cid}/{rel.as_posix()}"
        except ValueError:
            return None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/campaigns")
    def campaigns():
        return jsonify(record.q("SELECT id,name,mode,status,started,ended FROM campaigns WHERE id!='bench' "
                                "ORDER BY started DESC"))

    @app.get("/api/briefs")
    def briefs():
        return jsonify(sorted(p.name for p in (ROOT / "briefs").glob("*.yaml")))

    @app.get("/api/state")
    def state():
        cid = request.args.get("cid")
        if not cid:
            row = record.one("SELECT id FROM campaigns ORDER BY started DESC")
            if not row:
                return jsonify({"empty": True})
            cid = row["id"]
        since = int(request.args.get("since", 0))
        c = record.one("SELECT * FROM campaigns WHERE id=?", (cid,))
        if not c:
            return jsonify({"empty": True})
        c["brief"] = json.loads(c["brief"])
        tasks = record.q(f"SELECT {TASK_COLS} FROM tasks WHERE campaign_id=? ORDER BY created", (cid,))
        for t in tasks:
            m = json.loads(t.pop("meta") or "{}")
            t["family"] = m.get("family")
        events = record.q("SELECT id,task_id,role,ts,type,substr(content,1,600) content FROM events "
                          "WHERE campaign_id=? AND id>? ORDER BY id DESC LIMIT 120", (cid, since))
        for e in events:
            if e["type"] == "image":
                e["src"] = web_path(cid, e["content"])
        cands = record.candidates(cid)
        for k in cands:
            s = k.pop("stats")
            k["bbox"] = [s.get("lat_min"), s.get("lat_max"), s.get("lon_min"), s.get("lon_max")]
            k["cells"] = json.loads(s.get("cells") or "[]")[:400]
        reports = record.q("""SELECT r.id,r.task_id,r.family,r.title,r.created,t.rating,t.wins,t.played,t.rank
                              FROM reports r LEFT JOIN ratings t ON t.report_id=r.id WHERE r.campaign_id=?
                              ORDER BY COALESCE(t.rank, 999), r.created""", (cid,))
        for r in reports:
            cand = next((k for k in cands if k["id"] == r["family"]), None)
            r["outcome"] = cand["outcome"] if cand else None
        last = record.one("SELECT MAX(id) m FROM events WHERE campaign_id=?", (cid,))
        return jsonify({
            "campaign": c, "stats": record.stats(cid), "tasks": tasks, "events": events, "candidates": cands,
            "stages": record.q("SELECT key,name,status FROM stages WHERE campaign_id=? ORDER BY idx", (cid,)),
            "thresholds": record.q("SELECT key,value,rationale FROM thresholds WHERE campaign_id=?", (cid,)),
            "reports": reports, "last_event": last["m"] if last else 0,
            "kb": record.q("SELECT id,task_id,kind,title,substr(body,1,700) body,tags FROM kb WHERE campaign_id=? "
                           "ORDER BY id DESC LIMIT 80", (cid,)),
            "stop_requested": (cdir(cid) / "STOP").exists(),
            "models": record.q("SELECT role, model, COUNT(*) n FROM sessions WHERE campaign_id=? GROUP BY role, model",
                               (cid,)),
        })

    @app.get("/api/task/<tid>")
    def task(tid):
        t = record.task(tid)
        if not t:
            abort(404)
        ev = record.q("SELECT id,role,ts,type,content,meta FROM events WHERE task_id=? ORDER BY id", (tid,))
        for e in ev:
            if e["type"] == "image":
                e["src"] = web_path(t["campaign_id"], e["content"])
        t["figures_web"] = [web_path(t["campaign_id"], f) for f in t["meta"].get("figures", [])]
        children = record.q("SELECT id,title,status,origin FROM tasks WHERE parent_id=?", (tid,))
        return jsonify({"task": t, "events": ev, "children": children})

    @app.get("/api/report/<rid>")
    def report(rid):
        r = record.one("SELECT * FROM reports WHERE id=?", (rid,))
        if not r:
            abort(404)
        task_dir = cdir(r["campaign_id"]) / "tasks" / r["task_id"]

        def fix(m):
            src = m.group(2)
            if not src.startswith(("http", "/")):
                p = (task_dir / src).resolve()
                return f"![{m.group(1)}]({web_path(r['campaign_id'], str(p))})"
            return m.group(0)

        body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix, r["body"])
        try:
            import markdown

            html = markdown.markdown(body, extensions=["tables", "fenced_code"])
        except Exception:  # noqa: BLE001
            html = f"<pre>{body}</pre>"
        figs = [web_path(r["campaign_id"], f) for f in json.loads(r["figures"] or "[]")]
        return jsonify({"id": rid, "title": r["title"], "html": html, "figures": [f for f in figs if f],
                        "markdown": r["body"]})

    @app.get("/files/<cid>/<path:rel>")
    def files(cid, rel):
        base = cdir(cid).resolve()
        p = (base / rel).resolve()
        if not str(p).startswith(str(base)) or not p.is_file():
            abort(404)
        return send_file(p)

    @app.get("/api/basemap/<cid>")
    def basemap(cid):
        c = record.one("SELECT brief FROM campaigns WHERE id=?", (cid,))
        if not c:
            abort(404)
        reg = json.loads(c["brief"])["region"]
        out = cdir(cid) / "basemap.png"
        if not out.exists():
            from ..tools import nasa
            from ..tools.base import ToolContext

            ctx = ToolContext(record, cid, None, None, cdir(cid), cdir(cid))
            r = nasa.gibs_snapshot(ctx, "BlueMarble_ShadedRelief_Bathymetry", "2004-08-01", reg["lat_min"],
                                   reg["lat_max"], reg["lon_min"], reg["lon_max"], 1000)
            if r.is_error:
                abort(404)
            Path(r.files[0]).rename(out)
        return send_file(out)

    @app.get("/api/bench")
    def bench():
        rows = record.q("SELECT bench_id,model,level,case_id,detected,type_ok,onset_ok,location_ok,artifact_ok,score,"
                        "data_chars,longest_run FROM bench_attempts ORDER BY id")
        if not rows:
            return jsonify({"empty": True})
        latest = rows[-1]["bench_id"]
        bid = request.args.get("id", latest)
        rows = [r for r in rows if r["bench_id"] == bid]
        ids = sorted({r["bench_id"] for r in record.q("SELECT DISTINCT bench_id FROM bench_attempts")}, reverse=True)
        return jsonify({"id": bid, "ids": ids, "rows": rows})

    @app.post("/api/stop/<cid>")
    def stop(cid):
        (cdir(cid) / "STOP").write_text("stop requested from dashboard")
        record.event(cid, None, None, "harness", "log", "Stop requested from mission control")
        return jsonify({"ok": True})

    @app.post("/api/launch")
    def launch():
        body = request.get_json(force=True)
        brief = ROOT / "briefs" / Path(body.get("brief", "")).name
        if not brief.exists():
            return jsonify({"ok": False, "error": f"No brief named {brief.name}"}), 400
        cmd = [sys.executable, "-m", "synaptic", "--db", record.path, "run", str(brief)]
        if body.get("mock"):
            cmd.append("--mock")
        log = open(RUNS_DIR / "launch.log", "a")
        subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        return jsonify({"ok": True})

    return app
