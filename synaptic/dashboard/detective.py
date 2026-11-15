"""Trend Detective: the challenge's five steps as an app (variable -> region -> period -> time series ->
significance), on NASA monthly data fetched from Earthdata Cloud. Mounted on the mission-control server at /trend.

The AI detective button turns the current selection into a focused agent campaign (synaptic/briefs), so a human
finding and the autonomous research group use the same data and the same statistics.
"""
from __future__ import annotations

import io
import json
import threading
import time
import uuid

import numpy as np
import pandas as pd
from flask import Blueprint, Response, abort, jsonify, render_template, request

from .. import catalog, cloud, regions, trend

bp = Blueprint("detective", __name__)
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _months(v) -> list[int] | None:
    if not v:
        return None
    if isinstance(v, str):
        v = trend.SEASONS.get(v.upper()) or [int(x) for x in v.split(",") if x.strip()]
    m = sorted({int(x) for x in v if 1 <= int(x) <= 12})
    return None if len(m) in (0, 12) else m


def run_analysis(spec: dict, progress=lambda m: None) -> dict:
    """spec = {var, region, y0, y1, months}. Returns everything the page shows."""
    var = catalog.get(spec["var"])
    bbox, mask, label = regions.resolve(spec.get("region") or {"type": "world"})
    y0 = max(int(spec.get("y0") or var.start), var.start)
    y1 = int(spec.get("y1") or time.gmtime().tm_year)
    months = _months(spec.get("months"))
    progress(f"Fetching {var.label} from NASA ({var.source}) for {label}, {y0}-{y1}")
    f = cloud.field(var.id, bbox, y0, y1, progress=progress)
    w = cloud.area_weights(f.lat, f.lon)
    if mask is not None:
        w = w * mask(f.lat, f.lon)
    rows = []
    for k, ym in enumerate(f.times):
        a = f.data[k]
        ok = np.isfinite(a) & (w > 0)
        rows.append({"year": int(ym[:4]), "month": int(ym[5:7]),
                     "value": float((a[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None,
                     "coverage": float(w[ok].sum() / w.sum()) if w.sum() else 0.0})
    series = pd.DataFrame(rows)
    series["value"] = series["value"].astype(float)
    progress("Testing the trend")
    res = trend.analyse(series, months, var.units, var.label)
    tm = trend.trend_map(f, months)
    if mask is not None:  # grey out cells outside the outline
        mm = mask(np.array(tm["lat"]), np.array(tm["lon"]))
        tm["slope"] = np.where(mm > 0, tm["slope"], np.nan)
    outline = regions._all()[spec["region"]["name"]]["rings"] if (spec.get("region") or {}).get("type") == "named" \
        else None
    res.update({"series": series.to_dict("records"), "map_png": _map_png(tm, var, res, bbox, outline),
                "spec": {"var": var.id, "region": spec.get("region"), "region_label": label, "y0": y0, "y1": y1,
                         "months": months},
                "dataset": {"label": var.label, "units": var.units, "source": var.source, "doi": var.doi,
                            "note": var.note, "backend": var.backend}})
    return res


def _map_png(tm: dict, var, res: dict, bbox, outline=None) -> str:
    import base64

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(tm["slope"], float)
    if not np.isfinite(s).any():
        return ""
    lat, lon = np.asarray(tm["lat"]), np.asarray(tm["lon"])
    lim = float(np.nanpercentile(np.abs(s), 98)) or 1e-6
    w = 7.5
    h = max(3.0, min(5.2, 6.2 * (np.ptp(lat) + 1) / (np.ptp(lon) + 1) + 0.8))
    fig, ax = plt.subplots(figsize=(w, h))
    pc = ax.pcolormesh(lon, lat, s, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
    sig = np.asarray(tm["p"], float) < tm["alpha"]
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    stride = max(1, int(np.sqrt(sig.size / 900)))
    ax.scatter(xx[sig][::stride], yy[sig][::stride], s=4, c="k", alpha=0.6, linewidths=0)
    try:
        for v in regions._all().values():
            if v["kind"] == "country":
                for r in v["rings"]:
                    r = np.array(r)
                    ax.plot(r[:, 0], r[:, 1], lw=0.4, c="#666")
        for r in outline or []:
            r = np.array(r)
            ax.plot(r[:, 0], r[:, 1], lw=1.6, c="#111")
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
    ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    ax.set_aspect("equal" if np.ptp(lon) < 200 else "auto")
    cb = fig.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(f"trend per decade ({var.units})" if var.units else "trend per decade")
    ax.set_title(f"{res['season']} {tm['years'][0]}-{tm['years'][1]}: trend per grid cell\n"
                 f"(dots: significant at p<{tm['alpha']}, carry-over adjusted)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------- saved results (demo cases, cache)
# A finished analysis is saved, so the same question is answered at once the next time, even without a network.
# The demo cases ship with the repository (demo_cache/); everything else is kept in runs/cache/detective/.
_SHIPPED = __import__("pathlib").Path(__file__).with_name("demo_cache")
_LOCAL = cloud.CACHE.parent / "detective"

DEMOS = [
    {"id": "wisconsin_smoke", "label": "Summer smoke over Wisconsin (the challenge video's example)",
     "spec": {"var": "MODIS_AOD", "region": {"type": "named", "name": "Wisconsin (USA)"}, "y0": 2002, "y1": 2025,
              "months": [6, 7, 8]}},
    {"id": "morocco_rain", "label": "Morocco's winter rain since 2001 (GPM IMERG)",
     "spec": {"var": "IMERG_PRECIP", "region": {"type": "named", "name": "Morocco"}, "y0": 2001, "y1": 2025,
              "months": [11, 12, 1, 2, 3, 4]}},
    {"id": "lake_chad_water", "label": "Water stored around Lake Chad (GRACE, see Findings)",
     "spec": {"var": "GRACE_TWS", "region": {"type": "box", "bbox": [10, 17, 12, 18]}, "y0": 2002, "y1": 2026,
              "months": None}},
    {"id": "nw_africa_chlorophyll", "label": "Ocean plant life off north-west Africa (MODIS chlorophyll)",
     "spec": {"var": "MODIS_CHL", "region": {"type": "box", "bbox": [15, 35, -25, -10]}, "y0": 2003, "y1": 2025,
              "months": None}},
    {"id": "mississippi_chlorophyll", "label": "Chlorophyll off the Mississippi (NASA's reference paper, re-tested)",
     "spec": {"var": "MODIS_CHL", "region": {"type": "box", "bbox": [28, 29, -91.5, -90.5]}, "y0": 2003, "y1": 2025,
              "months": None}},
    {"id": "india_soil_moisture", "label": "India's soil moisture since 2000 (GLDAS)",
     "spec": {"var": "GLDAS_SOILM", "region": {"type": "named", "name": "India"}, "y0": 2000, "y1": 2025,
              "months": None}},    {"id": "stratosphere_cooling", "label": "The stratosphere, 20 km up, worldwide (AIRS)",
     "spec": {"var": "AIRS_T50", "region": {"type": "world"}, "y0": 2003, "y1": 2025, "months": None}},
    {"id": "china_no2", "label": "Nitrogen dioxide over eastern China (OMI)",
     "spec": {"var": "OMI_NO2", "region": {"type": "box", "bbox": [30, 40, 110, 122]}, "y0": 2005, "y1": 2025,
              "months": None}},
]


def _norm_spec(spec: dict) -> dict:
    var = catalog.get(spec["var"])
    reg = spec.get("region") or {"type": "world"}
    reg = {"type": reg.get("type", "world"), **({"name": reg["name"]} if reg.get("type") == "named" else {}),
           **({"bbox": [float(v) for v in reg["bbox"]]} if reg.get("type") == "box" else {})}
    return {"var": var.id, "region": reg, "y0": max(int(spec.get("y0") or var.start), var.start),
            "y1": int(spec.get("y1") or time.gmtime().tm_year), "months": sorted(_months(spec.get("months")) or [])}


def _key(spec: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(_norm_spec(spec), sort_keys=True).encode()).hexdigest()[:16]


def _dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o),
                      allow_nan=False).replace("NaN", "null")


def load_saved(spec: dict) -> dict | None:
    k = _key(spec)
    for d in (_LOCAL, _SHIPPED):
        f = d / f"{k}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
    return None


def save_result(spec: dict, out: dict, shipped: bool = False) -> None:
    d = _SHIPPED if shipped else _LOCAL
    d.mkdir(parents=True, exist_ok=True)
    out = {**out, "saved": time.strftime("%d %b %Y", time.gmtime())}
    try:
        (d / f"{_key(spec)}.json").write_text(_dumps(_to_plain(out)))
    except Exception:  # noqa: BLE001
        pass


def _to_plain(o):
    if isinstance(o, dict):
        return {str(k): _to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_plain(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _to_plain(o.tolist())
    return o


def _worker(jid: str, spec: dict):
    def progress(m):
        with _LOCK:
            JOBS[jid]["progress"] = m
    try:
        out = None if spec.get("fresh") else load_saved(spec)
        if out is None:
            out = run_analysis(spec, progress)
            save_result(spec, out)
        with _LOCK:
            JOBS[jid].update(status="done", result=out, progress="done")
    except Exception as e:  # noqa: BLE001
        with _LOCK:
            JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}")


@bp.get("/trend")
def page():
    return render_template("trend.html")


_EVID = __import__("pathlib").Path(__file__).resolve().parents[2] / "evidence"


@bp.get("/findings")
def findings():
    """What the AI research group measured, with the proof: both GRACE solutions, the pre-registered SMAP check, and
    the honest tally of everything that was tested and not shown."""
    def load(rel):
        f = _EVID / rel
        return json.loads(f.read_text()) if f.exists() else {}
    gains, smap = load("water_gains/results.json"), load("smap_confirmation/results.json")
    leads = []
    for key, g in gains.items():
        leads.append({"key": key, **g, "smap": (smap.get(key) or {}).get("primary"),
                      "smap_q": (smap.get(key) or {}).get("quality_only"),
                      "has_smap_fig": (_EVID / "figures" / f"{key}_smap.png").exists()})
    gio = load("giovanni_crosscheck/results.json")
    gio = [g for g in gio if "correlation" in g] if isinstance(gio, list) else []
    wk = load("weekend_shift/results.json").get("tests", {})
    uae = wk.get("UAE_2022")
    rivers = load("river_chlorophyll/results.json")
    return render_template("findings.html", leads=leads, has_smap=bool(smap), gio=gio, uae=uae, weekend=wk,
                           rivers=rivers)


@bp.get("/evidence/weekend/<name>")
def evidence_weekend_figure(name):
    from flask import send_from_directory

    return send_from_directory(_EVID / "weekend_shift" / "figures", name)


@bp.get("/evidence/figures/<name>")
def evidence_figure(name):
    from flask import send_from_directory

    return send_from_directory(_EVID / "figures", name)


@bp.get("/api/td/demos")
def api_demos():
    return jsonify([{"id": d["id"], "label": d["label"], "spec": _norm_spec(d["spec"]),
                     "saved": load_saved(d["spec"]) is not None} for d in DEMOS])


@bp.get("/api/td/catalog")
def api_catalog():
    return jsonify(catalog.menu())


@bp.get("/api/td/regions")
def api_regions():
    return jsonify(regions.names())


@bp.post("/api/td/run")
def api_run():
    spec = request.get_json(force=True)
    try:
        catalog.get(spec.get("var", ""))
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    jid = uuid.uuid4().hex[:10]
    with _LOCK:
        JOBS[jid] = {"status": "running", "progress": "starting", "spec": spec, "t": time.time()}
    threading.Thread(target=_worker, args=(jid, spec), daemon=True).start()
    return jsonify({"job": jid})


@bp.get("/api/td/job/<jid>")
def api_job(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404)
    return Response(json.dumps(j, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o),
                               allow_nan=False).replace("NaN", "null"), mimetype="application/json")


@bp.get("/api/td/job/<jid>/csv")
def api_csv(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done":
        abort(404)
    r = j["result"]
    df = pd.DataFrame(r["series"])
    head = (f"# {r['dataset']['label']} ({r['dataset']['units']}), {r['spec']['region_label']}, source: "
            f"{r['dataset']['source']}" + (f", doi:{r['dataset']['doi']}" if r['dataset']['doi'] else "") + "\n"
            f"# {r.get('verdict', '')}\n")
    return Response(head + df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=trend_{r['spec']['var']}.csv"})


@bp.post("/api/td/detective")
def api_detective():
    """Hand the current selection to the AI research group: writes a focused brief and launches a campaign."""
    import os
    import subprocess
    import sys

    from ..config import ROOT, RUNS_DIR

    spec = request.get_json(force=True)
    var = catalog.get(spec["var"])
    bbox, _, label = regions.resolve(spec.get("region") or {"type": "world"})
    brief = {
        "title": f"Trend Detective: {var.label} over {label}",
        "question": (f"Is {var.label.lower()} over {label} changing between {spec.get('y0')} and {spec.get('y1')}"
                     f" ({trend.season_label(_months(spec.get('months')))}), is the change real (not a data artifact), "
                     "and what drives it? Test it against independent NASA data and explain the mechanism."),
        "mode": "focused",
        "datasets": [var.id],
        "region": {"lat_min": bbox[0], "lat_max": bbox[1], "lon_min": bbox[2], "lon_max": bbox[3]},
        "start_year": int(spec.get("y0") or var.start), "end_year": int(spec.get("y1") or 2025),
    }
    path = RUNS_DIR / "briefs" / f"detective_{uuid.uuid4().hex[:6]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(brief, indent=1))
    cmd = [sys.executable, "-m", "synaptic", "run", str(path)]
    log = open(RUNS_DIR / "launch.log", "a")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    return jsonify({"ok": True, "brief": str(path)})
