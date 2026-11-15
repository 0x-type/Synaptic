"""Planted-anomaly benchmark (analogue of the paper's fixed-input benchmark, Fig. 4).

Real NASA POWER data for a box is detrended per cell and month (natural trends removed, natural
variability kept), then ONE anomaly with a known answer is planted:
  seasonal_step  physical-like: a step from year Y in 3 consecutive months over a 3x3 block
  phase_shift    physical-like: the seasonal cycle shifts by ~3 weeks from year Y in a block
  variance       physical-like: variability more than doubles from year Y in a block
  artifact_step  artifact: a uniform step in EVERY month at EVERY cell from year Y (instrument change)
  constant_run   artifact: 18 months of a repeated constant value in a block (gap filling)

Levels (information given to the model, as in the paper):
  1  summary statistics of the region only, no tools
  2  raw monthly tables of two cells as text in the prompt, no tools
  3  the whole dataset as a file + python/look (no images)
  4  + figures shown to the vision model
  5  + literature search and satellite imagery
Scoring is programmatic against the hidden truth: detected, type, onset (+-2 y), location, artifact call.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import synlib
from .config import MODELS, RUNS_DIR
from .llm import image_part, make_llm
from .protocol import parse_action
from .record import Record, new_id
from .tools import ToolContext, toolbox
from .tools import nasa

BOX = {"lat_min": 30.0, "lat_max": 34.0, "lon_min": -9.0, "lon_max": -4.0}
VARIABLES = ["T2M", "GWETROOT", "PRECTOTCORR"]
TYPES = ["seasonal_step", "phase_shift", "variance", "artifact_step", "constant_run"]
ARTIFACT = {"artifact_step", "constant_run"}
BINS = [(0, 1, "no DNA"), (1, 200, "<200"), (200, 600, "200-599"), (600, 2000, "600-1999"), (2000, 10**9, ">=2000")]

SUBMISSION = """{"anomaly_found": true|false, "type": "seasonal_step|phase_shift|variance|artifact_step|constant_run|trend|other|none",
 "onset_year": int|null, "lat": float|null, "lon": float|null, "months": [month numbers]|null,
 "is_artifact": true|false, "confidence": 0-1, "summary": "what you found, the evidence, and the experiment that would test it"}"""

TASK = """You receive NASA POWER monthly data ({var}, {units}) for a region of Morocco (lat {lat_min}..{lat_max},
lon {lon_min}..{lon_max}, {y0}-{y1}). Natural linear trends were removed from each cell. The data may contain one
notable anomaly: either a real change in the climate system or a data-production artifact. Characterize the data:
report every feature you judge important, with evidence and confidence, and decide whether it is physical or an
artifact. Give location as the centre of the affected area."""


# ---------------------------------------------------------------------------- data + cases

def base_data(cache: Path) -> dict[str, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for v in VARIABLES:
        f = cache / f"base_{v}.csv"
        if not f.exists():
            ctx = ToolContext(None, "bench", None, None, cache, cache)  # type: ignore[arg-type]
            r = nasa.power_region(ctx, parameters=[v], start_year=1981, end_year=2025, **BOX)
            if r.is_error:
                raise RuntimeError(r.text)
            df = pd.read_csv(r.files[0])
            df.to_csv(f, index=False)
        out[v] = pd.read_csv(f)
    return out


def _detrend(cube: synlib.Cube) -> np.ndarray:
    np.seterr(all="ignore")
    d = cube.data.copy()
    t = np.arange(len(cube.years), dtype=float)
    for m in range(12):
        x = d[..., m]
        mean = np.nanmean(x, axis=-1, keepdims=True)
        tc = t - t.mean()
        slope = np.nansum((x - mean) * tc, axis=-1, keepdims=True) / np.sum(tc ** 2)
        d[..., m] = x - slope * tc
    return d


def make_case(base: dict[str, pd.DataFrame], rng: random.Random, kind: str | None = None) -> tuple[pd.DataFrame, dict]:
    var = rng.choice(VARIABLES)
    kind = kind or rng.choice(TYPES)
    cube = synlib.to_cube(base[var], var)
    d = _detrend(cube)
    nl, nn, ny, _ = d.shape
    years = cube.years
    onset = rng.randint(1995, 2015)
    after = years >= onset
    complete = ~np.isnan(d).any(axis=(2, 3))  # cells with a full record (soil wetness is NaN over the ocean)
    centres = [(i, j) for i in range(1, nl - 1) for j in range(1, nn - 1) if complete[i - 1:i + 2, j - 1:j + 2].all()]
    ci, cj = rng.choice(centres)
    block = [(ci + a, cj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    with np.errstate(all="ignore"):
        sd = np.nanstd(d, axis=2)  # [lat, lon, 12]
    months = None
    if kind == "seasonal_step":
        m0 = rng.randint(0, 11)
        months = [((m0 + k) % 12) + 1 for k in range(3)]
        k = rng.choice([-1, 1]) * rng.uniform(1.5, 2.2)
        for (i, j) in block:
            for m in months:
                d[i, j, after, m - 1] += k * sd[i, j, m - 1]
    elif kind == "phase_shift":
        shift = rng.choice([-1, 1]) * 0.75  # months
        for (i, j) in block:
            clim = np.nanmean(d[i, j], axis=0)
            mm = np.arange(12)
            shifted = np.interp((mm - shift) % 12, np.r_[mm, 12], np.r_[clim, clim[0]])
            d[i, j, after, :] += shifted - clim
    elif kind == "variance":
        for (i, j) in block:
            clim = np.nanmean(d[i, j], axis=0)
            d[i, j, after, :] = clim + (d[i, j, after, :] - clim) * 2.4
    elif kind == "artifact_step":
        k = rng.choice([-1, 1]) * rng.uniform(1.2, 1.8)
        with np.errstate(all="ignore"):
            d[:, :, after, :] += k * np.nanmean(sd, axis=-1)[:, :, None, None]
    elif kind == "constant_run":
        start = int(np.argmax(after))
        flat = d.reshape(nl, nn, -1)
        for (i, j) in block:
            v0 = flat[i, j, start * 12]
            flat[i, j, start * 12: start * 12 + 18] = v0
        d = flat.reshape(d.shape)
    if var in ("PRECTOTCORR",):
        d = np.clip(d, 0, None)
    if var == "GWETROOT":
        d = np.clip(d, 0, 1)
    rows = []
    for i, la in enumerate(cube.lats):
        for j, lo in enumerate(cube.lons):
            for y_i, y in enumerate(years):
                for m in range(12):
                    rows.append((la, lo, var, int(y), m + 1, round(float(d[i, j, y_i, m]), 4)))
    df = pd.DataFrame(rows, columns=["lat", "lon", "variable", "year", "month", "value"])
    region_wide = kind == "artifact_step"
    truth = {"variable": var, "type": kind, "onset": onset, "is_artifact": kind in ARTIFACT, "months": months,
             "lat": float(cube.lats[ci]), "lon": float(cube.lons[cj]), "region_wide": region_wide,
             "block": [[float(cube.lats[a]), float(cube.lons[b])] for a, b in block],
             "dlat": float(np.diff(cube.lats).min()), "dlon": float(np.diff(cube.lons).min())}
    return df, truth


def score(sub: dict, truth: dict) -> dict:
    found = bool(sub.get("anomaly_found"))
    try:
        la, lo = float(sub.get("lat")), float(sub.get("lon"))
        if truth["region_wide"]:
            loc = True
        else:
            loc = abs(la - truth["lat"]) <= 2.01 * truth["dlat"] and abs(lo - truth["lon"]) <= 2.01 * truth["dlon"]
    except (TypeError, ValueError):
        loc = False
    t = str(sub.get("type", "")).strip().lower()
    type_ok = t == truth["type"] or (truth["type"] == "seasonal_step" and t in ("step", "shift") and sub.get("months"))
    try:
        onset_ok = abs(int(sub.get("onset_year")) - truth["onset"]) <= 2 if truth["type"] != "constant_run" \
            else abs(int(sub.get("onset_year")) - truth["onset"]) <= 2
    except (TypeError, ValueError):
        onset_ok = False
    art_ok = bool(sub.get("is_artifact")) == truth["is_artifact"]
    detected = found and loc
    return {"detected": int(detected), "type_ok": int(found and type_ok), "onset_ok": int(found and onset_ok),
            "location_ok": int(found and loc), "artifact_ok": int(found and art_ok),
            "score": float(detected + (found and type_ok) + (found and onset_ok) + (found and loc) + (found and art_ok))}


# ---------------------------------------------------------------------------- one attempt

def _summary_stats(df: pd.DataFrame) -> str:
    g = df.groupby("year").value.agg(["mean", "std"]).round(3)
    return ("Region-mean annual statistics (mean, spatial-temporal std) per year:\n" + g.to_string() +
            f"\nOverall: mean {df.value.mean():.3f}, std {df.value.std():.3f}, cells {df.groupby(['lat', 'lon']).ngroups}")


def _two_cells(df: pd.DataFrame, truth: dict, rng: random.Random) -> str:
    cells = df[["lat", "lon"]].drop_duplicates().values.tolist()
    inside = [truth["lat"], truth["lon"]]
    outside = rng.choice([c for c in cells if abs(c[0] - inside[0]) > 1.5 or abs(c[1] - inside[1]) > 1.5])
    pair = [inside, outside]
    rng.shuffle(pair)
    out = []
    for la, lo in pair:
        s = df[(df.lat == la) & (df.lon == lo)].pivot(index="year", columns="month", values="value").round(3)
        out.append(f"Cell lat {la}, lon {lo} (year x month):\n{s.to_string()}")
    return "\n\n".join(out)


def attempt(llm, model_role: str, level: int, case_dir: Path, df: pd.DataFrame, truth: dict,
            rng: random.Random, record: Record, max_steps: int = 16) -> dict:
    adir = case_dir / f"attempt_{new_id()}"
    adir.mkdir(parents=True)
    data = adir / "data.csv"
    df.to_csv(data, index=False)
    units = {"T2M": "C", "GWETROOT": "0-1", "PRECTOTCORR": "mm/day"}[truth["variable"]]
    task = TASK.format(var=truth["variable"], units=units, y0=int(df.year.min()), y1=int(df.year.max()), **BOX)
    sid = record.open_session("bench", None, f"bench-L{level}", model_role)
    state = {"data_chars": 0, "longest_run": 0}
    if level <= 2:
        context = _summary_stats(df) if level == 1 else _two_cells(df, truth, rng)
        n = sum(len(l) for l in context.splitlines() if sum(c.isdigit() for c in l) / max(len(l), 1) > 0.45)
        state.update({"data_chars": n if level == 2 else 0, "longest_run": n if level == 2 else 0})
        msgs = [{"role": "system", "content": "You are an expert Earth scientist. Answer with your analysis, then one "
                 f"fenced JSON block:\n```json\n{SUBMISSION}\n```"},
                {"role": "user", "content": f"{task}\n\n<data>\n{context}\n</data>"}]
        c = llm.complete(msgs, model_role)
        record.add_usage(sid, c.tokens_in, c.tokens_out, state["data_chars"])
        from .protocol import parse_json

        sub = parse_json(c.text) or {}
    else:
        tools = {k: v for k, v in toolbox("bench").items() if k in ("python", "look")}
        if level >= 5:
            tools.update({k: v for k, v in toolbox("bench").items()
                          if k in ("literature_search", "gibs_layers", "gibs_snapshot")})
        ctx = ToolContext(record, "bench", None, sid, adir, adir, mock=llm.name == "mock")
        ctx.state.update({"level": level})
        doc = "\n".join(t.doc() for t in tools.values())
        msgs = [{"role": "system", "content": f"You are an expert Earth scientist working alone with tools.\n"
                 f"The dataset is the file data.csv in your folder (columns lat, lon, variable, year, month, value). "
                 f"synlib (change detection: synlib.sweep, analyze_series, plot_series...) is importable.\n"
                 f"Tools:\n{doc}\nAct with one fenced JSON block per reply: "
                 '{"action": "<tool>", "args": {...}}. When done: {"action": "finish", "args": <submission>} with '
                 f"submission = {SUBMISSION}\n<bench_level>{level}</bench_level>"},
                {"role": "user", "content": task}]
        sub = {}
        for step in range(max_steps):
            if step == max_steps - 2:
                msgs.append({"role": "user", "content": "Last step: call finish now."})
            c = llm.complete(msgs, model_role)
            record.add_usage(sid, c.tokens_in, c.tokens_out)
            msgs.append({"role": "assistant", "content": c.text})
            action, _, err = parse_action(c.text)
            if err:
                msgs.append({"role": "user", "content": err})
                continue
            if action["action"] == "finish":
                sub = action["args"]
                break
            tool = tools.get(action["action"])
            if not tool:
                msgs.append({"role": "user", "content": f"Unknown tool. Available: {list(tools)} or finish."})
                continue
            res = tool.run(ctx, action["args"])
            images = res.images if (level >= 4 and MODELS.vision) else []
            text = res.text if level >= 4 or not res.images else \
                "Images are not available at this information level; inspect numbers instead."
            content = [{"type": "text", "text": f"Observation:\n{text}"}] + [image_part(p) for p in images] \
                if images else f"Observation:\n{text}"
            msgs.append({"role": "user", "content": content})
        state["data_chars"] = ctx.state.get("data_chars", 0)
        state["longest_run"] = ctx.state.get("longest_run", 0)
        record.add_data_chars(sid, state["data_chars"])
    record.close_session(sid)
    sc = score(sub, truth)
    return {**sc, "submission": sub, **state}


# ---------------------------------------------------------------------------- driver

def run_bench(models: list[str] | None = None, levels=(1, 2, 3, 4, 5), attempts: int = 10, mock: bool = False,
              db: str | None = None, seed: int = 11):
    record = Record(db or RUNS_DIR / "synaptic.db")
    bench_id = time.strftime("b%Y%m%d-%H%M%S")
    root = RUNS_DIR / "bench" / bench_id
    base = base_data(RUNS_DIR / "bench" / "base")
    models = models or ([MODELS.default] if not mock else ["mock"])
    llm = make_llm(mock)
    for model in models:
        if not mock:
            MODELS.by_role["bench"] = model
        for level in levels:
            rng = random.Random(seed + level)  # same cases for every model at a level
            for k in range(attempts):
                df, truth = make_case(base, rng, TYPES[k % len(TYPES)])
                case_dir = root / f"L{level}" / f"case{k:03d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                (root / "truth").mkdir(exist_ok=True)
                (root / "truth" / f"L{level}_case{k:03d}.json").write_text(json.dumps(truth))  # never inside case_dir
                try:
                    r = attempt(llm, "bench", level, case_dir, df, truth, rng, record)
                except Exception as e:  # noqa: BLE001
                    r = {"detected": 0, "type_ok": 0, "onset_ok": 0, "location_ok": 0, "artifact_ok": 0, "score": 0,
                         "submission": {"error": str(e)}, "data_chars": 0, "longest_run": 0}
                record.x("""INSERT INTO bench_attempts(bench_id,model,level,case_id,detected,onset_ok,type_ok,location_ok,
                            artifact_ok,score,data_chars,longest_run,submission,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (bench_id, model, level, f"{truth['type']}:{truth['variable']}", r["detected"], r["onset_ok"],
                          r["type_ok"], r["location_ok"], r["artifact_ok"], r["score"], r["data_chars"], r["longest_run"],
                          json.dumps(r["submission"], default=str)[:4000], time.time()))
                print(f"[{model} L{level} #{k}] {truth['type']:14s} {truth['variable']:12s} score {r['score']:.0f}/5 "
                      f"detected={r['detected']} type={r['type_ok']} artifact={r['artifact_ok']} read={r['data_chars']}",
                      flush=True)
    summarize(record, bench_id)
    return bench_id


def summarize(record: Record, bench_id: str):
    rows = pd.DataFrame(record.q("SELECT * FROM bench_attempts WHERE bench_id=?", (bench_id,)))
    if rows.empty:
        return
    print("\nMean score (0-5) by model and level:")
    print(rows.pivot_table(index="model", columns="level", values="score", aggfunc="mean").round(2).to_string())
    print("\nDetection rate by model and level:")
    print(rows.pivot_table(index="model", columns="level", values="detected", aggfunc="mean").round(2).to_string())
    t = rows[rows.level >= 3].copy()
    if len(t):
        t["read_bin"] = pd.cut(t.longest_run, [b[0] for b in BINS] + [10**9], right=False, labels=[b[2] for b in BINS])
        print("\nDetection vs longest contiguous raw data read (levels 3-5):")
        print(t.groupby("read_bin", observed=False).detected.agg(["mean", "count"]).round(2).to_string())
