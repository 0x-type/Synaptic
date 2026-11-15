"""MockLLM: a deterministic scripted 'scientist' that drives the full pipeline through the REAL
tools and REAL NASA data without any API key. Used for offline testing and as a fallback demo.
It is not intelligent: every decision is a fixed rule. The dashboard labels mock runs clearly."""
from __future__ import annotations

import json
import re

from .llm import Completion


def _txt(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    return c or ""


def _act(name: str, thought: str = "", **args) -> str:
    return f"[mock] {thought}\n```json\n{json.dumps({'action': name, 'args': args})}\n```"


def _family_obs(all_obs: str) -> dict | None:
    """Last family_info observation still intact in context (older ones may have been pruned)."""
    for m in reversed(list(re.finditer(r"Observation:\n(\{\n \"family\".*?\n\})", all_obs, re.S))):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def _json(obj: dict, thought: str = "") -> str:
    return f"[mock] {thought}\n```json\n{json.dumps(obj)}\n```"


class MockLLM:
    name = "mock"

    def complete(self, messages: list[dict], role: str, max_tokens=None, temperature=None, schema=None) -> Completion:
        fn = getattr(self, f"_{role}", None)
        text = fn(messages) if fn else _json({})
        tin = sum(len(_txt(m)) for m in messages) // 4
        return Completion(text, "", tin, len(text) // 4, "mock")

    # ------------------------------------------------------------------ one-shot roles
    def _planner(self, messages):
        brief = json.loads(_txt(messages[-1]).split("Research brief:\n", 1)[1])
        vars_ = [v.upper() for v in brief.get("variables", [])]
        links = {"PRECTOTCORR": "GWETROOT", "GWETROOT": "PRECTOTCORR", "T2M_MAX": "T2M", "T2M": "T2M_MAX",
                 "ALLSKY_SFC_SW_DWN": "CLOUD_AMT"}
        for v in list(vars_):
            if links.get(v) and links[v] not in vars_:
                vars_.append(links[v])
        return _json({"variables": vars_, "season": None,
                      "thresholds": {"q_max": [0.05, "field FDR"], "min_cells": [3, "spatial coherence"],
                                     "min_corroboration": [0.25, "linked variable co-change"]},
                      "sweep_brief": "Report gaps, constant runs and abrupt jumps.",
                      "classify_brief": "", "census_brief": "Prefer diverse variables and classes.",
                      "deep_dive_brief": "Check whether a step is uniform across months (artifact-like).",
                      "notes": "mock plan"}, "Planning variables plus linked variables; thresholds before data.")

    def _pi(self, messages):
        sysm = _txt(messages[0])
        if "lab meeting" in sysm.lower():
            board = _txt(messages[-1])
            ids = re.findall(r"^(H\w+) \[(\w+)\]", board, re.M)
            close = [{"id": i, "status": "inconclusive", "reason": "mock: no signal"} for i, st in ids
                     if st == "inconclusive"][:1]
            return _json({"notes": "mock lab meeting", "close": close, "deeper": [], "new": []})
        return _json({"rationale": "mock programme: one driver hypothesis per brief question",
                      "hypotheses": [
                          {"question": "3", "title": "ENSO drives interannual rain", "priority": 0.8,
                           "statement": "Annual-mean rain over the campaign region correlates with annual NINO34 at "
                                        "|r| > 0.3 over the record.",
                           "test": "Pearson r of region-mean annual PRECTOTCORR vs NINO34, 1981-2025, p<0.05",
                           "kill": "|r| < 0.3 or p > 0.05", "data": "PRECTOTCORR, NINO34, IMERG"},
                          {"question": "2", "title": "Soil-heat coupling strengthened", "priority": 0.6,
                           "statement": "The correlation between root-zone soil moisture and temperature became more "
                                        "negative after 2003.",
                           "test": "r(GWETROOT, T2M) 1981-2002 vs 2003-2025, Fisher z test p<0.05",
                           "kill": "no significant difference", "data": "GWETROOT, T2M, GLDAS"}]},
                     "Writing the research programme.")

    def _auditor(self, messages):
        return _json({"outcome": "new_signal", "reason": "mock audit keeps the verdict", "novel_claim": ""})

    def _supervisor(self, messages):
        sysm, user = _txt(messages[0]), _txt(messages[-1])
        if "PLAN" in sysm.split("\n")[0] or "Review the worker's PLAN" in sysm:
            return _json({"decision": "approve", "feedback": "Plan is sound."})
        followups = []
        m = re.search(r"Task \((\w+)\): (.*)", user)
        kind = m.group(1) if m else ""
        if kind == "deep_dive" and "literature_search" not in user:
            return _json({"decision": "revise", "soundness": 2, "followups": [],
                          "feedback": "You claim an outcome but never ran the literature novelty test. Run "
                                      "literature_search with precise descriptors and update the verdict."})
        if kind == "deep_dive" and "'outcome': 'new_signal'" in user:
            fam = re.search(r"family ([\w+\-]+:[\w+\-]+:\d+)", user)
            if fam:
                followups.append({"title": f"Follow-up: seasonal anatomy of {fam.group(1)}",
                                  "brief": f"Split the change in {fam.group(1)} by calendar month; identify which "
                                           f"season carries it and whether a linked variable changes in the same months.",
                                  "rationale": "A seasonal fingerprint discriminates physical causes."})
        return _json({"decision": "accept", "soundness": 4, "feedback": "Claims trace to tool outputs.",
                      "followups": followups})

    def _curator(self, messages):
        u = _txt(messages[-1])
        title = re.search(r"Task \(\w+\): (.*)", u)
        summ = u.split("Worker summary:\n", 1)[-1].split("\n\nVerdict:")[0]
        verdict = re.search(r"'outcome': '(\w+)'", u)
        kind = "rejection" if verdict and verdict.group(1) in ("artifact", "not_robust") else "finding"
        return _json({"entries": [{"kind": kind, "title": (title.group(1) if title else "task")[:120],
                                   "body": summ[:700], "tags": ["mock"]}]})

    def _triage(self, messages):
        n = len(re.findall(r"^\[\d+\]", _txt(messages[-1]), re.M))
        return _json({"decisions": [{"index": i, "kind": "science", "value": 2, "decision": "release",
                                     "reason": "specific and testable"} for i in range(n)]})

    def _editor(self, messages):
        body = _txt(messages[-1])
        t = body.strip().splitlines()[0].lstrip("# ") if body.strip() else "Report"
        return _json({"decision": "accept", "feedback": "Checks and limitations are stated.", "title": t})

    def _judge(self, messages):
        u = _txt(messages[-1])
        a, b = u.split("## Report B", 1)
        def score(r):
            new = "new_signal" in r
            return {"impact": 4 if new else 3, "novelty": 4 if new else 2, "soundness": 4,
                    "actionability": 3 + ("GPM" in r or "GRACE" in r)}
        return _json({"A": score(a), "B": score(b), "reason": "mock scoring by outcome"})

    # ------------------------------------------------------------------ worker
    def _worker(self, messages):
        sysm = _txt(messages[0])
        kind = re.search(r"stage \w+, kind (\w+)\)", sysm).group(1)
        meta = json.loads(re.search(r"<task_meta>(.*?)</task_meta>", sysm, re.S).group(1))
        di = re.search(r"# Data already downloaded by other agents \(variable -> file\)\n(.*)\n", sysm)
        data_index = json.loads(di.group(1)) if di and di.group(1).startswith("{") else {}
        brief_box = re.search(r"Region: (\{.*?\})\s+Period: (\d+)-(\d+)", sysm)
        done = []
        for m in messages[1:]:
            if m["role"] == "assistant":
                j = re.search(r"```json\n(.*?)\n```", _txt(m), re.S)
                if j:
                    done.append(json.loads(j.group(1))["action"])
        obs = [_txt(m) for m in messages if m["role"] == "user"]
        last = obs[-1] if obs else ""
        all_obs = "\n".join(obs)
        if "plan" not in done or "revise the plan" in last:
            return _act("plan", "Writing my plan.", steps=self._plan_steps(kind), rationale="Follow the stage brief.")
        return getattr(self, f"_w_{kind}")(meta, done, last, all_obs, data_index, brief_box)

    @staticmethod
    def _plan_steps(kind):
        return {"sweep": ["download region", "run synlib.sweep", "map + look", "look at raw extreme cell",
                          "register sweep", "finish"],
                "classify": ["build families", "look at field maps", "finish"],
                "census": ["apply filters", "inspect top families", "promote", "finish"],
                "deep_dive": ["family info", "family-mean series + per-month change", "look at figure and raw numbers",
                              "artifact checks", "literature novelty test", "verdict", "finish"],
                "followup": ["read parent task", "targeted analysis", "look", "verdict", "finish"],
                "report": ["read source task", "re-create figure", "look", "write report"],
                "science": ["family info", "driver indices", "correlate", "look", "record finding", "finish"],
                "hypothesis": ["board", "index", "test", "look", "literature", "record result", "finish"],
                "referee": ["recompute", "index", "literature", "look", "report", "finish"]}.get(kind, ["work"])

    def _w_sweep(self, meta, done, last, all_obs, di, box):
        v = meta["variable"]
        reg = meta["region"]
        if "power_region" not in done:
            y0, y1 = (int(box.group(2)), int(box.group(3))) if box else (1981, 2025)
            return _act("power_region", f"Downloading {v}.", lat_min=reg["lat_min"], lat_max=reg["lat_max"],
                        lon_min=reg["lon_min"], lon_max=reg["lon_max"], parameters=[v], start_year=y0, end_year=y1)
        path = re.search(r"Saved [\d,]+ rows to (\S+)", all_obs).group(1)
        if "python" not in done:
            code = f'''import synlib
df = synlib.load(r"{path}")
sw = synlib.sweep(df, "{v}")
sw.to_csv("sweep_{v}.csv", index=False)
synlib.plot_field("map_{v}_trend.png", sw, "sen_per_decade", "{v}: Sen slope per decade", q_column="mk_q")
ext = sw.loc[sw["mk_p"].idxmin()]
print("cells", len(sw))
print("EXTREME", ext["lat"], ext["lon"], "sen/decade", round(float(ext["sen_per_decade"]), 4))
for c in ["mk_q", "cp_q", "var_q", "phase_q", "recent_q"]:
    if c in sw: print(c, "cells q<=0.05:", int((sw[c] <= 0.05).sum()))
if "cp_year" in sw:
    sig = sw[sw["cp_q"] <= 0.05]
    if len(sig): print("most common break year among significant cells:", int(sig["cp_year"].mode()[0]), "n=", len(sig))
'''
            return _act("python", "Running the field sweep.", code=code)
        if done.count("look") == 0:
            return _act("look", "Looking at the trend map.", path=f"map_{v}_trend.png")
        if done.count("look") == 1:
            m = re.search(r"EXTREME ([\-\d\.]+) ([\-\d\.]+)", all_obs)
            return _act("look", "Reading the raw numbers at the most significant cell.", path=path, variable=v,
                        lat=float(m.group(1)), lon=float(m.group(2)))
        if "register_sweep" not in done:
            return _act("register_sweep", "Registering.", path=f"sweep_{v}.csv", variable=v)
        brk = re.search(r"most common break year among significant cells: (\d+) n= (\d+)", all_obs)
        if v in ("ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "ALLSKY_KT") and brk and "propose_followup" not in done \
                and abs(int(brk.group(1)) - 2001) <= 1:
            return _act("propose_followup", "Many cells break near 2001, when POWER changes solar source.",
                        title=f"Is the {brk.group(1)} break in {v} a data-source artifact?",
                        brief=f"{brk.group(2)} cells of {v} show a break near {brk.group(1)}. POWER switches from "
                              f"SRB to CERES SYN1deg in 2001. Test whether the step is uniform across months and "
                              f"cells (artifact-like) and compare with CLOUD_AMT and PRECTOTCORR.",
                        rationale="Breaks clustered on a production change are a classic artifact.")
        sig = re.findall(r"(\w+_q) cells q<=0.05: (\d+)", all_obs)
        return _act("finish", "Done.", summary=f"Sweep of {v}: " + ", ".join(f"{a}={b}" for a, b in sig) +
                    (f"; break years cluster at {brk.group(1)}" if brk else "") + ". Looked at the map and the raw "
                    "series of the most significant cell.", figures=[f"map_{v}_trend.png"], confidence=0.8)

    def _w_classify(self, meta, done, last, all_obs, di, box):
        if "build_families" not in done:
            return _act("build_families", "Classifying cells into families.")
        figs = re.findall(r"(\S+field_\S+\.png)", all_obs)
        if "look" not in done and figs:
            return _act("look", "Looking at a field map.", path=figs[0])
        n = re.search(r"(\d+) anomaly families", all_obs)
        return _act("finish", "Done.", summary=f"{n.group(1) if n else 0} families built; see families.csv.",
                    figures=re.findall(r"(\S+field_\S+\.png)", all_obs)[:2], confidence=0.8)

    def _w_census(self, meta, done, last, all_obs, di, box):
        if "apply_filters" not in done:
            return _act("apply_filters", "Applying the locked filters.")
        passing = re.findall(r"^\s*(\S+:\w+[+\-]:\d+)\s+\d+.*\byes\b", all_obs, re.M)
        if passing and "family_info" not in done:
            return _act("family_info", "Inspecting the top passing family.", family=passing[0])
        if "look" not in done:
            return _act("look", "Looking at the families table.", path="families.csv")
        if "promote" not in done and passing:
            # diversity: at most 2 per variable, prefer different classes
            picked, per = [], {}
            for f in passing:
                v = f.split(":")[0]
                if per.get(v, 0) < 2:
                    picked.append(f)
                    per[v] = per.get(v, 0) + 1
            return _act("promote", "Promoting a diverse set.", families=picked[:6],
                        rationale="Largest passing families, at most two per variable.")
        return _act("finish", "Done.", summary=f"{len(passing)} families passed; promoted the most diverse set.",
                    figures=[], confidence=0.75)

    def _family_code(self, fam_info: dict, path: str, tag: str) -> str:
        var, cls = fam_info["variable"], fam_info["class"]
        return f'''import synlib, json, numpy as np
df = synlib.load(r"{path}")
cube = synlib.to_cube(df, "{var}")
cells = {json.dumps(fam_info["cells"])}
idx = [cube.cell(a, b) for a, b in cells]
ann_all = cube.annual()
ann = np.nanmean([ann_all[i, j] for i, j in idx], axis=0)
mon = np.nanmean([cube.data[i, j] for i, j in idx], axis=0)
r = synlib.analyze_series(mon, cube.years)
cp = r.get("cp_year") if "{cls}" == "shift" else None
synlib.plot_series("{tag}.png", cube.years, ann, "{fam_info['family']} family mean", "", cp_year=cp, months=mon)
print("RESULT", json.dumps({{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}))
split = r.get("cp_year") or int(cube.years[len(cube.years) // 2])
b = cube.years < split
per_month = [float(np.nanmean(mon[~b, m]) - np.nanmean(mon[b, m])) for m in range(12)]
print("PER_MONTH_CHANGE", [round(x, 3) for x in per_month])
cv = float(np.std(per_month) / (abs(np.mean(per_month)) + 1e-9))
print("UNIFORMITY_CV", round(cv, 3), "(low = same change in every month, artifact-like)")
print(synlib.series_text(cube.years, ann))
'''

    def _w_deep_dive(self, meta, done, last, all_obs, di, box):
        fam = meta["family"]
        var = meta["variable"]
        info = _family_obs(all_obs)
        if "family_info" not in done or info is None:
            return _act("family_info", "Reading the family record.", family=fam)
        path = di.get(var)
        if "python" not in done:
            return _act("python", "Family-mean series and per-month anatomy of the change.",
                        code=self._family_code(info, path, "family_series"))
        if done.count("look") == 0:
            return _act("look", "Looking at the family series and monthly anomaly strip.", path="family_series.png")
        if done.count("look") == 1:
            return _act("look", "Reading raw numbers at the family centroid.", path=path, variable=var,
                        lat=info["centroid"][0], lon=info["centroid"][1])
        revising = "SUPERVISOR REVIEW" in all_obs
        skip_lit = (sum(map(ord, fam)) % 2 == 0) and not revising
        if "literature_search" not in done and not skip_lit:
            region = "Morocco" if -14 < info["centroid"][1] < 0 and 27 < info["centroid"][0] < 37 else "North Africa"
            words = {"PRECTOTCORR": "precipitation", "GWETROOT": "root zone soil moisture", "T2M": "temperature",
                     "T2M_MAX": "maximum temperature", "ALLSKY_SFC_SW_DWN": "surface solar radiation",
                     "CLOUD_AMT": "cloud cover", "SNODP": "snow depth", "GWETTOP": "soil moisture"}
            q = f"{words.get(var, var)} {info['class'].replace('_', ' ')} {region}"
            return _act("literature_search", "Novelty kill-test.", query=q, n=6)
        cv = re.search(r"UNIFORMITY_CV ([\d\.]+)", all_obs)
        flag = (info.get("filters") or {}).get("artifact_flag")
        lit_ran = "Literature for:" in all_obs
        if "set_verdict" not in done or (revising and done.count("set_verdict") < 2 and "literature_search" in done):
            if flag and (cv and float(cv.group(1)) < 0.6 or info["class"] == "shift"):
                outcome, note = "artifact", f"{flag}; change near-uniform across months (CV {cv.group(1) if cv else '?'})."
            elif var.startswith("T2M") and info["class"] in ("trend", "recent_extreme") and info["effect"] > 0:
                outcome, note = "known_phenomenon", "Warming consistent with documented regional warming and recent heat records."
            else:
                # The scripted demo never claims a discovery: novelty needs real judgement (the live model).
                outcome, note = "not_robust", (f"Candidate: {info['class']} passed the filters (per-month CV "
                                               f"{cv.group(1) if cv else '?'}; literature {'searched' if lit_ran else 'not searched'}). "
                                               f"The scripted demo cannot judge novelty; run with Nemotron.")
            return _act("set_verdict", "Recording verdict.", family=fam, outcome=outcome, note=note)
        res = re.search(r"RESULT (\{.*\})", all_obs)
        vnote = re.findall(r"Verdict recorded for \S+: (\w+)", all_obs)[-1]
        summary = (f"Family {fam} ({info['class']} in {var}, {info['n_cells']} cells near "
                   f"{info['centroid'][0]:.2f},{info['centroid'][1]:.2f}). Verdict: **{vnote}**.\n\n"
                   f"Key stats: {res.group(1)[:600] if res else ''}\n\n"
                   f"Checks run: per-month uniformity (CV {cv.group(1) if cv else '?'}), source-transition flag "
                   f"({flag or 'none'}), literature search ({'yes' if lit_ran else 'NOT run'}). Looked at the family "
                   f"series figure and the raw centroid table.")
        return _act("finish", "Done.", summary=summary, figures=["family_series.png"], confidence=0.6)

    def _w_followup(self, meta, done, last, all_obs, di, box):
        fam = meta.get("family")
        info = _family_obs(all_obs) if fam else None
        if fam and ("family_info" not in done or info is None):
            return _act("family_info", "Context.", family=fam)
        if not fam:
            if "kb_search" not in done:
                return _act("kb_search", "Context from the knowledge base.", query="break source 2001")
            if "look" not in done:
                return _act("look", "Inspect sweep table.", path=next(iter(di.values()), "plan.json") if di else "plan.json")
            return _act("finish", "Done.", summary="Follow-up reviewed the knowledge base and the data; see the "
                        "deep dives for verdicts.", figures=[], confidence=0.5)
        path = di.get(info["variable"])
        if "python" not in done:
            return _act("python", "Seasonal anatomy.", code=self._family_code(info, path, "followup_series"))
        if "look" not in done:
            return _act("look", "Looking.", path="followup_series.png")
        if "set_verdict" not in done:
            return _act("set_verdict", "Keeping the deep-dive verdict, adding seasonal detail.", family=fam,
                        outcome=info.get("current_outcome") or "not_robust",
                        note=(info.get("note") or "") + " | follow-up: seasonal split computed.")
        pm = re.search(r"PER_MONTH_CHANGE (\[.*?\])", all_obs)
        return _act("finish", "Done.", summary=f"Seasonal anatomy of {fam}: per-month change {pm.group(1) if pm else '?'}.",
                    figures=["followup_series.png"], confidence=0.6)

    def _w_science(self, meta, done, last, all_obs, di, box):
        fam = meta["family"]
        info = _family_obs(all_obs)
        if "family_info" not in done or info is None:
            return _act("family_info", "Reading the signal.", family=fam)
        if "climate_index" not in done:
            return _act("climate_index", "Candidate drivers.", names=["NINO34", "DMI"], start_year=1981, end_year=2025)
        idx = re.search(r"Saved [\d,]+ index-months to (\S+\.csv)", all_obs)
        path = di.get(info["variable"])
        if "python" not in done:
            code = self._family_code(info, path, "science_series") + f'''
import pandas as pd
ix = pd.read_csv(r"data/{idx.group(1) if idx else ""}")
for name, g in ix.groupby("index"):
    a = g.groupby("year").value.mean().reindex(cube.years)
    ok = ~(np.isnan(ann) | np.isnan(a.values))
    r = np.corrcoef(ann[ok], a.values[ok])[0, 1] if ok.sum() > 10 else float("nan")
    print("CORR", name, round(float(r), 3), "n", int(ok.sum()))
'''
            return _act("python", "Correlating the signal with ENSO and the Indian Ocean Dipole.", code=code)
        if "look" not in done:
            return _act("look", "Looking at the signal series.", path="science_series.png")
        corr = re.findall(r"CORR (\w+) ([\-\d\.nan]+) n (\d+)", all_obs)
        if "record_finding" not in done:
            best = max(corr, key=lambda c: abs(float(c[1])) if c[1] != "nan" else 0) if corr else ("none", "0", "0")
            stmt = (f"Annual {info['variable']} in {fam} correlates with {best[0]} at r = {best[1]} over {best[2]} "
                    f"years (scripted demo; the live model interprets drivers).")
            return _act("record_finding", "Recording the driver test.", family=fam, statement=stmt,
                        evidence="Family-mean annual series vs annual-mean NINO34 and DMI over the full record: " +
                                 (", ".join(f"{a} r={b} (n={c})" for a, b, c in corr) or "correlation script failed, "
                                  "so no driver could be tested in this run"), confidence=0.3,
                        novelty="Not assessed in the scripted demo.")
        return _act("finish", "Done.", summary=f"Driver test for {fam}: " + ", ".join(f"{a} r={b}" for a, b, c in corr),
                    figures=["science_series.png"], confidence=0.4)

    def _w_hypothesis(self, meta, done, last, all_obs, di, box):
        hid = meta.get("hypothesis")
        if "hypothesis_board" not in done:
            return _act("hypothesis_board", "Reading the programme.")
        if "climate_index" not in done:
            return _act("climate_index", "Driver index.", names=["NINO34"], start_year=1981, end_year=2025)
        path = next(iter(di.values()), None)
        idx = re.search(r"Saved [\d,]+ index-months to (\S+\.csv)", all_obs)
        if "python" not in done:
            code = f'''
import numpy as np, pandas as pd, synlib
df = synlib.load(r"{path}")
ann = df.groupby("year").value.mean()
ix = pd.read_csv(r"data/{idx.group(1) if idx else ""}")
a = ix.groupby("year").value.mean().reindex(ann.index)
ok = ~(np.isnan(ann.values) | np.isnan(a.values))
r = np.corrcoef(ann.values[ok], a.values[ok])[0, 1] if ok.sum() > 10 else float("nan")
print("CORR", round(float(r), 3), int(ok.sum()))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.plot(ann.index, ann.values); plt.savefig("hyp.png")
'''
            return _act("python", "Running the pre-registered test.", code=code)
        if "look" not in done:
            return _act("look", "Looking at the series.", path="hyp.png")
        if "literature_search" not in done:
            return _act("literature_search", "Literature check.", query="ENSO rainfall teleconnection", n=5)
        corr = re.search(r"CORR ([\-\d\.nan]+) (\d+)", all_obs)
        if "record_hypothesis_result" not in done:
            return _act("record_hypothesis_result", "Closing the test.", hypothesis_id=hid, status="inconclusive",
                        result=f"Scripted demo test: region-mean annual series vs NINO34 gave r = "
                               f"{corr.group(1) if corr else 'nan'}; the mock does not interpret it.",
                        evidence=f"Pearson correlation of annual means over {corr.group(2) if corr else '?'} years "
                                 "(scripted demo; no independent-data check was run, so the status is inconclusive).",
                        confidence=0.2)
        return _act("finish", "Done.", summary=f"Hypothesis {hid}: inconclusive (scripted demo).",
                    figures=["hyp.png"], confidence=0.2)

    def _w_referee(self, meta, done, last, all_obs, di, box):
        target = meta.get("target")
        path = next(iter(di.values()), None)
        if "python" not in done:
            return _act("python", "Recomputing.", code=f'import synlib\ndf = synlib.load(r"{path}")\nprint(len(df))\n'
                        'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt\n'
                        'plt.plot([0, 1]); plt.savefig("ref.png")')
        if "climate_index" not in done:
            return _act("climate_index", "Driver.", names=["NINO34"], start_year=1981, end_year=2025)
        if "literature_search" not in done:
            return _act("literature_search", "Literature.", query="ENSO teleconnection change", n=5)
        if "look" not in done:
            return _act("look", "Look.", path="ref.png")
        if "referee_report" not in done:
            return _act("referee_report", "Deciding.", target=target, decision="downgraded",
                        objections="Scripted referee: the claim was not reproduced with an independent dataset.",
                        checks_run="Recomputed the series from the campaign table (python), loaded NINO34 "
                                   "(climate_index), searched the literature; no independent-data check in the demo.")
        return _act("finish", "Done.", summary=f"Referee report on {target}: downgraded (scripted demo).",
                    figures=["ref.png"], confidence=0.3)

    def _w_report(self, meta, done, last, all_obs, di, box):
        if "read_task" not in done:
            return _act("read_task", "Reading the deep dive.", task_id=meta["source_task"])
        info = _family_obs(all_obs)
        if "family_info" not in done or info is None:
            return _act("family_info", "Family record.", family=meta["family"])
        path = di.get(info["variable"])
        if "python" not in done:
            return _act("python", "Re-creating the key figure.", code=self._family_code(info, path, "report_fig"))
        if "look" not in done:
            return _act("look", "Checking the figure.", path="report_fig.png")
        src = re.search(r"Summary: (.*?)(?:\nObservation:|\Z)", all_obs, re.S)
        src_txt = src.group(1).split("Key stats:")[0].strip()[:600] if src else ""
        checks = re.search(r"(Checks run:.*?)(?:\n|$)", all_obs)
        src_txt += ("\n\n" + checks.group(1)) if checks else ""
        res = re.search(r"RESULT (\{.*\})", all_obs)
        title = f"{info['class'].replace('_', ' ').title()} in {info['variable']} near {info['centroid'][0]:.1f}N {abs(info['centroid'][1]):.1f}W"
        body = f"""# {title}

**Summary.** The campaign found a {info['class'].replace('_', ' ')} in {info['variable']} over {info['n_cells']} grid
cells. Deep-dive verdict: {meta['outcome']}. This is a candidate for expert review, produced in MOCK mode.

## What was observed
Family {meta['family']}; centroid {info['centroid']}; onset {info['onset']}; effect {info['effect']:.2f};
min q {(info['q'] or float('nan')):.2g}.

![family series](report_fig.png)

## How the agents found it
Sweep of {info['variable']}, then family {meta['family']}, which passed the pre-registered filters
(corroboration {json.dumps((info['filters'] or {}).get('corroboration_detail'))}), then deep dive {meta['source_task']}.

## Checks that tried to kill it
{src_txt}

## Interpretation and hypotheses
To be written by the live model; the mock does not interpret.

## What NASA scientists should do next
Confirm with GPM IMERG (precipitation), GRACE-FO (water storage) and MODIS NDVI over the same cells.

## Limitations
Reanalysis-based; mock-mode judgements are rule-based placeholders.
"""
        return _act("finish", "Report written.", summary=body, figures=["report_fig.png"], confidence=0.6)


# ---------------------------------------------------------------------------- benchmark baseline
BASELINE_CODE = r"""
import json, numpy as np, synlib
df = synlib.load("data.csv"); var = df.variable.iloc[0]; cube = synlib.to_cube(df, var)
nl, nn, ny, _ = cube.data.shape
flat = cube.data.reshape(nl, nn, -1)
best = (0, None, None)
for i in range(nl):
    for j in range(nn):
        x = flat[i, j]; run = 1
        for t in range(1, len(x)):
            run = run + 1 if x[t] == x[t - 1] else 1
            if run > best[0]: best = (run, (i, j), t - run + 1)
out = {"anomaly_found": False, "type": "none", "onset_year": None, "lat": None, "lon": None, "months": None,
       "is_artifact": False, "confidence": 0.3}
if best[0] >= 6:
    (i, j), t0 = best[1], best[2]
    out.update(anomaly_found=True, type="constant_run", onset_year=int(cube.years[t0 // 12]), lat=float(cube.lats[i]),
               lon=float(cube.lons[j]), is_artifact=True, confidence=0.9)
else:
    sw = synlib.sweep(df, var)
    frac = float((sw["cp_q"] <= 0.05).mean())
    if frac >= 0.6:
        yr = int(sw.loc[sw["cp_q"] <= 0.05, "cp_year"].mode()[0])
        b = cube.years < yr
        pm = np.nanmean(cube.data[:, :, ~b, :], axis=2) - np.nanmean(cube.data[:, :, b, :], axis=2)
        out.update(anomaly_found=True, type="artifact_step", onset_year=yr, lat=float(cube.lats.mean()),
                   lon=float(cube.lons.mean()), is_artifact=True, confidence=0.8)
    else:
        cands = []
        for i in range(nl):
            for j in range(nn):
                for m in range(12):
                    pt = synlib.pettitt(cube.data[i, j, :, m])
                    if pt["cp"] is not None:
                        cands.append((pt["p"] * nl * nn * 12, "seasonal_step", i, j, int(cube.years[pt["cp"] + 1]), m + 1))
        r = sw.sort_values("var_p").iloc[0]
        cands.append((r["var_p"] * len(sw), "variance", *cube.cell(r["lat"], r["lon"]), int(r.get("cp_year", 0) or 0), None))
        r = sw.sort_values("phase_p").iloc[0]
        cands.append((r["phase_p"] * len(sw), "phase_shift", *cube.cell(r["lat"], r["lon"]), None, None))
        p, kind, i, j, yr, m = min(cands, key=lambda c: c[0])
        if p < 0.05:
            months = None
            if kind == "seasonal_step":
                ps = [synlib.pettitt(cube.data[i, j, :, mm]) for mm in range(12)]
                months = [mm + 1 for mm in range(12) if ps[mm]["p"] < 0.01]
            out.update(anomaly_found=True, type=kind, onset_year=yr, lat=float(cube.lats[i]), lon=float(cube.lons[j]),
                       months=months, is_artifact=False, confidence=0.6)
print("RESULT", json.dumps(out))
"""


def _bench(self, messages):
    sysm = _txt(messages[0])
    lvl = re.search(r"<bench_level>(\d)</bench_level>", sysm)
    if not lvl:  # levels 1-2: no tools. L1 has no usable data; L2 runs the baseline on the two cells shown.
        u = _txt(messages[-1])
        tables = re.findall(r"Cell lat ([\-\d\.]+), lon ([\-\d\.]+) \(year x month\):\n(.*?)(?=\n\nCell|\n</data>)", u, re.S)
        if not tables:
            return _json({"anomaly_found": False, "type": "none", "is_artifact": False, "summary": "No usable data."})
        import numpy as np
        import synlib
        best = None
        for la, lo, tab in tables:
            rows = [l.split() for l in tab.strip().splitlines()[2:]]
            years = np.array([int(r[0]) for r in rows])
            data = np.array([[float(v) for v in r[1:13]] for r in rows])
            for m in range(12):
                pt = synlib.pettitt(data[:, m])
                if pt["cp"] is not None and (best is None or pt["p"] < best[0]):
                    best = (pt["p"], float(la), float(lo), int(years[pt["cp"] + 1]), m + 1)
        if best and best[0] * 24 < 0.05:
            return _json({"anomaly_found": True, "type": "seasonal_step", "onset_year": best[3], "lat": best[1],
                          "lon": best[2], "months": [best[4]], "is_artifact": False, "summary": "baseline"})
        return _json({"anomaly_found": False, "type": "none", "is_artifact": False, "summary": "baseline: nothing"})
    level = int(lvl.group(1))
    done = [json.loads(j.group(1))["action"] for m in messages[1:] if m["role"] == "assistant"
            for j in [re.search(r"```json\n(.*?)\n```", _txt(m), re.S)] if j]
    obs = "\n".join(_txt(m) for m in messages if m["role"] == "user")
    if "python" not in done:
        return _act("python", "Running the statistical baseline.", code=BASELINE_CODE)
    res = re.search(r"RESULT (\{.*\})", obs)
    out = json.loads(res.group(1)) if res else {"anomaly_found": False}
    if "look" not in done and out.get("lat") is not None:
        return _act("look", "Reading the raw numbers at the candidate cell.", path="data.csv", lat=out["lat"], lon=out["lon"])
    out["summary"] = "Rule-based statistical baseline (mock)."
    return _act("finish", "Submitting.", **out)


MockLLM._bench = _bench
