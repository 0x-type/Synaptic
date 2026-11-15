"""The campaign orchestrator (paper Fig. 1A): research brief -> planner -> staged tasks run by
worker/supervisor pairs -> curator -> follow-up triage -> reports -> editor -> tournament."""
from __future__ import annotations

import json
import re
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import yaml

from . import prompts
from .protocol import _loads
from .agents import Runner, WorkerSession
from .config import HARNESS, RUNS_DIR
from .llm import make_llm
from .record import Record
from .tools.science import DEFAULT_THRESHOLDS
from .tournament import run_tournament

STAGES = [("plan", "Plan & pre-register"), ("screen", "Data screen"), ("learn", "GPU model"),
          ("program", "Research programme (PI)"), ("sweep", "Sweep"), ("classify", "Classify"),
          ("census", "Census & evidence"), ("deep_dive", "Deep dives"), ("science", "Science"),
          ("review", "Referees & audit"), ("report", "Reports"), ("rank", "Tournament")]
PARALLEL_STAGE = "program"  # hypothesis tests and referee reports run beside every other stage
# v4 focused campaigns: one open problem, a literature-first research programme, no survey of every anomaly
FOCUSED_STAGES = [("plan", "Plan & pre-register"), ("screen", "Data & cross-product screen"), ("learn", "GPU model"),
                  ("program", "Research programme (PI)"), ("review", "Referees & audit"), ("report", "Reports"),
                  ("rank", "Tournament")]

PIPELINE_KINDS = ("sweep", "classify", "census")  # mission steps whose outputs the next stage needs
VALID_POWER = {"T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "T2MDEW", "T2MWET", "TS", "RH2M", "QV2M",
               "PRECTOTCORR", "PS", "WS2M", "WS10M", "WS50M", "WD2M", "WD10M", "ALLSKY_SFC_SW_DWN",
               "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "ALLSKY_KT", "CLOUD_AMT", "GWETTOP", "GWETROOT",
               "GWETPROF", "SNODP", "EVPTRNS", "FROST_DAYS", "EVLAND"}


THRESHOLD_RANGES = {"q_max": (0.001, 0.1), "p_form": (0.01, 0.1), "min_cells": (2, 50), "min_effect": (0.5, 3), "min_phase_days": (5, 60),
                    "min_log_var": (0.2, 1.5), "min_recent_z": (2, 5), "min_corroboration": (0, 1),
                    "transition_window": (1, 5), "max_promoted": (1, 12)}


def load_brief(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return yaml.safe_load(Path(path_or_dict).read_text())


class Campaign:
    def __init__(self, brief, mock: bool = False, db: str | Path | None = None, name: str | None = None):
        self.brief = load_brief(brief)
        self.mock = mock
        self.db_path = Path(db) if db else RUNS_DIR / "synaptic.db"
        self.record = Record(self.db_path)
        self.llm = make_llm(mock)
        title = name or self.brief.get("title", "campaign")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.cdir = RUNS_DIR / f"{stamp}-{''.join(ch if ch.isalnum() else '-' for ch in title.lower())[:40]}"
        self.cdir.mkdir(parents=True, exist_ok=True)
        self.cid = self.record.create_campaign(title, self.brief, "mock" if mock else "live", str(self.cdir))
        self.runner = Runner(self.record, self.llm, self.cid, self.cdir, mock)
        self.record.set_stages(self.cid, FOCUSED_STAGES if self.focused else STAGES)
        self.plan: dict = {}
        self.data_index: dict = {}
        self.lock = threading.Lock()

    @property
    def focused(self) -> bool:
        return (self.brief or {}).get("mode") == "focused"

    @property
    def holdout(self) -> set:
        return set((self.brief or {}).get("holdout") or [])

    @property
    def confirm_sets(self) -> set:
        """Datasets confirmations may use: the held-out ones plus the time-held-out ones (explorers saw only their
        early years)."""
        return self.holdout | set(((self.brief or {}).get("time_holdout") or {}).get("datasets") or [])

    @property
    def campaign(self) -> dict:
        return {"id": self.cid, "brief": self.brief}

    def log(self, msg: str):
        self.record.event(self.cid, None, None, "harness", "log", msg)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def stopped(self) -> bool:
        return (self.cdir / "STOP").exists()

    # ---------------------------------------------------------------------- stage 0
    def stage_plan(self):
        self.record.set_stage_status(self.cid, "plan", "running")
        if self.focused:  # the datasets come from the brief (catalogue ids); thresholds are the locked defaults
            from . import catalog

            ds = [d for d in self.brief.get("datasets", []) if d in catalog.BY_ID and d not in self.holdout]
            th = {k: tuple(v) for k, v in DEFAULT_THRESHOLDS.items()}
            th["max_promoted"] = (HARNESS.max_promoted, "deep-dive budget")
            self.record.lock_thresholds(self.cid, th)
            self.plan = {"variables": ds, "holdout": sorted(self.holdout), "mode": "focused"}
            (self.cdir / "plan.json").write_text(json.dumps(self.plan, indent=1))
            self.record.kb_add(self.cid, None, "method", "Campaign plan (focused)",
                               json.dumps({"datasets": ds, "held_out_for_confirmation": sorted(self.holdout)}), ["plan"])
            self.log(f"Plan (focused): datasets {ds}; held out for confirmation {sorted(self.holdout)}")
            self.record.set_stage_status(self.cid, "plan", "done")
            return
        plan = self.runner.plan_campaign(self.brief) or {}
        variables = [v.upper() for v in (plan.get("variables") or self.brief.get("variables") or ["T2M", "PRECTOTCORR"])]
        variables = [v for v in dict.fromkeys(variables) if v in VALID_POWER][: int(self.brief.get("max_variables", 6))]
        for v in self.brief.get("variables", []):  # the brief's own variables are never dropped
            if v.upper() in VALID_POWER and v.upper() not in variables:
                variables.insert(0, v.upper())
        if self.brief["region"].get("global"):
            from .globaldata import VARIABLES as GV

            variables = [v for v in variables if v in GV] or list(GV)
            for v in GV:  # a global campaign always carries the whole water-heat set
                if v not in variables:
                    variables.append(v)
        plan["variables"] = variables
        th = {k: tuple(v) for k, v in DEFAULT_THRESHOLDS.items()}
        th["max_promoted"] = (HARNESS.max_promoted, "deep-dive budget")
        for k, v in (plan.get("thresholds") or {}).items():
            if k in th:
                try:
                    val, why = (v[0], v[1]) if isinstance(v, (list, tuple)) else (float(v), "planner")
                    lo, hi = THRESHOLD_RANGES[k]
                    clamped = min(max(float(val), lo), hi)
                    if k == "max_promoted":
                        clamped = min(clamped, HARNESS.max_promoted)
                    note = f" (planner proposed {val}; clamped to the allowed range {lo}-{hi})" if clamped != float(val) else ""
                    th[k] = (clamped, str(why) + note)
                except Exception:  # noqa: BLE001
                    pass
        self.record.lock_thresholds(self.cid, th)
        self.plan = plan
        (self.cdir / "plan.json").write_text(json.dumps(plan, indent=1))
        self.record.kb_add(self.cid, None, "method", "Campaign plan and pre-registered thresholds",
                           json.dumps({"variables": variables, "thresholds": {k: v[0] for k, v in th.items()},
                                       "notes": plan.get("notes", "")}), ["plan"])
        self.log(f"Plan: variables={variables}; thresholds locked ({len(th)})")
        self.record.set_stage_status(self.cid, "plan", "done")

    # ---------------------------------------------------------------------- stage 1: automatic data screen
    def stage_screen(self):
        """Deterministic quality check before any agent runs (minutes, not agent-hours): the trust map."""
        import pandas as pd

        from . import screen
        from .tools import nasa
        from .tools.base import ToolContext

        self.record.set_stage_status(self.cid, "screen", "running")
        if self.focused:
            self._focused_data()
            self.record.set_stage_status(self.cid, "screen", "done")
            return
        b, reg = self.brief, self.brief["region"]
        ctx = ToolContext(self.record, self.cid, None, None, self.cdir / "screen", self.cdir, self.mock)
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        data = {}
        if reg.get("global"):
            from . import globaldata

            info = globaldata.build(b["start_year"], b["end_year"], log=self.log)
            if info["missing"]:
                self.log(f"global data: months missing in the archive: {info['missing'][:12]}")
            paths = globaldata.write_tables(b["start_year"], b["end_year"], self.cdir / "data")
            for v in self.plan["variables"]:
                if v in paths:
                    df = pd.read_csv(paths[v])
                    data[v] = df
                    with self.lock:
                        self.data_index[v] = str(paths[v])
            (self.cdir / "data_index.json").write_text(json.dumps(self.data_index, indent=1))
            self.log(f"global data: {len(data)} variables, {next(iter(data.values()))[['lat', 'lon']].drop_duplicates().shape[0]} "
                     f"land cells at 2 x 2.5 degrees, {b['start_year']}-{b['end_year']}")
        for v in ([] if reg.get("global") else self.plan["variables"]):
            res = nasa.power_region(ctx, reg["lat_min"], reg["lat_max"], reg["lon_min"], reg["lon_max"], [v],
                                    b["start_year"], b["end_year"])
            if res.is_error or not res.files:
                self.log(f"screen: could not download {v}: {res.text[:160]}")
                continue
            df = pd.read_csv(res.files[0])
            data[v] = df[df["variable"] == v]
            with self.lock:
                self.data_index[v] = res.files[0]
                (self.cdir / "data_index.json").write_text(json.dumps(self.data_index, indent=1))
        self._screen_data = data
        t0 = time.time()
        trust = screen.run_screen(data, reg, log=self.log) if data else pd.DataFrame()
        trust.to_csv(self.cdir / "trust.csv", index=False)
        (self.cdir / "figures").mkdir(exist_ok=True)
        if not trust.empty:
            screen.plot_trust(trust, self.cdir / "figures" / "trust_map.png", reg)
            real = trust[trust.kind != "known_caveat"]
            by = real.groupby(["variable", "kind"]).size().to_string()
            cells = real.groupby("variable").apply(lambda g: g[["lat", "lon"]].drop_duplicates().shape[0]).to_dict()
            brk = real[real.kind == "break"].groupby(["variable", "year_from"]).size().sort_values(ascending=False).head(6)
            body = (f"Automatic screen of NASA POWER against GPCP rain, Berkeley Earth temperature, gap-fill and "
                    f"variability checks and the known-issues registry ({time.time() - t0:.0f} s). Untrusted cells per "
                    f"variable: {cells}.\nIssues by kind:\n{by}\nMost common break years (variable, year: cells):\n"
                    f"{brk.to_string()}\nAnomaly families inside untrusted data are labelled 'artifact (screened)' "
                    f"automatically and are NOT promoted. Do not spend tasks re-proving these issues; build science on "
                    f"trusted cells and say when a signal touches untrusted ones. Table: trust.csv.")
            self.record.kb_add(self.cid, None, "method", "Automatic data screen (trust map)", body, ["screen", "data"])
            for _, r in trust[trust.kind.str.startswith("known")].drop_duplicates("detail").iterrows():
                self.record.kb_add(self.cid, None, "rejection", f"Known data issue: {r['detail'][:80]}", r["detail"],
                                   ["screen", "known_issue", str(r["variable"])])
            self.log(f"screen: {len(real)} issues in {sum(cells.values())} variable-cells ({time.time() - t0:.0f} s)")
        if data:  # trusted copies: what agents analyse by default (untrusted values removed, longer segment kept)
            for v, df in data.items():
                try:
                    screen.apply_trust(df, trust, v).to_csv(self.cdir / "data" / f"trusted_{v}.csv", index=False)
                except Exception as e:  # noqa: BLE001
                    self.log(f"screen: trusted table for {v} skipped ({e})")
        self.record.set_stage_status(self.cid, "screen", "done")

    def _focused_data(self):
        """Fetch the campaign datasets from NASA Earthdata Cloud onto one grid and screen them against each other:
        two products of the same quantity whose DIFFERENCE breaks in some year point to a production change, not
        to the climate. Held-out datasets are never fetched here (they are for the confirmation step only)."""
        import pandas as pd

        from . import cloud
        from . import screen as scr

        b, reg = self.brief, self.brief["region"]
        bbox = (reg["lat_min"], reg["lat_max"], reg["lon_min"], reg["lon_max"])
        res = float(b.get("resolution", 2.0))
        (self.cdir / "data").mkdir(exist_ok=True)
        data = {}
        for v in self.plan["variables"]:
            out = self.cdir / "data" / f"{v}.csv"
            try:
                if not out.exists():
                    th = b.get("time_holdout") or {}
                    y_end = min(b["end_year"], int(th["explore_end"])) if v in (th.get("datasets") or []) else b["end_year"]
                    tab = cloud.table(v, bbox, b["start_year"], y_end, res, progress=self.log)
                    from . import catalog as _cat
                    if _cat.get(v).domain == "land" and v.startswith("GRACE"):
                        # GRACE mascons carry ocean values: keep land cells only (found live in the Track B sweep)
                        from .regions import land_mask
                        cells = tab[["lat", "lon"]].drop_duplicates()
                        la, lo = sorted(cells.lat.unique()), sorted(cells.lon.unique())
                        m = land_mask(np.array(la), np.array(lo))
                        keep = {(a, o) for i, a in enumerate(la) for j, o in enumerate(lo) if m[i, j] > 0}
                        tab = tab[[(a, o) in keep for a, o in zip(tab.lat, tab.lon)]]
                    tab.to_csv(out, index=False)
                data[v] = pd.read_csv(out)
                with self.lock:
                    self.data_index[v] = str(out)
                self.log(f"data: {v} {len(data[v])} rows ({data[v][['lat', 'lon']].drop_duplicates().shape[0]} cells)")
            except Exception as e:  # noqa: BLE001
                self.log(f"data: {v} unavailable ({type(e).__name__}: {str(e)[:200]})")
        (self.cdir / "data_index.json").write_text(json.dumps(self.data_index, indent=1))
        self._screen_data = data
        lines = []
        for a, c in b.get("screen_pairs", []):
            if a in data and c in data and a not in self.holdout and c not in self.holdout:
                try:
                    r = scr.pair_breaks(data[a], data[c])
                    lines.append(f"{a} vs {c}: {r}")
                except Exception as e:  # noqa: BLE001
                    lines.append(f"{a} vs {c}: not compared ({e})")
        known = scr.satellite_known_issues(list(data))
        body = ("Cross-product screen (difference series of two products of the same quantity; a break means a "
                "production change in one of them):\n" + ("\n".join(lines) or "(no product pairs to compare)")
                + "\n\nKnown data issues for these products:\n" + "\n".join(f"- {k}" for k in known))
        self.record.kb_add(self.cid, None, "method", "Automatic data screen (trust map)", body, ["screen", "data"])
        self.log("screen: " + ("; ".join(lines)[:300] if lines else "no product pairs") + f"; {len(known)} known issues")

    # ---------------------------------------------------------------------- stage 2: GPU model (background)
    def _load_campaign_data(self) -> dict:
        import pandas as pd

        data = getattr(self, "_screen_data", None) or {}
        if not data:
            for v in self.plan.get("variables", []):
                f = Path(str(self.data_index.get(v, "")))
                if f.is_file():
                    df = pd.read_csv(f)
                    data[v] = df[df["variable"] == v]
        return data

    def start_learn(self):
        """The GPU model trains while the agents sweep; science tasks wait for it."""
        self._learn_result = None

        def work():
            from .learn import run_learn

            self.record.set_stage_status(self.cid, "learn", "running")
            try:
                data = self._load_campaign_data()
                s = run_learn(self, data) if data else None
            except Exception as e:  # noqa: BLE001
                self.log(f"learn: failed ({type(e).__name__}: {str(e)[:200]})")
                s = None
            self._learn_result = s
            if s:
                (self.cdir / "learn" / "summary.json").write_text(json.dumps(s, indent=1))
                lines = [f"{c['driver']} -> {c['target']}: {c['cells']} cells, influence {c['early']:+.3f} -> "
                         f"{c['late']:+.3f} (stronger in {c['frac_stronger']:.0%}, weaker in {c['frac_weaker']:.0%}); "
                         f"95% CI of the change {c.get('ci95')}, {'significant' if c.get('significant') else 'NOT significant'}"
                         + ("" if c.get("survives_sample_control") is None else
                            f"; after controlling for event strength {c.get('change_ctrl_mean')} -> "
                            f"{'survives' if c.get('survives_sample_control') else 'DOES NOT survive (sample effect)'}")
                         + (f"; strongest 10-degree blocks {c.get('top_blocks')[:3]}" if c.get("top_blocks") else "")
                         for c in s.get("top_coupling_changes", [])[:8]]
                body = (f"GPU coupling model v2 ({s.get('backend')}, {s.get('minutes')} min, {s.get('folds')}-fold "
                        f"cross-fitted): networks learned every trusted cell's month-to-month behaviour from all variables "
                        f"and drivers {s.get('drivers')}; every year is scored out of sample. Forecast skill (early, late) "
                        f"by variable: {s.get('skill_mean')}. Influence of each input = loss of skill when it is "
                        f"scrambled, per year; change = late (from {s.get('split_year')}) minus early, with a year-bootstrap "
                        f"CI and a control for how strong the driver's events were in each period. Largest coherent "
                        f"changes:\n" + "\n".join(lines) +
                        "\nFiles: learn/importance.csv (per cell: early, late, change, change_ctrl, thirds), "
                        "learn/blocks.csv (10-degree blocks), learn/skill.csv, learn/surprise_top.csv, "
                        "learn/coupling_change.png (maps + 15-year window series).")
                self.record.kb_add(self.cid, None, "finding", "GPU coupling model: changing drivers", body,
                                   ["learn", "gpu", "coupling"])
                self.log("learn: done; " + (lines[0] if lines else "no coherent coupling change"))
            self.record.set_stage_status(self.cid, "learn", "done")

        self._learn_thread = threading.Thread(target=work, daemon=True)
        self._learn_thread.start()

    def _coupling_leads(self, limit: int = 2) -> list[dict]:
        """Turn the GPU model's largest coherent coupling changes into candidates for science tasks."""
        s = getattr(self, "_learn_result", None)
        if not s and (self.cdir / "learn" / "summary.json").exists():
            s = json.loads((self.cdir / "learn" / "summary.json").read_text())
        if not s:
            return []
        import pandas as pd

        imp_f = self.cdir / "learn" / "importance.csv"
        imp = pd.read_csv(imp_f) if imp_f.exists() else None
        leads = []
        for c in s.get("top_coupling_changes", []):
            frac = max(c["frac_stronger"], c["frac_weaker"])
            if c["cells"] < 30 or frac < 0.3 or c["driver"] == f"hist:{c['target']}":
                continue
            if c.get("significant") is False or c.get("survives_sample_control") is False:
                continue  # v2 model: not significant, or only a sample effect (more strong events in one period)
            fam = f"COUPLING:{c['driver']}->{c['target']}"
            sign = 1 if c["frac_stronger"] >= c["frac_weaker"] else -1
            cells = []
            if imp is not None:
                g = imp[(imp.target == c["target"]) & (imp.driver == c["driver"]) & (imp.change * sign > 0.02)]
                cells = [[float(a), float(b)] for a, b in zip(g.lat, g.lon)]
            la = float(np.mean([x[0] for x in cells])) if cells else 0.0
            lo = float(np.mean([x[1] for x in cells])) if cells else 0.0
            self.record.upsert_candidate(self.cid, {
                "id": fam, "task_id": None, "family": fam, "variable": c["target"], "cls": "coupling", "lat": la,
                "lon": lo, "n_cells": len(cells) or c["cells"], "onset": s.get("split_year"), "effect": sign * frac,
                "qvalue": None, "stats": {**c, "cells": json.dumps(cells[:2000])}, "filters": {}, "passed": 1,
                "promoted": 0, "outcome": None,
                "note": f"GPU coupling model: influence of {c['driver']} on {c['target']} "
                        f"{'strengthened' if sign > 0 else 'weakened'} after {s.get('split_year')} in {frac:.0%} of "
                        f"{c['cells']} cells ({c['early']:+.3f} -> {c['late']:+.3f})."})
            leads.append(self.record.candidates(self.cid, id=fam)[0])
            if len(leads) >= limit:
                break
        return leads

    # ---------------------------------------------------------------------- task execution
    def process(self, task_id: str):
        task = self.record.task(task_id)
        self.record.update_task(task_id, status="running", started=time.time())
        self.log(f"start {task['kind']} {task_id}: {task['title']}")
        ws = None
        try:
            ws = WorkerSession(self.runner, self.campaign, task, dict(self.data_index))
            result = ws.run()
            review = {"decision": "reject", "feedback": result.get("summary", "")}
            revisions = 0
            while result["status"] == "finished":
                self.record.update_task(task_id, status="review")
                review = self.runner.supervise_result(self.campaign, task, result, ws.digest(), result.get("figures", []),
                                                      revision=revisions, previous_feedback=review.get("feedback", "")
                                                      if revisions else "")
                self.record.event(self.cid, None, task_id, "supervisor", "review",
                                  f"{review.get('decision')}: {review.get('feedback', '')}", review)
                if (review.get("decision") == "reject" and task["origin"] == "mission"
                        and task["kind"] in PIPELINE_KINDS and revisions < HARNESS.max_revisions):
                    # a pipeline step cannot simply be dropped: a rejected write-up gets a revision instead
                    review = {**review, "decision": "revise"}
                if review.get("decision") != "revise":
                    break
                if task["stage"] == "report" and revisions >= 2 and result.get("summary"):
                    # third request on a report: the editor fixes the wording itself (no more stalled reports)
                    result["summary"] = self.runner.edit_fix(result["summary"], review.get("feedback", ""), task_id)
                    review = {**review, "decision": "accept", "feedback": "[fixed by the editor] " +
                              review.get("feedback", "")}
                    break
                if revisions >= HARNESS.max_revisions:
                    result["status"] = "stalled"
                    break
                revisions += 1
                self.record.update_task(task_id, status="revising", revisions=revisions)
                result = ws.run(feedback=review.get("feedback", "Please revise."))
            status = {"accept": "done", "reject": "rejected"}.get(review.get("decision"), "done") \
                if result["status"] == "finished" else result["status"]
            verdict = result.get("verdict") or {}
            self.record.update_task(task_id, status=status, ended=time.time(), summary=result.get("summary", ""),
                                    verdict=verdict.get("outcome") if isinstance(verdict, dict) else None,
                                    meta={**task["meta"], "figures": result.get("figures", []),
                                          "confidence": result.get("confidence"),
                                          "soundness": review.get("soundness")})
            if status == "done" or (task["origin"] == "mission" and task["kind"] in PIPELINE_KINDS):
                if status != "done":
                    self.log(f"{task_id}: {task['kind']} write-up not accepted ({status}); its tool outputs (sweep "
                             f"tables, families, filter results, promotions) are still used by the next stage")
                self._on_done(task, ws, result)
            self._queue_followups(task, result.get("followups") or [], "worker")
            self._queue_followups(task, review.get("followups") or [], "supervisor")
            self.runner.curate(task, result, review)
            self.log(f"{status} {task_id} ({revisions} revisions)")
        except Exception as e:  # noqa: BLE001
            if ("429" in str(e) or "Too Many Requests" in str(e)) and not task["meta"].get("requeued"):
                self.record.update_task(task_id, status="queued", meta={**task["meta"], "requeued": True})
                self.log(f"rate-limited: {task_id} requeued")
                return
            self.record.update_task(task_id, status="failed", ended=time.time(), summary=f"{type(e).__name__}: {e}")
            self.record.event(self.cid, None, task_id, "harness", "error", traceback.format_exc()[-4000:])
            self.log(f"FAILED {task_id}: {e}")
        finally:
            if ws:
                ws.close()
            self._tidy(task_id)

    def _tidy(self, task_id: str):
        """Agents download raw satellite files into their folders; a live run filled the disk. Keep results,
        drop bulky raw downloads once the task is over."""
        d = self.cdir / "tasks" / task_id
        if not d.is_dir():
            return
        for f in d.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink() and f.suffix.lower() in (".nc", ".nc4", ".h5", ".hdf", ".grib",
                                                                                 ".grb", ".tif", ".zip", ".gz", ".tar") \
                        and f.stat().st_size > 20_000_000:
                    f.unlink()
            except OSError:
                pass

    def _on_done(self, task: dict, ws: WorkerSession, result: dict):
        with self.lock:
            for k, v in ws.ctx.state.get("data_files", {}).items():
                # the campaign's own tables are fixed once set: an agent's small regional download of the same
                # variable must not replace them (found live: PRECTOTCORR pointed to one agent's box afterwards)
                self.data_index.setdefault(k, v)
            (self.cdir / "data_index.json").write_text(json.dumps(self.data_index, indent=1))
        if task["kind"] == "census":
            self._queue_deep_dives(task["id"])
        if task["kind"] in ("hypothesis", "followup"):
            self._safe(self.confirm_pending)

    def _queue_deep_dives(self, parent_id: str | None = None):
        """One deep dive per promoted family (idempotent). Called when the census finishes and again before the
        deep-dive stage, so a census whose write-up the supervisor rejected still hands its promotions on."""
        if parent_id is None:
            t = self.record.tasks(self.cid, kind="census")
            parent_id = t[0]["id"] if t else None
        for c in self.record.candidates(self.cid, promoted=1):
            if self.record.one("SELECT id FROM tasks WHERE campaign_id=? AND kind='deep_dive' AND meta LIKE ?",
                               (self.cid, f'%"family": "{c["family"]}"%')):
                continue
            f = c["filters"] or {}
            ev_txt = ""
            if f.get("independent"):
                ev_txt = (f"\nAutomatic evidence (already computed, do not redo it; build on it): independent data "
                          f"{f.get('independent')} {json.dumps(f.get('independent_detail', []))}; robustness "
                          f"{f.get('robustness')} {json.dumps(f.get('robustness_detail', {}))}; merged families on the "
                          f"same cells: {f.get('merged_members', [])}.\nClosest literature found automatically:\n"
                          + self._lit_precheck(f"{c['variable']} {c['cls']} {f.get('place', '')}"[:200], 4))
            brief = (f"Deep dive on anomaly family {c['family']} ({f.get('place') or 'see family_info'}): {c['cls']} in "
                     f"{c['variable']} over {c['n_cells']} grid cells centred near ({c['lat']:.2f}, {c['lon']:.2f})"
                     f"{', onset ~' + str(c['onset']) if c['onset'] else ''}; standardized effect {c['effect']:.2f}, "
                     f"min q {c['qvalue']:.2g}. Filter results: {json.dumps({k: f.get(k) for k in ('significance', 'coherence', 'magnitude', 'corroboration', 'data_quality', 'data_issue')})}.\n"
                     f"{self.plan.get('deep_dive_brief', '')}\nDecide: new_signal, known_phenomenon, artifact or "
                     f"not_robust, with evidence. Look at the raw series and at imagery where useful." + ev_txt)
            self.record.add_task(self.cid, title=f"Deep dive: {c['family']}", brief=brief, stage="deep_dive",
                                 kind="deep_dive", origin="mission", parent_id=parent_id, depth=0,
                                 meta={"family": c["family"], "variable": c["variable"], "cls": c["cls"]})

    def _queue_followups(self, parent: dict, proposals: list[dict], origin: str):
        if parent["stage"] == "report":  # reports are terminal; their ideas go to the KB via the curator
            return
        for p in proposals[: HARNESS.max_followups_per_task]:
            if not p.get("title"):
                continue
            stage = parent["stage"]
            meta = {"family": parent["meta"].get("family"), "rationale": p.get("rationale", ""),
                    "parent_title": parent["title"]}
            self.record.add_task(self.cid, title=str(p["title"])[:200], brief=str(p.get("brief", "")),
                                 stage=stage, kind="followup", origin=origin, parent_id=parent["id"],
                                 depth=parent["depth"] + 1, meta=meta, status="triage")

    def triage(self):
        pending = self.record.tasks(self.cid, status="triage")
        if not pending:
            return
        all_tasks = self.record.tasks(self.cid)
        n_total = len([t for t in all_tasks if t["status"] not in ("triage", "rejected")])  # declined ideas cost nothing
        survey_cap = max(2, int(HARNESS.max_tasks * HARNESS.survey_followup_share))  # keep budget for deep dives
        survey_used = len([t for t in all_tasks if t["kind"] == "followup" and t["status"] not in ("triage", "rejected")
                           and t["stage"] in ("sweep", "classify", "census")])
        verif_cap = max(2, int(HARNESS.max_tasks * HARNESS.verification_share))
        verif_used = len([t for t in all_tasks if t["meta"].get("work_kind") == "verification"
                          and t["status"] not in ("triage", "rejected")])
        # the survey stages get their own, smaller verification allowance: otherwise single-cell corroborations
        # fill the whole survey budget before a single family has been classified (seen live on the global run)
        survey_stages = ("sweep", "classify", "census")
        sv_cap = max(3, int(survey_cap * min(1.0, 2 * HARNESS.verification_share)))
        sv_used = len([t for t in all_tasks if t["meta"].get("work_kind") == "verification"
                       and t["stage"] in survey_stages and t["status"] not in ("triage", "rejected")])
        closed = {c["family"] for c in self.record.candidates(self.cid) if c["outcome"] == "artifact"}
        ok = []
        for t in pending:
            reason = None
            fam = t["meta"].get("family")
            if fam and fam in closed and t["stage"] != "science":
                reason = f"lead closed: {fam} is a data artifact (catalogued); budget goes to science"
            elif t["depth"] > HARNESS.max_depth:
                reason = f"depth {t['depth']} exceeds max {HARNESS.max_depth}"
            elif n_total >= HARNESS.max_tasks:
                reason = f"campaign budget of {HARNESS.max_tasks} tasks reached"
            elif t["stage"] in ("sweep", "classify", "census") and survey_used >= survey_cap:
                reason = f"survey-stage follow-up budget ({survey_cap}) reached; budget is kept for deep dives"
            if reason:
                self.record.update_task(t["id"], status="rejected", triage_note=reason)
            else:
                ok.append(t)
                n_total += 1
                if t["stage"] in ("sweep", "classify", "census"):
                    survey_used += 1
        if not ok:
            return
        # duplicates are judged against work that exists: declined ideas are not precedent (many were declined only
        # because a budget was full, and the triage model otherwise reads them as "already deprioritised")
        existing = "\n".join(f"- {t['title']} [{t['status']}]" for t in self.record.tasks(self.cid)
                             if t["status"] not in ("triage", "rejected"))[-4000:]
        decisions = self.runner.triage([{**t, "parent_title": t["meta"].get("parent_title", ""),
                                         "rationale": t["meta"].get("rationale", "")} for t in ok],
                                       f"{n_total} tasks so far (budget {HARNESS.max_tasks}). Variables in this "
                                       f"campaign's data: {', '.join(self.plan.get('variables') or self.brief.get('variables') or [])} "
                                       f"(an idea about any other variable is not feasible: reject it). "
                                       f"Existing tasks:\n{existing}")
        by_idx = {int(d.get("index", -1)): d for d in decisions if str(d.get("index", "")).lstrip("-").isdigit()}
        order = sorted(range(len(ok)), key=lambda i: -float(by_idx.get(i, {}).get("value", 1) or 0))
        for i in order:
            t = ok[i]
            d = by_idx.get(i, {"decision": "release", "reason": "no decision returned"})
            kind = str(d.get("kind", "science")).lower()
            note = f"[{kind}, value {d.get('value', '?')}] {d.get('reason', '')}"
            if d.get("decision") == "release" and kind == "housekeeping":
                d = {**d, "decision": "reject"}
                note += " | housekeeping is not research: rejected"
            elif d.get("decision") == "release" and kind == "verification":
                if verif_used >= verif_cap:
                    d = {**d, "decision": "reject"}
                    note += f" | verification budget ({verif_cap} tasks) used: the rest goes to science"
                elif t["stage"] in survey_stages and sv_used >= sv_cap:
                    d = {**d, "decision": "reject"}
                    note += (f" | survey-stage verification allowance ({sv_cap} tasks) used: signals are checked "
                             f"in the deep dives and the science stage")
                else:
                    verif_used += 1
                    if t["stage"] in survey_stages:
                        sv_used += 1
            meta = {**t["meta"], "work_kind": kind, "value": d.get("value")}
            if d.get("decision") == "release":
                self.record.update_task(t["id"], status="queued", triage_note=note, meta=meta)
            else:
                self.record.update_task(t["id"], status="rejected", triage_note=note, meta=meta)
        self.log(f"triage: {sum(1 for d in by_idx.values() if d.get('decision') == 'release')} released of {len(ok)}")

    # ---------------------------------------------------------------------- scheduler (v3: two tracks)
    def _pool_get(self) -> ThreadPoolExecutor:
        if getattr(self, "_pool", None) is None:
            self._pool = ThreadPoolExecutor(max_workers=HARNESS.concurrency)
            self._running: dict = {}
        return self._pool

    def _program_on(self) -> bool:
        return HARNESS.program and self._has_stage("program")

    def _start_tasks(self, stage: str):
        """Fill free worker slots. The research programme keeps a reserved share of the workers whenever it has
        queued work, so hypothesis tests run from the first hour instead of after the survey."""
        pool = self._pool_get()
        free = HARNESS.concurrency - len(self._running)
        if free <= 0 or self.stopped():
            return
        prog_on = self._program_on() and stage != PARALLEL_STAGE
        queued = self.record.tasks(self.cid, status="queued")
        prim = [t for t in queued if t["stage"] == stage]
        prog = sorted([t for t in queued if prog_on and t["stage"] == PARALLEL_STAGE],
                      key=lambda t: -float(t["meta"].get("priority", 0.5) or 0.5))
        running_stages = [st for st in self._running.values()]
        n_prog = sum(1 for st in running_stages if st == PARALLEL_STAGE)
        reserved = max(1, round(HARNESS.concurrency * HARNESS.program_slots)) if prog_on else 0
        order = []
        if prog:  # the programme first takes its reserved slots, the stage gets the rest, spare slots go to either
            order += prog[: max(0, reserved - n_prog)]
        order += prim
        order += [t for t in prog if t not in order]
        for t in order[:free]:
            self.record.update_task(t["id"], status="starting")
            self._running[pool.submit(self.process, t["id"])] = t["stage"]

    def run_stage(self, stage: str):
        self.record.set_stage_status(self.cid, stage, "running")
        self._pool_get()
        while True:
            self.triage()
            self._pi_tick()
            self._start_tasks(stage)
            mine = [f for f, st in self._running.items() if st == stage]
            if not mine:
                more = [t for t in self.record.tasks(self.cid) if t["stage"] == stage
                        and t["status"] in ("queued", "triage", "starting")]
                if stage == PARALLEL_STAGE and self._pi_busy():
                    more = more or ["pi"]
                if not more or self.stopped():
                    break
                if not self._running:
                    time.sleep(0.5)
                    continue
            done, _ = wait(list(self._running), timeout=2, return_when=FIRST_COMPLETED)
            for f in done:
                self._running.pop(f, None)
                self._done_count = getattr(self, "_done_count", 0) + 1
        left = [t for t in self.record.tasks(self.cid) if t["stage"] == stage and t["status"] in ("queued", "triage")]
        # a STOP drains running tasks; the stage stays open so `resume` starts the queued ones
        self.record.set_stage_status(self.cid, stage, "pending" if (left and self.stopped()) else "done")

    def _drain_all(self):
        """Wait for every running task (any stage), e.g. before the campaign ends."""
        while getattr(self, "_running", None):
            done, _ = wait(list(self._running), timeout=5, return_when=FIRST_COMPLETED)
            for f in done:
                self._running.pop(f, None)

    # ---------------------------------------------------------------------- v3: the PI and the programme
    def _pi_busy(self) -> bool:
        t = getattr(self, "_pi_thread", None)
        return bool(t and t.is_alive())

    def _pi_tick(self):
        """Called from the scheduler loop: write the programme once the GPU results are in (or after a maximum
        wait), hold lab meetings every few finished tasks, and send supported claims to referees."""
        if not self._program_on():
            return
        if self._pi_busy():
            return
        self._queue_referees()
        if not self.record.hypotheses(self.cid):
            if self._stage_done("program"):
                return
            learn_alive = getattr(self, "_learn_thread", None) is not None and self._learn_thread.is_alive()
            waited = (time.time() - getattr(self, "_t_start", time.time())) / 60
            if learn_alive and waited < HARNESS.program_wait_min:
                return
            self._pi_thread = threading.Thread(target=self._safe, args=(self.write_program,), daemon=True)
            self._pi_thread.start()
            return
        self._promote_reserve()
        if self.record.hypotheses(self.cid, status="awaiting_confirmation") or self._untestable_by_explorers():
            threading.Thread(target=self._safe, args=(self.confirm_pending,), daemon=True).start()
        n_meet = getattr(self, "_meetings", None)
        if n_meet is None:
            n_meet = len(self.record.q("SELECT id FROM kb WHERE campaign_id=? AND title LIKE 'Lab meeting%'", (self.cid,)))
            self._meetings = n_meet
        if n_meet >= HARNESS.max_meetings or self._program_budget_left() <= 0:
            return
        since = getattr(self, "_done_count", 0) - getattr(self, "_done_at_meeting", 0)
        prog = [t for t in self.record.tasks(self.cid) if t["stage"] == PARALLEL_STAGE]
        idle = not [t for t in prog if t["status"] in ("queued", "starting", "running", "review", "revising", "triage")]
        last = getattr(self, "_last_meeting_ts", 0)
        new_results = sum(1 for t in prog if t["status"] in ("done", "stalled") and (t["ended"] or 0) > last)
        gap = time.time() - last
        # a lab meeting needs programme results to discuss: after a few finished tests, or when the programme
        # has run dry
        if new_results and ((new_results >= max(2, HARNESS.meeting_every // 2) and gap > 120)
                            or (since >= HARNESS.meeting_every and gap > 120) or (idle and gap > 120)):
            self._done_at_meeting = getattr(self, "_done_count", 0)
            self._pi_thread = threading.Thread(target=self._safe, args=(self.lab_meeting,), daemon=True)
            self._pi_thread.start()

    def _promote_reserve(self):
        """No reserve idea is wasted: when no programme test is queued or running, the next reserve hypothesis is
        started (the lab meeting can also promote them by id)."""
        active = [t for t in self.record.tasks(self.cid) if t["stage"] == PARALLEL_STAGE and t["kind"] == "hypothesis"
                  and t["status"] in ("queued", "starting", "running", "review", "revising")]
        if active or self._program_budget_left() <= 0:
            return
        res = [h for h in self.record.hypotheses(self.cid) if h["status"] == "reserve" and not self.spec_problem(
            {**h, "_storage_kind": self.brief.get("storage_kind"), "_time_holdout": self.brief.get("time_holdout")})]
        # found live (Track B): the first reserve ideas had no valid spec, were refused again and again, and the
        # campaign ended without a single test; only testable reserve ideas are promoted
        for h in res[: max(1, round(HARNESS.concurrency * HARNESS.program_slots))]:
            self._add_hypothesis_task(h, h["id"])
            self.log(f"programme: reserve hypothesis {h['id']} started ({h['title'][:60]})")

    def confirm_pending(self):
        """Run the pre-registered held-out confirmation for every claimed success (code only; agents cannot touch
        it), then the novelty gate."""
        import os

        from . import confirm as _confirm

        if not hasattr(self, "_confirm_lock"):
            self._confirm_lock = threading.Lock()
        if not self._confirm_lock.acquire(blocking=False):
            return  # another thread is already confirming
        try:
            self._confirm_all(_confirm)
        finally:
            self._confirm_lock.release()

    def _untestable_by_explorers(self) -> list[dict]:
        """Inconclusive hypotheses with a valid pre-registered spec that was never run: typically the claim lives in
        held-out data the explorers cannot see (their result says so). Each gets its one held-out test anyway."""
        from . import confirm as _confirm

        if not self.confirm_sets:
            return []
        return [h for h in self.record.hypotheses(self.cid) if h["status"] in ("inconclusive", "not_shown")
                and not h.get("confirmation") and _confirm.validate(h.get("confirm") or "")[0] is not None]

    def _confirm_all(self, _confirm):
        import os

        for h in self.record.hypotheses(self.cid, status="awaiting_confirmation") + self._untestable_by_explorers():
            spec = self._apply_alpha_rules(h.get("confirm") or "")
            bad = self.spec_problem({**h, "_storage_kind": self.brief.get("storage_kind"), "_time_holdout": self.brief.get("time_holdout")}) if spec and spec != "{}" else ""
            if bad:
                self.record.update_hypothesis(self.cid, h["id"], actor="confirmation", status="inconclusive",
                                              confirmation=f"not run: the pre-registered spec cannot test this claim ({bad})")
                self.log(f"held-out confirmation {h['id']}: not run (invalid spec: {bad[:120]})")
                continue
            try:
                res = _confirm.run(spec, progress=None) if spec and spec != "{}" else \
                    {"passed": False, "outcome": "inconclusive", "detail": "no pre-registered confirmation: not confirmable"}
                if res["passed"]:
                    res = self._replicate(_confirm, spec, res)
            except Exception as e:  # noqa: BLE001
                res = {"passed": False, "outcome": "inconclusive", "detail": f"confirmation could not run ({e})"}
            self.record.update_hypothesis(self.cid, h["id"], actor="confirmation", status=res["outcome"],
                                          confirmation=res["detail"][:2000])
            self.log(f"held-out confirmation {h['id']}{' (explorers could not test it)' if h['status'] == 'inconclusive' else ''}"
                     f": {res['outcome']} ({res['detail'][:160]})")
            if res["passed"]:
                # every confirmed result gets the real ADS check (agents under-claim as well as over-claim); only an
                # ADS search that finds no paper reporting it makes it "new"
                m = self._n_tests()
                if res.get("p") is not None and res["p"] > 0.05 / m:
                    self.record.update_hypothesis(self.cid, h["id"], novelty=(h.get("novelty") or "") + (
                        f"\n[NOT NEW: p = {res['p']:.3g} does not survive the correction for {m} pre-registered held-out "
                        f"tests in this campaign (needs p <= {0.05 / m:.2g}); it stays 'supported']"))
                    self.log(f"novelty gate {h['id']}: skipped, p = {res['p']:.3g} > 0.05/{m}")
                    continue
                self.novelty_gate(h["id"], res)

    TWIN = {"GRACE_TWS": "GRACE_GSFC_TWS", "GRACE_GSFC_TWS": "GRACE_TWS"}

    @staticmethod
    def spec_problem(h: dict) -> str:
        """Why a hypothesis's confirm spec cannot test it ("" when it can). Besides confirm.validate: a claim about
        a specific recent year must not be tested with a long-term trend (found live: "northern Morocco's 2026
        storage is above the pre-drought trend" was registered as a 2002-2026 trend test)."""
        import re as _re

        from . import confirm as _confirm

        spec, problem = _confirm.validate(h.get("confirm") or "")
        if spec is None:
            return problem
        if (h.get("_storage_kind") == "gain" and spec["dataset"] in ("GRACE_TWS", "GRACE_GSFC_TWS")
                and spec["kind"] != "gain"):
            return ("this campaign tests water storage as seasonal GAINS (levels carry over from year to year): use "
                    "\"kind\": \"gain\" with \"from_months\", \"to_months\", \"test\" and event_years")
        th = h.get("_time_holdout") or {}
        if th:
            cut = int(th["explore_end"])
            if spec["dataset"] not in (th.get("datasets") or []):
                return f"this campaign confirms on {th.get('datasets')} after {cut} (time holdout): use one of them"
            if spec.get("test") == "later":
                if int(spec.get("after", 0)) < cut or spec["fit_years"][1] > cut:
                    return f"test 'later' must use \"after\": {cut} (the explorers saw the years up to {cut})"
            elif spec["years"][0] <= cut:
                return (f"the explorers saw {spec['dataset']} up to {cut}: the confirmation must test only later years "
                        f"(\"years\": [{cut + 1}, ...]) or use \"test\": \"later\" with \"after\": {cut}")
        test = spec.get("test") if spec["kind"] in ("contrast", "residual", "gain") else spec["kind"]
        years = sorted({int(y) for y in _re.findall(r"\b(20[12]\d)\b", h.get("title") or "")})
        if test == "trend" and years and years[-1] >= 2015:
            return (f"the claim is about {years} but the test is a long-term trend over {spec['years']}: use an "
                    f"event test (\"event_years\": {[y for y in years if y >= 2015]}) or a residual/contrast with "
                    f"\"test\": \"event\"")
        return ""

    def _apply_alpha_rules(self, spec):
        """Brief-level stricter alphas, enforced in code: a box an earlier round already looked at pays for it
        ("alpha_rules": [{"box": [S, N, W, E], "from_year": 2024, "alpha": 0.01}])."""
        rules = self.brief.get("alpha_rules") or []
        if not rules or not spec or spec == "{}":
            return spec
        try:
            s = json.loads(spec) if isinstance(spec, str) else dict(spec)
            S, N, W, E = (float(v) for v in s["box"])
        except Exception:  # noqa: BLE001
            return spec
        yrs = [int(y) for y in (s.get("event_years") or [])] or [int((s.get("years") or [0, 0])[1])]
        for r in rules:
            s2, n2, w2, e2 = (float(v) for v in r["box"])
            overlap = S < n2 and s2 < N and W < e2 and w2 < E
            if overlap and max(yrs) >= int(r.get("from_year", 0)):
                s["alpha"] = min(float(s.get("alpha", 0.05)), float(r["alpha"]))
        return json.dumps(s)

    def _n_tests(self) -> int:
        """Number of pre-registered held-out tests this campaign makes (at least 10): the family a "new" claim must
        survive. With 30 tests at alpha 0.05, one or two pass by chance."""
        from . import confirm as _confirm

        n = sum(1 for h in self.record.hypotheses(self.cid) if _confirm.validate(h.get("confirm") or "")[0] is not None)
        # tests run by earlier rounds on the same question count too: the operator has seen their results
        return max(10, n) + int(self.brief.get("prior_tests", 0) or 0)

    def _replicate(self, _confirm, spec, res: dict) -> dict:
        """A water-storage result must hold on BOTH GRACE solutions (JPL and GSFC mascons, processed independently):
        run the same pre-registered spec on the twin, in code. Both must pass; the larger p counts."""
        s = json.loads(spec) if isinstance(spec, str) else dict(spec)
        twin = self.TWIN.get(s.get("dataset"))
        if not twin:
            return res
        s2 = {**s, "dataset": twin}
        if s2.get("kind") in ("correlation", "correlation_change", "residual") and s2.get("driver") in self.TWIN:
            s2["driver"] = self.TWIN[s2["driver"]]
        ok, problem = _confirm.validate(s2)
        if ok is None:
            return {**res, "passed": False, "outcome": "inconclusive",
                    "detail": res["detail"] + f" REPLICATION on {twin} could not run ({problem}): not confirmed."}
        r2 = _confirm.run(s2, progress=None)
        both = r2["passed"]
        p = max(res.get("p") or 0, r2.get("p") or 0) if both else res.get("p")
        return {**res, "passed": both, "outcome": "supported" if both else (r2["outcome"] if r2["outcome"] in
                ("refuted",) else "not_shown"), "p": p,
                "detail": res["detail"] + f" REPLICATION on the independent {twin}: {r2['detail']}"}

    def novelty_gate(self, hid: str, res: dict | None = None):
        import os

        from . import novelty

        h = self.record.hypotheses(self.cid, id=hid)[0]
        basis = (h.get("novelty") or "").strip()
        if self.mock or not os.environ.get("ADS_API_TOKEN"):
            self.record.update_hypothesis(self.cid, hid, novelty=basis + " [novelty UNVERIFIED: no NASA ADS access]")
            return
        if res:
            h = {**h, "confirmation": res.get("detail", h.get("confirmation", ""))}
        try:
            out = novelty.check(h, self.llm)
        except Exception as e:  # noqa: BLE001
            out = {"verdict": "unverified", "reason": f"novelty check failed: {type(e).__name__}", "n_ok": 0,
                   "queries": [], "papers": [], "closest": []}
        (self.cdir / "novelty").mkdir(exist_ok=True)
        (self.cdir / "novelty" / f"{hid}.json").write_text(novelty.dumps(out))
        note = (basis + "\n\n" if basis else "") + novelty.summary(out)
        self.record.update_hypothesis(self.cid, hid, novelty=note[:6000])
        self.log(f"novelty gate {hid}: ADS {out['verdict']} ({out.get('n_ok', 0)} searches, "
                 f"{len(out.get('papers', []))} papers)")
        if out["verdict"] == "not_found":
            self.record.update_hypothesis(self.cid, hid, actor="novelty_gate", status="supported_new")

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.log(f"PI: {fn.__name__} failed ({type(e).__name__}: {str(e)[:200]})")
            self.record.event(self.cid, None, None, "harness", "error", traceback.format_exc()[-3000:])

    def _program_budget_left(self) -> int:
        used = len([t for t in self.record.tasks(self.cid) if t["stage"] == PARALLEL_STAGE
                    and t["status"] not in ("rejected", "triage")])
        share = 1.0 if self.focused else HARNESS.program_budget  # a focused campaign IS its programme
        return int(HARNESS.max_tasks * share) - used

    def _lit_precheck(self, query: str, n: int = 5) -> str:
        from .tools import compute
        from .tools.base import ToolContext

        if self.mock:
            return "(literature pre-check skipped in mock mode)"
        try:
            ctx = ToolContext(self.record, self.cid, None, None, self.cdir, self.cdir, self.mock)
            r = compute.literature_search(ctx, query[:300], n)
            return r.text[:2500]
        except Exception as e:  # noqa: BLE001
            return f"(literature pre-check failed: {e})"

    def _literature_scan(self) -> str:
        """Literature first (v4): what is published about the brief's open problem and what the papers themselves
        say is unresolved. Runs once, cached in the knowledge base."""
        row = self.record.one("SELECT body FROM kb WHERE campaign_id=? AND title='Literature scan (open problem)'",
                              (self.cid,))
        if row:
            return row["body"]
        qs = list(self.brief.get("literature_queries") or [])
        if not qs:
            qs = [f"{self.brief.get('title', '')} unexplained cause", f"{self.brief.get('title', '')} open questions"]
        parts = [f"## {q}\n{self._lit_precheck(q, 8)}" for q in qs[:8]]
        known = self.brief.get("known_literature") or []
        if known:
            parts.insert(0, "## Key papers named in the brief (read their abstracts)\n" + "\n".join(f"- {k}" for k in known))
        body = "\n\n".join(parts)[:16000]
        self.record.kb_add(self.cid, None, "method", "Literature scan (open problem)", body, ["literature", "pi"])
        return body

    def _data_menu(self) -> str:
        from . import catalog

        ds = self.plan.get("variables") or []
        lines = [f"- {d}: {catalog.get(d).label} [{catalog.get(d).units}] {catalog.get(d).source}"
                 for d in ds if d in catalog.BY_ID]
        hold = [f"- {d}: {catalog.get(d).label} [{catalog.get(d).units}] {catalog.get(d).source}"
                for d in sorted(self.holdout) if d in catalog.BY_ID]
        th = (self.brief or {}).get("time_holdout") or {}
        if th:
            hold += [f"- {d}: {catalog.get(d).label} [{catalog.get(d).units}] {catalog.get(d).source}: TIME-HELD-OUT "
                     f"after {th['explore_end']} (explorers see it up to {th['explore_end']}; confirmations test only "
                     f"the later years)" for d in th.get("datasets") or [] if d in catalog.BY_ID]
        return ("Campaign datasets (in data/, explorers use these):\n" + ("\n".join(lines) or "(survey tables)") +
                "\nHELD-OUT datasets (confirmation only; use these ids in each hypothesis's confirm.dataset):\n" +
                ("\n".join(hold) or "(none: confirmation uses the independent products of the evidence engine)"))

    def _program_context(self) -> str:
        kb = {k["title"]: k["body"] for k in self.record.q(
            "SELECT title, body FROM kb WHERE campaign_id=? AND title IN ('Automatic data screen (trust map)', "
            "'GPU coupling model: changing drivers') ORDER BY id", (self.cid,))}
        sig = self._top_signals(12)
        b = self.brief
        return (f"# Research brief\n{b.get('title')}\n{b.get('question')}\nRegion: {b.get('region')} Period: "
                f"{b.get('start_year')}-{b.get('end_year')}\nNotes: {b.get('notes', '')}\n\n# Data\n{self._data_menu()}"
                f"\n\n# Literature scan (what is published, what is still open)\n{self._literature_scan()}"
                f"\n\n# Data screen\n{kb.get('Automatic data screen (trust map)', '(not run)')[:3000]}"
                f"\n\n# GPU coupling model\n{kb.get('GPU coupling model: changing drivers', '(not available yet)')[:3000]}"
                f"\n\n# Strongest independently confirmed survey signals so far\n{sig or '(survey not finished yet)'}"
                + (f"\n\n# {ex['title']}\n{ex['body'][:9000]}" if (ex := self.record.one(
                    "SELECT title, body FROM kb WHERE campaign_id=? AND title LIKE 'Exploration sweep%'", (self.cid,)))
                   else ""))

    def _top_signals(self, n: int) -> str:
        rows = []
        for c in self.record.candidates(self.cid, passed=1):
            f = c["filters"] or {}
            if f.get("merged_into") or c["outcome"] in ("artifact", "not_robust") or f.get("score") is None:
                continue
            rows.append((f.get("score", 0), c, f))
        rows.sort(key=lambda r: -r[0])
        return "\n".join(f"- {c['family']} score {sc}: {c['cls']} in {c['variable']}, {c['n_cells']} cells, effect "
                         f"{c['effect']:.2f}" + (f", onset {c['onset']}" if c['onset'] else "") +
                         f"; {f.get('place', '')}; independent: {f.get('independent')}; robustness {f.get('robustness')}"
                         for sc, c, f in rows[:n])

    def _confirmation_only(self, h: dict, hid: str) -> bool:
        """A hypothesis whose test needs ONLY held-out data cannot be explored by agents (they are refused those
        datasets; v4 live run: such tests ended 'inconclusive, data held out'). Its pre-registered test is run by
        code directly instead: a purely confirmatory test, never seen by an explorer."""
        from . import catalog
        from . import confirm as _confirm

        if not self.holdout:
            return False
        txt = f"{h.get('data', '')} {h.get('test', '')}"
        ids = {v.id for v in catalog.CATALOG if v.id in txt}
        if not ids or not ids <= self.holdout:
            return False
        hh = self.record.hypotheses(self.cid, id=hid)
        if not hh or _confirm.validate(hh[0].get("confirm") or "")[0] is None:
            return False
        self.record.update_hypothesis(self.cid, hid, actor="pi", status="awaiting_confirmation",
                                      result="Direct pre-registered test on held-out data (no exploring agent).",
                                      evidence="[confirmation-only hypothesis: its data are all held out]")
        self.log(f"programme: {hid} goes straight to its held-out confirmation (its data are all held out)")
        threading.Thread(target=self._safe, args=(self.confirm_pending,), daemon=True).start()
        return True

    def _add_hypothesis_task(self, h: dict, hid: str, extra: str = "", parent: str | None = None):
        if not extra and self.confirm_sets:
            from . import confirm as _confirm

            hh = self.record.hypotheses(self.cid, id=hid)
            if hh and self.spec_problem({**hh[0], '_storage_kind': self.brief.get('storage_kind'), '_time_holdout': self.brief.get('time_holdout')}):
                # found live (round 1): 25 hypotheses without any pre-registered test were queued and used ~40 agent
                # tasks that could never confirm anything. A hypothesis is tested only once it has a valid spec.
                self.record.update_hypothesis(self.cid, hid, actor="pi", status="reserve")
                self.log(f"programme: {hid} not queued (no valid held-out confirmation spec)")
                return None
        if not extra and self._confirmation_only(h, hid):
            return None
        lit = self._lit_precheck(f"{h.get('title', '')} {h.get('statement', '')}"[:240])
        brief = prompts.HYPOTHESIS_TASK.format(hid=hid, question=h.get("question", "?"), statement=h.get("statement", ""),
                                               test=h.get("test", ""), kill=h.get("kill", ""), data=h.get("data", ""),
                                               extra=extra, literature=lit,
                                               novelty_basis=h.get("novelty_basis") or "(not stated)")
        tid = self.record.add_task(self.cid, title=f"Test {hid}: {h.get('title', '')}"[:200], brief=brief,
                                   stage=PARALLEL_STAGE, kind="hypothesis", origin="pi", parent_id=parent, depth=0,
                                   meta={"hypothesis": hid, "priority": float(h.get("priority", 0.5) or 0.5)})
        hh = self.record.hypotheses(self.cid, id=hid)
        if hh:  # a deeper test keeps the hypothesis's current result and status until it reports
            st = "testing" if hh[0]["status"] in ("open", "testing", "dropped", "reserve") else hh[0]["status"]
            self.record.update_hypothesis(self.cid, hid, actor="pi", status=st, task_ids=hh[0]["task_ids"] + [tid])
        return tid

    def exploration_sweep(self):
        """Time-holdout campaigns: a code search over the exploration period only (no later data exists in the
        campaign tables), so the PI writes hypotheses from what the data showed, not from memory of the literature."""
        from . import explore

        if self.record.one("SELECT 1 FROM kb WHERE campaign_id=? AND title LIKE 'Exploration sweep%'", (self.cid,)):
            return
        th = self.brief["time_holdout"]
        body = explore.water_balance_sweep(self._load_campaign_data(), int(th["explore_end"]), log=self.log)
        self.record.kb_add(self.cid, None, "method", f"Exploration sweep (code, up to {th['explore_end']})", body,
                           ["explore", "pi"])
        self.log("exploration sweep: " + body.split("\n")[0][:200])

    def write_program(self):
        self.record.set_stage_status(self.cid, "program", "running")
        out = self.runner.pi_program(self._program_context())
        hyps = [h for h in (out.get("hypotheses") or []) if h.get("statement")][:16]
        if not hyps:
            self.log("PI: no research programme returned; the survey continues alone")
            self.record.set_stage_status(self.cid, "program", "done")
            return
        from . import confirm as _confirm

        lines = []
        hyps.sort(key=lambda h: -float(h.get("priority", 0.5) or 0.5))
        n_now = max(3, int(HARNESS.concurrency * max(HARNESS.program_slots, 0.5) * 2))
        to_queue = []
        for k, h in enumerate(hyps):
            spec, problem = _confirm.validate(h.get("confirm"))
            if spec and spec["dataset"] not in self.confirm_sets and self.confirm_sets:
                spec, problem = None, f"confirm.dataset {spec['dataset']} is not a held-out dataset"
            h = {**h, "confirm": spec or {}, "status": "open" if k < n_now else "reserve"}
            hid = self.record.add_hypothesis(self.cid, {**h, "origin": "pi"})
            if k < n_now:
                to_queue.append((h, hid))
            lines.append(f"{hid} [{'queued' if k < n_now else 'reserve'}] (p={float(h.get('priority', 0.5) or 0.5):.2f}) "
                         f"{h.get('title')}: {h.get('statement')}\n  test: {h.get('test')}\n  kill: {h.get('kill')}"
                         f"\n  novelty basis: {h.get('novelty_basis', '-')}\n  confirm: "
                         + (json.dumps(spec) if spec else f"NONE ({problem}): cannot become supported"))
        self.repair_confirm_specs()  # every spec is completed BEFORE the first test starts
        # queue the best TESTABLE hypotheses first (a hypothesis without a valid spec waits in reserve)
        ok = [x for x in self.record.hypotheses(self.cid) if not self.spec_problem(
            {**x, "_storage_kind": self.brief.get("storage_kind"), "_time_holdout": self.brief.get("time_holdout")})]
        for x in ok[:n_now]:
            self._add_hypothesis_task(x, x["id"])
        self.record.kb_add(self.cid, None, "method", "Research programme (PI)",
                           f"{out.get('rationale', '')}\n\n" + "\n".join(lines), ["programme", "hypothesis", "pi"])
        self.log(f"PI: research programme with {len(hyps)} hypotheses ({min(len(hyps), n_now)} queued, "
                 f"{max(0, len(hyps) - n_now)} in reserve)")

    def repair_confirm_specs(self):
        """Hypotheses without a valid held-out confirmation can never be supported: the PI fixes their specs once,
        right after writing the programme and before any result exists (pre-registration stays before results)."""
        from . import confirm as _confirm

        # only hypotheses without a result of their own; a spec is never changed after its hypothesis has a result
        bad = [h for h in self.record.hypotheses(self.cid) if h["status"] in ("open", "testing", "reserve")
               and self.spec_problem({**h, '_storage_kind': self.brief.get('storage_kind'), '_time_holdout': self.brief.get('time_holdout')})]
        if not bad:
            return
        ctx = (f"# Brief\n{self.brief.get('title')}\n{self.brief.get('question')}\n\n# Data\n{self._data_menu()}\n\n"
               "# Hypotheses without a valid confirm spec\n" + "\n\n".join(
                   f"{h['id']}: {h['title']}\n{h['statement']}\ntest: {h['test']}\ncurrent spec: "
                   f"{h.get('confirm') or 'none'}\nwhy it is not valid: {self.spec_problem({**h, '_storage_kind': self.brief.get('storage_kind'), '_time_holdout': self.brief.get('time_holdout')})}"
                   for h in bad))
        out = self.runner.pi_confirm_fix(ctx)
        fixed = []
        for f in out.get("fixes") or []:
            h = next((x for x in bad if x["id"] == f.get("id")), None)
            spec, _ = _confirm.validate(f.get("confirm")) if h and f.get("confirm") else (None, "")
            if spec and self.spec_problem({**h, "confirm": spec, "_storage_kind": self.brief.get("storage_kind"), "_time_holdout": self.brief.get("time_holdout")}):
                spec = None
            if spec and (not self.confirm_sets or spec["dataset"] in self.confirm_sets):
                n = self.record.x("UPDATE hypotheses SET confirm=? WHERE campaign_id=? AND id=? AND status IN "
                                  "('open','testing','reserve')", (json.dumps(spec), self.cid, h["id"])).rowcount
                if n:  # a hypothesis that got its result meanwhile keeps no spec: pre-registration is before results
                    fixed.append(h["id"])
        self.record.kb_add(self.cid, None, "method", "Pre-registration completed (PI)",
                           f"Confirmation specs added before any result for {fixed}; still unconfirmable: "
                           f"{[h['id'] for h in bad if h['id'] not in fixed]}", ["programme", "preregistration"])
        self.log(f"PI: confirmation specs completed for {len(fixed)}/{len(bad)} hypotheses (before any result)")

    def lab_meeting(self):
        hs = self.record.hypotheses(self.cid)
        # full memory (v4): complete results, held-out confirmations and referee reports, not 400-character excerpts
        board = "\n\n".join(
            f"{h['id']} [{h['status']}] p={h['priority']:.2f} {h['title']}: {h['statement']}"
            f"\n   novelty basis: {h.get('novelty_basis') or '-'}"
            f"\n   result: {(h['result'] or '-')[:2500]}"
            + (f"\n   evidence: {h['evidence'][:1500]}" if h['evidence'] else "")
            + (f"\n   held-out confirmation: {h.get('confirmation')}" if h.get('confirmation') else "")
            + (f"\n   referee: {h['review'][:1500]}" if h['review'] else "")
            for h in hs if h["status"] != "reserve")
        reserve = "\n".join(f"{h['id']} p={h['priority']:.2f} {h['title']}: {h['statement'][:400]}"
                             for h in hs if h["status"] == "reserve")
        done = self.record.q("SELECT title, summary FROM tasks WHERE campaign_id=? AND stage=? AND status IN "
                             "('done','stalled') ORDER BY ended", (self.cid, PARALLEL_STAGE))
        rec = "\n\n".join(f"## {r['title']}\n{(r['summary'] or '')[:1500]}" for r in done)[-60000:]
        past = self.record.q("SELECT title, body FROM kb WHERE campaign_id=? AND title LIKE 'Lab meeting%' ORDER BY id",
                             (self.cid,))
        mins = "\n".join(f"- {m['title']}: {m['body'][:1200]}" for m in past)
        ctx = (f"# Brief\n{self.brief.get('title')}\n{self.brief.get('question')}\n\n# Hypothesis board\n{board}\n\n"
               f"# Reserve pool (not yet tested)\n{reserve or '(empty)'}\n\n# Every programme result so far\n"
               f"{rec or '(none)'}\n\n# Decisions of earlier meetings\n{mins or '(first meeting)'}\n\n"
               f"# Strongest confirmed survey signals\n{self._top_signals(10) or '(no survey in this campaign)'}\n\n"
               f"# Budget\n{self._program_budget_left()} programme tasks left.")
        self._last_meeting_ts = time.time()
        out = self.runner.pi_meeting(ctx)
        known = {h["id"]: h for h in hs}
        closed, deeper, new = [], [], []
        active = {}
        for t in self.record.tasks(self.cid):
            if t["stage"] == PARALLEL_STAGE and t["meta"].get("hypothesis"):
                active.setdefault(t["meta"]["hypothesis"], []).append(t)
        for c in out.get("close") or []:
            h = known.get(c.get("id"))
            if not h or h["status"] in ("supported", "supported_new", c.get("status")) or c.get("status") not in (
                    "not_shown", "refuted", "known", "inconclusive"):
                continue
            ts = active.get(h["id"], [])
            if any(t["status"] in ("starting", "running", "review", "revising") for t in ts):
                continue  # never close a hypothesis while its test is running: wait for the result
            for t in ts:  # a queued test of a hypothesis the PI drops is not run
                if t["status"] == "queued":
                    self.record.update_task(t["id"], status="rejected", triage_note=f"PI closed {h['id']} before the "
                                            f"test started: {c.get('reason', '')}")
            tested = any(t["status"] in ("done", "stalled") for t in ts)
            self.record.update_hypothesis(self.cid, h["id"], actor="pi", status=c["status"] if tested else "dropped",
                                          result=(h["result"] or "") + f" [PI closed: {c.get('reason', '')}]")
            closed.append(h["id"])
        promoted = []
        for pid in (out.get("promote") or [])[:4]:
            h = known.get(pid)
            if h and h["status"] == "reserve" and self._program_budget_left() > 0:
                self._add_hypothesis_task(h, h["id"])
                promoted.append(h["id"])
        for d in (out.get("deeper") or [])[:3]:
            h = known.get(d.get("id"))
            if not h or self._program_budget_left() <= 0 or not h["result"] or h["status"] in ("known", "refuted",
                                                                                                  "not_shown", "dropped"):
                continue  # deeper tests build on a live result; settled or untested hypotheses get none
            if h["id"] in deeper or any(t["status"] in ("queued", "starting", "running", "review", "revising")
                                        for t in active.get(h["id"], [])):
                continue
            ask = f"{d.get('title', '')} {d.get('brief', '')}"
            own = {x.split("_")[0] for x in (self.brief.get("datasets") or [])}
            names = set(self.holdout or []) | {x.split("_")[0] for x in (self.holdout or [])} - own
            if self.holdout and any(x.lower() in ask.lower() for x in names):
                # the explorers cannot see held-out data: a deeper test that needs it would only end "blocked"
                self.log(f"lab meeting: deeper test on {h['id']} skipped (it needs held-out data: {ask[:120]})")
                continue
            extra = (f"This is a DEEPER test requested at the lab meeting: {d.get('title')}\n{d.get('brief')}\n"
                     f"Previous result on this hypothesis: {h['result'][:1200]}"
                     + (f"\nHeld-out confirmation already run by code (you cannot see that data and must not try to "
                        f"re-run it): {h.get('confirmation')[:800]}\nYour job is the analysis above on the datasets you "
                        f"CAN use; record what it shows about the mechanism." if h.get("confirmation") else ""))
            self._add_hypothesis_task({**h, "priority": d.get("priority", h["priority"])}, h["id"], extra=extra)
            deeper.append(h["id"])
        from . import confirm as _confirm

        backlog = len([t for t in self.record.tasks(self.cid, status="queued") if t["stage"] == PARALLEL_STAGE])
        for nh in (out.get("new") or [])[:5]:
            if not nh.get("statement"):
                continue
            spec, _ = _confirm.validate(nh.get("confirm"))
            if spec and self.confirm_sets and spec["dataset"] not in self.confirm_sets:
                spec = None
            hid = self.record.add_hypothesis(self.cid, {**nh, "confirm": spec or {}, "origin": "lab meeting",
                                                        "status": "reserve"})
            new.append(hid)
        if new:
            self._safe(self.repair_confirm_specs)  # pre-registration for the new ones too, before any result
        for hid in new:
            queue_now = self._program_budget_left() > 0 and backlog < 2 * HARNESS.concurrency
            if queue_now:  # otherwise it waits in the reserve pool instead of being thrown away
                h_new = self.record.hypotheses(self.cid, id=hid)[0]
                if self._add_hypothesis_task(h_new, hid):
                    backlog += 1
        self._meetings = getattr(self, "_meetings", 0) + 1
        self.record.kb_add(self.cid, None, "decision", f"Lab meeting {self._meetings}",
                           f"{out.get('notes', '')}\nClosed: {closed}; deeper tests: {deeper}; promoted from reserve: "
                           f"{promoted}; new hypotheses: {new}",
                           ["programme", "pi", "meeting"])
        self.log(f"PI lab meeting {self._meetings}: closed {closed}, deeper {deeper}, promoted {promoted}, new {new}")

    def _queue_referees(self):
        """Every claimed supported result (hypothesis or new_signal family) gets one independent referee."""
        have = {t["meta"].get("target") for t in self.record.tasks(self.cid, kind="referee")}
        if self._program_budget_left() <= 0 and not self.mock:
            pass  # referees are never skipped for budget: unreviewed claims cannot be reported
        refs: dict[str, list] = {}
        for t in self.record.tasks(self.cid, kind="referee"):
            refs.setdefault(t["meta"].get("target"), []).append(t)
        for h in self.record.hypotheses(self.cid):
            # a stronger claim recorded after a review (e.g. a deeper test now says "new") gets a second referee
            rv = {}
            try:
                rv = json.loads(h["review"]) if h["review"] else {}
            except Exception:  # noqa: BLE001
                pass
            prev = refs.get(h["id"], [])
            last_ref = max([t["ended"] or 0 for t in prev] or [0])
            re_review = (h["status"] == "supported_new" and rv and rv.get("decision") != "upheld" and len(prev) < 2
                         and all(t["status"] in ("done", "stalled", "failed") for t in prev)
                         and (h["updated"] or 0) > last_ref)
            if re_review:
                have.discard(h["id"])
            if h["status"] in ("supported", "supported_new") and (not h["review"] or re_review) and h["id"] not in have:
                brief = prompts.REFEREE_TASK.format(target=h["id"], claim=f"{h['statement']}\nResult: {h['result']}",
                                                    evidence=h["evidence"][:4000], novelty=h["novelty"] or "(none claimed)")
                self.record.add_task(self.cid, title=f"Referee: {h['id']} {h['title']}"[:200], brief=brief,
                                     stage=PARALLEL_STAGE, kind="referee", origin="harness", depth=0,
                                     meta={"target": h["id"], "priority": 0.95})
                self.log(f"referee queued for {h['id']} ({h['status']})")
        for c in self.record.candidates(self.cid, promoted=1):
            if c["outcome"] == "new_signal" and c["family"] not in have and "[referee" not in (c["note"] or ""):
                brief = prompts.REFEREE_TASK.format(target=c["family"], claim=f"{c['family']}: {c['cls']} in "
                                                    f"{c['variable']} ({(c['filters'] or {}).get('place', '')})",
                                                    evidence=(c["note"] or "")[:4000], novelty=(c["note"] or "")[-1500:])
                self.record.add_task(self.cid, title=f"Referee: {c['family']}"[:200], brief=brief, stage=PARALLEL_STAGE,
                                     kind="referee", origin="harness", depth=0,
                                     meta={"target": c["family"], "family": c["family"], "priority": 0.95})
                self.log(f"referee queued for family {c['family']}")

    def _precompute_evidence(self):
        """Run the evidence engine once, before the census agent starts, so its apply_filters call is instant
        and the PI sees the confirmed signals at the next lab meeting."""
        from .tools import science
        from .tools.base import ToolContext

        if self.mock or (self.cdir / "evidence.json").exists():
            return
        t0 = time.time()
        try:
            ctx = ToolContext(self.record, self.cid, None, None, self.cdir, self.cdir, self.mock)
            r = science.apply_filters(ctx)
            head = r.text.split("\n")[1] if "\n" in r.text else r.text[:200]
            self.log(f"evidence engine: {head} ({time.time() - t0:.0f} s)")
            self.record.kb_add(self.cid, None, "method", "Evidence engine (automatic independent checks)", r.text[:6000],
                               ["evidence", "census"])
        except Exception as e:  # noqa: BLE001
            self.log(f"evidence engine failed ({type(e).__name__}: {str(e)[:200]})")
            self.record.event(self.cid, None, None, "harness", "error", traceback.format_exc()[-3000:])

    # ---------------------------------------------------------------------- stages
    # ---------------------------------------------------------------------- resume support
    @classmethod
    def attach(cls, cid: str, db: str | Path | None = None) -> "Campaign":
        """Reopen an existing campaign after a crash, reboot or lost connection: interrupted tasks go back to the
        queue and the pipeline continues from the first unfinished stage, keeping all completed work."""
        self = cls.__new__(cls)
        self.db_path = Path(db) if db else RUNS_DIR / "synaptic.db"
        self.record = Record(self.db_path)
        row = self.record.one("SELECT * FROM campaigns WHERE id=?", (cid,))
        if not row:
            raise SystemExit(f"No campaign {cid} in {self.db_path}")
        self.cid, self.brief, self.mock = cid, json.loads(row["brief"]), row["mode"] == "mock"
        self.cdir = Path(row["workdir"])
        self.llm = make_llm(self.mock)
        self.runner = Runner(self.record, self.llm, self.cid, self.cdir, self.mock)
        self.lock = threading.Lock()
        self.plan = json.loads((self.cdir / "plan.json").read_text()) if (self.cdir / "plan.json").exists() else {}
        self.data_index = json.loads((self.cdir / "data_index.json").read_text()) \
            if (self.cdir / "data_index.json").exists() else {}
        n = self.record.x("UPDATE tasks SET status='queued' WHERE campaign_id=? AND status IN "
                          "('running','review','revising','starting')", (cid,)).rowcount
        for st in self.record.q("SELECT key FROM stages WHERE campaign_id=? AND status='running'", (cid,)):
            self.record.set_stage_status(cid, st["key"], "pending")
        (self.cdir / "STOP").unlink(missing_ok=True)
        self.record.x("UPDATE campaigns SET status='running', ended=NULL WHERE id=?", (cid,))
        self.log(f"resumed campaign: {n} interrupted task(s) re-queued")
        self._t_start = time.time()
        lm = self.record.one("SELECT created FROM kb WHERE campaign_id=? AND title LIKE 'Lab meeting%' ORDER BY id DESC "
                             "LIMIT 1", (cid,))
        if lm:  # results already discussed before the restart are not "new" again
            self._last_meeting_ts = lm["created"]
            self._meetings = len(self.record.q("SELECT id FROM kb WHERE campaign_id=? AND title LIKE 'Lab meeting%'", (cid,)))
        return self

    def _has_stage(self, key: str) -> bool:
        """Campaigns created by an older version lack newer stages; resuming them must not insert work midway."""
        return bool(self.record.one("SELECT 1 FROM stages WHERE campaign_id=? AND key=?", (self.cid, key)))

    def _stage_done(self, key: str) -> bool:
        return bool(self.record.one("SELECT 1 FROM stages WHERE campaign_id=? AND key=? AND status='done'",
                                    (self.cid, key)))

    def _ensure_task(self, kind: str, **kw):
        if not self.record.tasks(self.cid, kind=kind):
            self.record.add_task(self.cid, kind=kind, origin="mission", **kw)

    def _survey_after_sweep(self) -> bool:
        """Classify -> census -> deep dives (the survey track). Returns False when a STOP was requested."""
        self._ensure_task("classify", title="Classify anomalies into families", stage="classify",
                          brief=("Call build_families to classify significant cells (trend, shift, variance, "
                                 "seasonality, recent_extreme) and group them into spatially connected families. Look at "
                                 "the field maps, describe the main structures, and flag families that look suspicious. "
                                 f"{self.plan.get('classify_brief', '')}"), meta={})
        if not self._stage_done("classify"):
            self.run_stage("classify")
        if self.stopped():
            return False
        self._ensure_task("census", title="Census: apply pre-registered filters and promote", stage="census",
                          brief=("Call apply_filters (locked thresholds). Review the passing families: read family_info, "
                                 "look at at least the top candidates' raw series, discard obvious duplicates, then promote "
                                 f"the most promising (respect the cap). {self.plan.get('census_brief', '')}"), meta={})
        if not self._stage_done("census"):
            self._precompute_evidence()
            self.run_stage("census")
        if self.stopped():
            return False
        if not self._stage_done("deep_dive"):
            self._queue_deep_dives()
            self.run_stage("deep_dive")
        return not self.stopped()

    # ---------------------------------------------------------------------- stages
    def run(self):
        try:
            b = self.brief
            reg = b["region"]
            self._t_start = time.time()
            if not self._stage_done("plan"):
                self.stage_plan()
            if self._has_stage("screen") and not self._stage_done("screen"):
                self.stage_screen()
            if self._has_stage("learn") and not self._stage_done("learn"):
                if self.focused and not b.get("learn"):
                    self.record.set_stage_status(self.cid, "learn", "done")  # focused briefs opt in to the GPU model
                else:
                    self.start_learn()
            if self._has_stage("sweep") and not self.record.tasks(self.cid, kind="sweep"):
                for v in self.plan["variables"]:
                    get = (f"The whole-Earth table for {v} is already prepared from NASA MERRA-2 (land cells, 2 x 2.5 "
                           f"degrees, {b['start_year']}-{b['end_year']}): {self.data_index.get(v)}. Load it (do not "
                           f"download). Run " if reg.get("global") else
                           f"Download NASA POWER monthly {v} for the box lat {reg['lat_min']}..{reg['lat_max']}, lon "
                           f"{reg['lon_min']}..{reg['lon_max']}, {b['start_year']}-{b['end_year']} (power_region). Run ")
                    self.record.add_task(
                        self.cid, title=f"Sweep {v}", stage="sweep", kind="sweep", origin="mission",
                        brief=(get + 
                               f"synlib.sweep(df, '{v}'{', months=' + str(self.plan.get('season')) if self.plan.get('season') else ''}) "
                               f"on it, save the table as CSV, register it with register_sweep, map at least one statistic, "
                               f"LOOK at the map and at the raw series of the most extreme cell, and report anything odd "
                               f"(gaps, constant runs, edge effects, abrupt jumps). Do NOT classify, filter or promote: later stages do that."
                               + (f"\nPlanner guidance for this stage: {self.plan.get('sweep_brief')}" if self.plan.get('sweep_brief') else "")),
                        meta={"variable": v, "region": reg, "season": self.plan.get("season")})
            if self._has_stage("sweep") and not self._stage_done("sweep"):
                self.run_stage("sweep")
            if self.stopped():
                return self._end("stopped")
            if self._has_stage("classify") and not self._survey_after_sweep():
                return self._end("stopped")
            if self._has_stage("science") and not self._stage_done("science"):
                self.stage_science()
            if self.stopped():
                return self._end("stopped")
            if self._program_on() and not self._stage_done("program"):
                # the research programme finishes its tests, lab meetings and referee reports
                if not self.record.hypotheses(self.cid) and not self._pi_busy():
                    if self.brief.get("time_holdout"):
                        self._safe(self.exploration_sweep)
                    self._safe(self.write_program)
                elif self.record.hypotheses(self.cid):
                    self._safe(self.repair_confirm_specs)  # on resume: fix untestable specs before any result
                self.run_stage(PARALLEL_STAGE)
                if self.stopped():
                    return self._end("stopped")
            if self._has_stage("review") and not self._stage_done("review"):
                self.record.set_stage_status(self.cid, "review", "running")
                self._queue_referees()
                self.run_stage(PARALLEL_STAGE) if [t for t in self.record.tasks(self.cid, status="queued")
                                                   if t["stage"] == PARALLEL_STAGE] else None
                self.audit_verdicts()
                self.record.set_stage_status(self.cid, "review", "done")
            self._drain_all()
            if not self._stage_done("report"):
                self.stage_reports()
            try:
                self.screen_report()
            except Exception as e:  # noqa: BLE001
                self.log(f"screen report skipped: {e}")
            if not self._stage_done("rank"):
                self.record.set_stage_status(self.cid, "rank", "running")
                run_tournament(self.record, self.runner, self.cid)
                self.record.set_stage_status(self.cid, "rank", "done")
            try:
                from .export import export

                page = export(self.record, self.cid)
                self.log(f"results page: {page} (+ NASA_data_issues.md)")
            except Exception as e:  # noqa: BLE001
                self.log(f"results page skipped: {type(e).__name__}: {e}")
            self._end("done")
        except Exception as e:  # noqa: BLE001
            self.record.event(self.cid, None, None, "harness", "error", traceback.format_exc()[-4000:])
            self._end("failed")
            raise

    def verdict_history(self, family: str) -> list[dict]:
        rows = self.record.q("SELECT e.task_id, t.title, t.kind, e.content FROM events e JOIN tasks t ON t.id = e.task_id "
                             "WHERE e.campaign_id=? AND e.type='action' AND e.content LIKE 'set_verdict%' ORDER BY e.id",
                             (self.cid,))
        out = {}
        for r in rows:
            raw = r["content"][len("set_verdict("):]
            raw = raw[:-1] if raw.endswith(")") else raw  # long actions are stored truncated
            try:
                a = _loads(raw)
            except Exception:  # noqa: BLE001
                m = re.search(r'"outcome":\s*"(\w+)"', raw)
                f = re.search(r'"family":\s*"([^"]+)"', raw)
                if not (m and f):
                    continue
                a = {"family": f.group(1), "outcome": m.group(1), "note": raw[:1200]}
            if a.get("family") == family:
                out[r["task_id"]] = {"task": r["title"], "kind": r["kind"], "outcome": a.get("outcome"),
                                     "note": str(a.get("note", ""))[:1200]}  # last call per task counts
        # referee decisions are part of the history (found live: the auditor never saw a referee's downgrade)
        for k in self.record.q("SELECT task_id, title, body FROM kb WHERE campaign_id=? AND kind='review' AND title LIKE ?",
                               (self.cid, f"Referee %: {family}")):
            dec = k["title"].split(":")[0].replace("Referee", "").strip()
            out[f"ref_{k['task_id']}"] = {"task": k["title"], "kind": "referee", "outcome": f"referee {dec}",
                                         "note": k["body"][:1500]}
        return list(out.values())

    def audit_verdicts(self):
        """Skeptical final decision on every promoted family before reporting: agents over-claim novelty, and
        follow-ups can overwrite each other's verdicts, so the auditor reads the whole verdict history."""
        for c in self.record.candidates(self.cid):
            if not c.get("promoted") or not c["outcome"] or "[audit" in (c["note"] or ""):
                continue
            hist = self.verdict_history(c["family"])
            if c["outcome"] != "new_signal" and len({h["outcome"] for h in hist}) <= 1:
                continue  # nothing claimed and nothing contested
            src = self.record.one("SELECT id, summary FROM tasks WHERE campaign_id=? AND meta LIKE ? AND kind='deep_dive' "
                                  "AND status IN ('done','stalled') ORDER BY created",
                                  (self.cid, f'%"family": "{c["family"]}"%'))
            fam = {k: c[k] for k in ("family", "variable", "cls", "n_cells", "lat", "lon", "onset", "effect", "qvalue")}
            note = (c["note"] or "") + "\n\nVERDICT HISTORY:\n" + "\n".join(
                f"- [{h['kind']}] {h['task']}: {h['outcome']} | {h['note']}" for h in hist)
            a = self.runner.audit(fam, note[:12000], (src or {}).get("summary") or "", (src or {}).get("id"))
            out = a.get("outcome")
            if out not in ("new_signal", "known_phenomenon", "artifact", "not_robust"):
                continue
            new_note = (c["note"] or "") + f" [audit: {out}. {a.get('reason', '')}" + \
                (f" Claim: {a['novel_claim']}" if a.get("novel_claim") else "") + "]"
            refused = self.record.update_candidate(self.cid, c["family"], actor="auditor", outcome=out, note=new_note)
            self.log(f"audit {c['family']}: {c['outcome']} -> {out}" + (f" REFUSED ({refused})" if refused else ""))

    def stage_science(self):
        """Turn verified real signals into science: drivers, mechanism, predictability, impacts."""
        self.audit_verdicts()
        th = getattr(self, "_learn_thread", None)
        if th is not None and th.is_alive():
            self.log("science: waiting for the GPU model")
            th.join(timeout=3600)
        cands = [c for c in self.record.candidates(self.cid) if c.get("promoted")
                 and c["outcome"] in ("new_signal", "known_phenomenon")]
        cands += self._coupling_leads()
        if len(cands) < 2:  # few real promoted signals: take the strongest families the screen left untouched
            extra = [c for c in self.record.candidates(self.cid, passed=1)
                     if not c.get("promoted") and c["outcome"] is None]
            extra.sort(key=lambda c: (-int(c["n_cells"] or 0), abs(c["effect"] or 0) * -1))
            cands += extra[: 2 - len(cands)]
        have = {t["meta"].get("family") for t in self.record.tasks(self.cid, kind="science")}
        for c in cands[: HARNESS.max_promoted]:
            if c["family"] in have:
                continue
            if c["cls"] == "coupling":
                what = (c["note"] + " Test this changed relationship directly (sliding-window and partial correlations "
                        "with climate_index on trusted cells, independent rain/temperature data), explain it, and say "
                        "whether it is new. Per-cell numbers: learn/importance.csv")
            else:
                what = (f"{c['cls']} in {c['variable']} over {c['n_cells']} cells near ({c['lat']:.2f}, {c['lon']:.2f})"
                        + (f", onset ~{c['onset']}" if c["onset"] else "") + f", standardized effect {c['effect']:.2f}")
            self.record.add_task(self.cid, title=f"Science: {c['family']}", stage="science", kind="science",
                                 origin="mission", brief=prompts.SCIENCE_TASK.format(
                                     family=c["family"], what=what, outcome=c["outcome"] or "passed the filters"),
                                 meta={"family": c["family"], "variable": c["variable"], "cls": c["cls"]})
        self.run_stage("science")

    def stage_reports(self):
        self.audit_verdicts()
        # science first: signals with science-stage results, then new signals, then proven data artifacts
        sci = {t["meta"].get("family"): t["id"] for t in self.record.tasks(self.cid, kind="science") if t["status"] == "done"}
        allc = self.record.candidates(self.cid)
        fams = [c for c in allc if c["family"] in sci]
        fams += [c for c in allc if c.get("promoted") and c["outcome"] == "new_signal" and c not in fams]
        fams += [c for c in allc if c.get("promoted") and c["outcome"] == "artifact" and c not in fams]
        fams = fams[:HARNESS.max_promoted]
        if len(fams) < 2:  # also document rediscovered known phenomena: they validate the method
            fams += [c for c in allc if c.get("promoted") and c["outcome"] == "known_phenomenon"
                     and c not in fams][: 2 - len(fams)]
        have = {t["meta"].get("family") for t in self.record.tasks(self.cid, kind="report")}
        for c in fams:
            if c["family"] in have:
                continue
            src = self.record.one("SELECT id FROM tasks WHERE campaign_id=? AND meta LIKE ? AND kind IN "
                                  "('deep_dive','science','followup') AND status='done' ORDER BY created",
                                  (self.cid, f'%"family": "{c["family"]}"%'))
            if not src:
                continue
            sci_ids = [t["id"] for t in self.record.tasks(self.cid) if t["stage"] == "science"
                       and t["status"] == "done" and t["meta"].get("family") == c["family"]][:4]
            self.record.add_task(self.cid, title=f"Report: {c['family']}", stage="report", kind="report",
                                 origin="mission", brief=prompts.REPORT_TASK.format(
                                     family=c["family"], outcome=c["outcome"], source_task=src["id"],
                                     science_tasks=(" and the science task(s) " + ", ".join(sci_ids)) if sci_ids else ""),
                                 meta={"family": c["family"], "outcome": c["outcome"], "source_task": src["id"]})
        # research-programme results: reviewed supported claims first, then the clearest refutations (they are
        # results too), at most max_promoted dossiers
        rank = {"supported_new": 0, "supported": 1, "known": 2, "refuted": 3, "not_shown": 4}
        hyps = [h for h in self.record.hypotheses(self.cid) if h["status"] in rank and h["result"]]
        hyps.sort(key=lambda h: (rank[h["status"]], -h["priority"]))
        refs = {t["meta"].get("target"): t["id"] for t in self.record.tasks(self.cid, kind="referee") if t["status"] == "done"}
        for h in hyps[:HARNESS.max_promoted]:
            if h["id"] in have or not h["task_ids"]:
                continue
            done_t = [t for t in h["task_ids"] if (self.record.task(t) or {}).get("status") in ("done", "stalled")]
            if not done_t:
                continue
            self.record.add_task(self.cid, title=f"Dossier: {h['id']} {h['title']}"[:200], stage="report", kind="report",
                                 origin="mission", brief=prompts.HYPOTHESIS_REPORT_TASK.format(
                                     hid=h["id"], status=h["status"], statement=h["statement"],
                                     tasks=", ".join(done_t), referees=refs.get(h["id"], "(none)")),
                                 meta={"family": h["id"], "outcome": h["status"], "source_task": done_t[-1],
                                       "hypothesis": h["id"]})
        self.run_stage("report")
        filed = {r["task_id"] for r in self.record.q("SELECT task_id FROM reports WHERE campaign_id=?", (self.cid,))}
        for t in self.record.tasks(self.cid, kind="report"):
            if t["status"] not in ("done", "stalled") or not t["summary"] or t["id"] in filed:
                continue
            if t["status"] == "stalled" and t["summary"].startswith("Step budget exhausted"):
                continue
            body = t["summary"]
            if t["status"] == "stalled":  # the supervisor never accepted it: file it as a draft, objections attached
                last = self.record.one("SELECT content FROM events WHERE task_id=? AND role='supervisor' AND type='review' "
                                       "ORDER BY id DESC LIMIT 1", (t["id"],))
                body += ("\n\n---\n**DRAFT: not accepted by the supervisor after the maximum number of revisions.** "
                         "Last objections:\n\n" + ((last or {}).get("content") or "(none recorded)")[:3000])
            ed = self.runner.edit(body, t["id"])
            title = ed.get("title") or t["title"]
            first = body.strip().splitlines()[0] if body.strip() else ""
            if first.startswith("# "):
                title = first[2:].strip()
            if t["status"] == "stalled":
                title = "[Draft] " + title
            self.record.add_report(self.cid, t["id"], t["meta"].get("family", ""), title, body,
                                   t["meta"].get("figures", []))
            self.record.event(self.cid, None, t["id"], "editor", "review", f"{ed.get('decision')}: {ed.get('feedback', '')}")

    def screen_report(self):
        """File the automatic data screen as its own report: the trust map is a deliverable for data users."""
        import pandas as pd

        f = self.cdir / "trust.csv"
        if not f.exists() or self.record.one("SELECT 1 FROM reports WHERE campaign_id=? AND family='DATA_SCREEN'",
                                             (self.cid,)):
            return
        t = pd.read_csv(f)
        t = t[t.kind != "known_caveat"]
        if t.empty:
            return
        by = t.groupby(["variable", "kind"]).size().rename("cell-issues").reset_index()
        brk = (t[t.kind == "break"].groupby(["variable", "year_from"]).size().sort_values(ascending=False)
               .head(8).rename("cells").reset_index())
        bad = (t[t.kind.isin(["bad_year", "bad_month", "known_period"])].groupby(["variable", "kind", "year_from"])
               .size().sort_values(ascending=False).head(8).rename("cells").reset_index())
        closed = [c for c in self.record.candidates(self.cid) if "[screened automatically]" in (c["note"] or "")]
        body = (f"# Automatic data screen: where the record cannot be trusted\n\n**Summary** Before any agent ran, "
                f"plain statistics compared NASA POWER cell by cell with independent references (GPCP satellite-gauge "
                f"rain, Berkeley Earth station temperature), searched for gap fills (zero runs, frozen values) and "
                f"suppressed variability, and added the known-issues registry. {t[['lat', 'lon']].drop_duplicates().shape[0]} "
                f"grid cells carry at least one issue; {len(closed)} anomaly families were closed as data artifacts "
                f"without spending agent time.\n\n## Issues by variable and kind\n\n{by.to_markdown(index=False)}\n\n"
                f"## Most common break years\n\n{brk.to_markdown(index=False) if len(brk) else '(none)'}\n\n"
                f"## Most widespread bad years or months\n\n{bad.to_markdown(index=False) if len(bad) else '(none)'}\n\n"
                f"## Families closed by the screen\n\n" + ("\n".join(f"- {c['family']}: {c['note'][:220]}" for c in closed[:20])
                                                             or "(none)") +
                "\n\n## Limitations\n\nGPCP is a 2.5-degree product and Berkeley Earth 1 degree: single-cell issues near "
                "coasts or mountains can be resolution effects. Breaks are flagged at p < 0.01 with a 20% (rain) or "
                "0.5 C (temperature) step. Full table: trust.csv; map: figures/trust_map.png.")
        figs = [str(self.cdir / "figures" / "trust_map.png")] if (self.cdir / "figures" / "trust_map.png").exists() else []
        self.record.add_report(self.cid, None, "DATA_SCREEN", "Automatic data screen: where the record cannot be trusted",
                               body, figs)

    def _end(self, status: str):
        self.record.set_campaign_status(self.cid, status)
        s = self.record.stats(self.cid)
        self.log(f"campaign {status}: {s['sessions']} sessions, {s['agent_hours']} agent-hours, "
                 f"{s['tokens_in'] + s['tokens_out']:,} tokens")
