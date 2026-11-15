"""Pipeline tools: the deterministic parts of the paper's survey (profiles -> search -> classify ->
census with pre-registered promotion filters) exposed to agents, plus the knowledge base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import synlib
from .. import screen
from .base import ToolContext, ToolResult

DEFAULT_THRESHOLDS = {
    "q_max": (0.05, "Benjamini-Hochberg FDR across the whole field (seasonality cells only)"),
    "p_form": (0.05, "cluster-forming p per cell (carry-over adjusted); families must then beat surrogate fields"),
    "min_cells": (3, "a real climate signal is spatially coherent; single cells are noise or artifacts"),
    "min_effect": (0.8, "standardized size for trend/shift (change divided by interannual SD)"),
    "min_phase_days": (10, "seasonal-cycle shifts smaller than ~10 days are within sampling noise"),
    "min_log_var": (0.4, "log variance ratio (~1.5x change in variability)"),
    "min_recent_z": (2.5, "recent-years anomaly t statistic"),
    "min_corroboration": (0.3, "fraction of cells where a physically linked variable co-changes"),
    "transition_window": (3, "change-point dates are uncertain by a few years; flag breaks within this many "
                          "years of a known data-production change"),
}


def _sweep_dir(ctx: ToolContext) -> Path:
    d = ctx.campaign_dir / "sweeps"
    d.mkdir(exist_ok=True)
    return d


def _trust(ctx: ToolContext) -> pd.DataFrame | None:
    f = ctx.campaign_dir / "trust.csv"
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return None


def _end_year(ctx: ToolContext) -> int:
    camp = ctx.record.one("SELECT brief FROM campaigns WHERE id=?", (ctx.campaign_id,)) if ctx.record else None
    try:
        return int(json.loads(camp["brief"]).get("end_year", 2025))
    except Exception:  # noqa: BLE001
        return 2025


def _load_sweeps(ctx: ToolContext) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(_sweep_dir(ctx).glob("*.csv"))}


def register_sweep(ctx: ToolContext, path: str, variable: str) -> ToolResult:
    """Register the output of synlib.sweep(...).to_csv(path) for one variable."""
    df = pd.read_csv(ctx.resolve(path))
    need = {"lat", "lon", "mk_p", "cp_p", "sen_per_decade"}
    if not need <= set(df.columns):
        return ToolResult(f"Not a sweep table; missing {sorted(need - set(df.columns))}. "
                          "Produce it with synlib.sweep(df, variable).to_csv(path, index=False).", is_error=True)
    df.to_csv(_sweep_dir(ctx) / f"{variable.upper()}.csv", index=False)
    th = ctx.record.thresholds(ctx.campaign_id)
    q = th.get("q_max", 0.05)
    sig = {c: int((df[c] <= q).sum()) for c in ("mk_q", "cp_q", "var_q", "phase_q", "recent_q") if c in df}
    ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "method", f"Sweep registered: {variable}",
                      f"{len(df)} cells; significant cells at q<={q}: {sig}", [variable, "sweep"])
    return ToolResult(f"Registered sweep for {variable}: {len(df)} cells. Significant at q<={q}: {sig}")


def build_families(ctx: ToolContext) -> ToolResult:
    """Classify significant cells of every registered sweep and group them into spatially connected
    anomaly families. Writes families.csv, a map per variable, and the candidate rows of the record."""
    sweeps = _load_sweeps(ctx)
    if not sweeps:
        return ToolResult("No sweeps registered yet.", is_error=True)
    th = ctx.record.thresholds(ctx.campaign_id)
    # cluster inference: cells join candidate families at their own carry-over-adjusted p <= p_form; the family's
    # size is then tested against surrogate fields at the census (per-cell FDR had no power for regional signals
    # and treated neighbouring cells as independent)
    cells = pd.concat([synlib.classify_cells(sw, th.get("q_max", 0.05), th.get("p_form", 0.05))
                       for sw in sweeps.values()], ignore_index=True)
    fams = synlib.families(cells) if not cells.empty else pd.DataFrame()
    out = ctx.campaign_dir / "families.csv"
    fams.to_csv(out, index=False)
    figs = []
    for var, sw in sweeps.items():
        col = "sen_per_decade"
        f = ctx.campaign_dir / "figures" / f"field_{var}_trend.png"
        f.parent.mkdir(exist_ok=True)
        synlib.plot_field(str(f), sw, col, f"{var}: Sen slope per decade", q_column="mk_q", q_max=th.get("q_max", 0.05))
        figs.append(str(f))
        if "shift_effect" in sw:
            f2 = ctx.campaign_dir / "figures" / f"field_{var}_shift.png"
            synlib.plot_field(str(f2), sw, "shift_effect", f"{var}: step-change effect (break year varies)",
                              q_column="cp_q", q_max=th.get("q_max", 0.05))
            figs.append(str(f2))
    for _, r in fams.iterrows():
        ctx.record.upsert_candidate(ctx.campaign_id, {
            "id": r["family"], "task_id": ctx.task_id, "family": r["family"], "variable": r["variable"],
            "cls": r["cls"], "lat": r["lat"], "lon": r["lon"], "n_cells": int(r["n_cells"]),
            "onset": None if np.isnan(r["onset"]) else int(r["onset"]), "effect": r["effect"], "qvalue": r["q"],
            "stats": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()},
            "filters": {}, "passed": 0, "promoted": 0, "outcome": None})
    if fams.empty:
        return ToolResult("0 anomaly families: no cell passed the field significance threshold "
                          f"(q <= {th.get('q_max', 0.05)}). That is a valid result.\nField maps (look at them):\n"
                          + "\n".join(figs), files=figs)
    view = fams[["family", "n_cells", "lat", "lon", "onset", "effect", "q"]].head(40)
    by = fams.groupby(["variable", "cls"]).size().to_string()
    return ToolResult(f"{len(fams)} anomaly families saved to {out.name}.\nBy variable/class:\n{by}\n\n"
                      f"Largest families:\n{view.round(3).to_string(index=False)}\n\nField maps (look at them):\n"
                      + "\n".join(figs), files=figs)


def _evidence(ctx: ToolContext):
    """The automatic evidence engine for this campaign (independent data, robustness, field significance)."""
    from .. import verify

    camp = ctx.record.one("SELECT brief FROM campaigns WHERE id=?", (ctx.campaign_id,))
    brief = json.loads(camp["brief"]) if camp else {}
    di_path = ctx.campaign_dir / "data_index.json"
    di = json.loads(di_path.read_text()) if di_path.exists() else {}
    plan_path = ctx.campaign_dir / "plan.json"
    season = json.loads(plan_path.read_text()).get("season") if plan_path.exists() else None
    return verify.Evidence(ctx.campaign_dir, di, int(brief.get("start_year", 1981)), int(brief.get("end_year", 2025)),
                           bool(brief.get("region", {}).get("global")), months=season or None,
                           log=lambda m: ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "caveat", "Evidence engine", m,
                                                           ["evidence"]) if ctx.record else None)


def apply_filters(ctx: ToolContext) -> ToolResult:
    """Score every family against the PRE-REGISTERED thresholds (locked before any analysis ran), then run the
    automatic evidence engine on the families that pass them: independent recomputation (GPCP, IMERG, Berkeley
    Earth, GLDAS), the robustness suite, field significance, place names, and merging of families on the same cells.
    """
    from .. import verify

    th = ctx.record.thresholds(ctx.campaign_id)
    fams_path = ctx.campaign_dir / "families.csv"
    if not fams_path.exists():
        return ToolResult("Run build_families first.", is_error=True)
    try:
        fams = pd.read_csv(fams_path)
    except pd.errors.EmptyDataError:
        fams = pd.DataFrame()
    if fams.empty:
        return ToolResult(f"Filters applied with locked thresholds {th}.\n0 of 0 families pass all four.\n"
                          "(no families: the sweep found no field-significant signal; finish and say so)")
    sweeps = _load_sweeps(ctx)
    trust, end_year = _trust(ctx), _end_year(ctx)
    ev = None if ctx.mock else _evidence(ctx)
    counts = {"screened": 0, "contradicted": 0, "not_robust": 0, "field": 0}
    passing, rows = [], []
    for _, r in fams.iterrows():
        fam = r.to_dict()
        eff = abs(fam["effect"])
        size_rule = {"trend": th["min_effect"], "shift": th["min_effect"], "seasonality": th["min_phase_days"],
                     "variance": th["min_log_var"], "recent_extreme": th["min_recent_z"]}[fam["cls"]]
        corr = synlib.corroboration(fam, sweeps)
        best_corr = max(corr.values()) if corr else 0.0
        flag = synlib.source_transition_flag(fam["variable"], fam["onset"], int(th["transition_window"]))
        bad_frac, why = screen.family_issue(fam, trust, end_year) if trust is not None else (0.0, "")
        f = {"significance": bool(fam.get("p", fam["q"]) <= th.get("p_form", 0.05) if fam["cls"] != "seasonality"
                                  else fam["q"] <= th["q_max"]),
             "coherence": bool(fam["n_cells"] >= th["min_cells"]),
             "magnitude": bool(eff >= size_rule),
             "corroboration": bool(best_corr >= th["min_corroboration"]),
             "data_quality": bool(bad_frac < 0.5),
             "data_issue_fraction": round(bad_frac, 2), "data_issue": why,
             "corroboration_detail": {k: round(v, 2) for k, v in corr.items()},
             "artifact_flag": flag}
        basic = all(f[k] for k in ("significance", "coherence", "magnitude", "data_quality"))
        outcome, note, sc = None, None, None
        if not f["data_quality"]:  # explained by untrusted data: closed without spending agent time
            outcome, note = "artifact", f"[screened automatically] {bad_frac:.0%} of cells sit in untrusted data: {why}"
            counts["screened"] += 1
        passed = False
        if basic and ev is not None:
            e = ev.family(fam)
            fs = ev.field_for(fam)
            rb = e.get("robust") or {}
            f.update({"place": e.get("place"), "independent": e.get("status"), "support": e.get("support"),
                      "independent_detail": [{k: c.get(k) for k in ("reference", "kind", "verdict", "years", "note")
                                              if c.get(k) is not None} for c in e.get("independent", [])],
                      "robustness": f"{rb.get('passed', 0)}/{rb.get('n_checks', 0)}",
                      "robustness_detail": {k: v for k, v in rb.items() if isinstance(v, (bool, list))},
                      "field_significance": bool(fs.get("field_significant", False)),
                      "field_detail": {k: fs.get(k) for k in ("observed", "null_p95", "p_field", "alpha", "note")
                                       if k in fs}})
            robust_ok = rb.get("passed", 0) >= rb.get("n_checks", 1) - 1
            contradicted = e.get("status") == "contradicted_by_observations"
            supported = (e.get("support") or 0) >= 0.6
            f["robust"] = bool(robust_ok)
            passed = (basic and f["field_significance"] and robust_ok and not contradicted
                      and (f["corroboration"] or supported))
            if contradicted:
                outcome = "artifact"
                det = "; ".join(f"{c['reference']}: {c['verdict']}" + (f" ({c['note']})" if c.get("note") else "")
                                for c in e.get("independent", []))
                note = f"[independent check] the signal is in MERRA-2 only. {det}"
                counts["contradicted"] += 1
            elif rb and rb.get("passed", 0) <= rb.get("n_checks", 0) - 2:
                outcome = "not_robust"
                note = (f"[robustness suite] {f['robustness']} checks passed: "
                        + ", ".join(k for k, v in rb.items() if v is False))
                counts["not_robust"] += 1
            elif not f["field_significance"]:
                counts["field"] += 1
            sc = verify.score(fam, e, th)
        elif basic and ev is None:  # mock: original rule
            passed = basic and f["corroboration"]
        f["score"] = sc
        ctx.record.update_candidate(ctx.campaign_id, fam["family"], filters=f, passed=int(passed))
        if outcome:
            ctx.record.update_candidate(ctx.campaign_id, fam["family"], actor="engine", outcome=outcome, note=note)
        if passed:
            passing.append({**fam, "score": sc or 0.0})
        rows.append({"family": fam["family"], "cells": fam["n_cells"], "|effect|": round(eff, 2),
                     "independent": (f.get("independent") or "-")[:32], "robust": f.get("robustness", "-"),
                     "score": sc if sc is not None else "-", "PASS": "yes" if passed else "-",
                     "closed": outcome or "", "place": (f.get("place") or "")[:70]})
    # one signal, one deep dive: families on the same cells merge under the best-scoring one
    groups = verify.merge_groups(passing) if passing else {}
    members: dict[str, list[str]] = {}
    for m, rep in groups.items():
        if m != rep:
            members.setdefault(rep, []).append(m)
            c = ctx.record.candidates(ctx.campaign_id, id=m)[0]
            ctx.record.update_candidate(ctx.campaign_id, m, filters={**c["filters"], "merged_into": rep})
    for rep, ms in members.items():
        c = ctx.record.candidates(ctx.campaign_id, id=rep)[0]
        ctx.record.update_candidate(ctx.campaign_id, rep, filters={**c["filters"], "merged_members": ms})
    for row in rows:
        if row["family"] in groups and groups[row["family"]] != row["family"]:
            row["PASS"] = f"merged->{groups[row['family']]}"
    if ev is not None:
        ev.save()
    t = pd.DataFrame(rows)
    t["_s"] = pd.to_numeric(t["score"], errors="coerce").fillna(-1)
    t = t.sort_values(["_s"], ascending=False).drop(columns="_s")
    n_pass = int((t["PASS"] == "yes").sum())
    top = t[t["PASS"] == "yes"].head(25)
    rest = t[(t["PASS"] != "yes")].head(15)
    return ToolResult(
        f"Filters applied with locked thresholds {th}, plus the automatic evidence engine (same statistic recomputed "
        f"in independent data: GPCP/IMERG rain, Berkeley Earth temperature, GLDAS soil moisture/evaporation/humidity; "
        f"robustness suite; field significance with spatial correlation kept; merging of families on the same cells)."
        f"\n{n_pass} distinct signals pass of {len(t)} families. Closed automatically, do not investigate them: "
        f"{counts['screened']} by the data screen, {counts['contradicted']} contradicted by independent observations, "
        f"{counts['not_robust']} failing the robustness suite; {counts['field']} lack field significance.\n\n"
        f"PASSING, ranked by score (independent support 45%, robustness 30%, size 15%, coherence 10%; documented "
        f"warming x0.5). Promote from the top unless you see a reason in family_info:\n"
        f"{top.to_string(index=False) if len(top) else '(none)'}\n\nOther families (sample):\n"
        f"{rest.to_string(index=False) if len(rest) else '(none)'}\n\nPlace names come from GeoNames; use them in "
        "your write-up instead of guessing regions.")


def promote(ctx: ToolContext, families: list[str], rationale: str) -> ToolResult:
    """Promote filter-passing families to deep dives (one worker each)."""
    from ..config import HARNESS

    th = ctx.record.thresholds(ctx.campaign_id)
    cap = int(th.get("max_promoted", HARNESS.max_promoted))
    ok, bad = [], []
    for fam in families:
        c = ctx.record.candidates(ctx.campaign_id, id=fam)
        if not c:
            bad.append(f"{fam} (unknown)")
        elif not c[0]["passed"]:
            bad.append(f"{fam} (did not pass the pre-registered filters and evidence checks)")
        elif (c[0]["filters"] or {}).get("merged_into"):
            bad.append(f"{fam} (same cells as {c[0]['filters']['merged_into']}: promote that one, it covers both)")
        elif c[0]["outcome"] in ("artifact", "not_robust"):
            bad.append(f"{fam} (closed automatically: {c[0]['outcome']})")
        else:
            ok.append(fam)
    ok = ok[:cap]
    for fam in ok:
        ctx.record.update_candidate(ctx.campaign_id, fam, promoted=1)
    ctx.state.setdefault("promoted", []).extend(ok)
    ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "decision", "Promotion to deep dives",
                      f"Promoted {ok}. Rationale: {rationale}. Refused: {bad}", ["census"])
    return ToolResult(f"Promoted {len(ok)}: {ok}" + (f"\nRefused: {bad}" if bad else "") +
                      f"\n(cap {cap}). Deep-dive tasks are created when this task is accepted.")


DOCUMENTED_BY_DEFAULT = {("T2M", "trend"), ("T2M_MAX", "trend"), ("T2M_MIN", "trend"), ("T2M", "recent_extreme"),
                         ("T2M_MAX", "recent_extreme"), ("T2M_MIN", "recent_extreme")}


def set_verdict(ctx: ToolContext, family: str, outcome: str, note: str, novelty_basis: str = "") -> ToolResult:
    """Deep-dive verdict. outcome: new_signal | known_phenomenon | artifact | not_robust.
    new_signal is gated (the paper's lesson: agents over-claim novelty): it needs at least two literature searches
    in this task and a novelty_basis naming the specific feature the closest papers do not report. Near-surface
    warming trends and recent heat extremes are documented by default, so those families get one extra check."""
    allowed = {"new_signal", "known_phenomenon", "artifact", "not_robust"}
    if outcome not in allowed:
        return ToolResult(f"outcome must be one of {sorted(allowed)}", is_error=True)
    cand = ctx.record.candidates(ctx.campaign_id, id=family)
    if not cand:
        return ToolResult(f"Unknown family {family}", is_error=True)
    if outcome == "new_signal":
        queries = ctx.state.get("lit_queries", [])
        if len(queries) < 2:
            return ToolResult("new_signal refused: run at least two precise literature_search queries first "
                              "(skill novelty-test).", is_error=True)
        if len(novelty_basis.strip()) < 40:
            return ToolResult("new_signal refused: give novelty_basis = the specific feature (where / when / season / "
                              "rate / decoupling) that the closest papers you found do NOT report, citing them by "
                              "title and year. If you cannot name one, the outcome is known_phenomenon.", is_error=True)
        c = cand[0]
        if (c["variable"], c["cls"]) in DOCUMENTED_BY_DEFAULT and not ctx.state.get("documented_ack"):
            ctx.state["documented_ack"] = True
            return ToolResult(
                f"Check before recording: {c['variable']} {c['cls']} is documented by default. Warming of near-surface "
                "temperature and more frequent heat extremes over West Africa, the Sahel and most land regions are "
                "established (IPCC AR6 WGI chapter 11 and Atlas; many regional station studies). 'It warmed, "
                "significantly, everywhere' is known_phenomenon even if your searches returned nothing relevant. "
                "new_signal needs a feature beyond that, e.g. an abrupt step at a dated year that an independent "
                "product does not show, a season or sub-region that departs from the regional pattern, or a "
                "decoupling between variables. If novelty_basis names such a feature, call set_verdict again with "
                "the same arguments; otherwise record known_phenomenon.", is_error=True)
        weak = set(ctx.state.get("lit_sources", [])) <= {"crossref", "europepmc", "doaj", "cache"}
        if weak:
            note = (note + " [Novelty provisional: literature coverage was weak (" +
                    ", ".join(sorted(set(ctx.state.get('lit_sources', [])))) + " only)]")
        note = note + f" | Novelty basis: {novelty_basis.strip()}"
    import re as _re

    keep = _re.findall(r"\[(?:referee|audit|operator)[^\]]*\]", cand[0]["note"] or "")
    if keep:  # a later verdict must not erase what a referee or the auditor found
        note = note + " " + " ".join(keep)
    task = ctx.record.task(ctx.task_id) if ctx.task_id else None
    actor = "deep_dive" if task and task["kind"] == "deep_dive" else (task["kind"] if task else "agent")
    refused = ctx.record.update_candidate(ctx.campaign_id, family, actor=actor, outcome=outcome, note=note)
    if refused:
        return ToolResult(f"Verdict NOT changed: {refused}. Your note was added to the family. A later check can only "
                          "lower a verdict, never raise it.", is_error=True)
    ctx.state["verdict"] = {"family": family, "outcome": outcome, "note": note}
    return ToolResult(f"Verdict recorded for {family}: {outcome}")


def family_info(ctx: ToolContext, family: str) -> ToolResult:
    c = ctx.record.candidates(ctx.campaign_id, id=family)
    if not c:
        return ToolResult(f"Unknown family {family}", is_error=True)
    c = c[0]
    s = dict(c["stats"])
    cells = json.loads(s.pop("cells", "[]"))
    head = {"family": family, "variable": c["variable"], "class": c["cls"], "n_cells": c["n_cells"],
            "centroid": [round(c["lat"], 3), round(c["lon"], 3)],
            "bbox": [s.get("lat_min"), s.get("lat_max"), s.get("lon_min"), s.get("lon_max")],
            "onset": c["onset"], "effect": round(c["effect"], 4), "q": c["qvalue"],
            "filters": c["filters"], "current_outcome": c["outcome"], "note": c["note"]}
    body = "{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v, default=str)}" for k, v in head.items())
    body += f',\n "cells": {json.dumps(cells[:80])}\n}}'
    extra = ""
    if len(cells) > 80:
        out = ctx.workdir / f"family_cells_{family.replace(':', '_').replace('+', 'p').replace('-', 'm')}.csv"
        try:
            pd.DataFrame(cells).to_csv(out, index=False)
            extra = f"\n({len(cells)} cells in total; first 80 listed; ALL cells saved to {out.name})"
        except Exception:  # noqa: BLE001
            extra = f"\n({len(cells)} cells in total; first 80 listed)"
    return ToolResult(body + extra)


def kb_search(ctx: ToolContext, query: str) -> ToolResult:
    hits = ctx.record.kb_search(ctx.campaign_id, query)
    if not hits:
        return ToolResult("Knowledge base: nothing relevant yet.")
    return ToolResult("\n\n".join(f"[{h['kind']}] {h['title']} (task {h['task_id']})\n{h['body'][:900]}" for h in hits))


def read_task(ctx: ToolContext, task_id: str) -> ToolResult:
    t = ctx.record.task(task_id)
    if not t:
        return ToolResult(f"No task {task_id}", is_error=True)
    return ToolResult(f"{t['title']} [{t['status']}] verdict={t['verdict']}\nBrief: {t['brief'][:1500]}\n"
                      f"Summary: {(t['summary'] or '(none)')[:3000]}")


# ---------------------------------------------------------------------------- v3: hypothesis board and referees

INDEPENDENT_TOOLS = {"independent_series", "station_series", "nasa_series"}
HYP_STATUS = {"supported", "supported_new", "refuted", "not_shown", "known", "inconclusive"}  # agent-claimable outcomes


def record_hypothesis_result(ctx: ToolContext, hypothesis_id: str, status: str, result: str, evidence: str,
                             novelty_basis: str = "", confidence: float = 0.5) -> ToolResult:
    """Close a hypothesis test. status: supported | supported_new (supported AND not in the literature) |
    not_shown (the test ran and was not significant) | refuted (a significant result CONTRADICTS the claim) |
    known (true but already published) | inconclusive (could not be tested). All are valid results."""
    status = str(status).strip().lower()
    if status not in HYP_STATUS:
        return ToolResult(f"status must be one of {sorted(HYP_STATUS)}", is_error=True)
    h = ctx.record.hypotheses(ctx.campaign_id, id=hypothesis_id)
    if not h:
        return ToolResult(f"Unknown hypothesis {hypothesis_id}", is_error=True)
    if len(result.strip()) < 40 or len(evidence.strip()) < 80:
        return ToolResult("Give the result in one or two precise sentences and the evidence: numbers, tests, data "
                          "sets, periods, cells.", is_error=True)
    if status == "supported_new":
        if len(ctx.state.get("lit_queries", [])) < 2:
            return ToolResult("supported_new needs at least two precise literature_search queries in this task.",
                              is_error=True)
        if len(novelty_basis.strip()) < 60:
            return ToolResult("supported_new needs novelty_basis: the specific feature the closest papers (cite title "
                              "and year) do not report. If you cannot name one, the status is known.", is_error=True)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.5
    task = ctx.record.task(ctx.task_id) if ctx.task_id else None
    import re as _re

    if _re.search(r"\b(drive[sn]?|driven by|because|caus(e|es|ed|ing)|mechanism|due to|responsible for)\b",
                  result, _re.I):
        ablation = ctx.record.one("SELECT 1 FROM events WHERE task_id=? AND type='action' AND content LIKE "
                                  "'%mediation_test%'", (ctx.task_id,)) if ctx.task_id else None
        if not ablation:  # a mechanism needs an ablation; otherwise it is an association
            result = result.strip() + " [association only: no ablation test (mediation_test) was run]"
    if status == "not_shown" and _re.search(
            r"(could not|cannot|can't|can not) (be )?(executed|computed|run|evaluated|completed|tested|fetched)|"
            r"uncomputable|not (be )?executable|held[- ]out (for|and)", result + " " + evidence, _re.I):
        # found live: agents closed hypotheses they could not test (decisive data held out) as not_shown. Not shown
        # means a test RAN and was not significant; an untested hypothesis is inconclusive (its held-out test still
        # runs in code if it has one).
        status = "inconclusive"
    if task and task["kind"] == "followup":
        # a narrow follow-up (one robustness check) adds to the evidence; it does not rewrite the hypothesis's
        # headline result or status (found live: a leave-one-year-out check replaced the main result)
        h0 = h[0]
        ctx.record.update_hypothesis(ctx.campaign_id, hypothesis_id, evidence=(
            (h0["evidence"] or "") + f"\n\nFOLLOW-UP ({status}): {result.strip()}")[-6000:])
    else:
        # a claimed success is NOT a result yet: it waits for the pre-registered confirmation that code runs on data
        # the exploring agents never saw (held-out products, the pre-registered split and statistic)
        claimed = status
        if status in ("supported", "supported_new"):
            status = "awaiting_confirmation"
        refused = ctx.record.update_hypothesis(
            ctx.campaign_id, hypothesis_id, actor="agent", status=status, result=result.strip(),
            evidence=(evidence.strip() + (f"\n[claimed: {claimed}]" if claimed != status else ""))[:6000],
            novelty=novelty_basis.strip()[:3000])
        if refused:
            return ToolResult(f"Status NOT changed: {refused}. Your result was kept in the evidence.", is_error=True)
    ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "finding", f"Hypothesis {hypothesis_id} {status}: {result[:90]}",
                      f"{result.strip()}\nEvidence: {evidence.strip()[:3000]}\nConfidence: {conf:.2f}"
                      + (f"\nNovelty basis: {novelty_basis.strip()}" if novelty_basis.strip() else ""),
                      ["hypothesis", hypothesis_id, status])
    ctx.state.setdefault("findings", []).append({"hypothesis": hypothesis_id, "status": status})
    ctx.state["hypothesis_result"] = {"id": hypothesis_id, "status": status, "confidence": conf}
    return ToolResult(f"Hypothesis {hypothesis_id} recorded as {status}."
                      + (" The harness now runs its pre-registered confirmation on held-out data; if that passes, an "
                         "independent referee reviews it." if status == "awaiting_confirmation" else ""))


def referee_report(ctx: ToolContext, target: str, decision: str, objections: str, checks_run: str) -> ToolResult:
    """Referee decision on a claimed result (hypothesis id or family id): upheld | downgraded | not_shown | refuted."""
    decision = str(decision).strip().lower().replace(" ", "_")
    if decision not in ("upheld", "downgraded", "not_shown", "refuted"):
        return ToolResult("decision must be upheld, downgraded (true but weaker / not new), not_shown (your "
                          "recomputation is not significant) or refuted (your checks significantly contradict it)",
                          is_error=True)
    if len(checks_run.strip()) < 80:
        return ToolResult("List the checks you actually ran (data, tests, numbers). A referee who ran nothing cannot "
                          "uphold or refute.", is_error=True)
    if len([t for t in ctx.state.get("ok_tools", []) if t in ("python", "nasa_series", "independent_series",
                                                           "station_series", "climate_index", "literature_search",
                                                           "merra2_series", "gpu_submit")]) < 3:
        return ToolResult("Run at least three real checks (code, independent data, literature) before deciding.",
                          is_error=True)
    review = {"decision": decision, "objections": objections.strip()[:4000], "checks": checks_run.strip()[:3000]}
    h = ctx.record.hypotheses(ctx.campaign_id, id=target)
    if h:
        # downgraded = true but weaker or narrower than claimed, or not new: it stays supported (never "new")
        cur = h[0]["status"]
        new = {"upheld": cur, "downgraded": "supported" if cur == "supported_new" else cur, "refuted": "refuted",
               "not_shown": "not_shown"}[decision]
        ctx.record.update_hypothesis(ctx.campaign_id, target, actor="referee", review=json.dumps(review), status=new)
    else:
        c = ctx.record.candidates(ctx.campaign_id, id=target)
        if not c:
            return ToolResult(f"Unknown target {target}", is_error=True)
        note = (c[0]["note"] or "") + f" [referee: {decision}. {objections.strip()[:600]}]"
        out = c[0]["outcome"]
        if decision in ("refuted", "not_shown"):
            out = "not_robust"
        elif decision == "downgraded" and out == "new_signal":
            out = "known_phenomenon"
        ctx.record.update_candidate(ctx.campaign_id, target, actor="referee", note=note, outcome=out)
    ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "review", f"Referee {decision}: {target}",
                      f"Objections: {objections.strip()[:2000]}\nChecks: {checks_run.strip()[:1500]}",
                      ["referee", target, decision])
    ctx.state["referee"] = review
    return ToolResult(f"Referee report filed: {target} {decision}.")


def hypothesis_board(ctx: ToolContext) -> ToolResult:
    """The PI's research programme: every hypothesis with its status and result."""
    hs = ctx.record.hypotheses(ctx.campaign_id)
    if not hs:
        return ToolResult("No hypotheses yet.")
    return ToolResult("\n".join(f"{h['id']} [{h['status']}] (priority {h['priority']:.2f}) {h['title']}\n  claim: "
                                f"{h['statement'][:300]}\n  test: {h['test'][:300]}\n  result: {h['result'][:300]}"
                                for h in hs))


def where(ctx: ToolContext, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> ToolResult:
    """Real place names for a box (GeoNames, offline): the centre and the countries/provinces the land cells fall
    in. Use it before naming a region in a result (live runs called Qinghai, China 'Central Asia')."""
    from .. import verify

    la = np.arange(float(lat_min), float(lat_max) + 1e-9, max((float(lat_max) - float(lat_min)) / 6, 0.5))
    lo = np.arange(float(lon_min), float(lon_max) + 1e-9, max((float(lon_max) - float(lon_min)) / 6, 0.5))
    cells = [[float(a), float(b)] for a in la for b in lo]
    fam = {"lat": (float(lat_min) + float(lat_max)) / 2, "lon": (float(lon_min) + float(lon_max)) / 2, "cells": cells}
    try:
        import reverse_geocoder as rg

        if verify._GEO is None:
            verify._GEO = rg.RGeocoder(mode=1, verbose=False)
        res = verify._GEO.query([tuple(c) for c in cells])
        admin = pd.Series([f"{r['admin1']}, {verify._country(r['cc'])}" for r in res]).value_counts().head(8)
        extra = "\nProvinces/states sampled: " + "; ".join(f"{k} ({v})" for k, v in admin.items())
    except Exception:  # noqa: BLE001
        extra = ""
    return ToolResult(f"Box lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}: {verify.family_place(fam)}{extra}\n"
                      "(points over the ocean take the nearest coastal town: check the distance.)")
