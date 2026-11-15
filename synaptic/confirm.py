"""Held-out confirmation: the step between "an agent found it" and "the group reports it".

Every hypothesis is pre-registered by the PI with a machine-readable `confirm` spec BEFORE any test runs: which
independent dataset, region, season, statistic, split year and direction. Exploring agents never get the held-out
datasets (their data tool refuses them). When an agent claims a success, this module runs the pre-registered test
in code on the held-out data; only a pass makes the hypothesis "supported". Nothing an agent writes can change the
spec or the verdict.

spec = {
  "dataset": "CERES_SW_UP",            # catalogue id of the held-out product
  "box": [S, N, W, E],
  "months": [6, 7, 8] | null,
  "kind": "trend" | "step" | "event" | "correlation" | "correlation_change" | "contrast" | "residual",
  "control_box": [S, N, W, E], "test": "step" | "event" | "trend",   # contrast: box minus control box, then test
  # residual: the target minus what a driver explains (OLS fitted on "fit_years", default all years), then `test`;
  #           "accumulate": true uses the running sum of driver anomalies (storage integrates rain/evaporation)
  "years": [2001, 2025],
  "split": 2015 | "scan",              # step / correlation_change
  "event_years": [2023, 2024],         # event
  "driver": "NINO34" | catalogue id,   # correlation kinds
  "driver_box": [S, N, W, E], "driver_months": [12, 1, 2], "driver_lag_years": 0,
  "direction": "+" | "-",
  "alpha": 0.05
}
kind "gain" (storage integrates its inputs, so its LEVEL carries over from year to year, e.g. carry-over 0.9 over
northern Morocco, and a level test has few independent years; the GAIN between two seasons does not carry over):
  "from_months": [9, 10], "to_months": [4, 5]   # gain = mean(to months of year Y) - mean(from months, of Y-1 when
                                                #  they come later in the calendar than the to months, else of Y)
  "driver": ["MERRA2_PRECIP", "GLDAS_EVAP"]     # optional: remove what the drivers' means over the window explain
  "fit_years": [2003, 2017], "test": "event" | "trend", "event_years": [2026]
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import synlib

from . import catalog, cloud, trend

KINDS = ("trend", "step", "event", "correlation", "correlation_change", "contrast", "residual", "gain")
CONTRAST_TESTS = ("trend", "step", "event")


def validate(spec) -> tuple[dict | None, str]:
    """Return (normalised spec, problem). A hypothesis without a valid spec can never become supported."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:  # noqa: BLE001
            return None, "confirm spec is not JSON"
    if not isinstance(spec, dict):
        return None, "no confirm spec"
    s = dict(spec)
    if s.get("dataset") not in catalog.BY_ID:
        return None, f"dataset must be one of the catalogue ids, got {s.get('dataset')!r}"
    if s.get("kind") not in KINDS:
        return None, f"kind must be one of {KINDS}"
    try:
        box = [float(v) for v in s["box"]]
        assert len(box) == 4 and box[0] < box[1]
        s["box"] = box
        y = s.get("years") or [catalog.get(s["dataset"]).start, 2025]
        s["years"] = [int(y[0]), int(y[1])]
    except Exception:  # noqa: BLE001
        return None, "box must be [S, N, W, E] and years [first, last]"
    if s["years"][1] - s["years"][0] < 7:
        return None, "years must span at least 8 years"
    if s["kind"] == "contrast":
        try:
            cb = [float(v) for v in s["control_box"]]
            assert len(cb) == 4 and cb[0] < cb[1]
            s["control_box"] = cb
        except Exception:  # noqa: BLE001
            return None, "contrast needs control_box [S, N, W, E]"
        if s.get("test") not in CONTRAST_TESTS:
            return None, f"contrast needs test in {CONTRAST_TESTS}"
    if s.get("test") == "later":
        try:
            s["after"] = int(s["after"])
            fy = s.get("fit_years") or [s["years"][0], s["after"]]
            s["fit_years"] = [int(fy[0]), min(int(fy[1]), s["after"])]
        except Exception:  # noqa: BLE001
            return None, "test 'later' needs \"after\": the last exploration year (fit on years up to it, test after it)"
        if s["kind"] not in ("residual", "gain"):
            return None, "test 'later' is for kind residual or gain"
        if s["years"][1] - s["after"] < 5:
            return None, "test 'later' needs at least 5 years after \"after\""
    if s["kind"] == "residual":
        if s.get("test") not in CONTRAST_TESTS + ("later",):
            return None, f"residual needs test in {CONTRAST_TESTS}"
        drvs = s.get("driver") if isinstance(s.get("driver"), list) else [s.get("driver")]
        if not drvs or not all(isinstance(d, str) and d for d in drvs) or len(drvs) > 3:
            return None, "residual needs a driver: a catalogue id / climate index, or a list of up to 3"
        if s.get("fit_years") is not None:
            try:
                fy = [int(v) for v in s["fit_years"]]
                assert len(fy) == 2 and fy[1] - fy[0] >= 5
                s["fit_years"] = fy
            except Exception:  # noqa: BLE001
                return None, "fit_years must be [first, last] spanning at least 6 years"
    if s["kind"] == "gain":
        try:
            fm = [int(m) for m in s["from_months"]]
            tm = [int(m) for m in s["to_months"]]
            assert fm and tm and all(1 <= m <= 12 for m in fm + tm)
            s["from_months"], s["to_months"] = fm, tm
        except Exception:  # noqa: BLE001
            return None, "gain needs from_months and to_months (lists of month numbers)"
        if s.get("test") not in ("event", "trend", "later"):
            return None, "gain needs test 'event', 'trend' or 'later'"
        if s.get("driver") is not None:
            drvs = s["driver"] if isinstance(s["driver"], list) else [s["driver"]]
            if not all(d in catalog.BY_ID for d in drvs) or len(drvs) > 3:
                return None, "gain driver must be up to 3 catalogue ids"
        s["months"] = sorted(set(tm))  # for the end-date check: the to months of the last year must exist
    if s["kind"] in ("step", "correlation_change") and s.get("split") is None:
        return None, "step / correlation_change need split (a year or 'scan')"
    needs_split = s["kind"] in ("step", "correlation_change") or (s["kind"] in ("contrast", "residual")
                                                                  and s.get("test") == "step")
    needs_event = s["kind"] == "event" or (s["kind"] in ("contrast", "residual", "gain") and s.get("test") == "event")
    if needs_split and s.get("split") is None:
        return None, "a step test needs split (a year or 'scan')"
    if needs_event and not s.get("event_years"):
        return None, "an event test needs event_years"
    if needs_split and str(s.get("split")) != "scan":
        try:
            sp = int(s["split"])
        except Exception:  # noqa: BLE001
            return None, "split must be a year or 'scan'"
        # a pre-registered split needs 4 years each side (a scanned one needs 8): IMO 2020 leaves only 2020-2025
        if sp - s["years"][0] < 4 or s["years"][1] - sp + 1 < 4:
            hint = (f"; for a change in the last years use \"test\"/\"kind\": \"event\" with \"event_years\": "
                    f"{list(range(sp, s['years'][1] + 1))}" if s["years"][1] - sp + 1 < 4 else "")
            return None, f"split {sp} leaves fewer than 4 years on one side of {s['years']}: not testable{hint}"
        s["split"] = sp
    if s["kind"] == "event" and not s.get("event_years"):
        return None, "event needs event_years"
    if s["kind"] in ("correlation", "correlation_change", "residual") and not s.get("driver"):
        return None, "correlation kinds need a driver (climate index name or catalogue id)"
    if s.get("direction") not in ("+", "-", None):
        return None, "direction must be '+' or '-'"
    # found live: a residual test of 2026 used IMERG as driver, but IMERG stops in Sep 2025, so the event year vanished
    # and the test could not run. Refuse such specs while they can still be fixed (before any result).
    need = list(s.get("event_years") or []) + [s["years"][1]]
    drivers = s.get("driver") if isinstance(s.get("driver"), list) else [s.get("driver")]
    checks = [(s["dataset"], s.get("months"))] + [(d, s.get("driver_months") or s.get("months")) for d in drivers
                                                  if d in catalog.BY_ID]
    for ds, months in checks:
        end = catalog.get(ds).end
        if not end:
            continue
        months = months or list(range(1, 13))
        ey, em = int(end[:4]), int(end[5:7])
        wraps = 12 in months and 1 in months and len(months) < 12
        for y in need:
            y = int(y)
            # a (season) year counts only with 2/3 of its months (trend.yearly); December belongs to the next year
            avail = sum(1 for mo in months if ((y - 1) if (wraps and mo == 12) else y, mo) <= (ey, em))
            if avail < math.ceil(2 * len(months) / 3):
                return None, (f"{ds} ends in {end}: year {y} would have only {avail} of the {len(months)} months "
                              f"{months}, so it drops out; use months that exist (e.g. a season ending by {end}) "
                              f"or an earlier year")
    s.setdefault("alpha", 0.05)
    return s, ""


def _annual_dataset(var_id: str, box, months, y0, y1, progress=None) -> pd.DataFrame:
    from .regions import land_mask

    mask = land_mask if catalog.get(var_id).domain == "land" else None  # GRACE has ocean values too
    ser = cloud.regional_series(var_id, tuple(box), y0, y1, mask=mask, progress=progress)
    return trend.yearly(ser, months)


def _annual_index(name: str, months, y0, y1) -> pd.DataFrame:
    from .tools.climate import load_index

    d = load_index(name)
    d.columns = [c.lower() for c in d.columns]
    d = d.assign(coverage=1.0)
    return trend.yearly(d[(d.year >= y0 - 1) & (d.year <= y1)], months)


def run(spec: dict, progress=None) -> dict:
    """Run a validated spec. Returns {passed, outcome: supported|not_shown|refuted|inconclusive, p, effect, detail}.
    refuted = significant in the OPPOSITE direction; not_shown = computed but not significant;
    inconclusive = could not be computed."""
    s, problem = validate(spec)
    if s is None:
        return {"passed": False, "outcome": "inconclusive", "detail": f"not confirmable: {problem}"}
    y0, y1 = s["years"]
    if s["kind"] == "gain":
        return _run_gain(s, progress)
    tgt = _annual_dataset(s["dataset"], s["box"], s.get("months"), y0, y1, progress)
    y, t = tgt.value.values.astype(float), tgt.year.values.astype(float)
    kind, alpha = s["kind"], float(s["alpha"])
    if kind == "contrast":  # the box minus a control box: a regional fingerprint, not a basin-wide change
        ctl = _annual_dataset(s["dataset"], s["control_box"], s.get("months"), y0, y1, progress)
        m = pd.merge(tgt, ctl, on="year", suffixes=("", "_c"))
        y, t = (m.value - m.value_c).values.astype(float), m.year.values.astype(float)
        kind = s["test"]
    resid_note = ""
    def _driver(drv, lo):
        dm = s.get("driver_months") or s.get("months")
        if drv in catalog.BY_ID:
            d = _annual_dataset(drv, s.get("driver_box") or s["box"], dm, lo, y1, progress)
        else:
            d = _annual_index(drv, dm, y0, y1)
        lag = int(s.get("driver_lag_years") or 0)
        return d.assign(year=d.year + lag)[["year", "value"]]

    if kind in ("correlation", "correlation_change"):
        drv = s["driver"][0] if isinstance(s["driver"], list) else s["driver"]
        m = pd.merge(tgt, _driver(drv, y0 - 1), on="year", suffixes=("", "_d"))
        y, t, x = m.value.values.astype(float), m.year.values.astype(float), m.value_d.values.astype(float)
    if kind == "residual":  # what the driver(s) cannot explain
        drvs = s["driver"] if isinstance(s["driver"], list) else [s["driver"]]
        m = tgt[["year", "value"]]
        for i, drv in enumerate(drvs):
            m = pd.merge(m, _driver(drv, y0).rename(columns={"value": f"x{i}"}), on="year")
        y, t = m.value.values.astype(float), m.year.values.astype(float)
        X = m[[f"x{i}" for i in range(len(drvs))]].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, t, X = y[ok], t[ok], X[ok]
        fy = s.get("fit_years") or [y0, y1]
        fit = (t >= fy[0]) & (t <= fy[1])
        if s.get("accumulate"):
            X = np.cumsum(X - X[fit].mean(axis=0), axis=0)
        if fit.sum() < len(drvs) + 5 or (X[fit].std(axis=0) == 0).any():
            return {"passed": False, "outcome": "inconclusive", "p": None, "effect": None, "n_years": int(len(y)),
                    "detail": "Held-out confirmation could not be computed: too few fit years for the residual",
                    "spec": s}
        A = np.column_stack([np.ones(len(y)), X])
        coef, *_ = np.linalg.lstsq(A[fit], y[fit], rcond=None)
        y = y - A @ coef
        resid_note = (f"residual after removing {'accumulated ' if s.get('accumulate') else ''}"
                      f"{' + '.join(drvs)} (coefficients {', '.join(f'{c:+.3g}' for c in coef[1:])}, "
                      f"fitted {fy[0]}-{fy[1]}); ")
        kind = s["test"]
    if kind == "later":
        effect, p, detail = _later(y, t, s["after"])
        r = {"valid": np.isfinite(p)}
    elif kind == "trend":
        mk = synlib.mann_kendall(y)
        effect, p = synlib.sen_slope(y, t) * 10, mk["p"]
        detail = f"Sen slope {effect:+.4g} per decade, Mann-Kendall p = {p:.3g} (carry-over {mk['r1']:.2f})"
    elif kind == "step":
        r = synlib.step_change_test(y, t, split=s["split"], n_sim=4000, min_side=8 if s["split"] == "scan" else 4)
        effect, p = r.get("step_beyond_trend", np.nan), r.get("p_selection_aware", 1.0)
        detail = f"step beyond trend at {r.get('split')}: {effect:+.4g}, p = {p:.3g} ({'scanned' if r.get('scanned') else 'pre-registered'} split)"
    elif kind == "event":
        r = synlib.event_departure(y, t, s["event_years"])
        effect, p = r.get("departure", np.nan), r.get("p", 1.0)
        detail = f"{r.get('event_years')} depart from the trend of the other years by {effect:+.4g}, p = {p:.3g}"
    elif kind == "correlation":
        r = synlib.correlation_test(y, x)
        effect, p = r.get("r", np.nan), r.get("p", 1.0)
        detail = f"r = {effect:+.3f} with {s['driver']} (n_eff {r.get('n_eff')}), p = {p:.3g}"
    else:
        r = synlib.correlation_change_test(y, x, t, split=s["split"], n_sim=4000,
                                          min_side=8 if s["split"] == "scan" else 4)
        effect, p = r.get("change_abs_r", np.nan), r.get("p_selection_aware", 1.0)
        detail = (f"|r| {r.get('r_early', np.nan):+.2f} -> {r.get('r_late', np.nan):+.2f} at {r.get('split')} "
                  f"({'scanned' if r.get('scanned') else 'pre-registered'}), p = {p:.3g}")
    detail = resid_note + detail
    if not (isinstance(r, dict) and r.get("valid", True)) if kind != "trend" else False:
        return {"passed": False, "outcome": "inconclusive", "p": None, "effect": None, "n_years": int(len(y)),
                "detail": f"Held-out confirmation could not be computed: {r.get('note', 'invalid input')}", "spec": s}
    sign_ok = s.get("direction") is None or (np.sign(effect) == (1 if s["direction"] == "+" else -1))
    sig = bool(np.isfinite(p) and p < alpha)
    if sig and sign_ok:
        outcome = "supported"
    elif sig and not sign_ok:
        outcome = "refuted"
    else:
        outcome = "not_shown"
    return {"passed": outcome == "supported", "outcome": outcome, "p": float(p) if np.isfinite(p) else None,
            "effect": float(effect) if np.isfinite(effect) else None, "n_years": int(len(y)),
            "detail": f"Held-out confirmation on {catalog.get(s['dataset']).source}, {s['box']}"
                      + (f" minus control {s['control_box']}" if s["kind"] == "contrast" else "") + ", "
                      f"{trend.season_label(s.get('months'))} {y0}-{y1}: {detail}. Pre-registered direction "
                      f"{s.get('direction') or 'any'}, alpha {alpha}: {outcome.upper()}.",
            "spec": s}


def _later(y, t, after: int):
    """Time holdout: residuals of a model fitted on the years up to `after` (the exploration period); is their mean
    over the later years, which the explorers never saw, different from zero? t-test with the effective sample size
    reduced for carry-over (lag-1 autocorrelation of all residuals, each period around its own mean, never below 0)."""
    import math

    from scipy import stats as _st

    y, t = np.asarray(y, float), np.asarray(t, float)
    later = y[t > after]
    if len(later) < 5:
        return np.nan, np.nan, "fewer than 5 later years"
    early = y[t <= after]
    # carry-over from ALL years (each period around its own mean): 10 later years alone give a noisy estimate
    resid = np.concatenate([early - early.mean(), later - later.mean()]) if len(early) > 2 else later - later.mean()
    r1 = max(0.0, float(synlib.ar1(resid)))
    neff = max(2.0, len(later) * (1 - r1) / (1 + r1))
    sd = float(np.std(later, ddof=1))
    tval = later.mean() / (sd / math.sqrt(neff)) if sd > 0 else 0.0
    p = float(2 * _st.t.sf(abs(tval), neff - 1))
    return float(later.mean()), p, (f"years after {after} (held out from the explorers) depart from the model fitted "
                                    f"up to {after} by {later.mean():+.4g} on average ({len(later)} years, carry-over "
                                    f"{r1:.2f}), p = {p:.3g}")


def _monthly(var_id: str, box, y0, y1, progress=None) -> pd.Series:
    from .regions import land_mask

    mask = land_mask if catalog.get(var_id).domain == "land" else None
    ser = cloud.regional_series(var_id, tuple(box), y0, y1, mask=mask, progress=progress)
    ser.loc[ser["coverage"] < 0.5, "value"] = np.nan
    return ser.set_index(["year", "month"])["value"]


def _gain_window(fm, tm, y):
    """(year, month) pairs of the from-period, the to-period, and the window between (after the from-period,
    through the end of the to-period)."""
    prev = min(fm) > max(tm)
    frm = [((y - 1) if prev else y, m) for m in fm]
    to = [(y, m) for m in tm]
    start, end = max(frm), max(to)
    win, cur = [], start
    while cur < end:
        cur = (cur[0] + 1, 1) if cur[1] == 12 else (cur[0], cur[1] + 1)
        win.append(cur)
    return frm, to, win


def _run_gain(s: dict, progress=None) -> dict:
    y0, y1 = s["years"]
    fm, tm = s["from_months"], s["to_months"]
    tgt = _monthly(s["dataset"], s["box"], y0 - 1, y1, progress)
    drvs = s["driver"] if isinstance(s.get("driver"), list) else ([s["driver"]] if s.get("driver") else [])
    dser = [_monthly(d, s.get("driver_box") or s["box"], y0 - 1, y1, progress) for d in drvs]
    rows = []
    for y in range(y0, y1 + 1):
        frm, to, win = _gain_window(fm, tm, y)
        a = [tgt.get(k, np.nan) for k in frm]
        b = [tgt.get(k, np.nan) for k in to]
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        g = float(np.nanmean(b) - np.nanmean(a))
        xs = []
        for d in dser:
            v = [d.get(k, np.nan) for k in win]
            xs.append(float(np.mean(v)) if win and np.isfinite(v).sum() >= max(1, int(np.ceil(2 * len(win) / 3)))
                      else np.nan)
        rows.append([y, g] + xs)
    if not rows:
        return {"passed": False, "outcome": "inconclusive", "p": None, "effect": None, "n_years": 0,
                "detail": "Held-out confirmation could not be computed: no years with both periods", "spec": s}
    arr = np.array(rows, float)
    ok = np.isfinite(arr).all(axis=1)
    arr = arr[ok]
    t, y = arr[:, 0], arr[:, 1]
    note = f"storage gain from months {fm} to {tm}"
    if drvs:
        X = arr[:, 2:]
        fy = s.get("fit_years") or [y0, y1]
        fit = (t >= fy[0]) & (t <= fy[1])
        if fit.sum() < len(drvs) + 5:
            return {"passed": False, "outcome": "inconclusive", "p": None, "effect": None, "n_years": int(len(y)),
                    "detail": "Held-out confirmation could not be computed: too few fit years for the gain residual",
                    "spec": s}
        A = np.column_stack([np.ones(len(y)), X])
        coef, *_ = np.linalg.lstsq(A[fit], y[fit], rcond=None)
        y = y - A @ coef
        note += (f", minus what {' + '.join(drvs)} over the window explain (coefficients "
                 f"{', '.join(f'{c:+.3g}' for c in coef[1:])}, fitted {fy[0]}-{fy[1]})")
    alpha = float(s["alpha"])
    if s["test"] == "later":
        effect, p, d2 = _later(y, t, s["after"])
        detail = f"{note}: {d2}"
    elif s["test"] == "trend":
        mk = synlib.mann_kendall(y)
        effect, p = synlib.sen_slope(y, t) * 10, mk["p"]
        detail = f"{note}: Sen slope {effect:+.4g} per decade, Mann-Kendall p = {p:.3g} (carry-over {mk['r1']:.2f})"
    else:
        r = synlib.event_departure(y, t, s["event_years"])
        if not r.get("valid", True):
            return {"passed": False, "outcome": "inconclusive", "p": None, "effect": None, "n_years": int(len(y)),
                    "detail": "Held-out confirmation could not be computed: event years missing", "spec": s}
        effect, p = r["departure"], r["p"]
        detail = (f"{note}: {r['event_years']} depart from the trend of the other years by {effect:+.4g}, p = {p:.3g} "
                  f"(carry-over {r['carryover']:.2f}, {len(y)} years)")
    sign_ok = s.get("direction") is None or (np.sign(effect) == (1 if s["direction"] == "+" else -1))
    sig = bool(np.isfinite(p) and p < alpha)
    outcome = "supported" if sig and sign_ok else ("refuted" if sig else "not_shown")
    return {"passed": outcome == "supported", "outcome": outcome, "p": float(p), "effect": float(effect),
            "n_years": int(len(y)),
            "detail": f"Held-out confirmation on {catalog.get(s['dataset']).source}, {s['box']}, {y0}-{y1}: {detail}. "
                      f"Pre-registered direction {s.get('direction') or 'any'}, alpha {alpha}: {outcome.upper()}.",
            "spec": s}
