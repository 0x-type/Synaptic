"""Trend Detective analysis: the five steps of the challenge (variable, region, period, time series, significance)
on a monthly regional series, with statistics that do not over-claim.

- Season: any set of months (e.g. Jun-Aug). Dec in a DJF season belongs to the following year's winter.
- Trend: Sen slope with a Mann-Kendall test whose variance is inflated for year-to-year carry-over, plus an
  ordinary least-squares slope with an AR(1)-adjusted standard error and 95% interval.
- Extras: an abrupt-shift test (Pettitt, carry-over adjusted), a 'recent years beyond the trend' test, per-month
  trends (seasonal trends) and a shift in the timing of the seasonal cycle.
- Plain-language verdict, including what 'not significant' means.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st

import synlib

MONTHS = "JFMAMJJASOND"
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def season_label(months: list[int] | None) -> str:
    if not months or len(months) == 12:
        return "all months"
    for k, v in SEASONS.items():
        if sorted(v) == sorted(months):
            return k
    return "".join(MONTHS[m - 1] for m in months) if len(months) <= 6 else f"{len(months)} months"


def yearly(series: pd.DataFrame, months: list[int] | None, min_coverage: float = 0.5) -> pd.DataFrame:
    """Mean over the selected months per (season) year. A year needs at least 2/3 of its months, each with data
    over at least `min_coverage` of the region."""
    s = series.copy()
    s.loc[s["coverage"] < min_coverage, "value"] = np.nan if "coverage" in s else s["value"]
    months = months or list(range(1, 13))
    s = s[s.month.isin(months)].copy()
    wraps = 12 in months and 1 in months and len(months) < 12
    s["syear"] = np.where(wraps & (s.month == 12), s.year + 1, s.year)
    g = s.groupby("syear").value.agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= math.ceil(2 * len(months) / 3)]
    return g.rename(columns={"syear": "year", "mean": "value"})[["year", "value", "count"]]


def ols_ar1(y: np.ndarray, t: np.ndarray) -> dict:
    """OLS slope with its standard error inflated for lag-1 carry-over (effective sample size)."""
    n = len(y)
    A = np.vstack([np.ones(n), t - t.mean()]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    r = synlib.ar1(res)
    neff = max(3.0, n * (1 - r) / (1 + r))
    s2 = (res ** 2).sum() / max(n - 2, 1)
    se = math.sqrt(s2 / ((t - t.mean()) ** 2).sum()) * math.sqrt(n / neff)
    df = max(neff - 2, 1)
    tval = c[1] / se if se > 0 else 0.0
    q = st.t.ppf(0.975, df)
    return {"slope": float(c[1]), "se": se, "ci95": [float(c[1] - q * se), float(c[1] + q * se)],
            "p": float(2 * st.t.sf(abs(tval), df)), "carryover": r, "n_eff": round(neff, 1), "intercept": float(c[0])}


def analyse(series: pd.DataFrame, months: list[int] | None = None, units: str = "", label: str = "",
            alpha: float = 0.05) -> dict:
    """Full analysis of a monthly regional series (columns year, month, value, coverage)."""
    if "coverage" not in series:
        series = series.assign(coverage=1.0)
    yr = yearly(series, months)
    y, t = yr.value.values.astype(float), yr.year.values.astype(float)
    out: dict = {"season": season_label(months), "years": [int(t.min()), int(t.max())] if len(t) else [],
                 "n_years": int(len(t)), "units": units, "label": label,
                 "yearly": [{"year": int(a), "value": float(b)} for a, b in zip(t, y)]}
    if len(y) < 8:
        out["verdict"] = ("Not enough years with data for a trend test (need at least 8). Choose a longer period "
                          "or a dataset with a longer record.")
        out["significant"] = None
        return out
    mk = synlib.mann_kendall(y)
    sen = synlib.sen_slope(y, t)
    ols = ols_ar1(y, t)
    span = t.max() - t.min()
    lo, hi = synlib.sen_ci(y, t, alpha)
    out["trend"] = {"sen_per_decade": sen * 10, "sen_ci95_per_decade": [lo * 10, hi * 10],
                    "mk_p": mk["p"], "mk_tau": mk["tau"], "carryover": mk["r1"],
                    "ols_per_decade": ols["slope"] * 10, "ols_ci95_per_decade": [v * 10 for v in ols["ci95"]],
                    "ols_p": ols["p"], "n_eff": ols["n_eff"], "total_change": sen * span,
                    "line": [{"year": int(a), "value": float(ols["intercept"] + ols["slope"] * (a - t.mean()))}
                             for a in (t.min(), t.max())]}
    p = mk["p"]
    sig = p < alpha
    out["significant"] = bool(sig)
    pt = synlib.pettitt(y)
    if pt["cp"] is not None:
        cp_year = int(t[pt["cp"] + 1])
        stp = synlib.step_change_test(y, t, split="scan", n_sim=1000)
        out["shift"] = {"year": cp_year, "pettitt_p": pt["p"],
                        "step_beyond_trend": stp.get("step_beyond_trend"), "p_selection_aware": stp.get("p_selection_aware"),
                        "before": float(np.mean(y[: pt["cp"] + 1])), "after": float(np.mean(y[pt["cp"] + 1:]))}
    rd = synlib.recent_departure(y, 3)
    if rd.get("valid"):
        out["recent"] = {"departure": rd["departure"], "p": rd["p"], "years": [int(v) for v in t[-3:]]}
    # trends month by month (a trend in one season can hide in the annual mean)
    if not months or len(months) == 12:
        per = []
        for m in range(1, 13):
            ym = series[series.month == m]
            ym = ym[ym.coverage >= 0.5].dropna(subset=["value"])
            if len(ym) >= 8:
                mm = synlib.mann_kendall(ym.value.values)
                per.append({"month": m, "sen_per_decade": synlib.sen_slope(ym.value.values, ym.year.values.astype(float)) * 10,
                            "p": mm["p"]})
        out["by_month"] = per
        # timing of the seasonal cycle: first vs second half of the record
        piv = series.pivot_table(index="year", columns="month", values="value")
        piv = piv[piv.notna().sum(1) == 12]
        if len(piv) >= 16:
            h = len(piv) // 2
            a1, ph1 = synlib.seasonal_phase(piv.iloc[:h].mean().values)
            a2, ph2 = synlib.seasonal_phase(piv.iloc[h:].mean().values)
            out["season_timing"] = {"peak_shift_days": float(((ph2 - ph1 + 182.6) % 365.25) - 182.6),
                                    "amplitude_ratio": float(a2 / a1) if a1 else None,
                                    "halves": [f"{int(piv.index[0])}-{int(piv.index[h - 1])}",
                                               f"{int(piv.index[h])}-{int(piv.index[-1])}"]}
    out["verdict"] = verdict_text(out, alpha)
    return out


def _fmt(v: float, units: str) -> str:
    a = abs(v)
    d = 3 if a < 0.01 else 2 if a < 1 else 1
    return f"{v:+.{d}f}{(' ' + units) if units else ''}"


def verdict_text(o: dict, alpha: float = 0.05) -> str:
    tr = o["trend"]
    y0, y1 = o["years"]
    s = (f"{o['label'] or 'The variable'} ({o['season']}, {y0}-{y1}): the trend is {_fmt(tr['sen_per_decade'], o['units'])} "
         f"per decade, about {_fmt(tr['total_change'], o['units'])} over the whole period. ")
    if o["significant"]:
        s += (f"The trend is statistically significant: p = {tr['mk_p']:.3g} < {alpha}. ")
    else:
        s += (f"It is NOT statistically significant: p = {tr['mk_p']:.3g} >= {alpha}. That does not prove there is no "
              "trend; the change is not distinguishable from year-to-year variability with this many years. A longer "
              "record or a larger region can make a real trend significant. ")
    if tr["carryover"] > 0.2:
        s += (f"Neighbouring years are correlated (carry-over r = {tr['carryover']:.2f}), so the test counts about "
              f"{tr['n_eff']:.0f} independent years instead of {o['n_years']}; ignoring that would overstate "
              "significance. ")
    sh = o.get("shift")
    if sh and sh.get("p_selection_aware") is not None and sh["p_selection_aware"] < alpha:
        s += (f"The change looks abrupt rather than gradual: a step around {sh['year']} beyond the trend "
              f"(p = {sh['p_selection_aware']:.2g}, accounting for the search over all years). Steps can be real "
              "(a regime shift) or come from a change in how the data were produced; check an independent dataset. ")
    rc = o.get("recent")
    if rc and rc["p"] < alpha:
        s += (f"The last three years ({rc['years'][0]}-{rc['years'][-1]}) depart from the trend by "
              f"{_fmt(rc['departure'], o['units'])} (p = {rc['p']:.2g}). ")
    return s.strip()


def trend_map(f, months: list[int] | None = None, alpha: float = 0.05, max_cells: int = 6000) -> dict:
    """Per-pixel trend (OLS slope per decade) and carry-over-adjusted Mann-Kendall significance on a Field."""
    months = months or list(range(1, 13))
    years = sorted({int(t[:4]) for t in f.times})
    idx = {t: k for k, t in enumerate(f.times)}
    wraps = 12 in months and 1 in months and len(months) < 12
    ny, nx = len(f.lat), len(f.lon)
    stride = max(1, int(math.ceil(math.sqrt(ny * nx / max_cells))))
    lat, lon = f.lat[::stride], f.lon[::stride]
    data = f.data[:, ::stride, ::stride]
    Y = np.full((len(years), len(lat), len(lon)), np.nan)
    for yi, y in enumerate(years):
        ks = [idx[f"{y - 1 if (wraps and m == 12) else y}-{m:02d}"] for m in months
              if f"{y - 1 if (wraps and m == 12) else y}-{m:02d}" in idx]
        if len(ks) >= math.ceil(2 * len(months) / 3):
            with np.errstate(all="ignore"):
                Y[yi] = np.nanmean(data[ks], axis=0)
    M = Y.reshape(len(years), -1).T
    ok = np.isfinite(M).sum(1) >= max(8, int(0.8 * len(years)))
    slope = np.full(M.shape[0], np.nan)
    pval = np.full(M.shape[0], np.nan)
    if ok.any():
        Mo = M[ok]
        mu = np.nanmean(Mo, axis=1, keepdims=True)
        Mo = np.where(np.isnan(Mo), mu, Mo)
        t = np.array(years, float) - np.mean(years)
        slope[ok] = (Mo * t).sum(1) / (t * t).sum() * 10
        pval[ok] = synlib._mk_p_matrix(Mo)[0]
    return {"lat": lat.tolist(), "lon": lon.tolist(), "slope": slope.reshape(len(lat), len(lon)),
            "p": pval.reshape(len(lat), len(lon)), "alpha": alpha, "years": [years[0], years[-1]]}
