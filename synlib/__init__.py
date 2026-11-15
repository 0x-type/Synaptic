"""synlib: the analysis toolkit agents import from the code they write.

It plays the role HMMER/MMseqs2/BLAST played in the paper: reliable, fast, well-tested building
blocks so the model spends its reasoning on *judgement*, not on re-implementing statistics.

Data convention (what the harness data tools write): a long CSV with columns
    lat, lon, variable, year, month, value     (month 1..12, NaN for missing)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

# Physical links used to look for corroboration of a signal in an independent variable.
LINKED_VARIABLES = {
    "PRECTOTCORR": ["GWETROOT", "GWETTOP", "RH2M"],
    "GWETROOT": ["PRECTOTCORR", "GWETPROF", "T2M"],
    "GWETTOP": ["PRECTOTCORR", "GWETROOT"],
    "T2M": ["T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT"],
    "T2M_MAX": ["T2M", "GWETROOT"],
    "T2M_MIN": ["T2M", "RH2M"],
    "RH2M": ["T2M", "PRECTOTCORR", "QV2M"],
    "QV2M": ["RH2M", "PRECTOTCORR"],
    "ALLSKY_SFC_SW_DWN": ["CLOUD_AMT", "PRECTOTCORR"],
    "CLOUD_AMT": ["ALLSKY_SFC_SW_DWN", "PRECTOTCORR"],
    "WS2M": ["WS10M", "PS"],
    "WS10M": ["WS2M", "PS"],
}

# Places where the *data production* changes and a step can appear for non-physical reasons.
# Verified from POWER API headers (Oct 2026): solar/cloud fields report source "SRB" up to 2000 and
# "SYN1DEG" (CERES) from 2001, and start in 1984. Meteorology reports "MERRA2" throughout; the most
# recent months may come from a near-real-time product. Confirm details in the POWER documentation.
_M2_STREAMS = [(1992, "MERRA-2 production stream change 100 -> 200 (separately spun-up land states)"),
               (2001, "MERRA-2 production stream change 200 -> 300"),
               (2011, "MERRA-2 production stream change 300 -> 400")]
KNOWN_SOURCE_TRANSITIONS = {
    **{v: list(_M2_STREAMS) for v in ("GWETROOT", "GWETTOP", "GWETPROF", "EVLAND", "EVPTRNS", "SNODP", "PRECTOTCORR",
                                      "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "QV2M")},
    "ALLSKY_SFC_SW_DWN": [(2001, "solar source change GEWEX SRB -> CERES SYN1deg")],
    "CLRSKY_SFC_SW_DWN": [(2001, "solar source change GEWEX SRB -> CERES SYN1deg")],
    "ALLSKY_SFC_LW_DWN": [(2001, "solar source change GEWEX SRB -> CERES SYN1deg")],
    "ALLSKY_KT": [(2001, "solar source change GEWEX SRB -> CERES SYN1deg")],
    "CLOUD_AMT": [(2001, "cloud fields from the radiation products, check source change")],
}


# ---------------------------------------------------------------------------- I/O

def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["value"] <= -998, "value"] = np.nan
    return df


@dataclass
class Cube:
    variable: str
    lats: np.ndarray
    lons: np.ndarray
    years: np.ndarray
    data: np.ndarray  # [nlat, nlon, nyear, 12]

    def annual(self, months: list[int] | None = None) -> np.ndarray:
        """Annual (or seasonal) mean, [nlat, nlon, nyear]. Seasons spanning Dec use the same year."""
        idx = [m - 1 for m in (months or range(1, 13))]
        with np.errstate(all="ignore"):
            return np.nanmean(self.data[..., idx], axis=-1)

    def cell(self, lat: float, lon: float) -> tuple[int, int]:
        return int(np.abs(self.lats - lat).argmin()), int(np.abs(self.lons - lon).argmin())


def to_cube(df: pd.DataFrame, variable: str | None = None, complete_years_only: bool = True) -> Cube:
    if variable is None:
        variable = str(df["variable"].iloc[0])
    d = df[df["variable"] == variable]
    if complete_years_only:
        counts = d.groupby("year")["month"].nunique()
        d = d[d["year"].isin(counts[counts == 12].index)]
    lats = np.sort(d["lat"].unique())
    lons = np.sort(d["lon"].unique())
    years = np.sort(d["year"].unique())
    cube = np.full((len(lats), len(lons), len(years), 12), np.nan)
    li = {v: i for i, v in enumerate(lats)}
    lo = {v: i for i, v in enumerate(lons)}
    yi = {v: i for i, v in enumerate(years)}
    cube[d["lat"].map(li).values, d["lon"].map(lo).values, d["year"].map(yi).values,
         d["month"].values.astype(int) - 1] = d["value"].values
    return Cube(variable, lats, lons, years, cube)


# ---------------------------------------------------------------------------- 1-D statistics

def _lag1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 4 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def ar1(resid: np.ndarray, floor: float = 0.0) -> float:
    """Lag-1 autocorrelation of residuals with the small-sample bias removed (the raw estimate from ~45 years is
    biased low by about (1 + 4 r) / n, which made every carry-over correction too weak). Clipped to [floor, 0.9]."""
    r = np.asarray(resid, float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 8:
        return float(max(floor, 0.0))
    r1 = _lag1(r)
    r1 = r1 + (1 + 4 * max(r1, 0.0)) / n
    return float(min(0.9, max(floor, r1, 0.0)))


def inflation(r: float) -> float:
    """Long-run variance inflation of an AR(1) series (how much carry-over shrinks the information)."""
    r = min(max(r, 0.0), 0.9)
    return (1 + r) / (1 - r)


def sen_slope(x: np.ndarray, t: np.ndarray | None = None) -> float:
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, float)
    ok = ~np.isnan(x)
    x, t = x[ok], t[ok]
    i, j = np.triu_indices(len(x), 1)
    return float(np.median((x[j] - x[i]) / (t[j] - t[i]))) if len(x) > 2 else float("nan")


def mann_kendall(x: np.ndarray, r_floor: float = 0.0) -> dict:
    """Mann-Kendall with a lag-1 variance inflation: residual autocorrelation after a Sen detrend, bias-corrected,
    never below `r_floor` (the sweep passes the regional carry-over so one noisy cell estimate cannot undo it)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return {"tau": np.nan, "p": 1.0, "z": 0.0, "r1": 0.0}
    s = np.sign(x[None, :] - x[:, None])[np.triu_indices(n, 1)].sum()
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    resid = x - sen_slope(x) * np.arange(n)
    r1 = ar1(resid, r_floor)
    var *= inflation(r1)
    z = (s - np.sign(s)) / math.sqrt(var) if var > 0 else 0.0
    return {"tau": float(s / (n * (n - 1) / 2)), "z": float(z), "p": float(2 * _st.norm.sf(abs(z))), "r1": r1}


def pettitt(x: np.ndarray, r_floor: float = 0.0, adjust: bool = True) -> dict:
    """Pettitt change-point test. Returns the index of the last point of the first segment.

    With `adjust` (default) the p-value accounts for year-to-year carry-over: the statistic's null variance is
    inflated by the AR(1) long-run factor of the series around a linear trend (bias-corrected, at least `r_floor`).
    Without it, carry-over alone made 35% of pure-noise cells "significant" (verified on synthetic AR(1) fields)."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    xx = x[ok]
    n = len(xx)
    if n < 10:
        return {"cp": None, "p": 1.0, "K": 0, "r1": 0.0}
    sgn = np.sign(xx[None, :] - xx[:, None])  # sgn[i, j] = sign(x_j - x_i)
    # U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)
    U = np.array([-sgn[: t + 1, t + 1:].sum() for t in range(n - 1)])
    t = int(np.abs(U).argmax())
    K = float(np.abs(U).max())
    f, r1 = 1.0, 0.0
    if adjust:
        # carry-over is estimated around the fitted step + trend: around a trend alone, a real step looks like
        # persistence and the test would explain its own signal away
        tt = np.arange(n, dtype=float)
        A = np.vstack([np.ones(n), (tt > t).astype(float), tt - tt.mean()]).T
        resid = xx - A @ np.linalg.lstsq(A, xx, rcond=None)[0]
        r1 = ar1(resid, r_floor)
        f = inflation(r1)
    p = min(1.0, 2 * math.exp(-6 * K ** 2 / (f * (n ** 3 + n ** 2))))
    return {"cp": int(np.flatnonzero(ok)[t]), "p": p, "K": K, "r1": r1}


def var_change_p(va: float, vb: float, na: int, nb: int, r: float) -> float:
    """Two-sided test of a variance change: log variance ratio against its standard error, inflated for carry-over
    (the variance of a variance estimate grows by (1 + r^2) / (1 - r^2) for AR(1) data)."""
    if va <= 0 or vb <= 0 or na < 4 or nb < 4:
        return 1.0
    f2 = (1 + r * r) / (1 - r * r)
    z = math.log(vb / va) / math.sqrt(f2 * (2 / (na - 1) + 2 / (nb - 1)))
    return float(2 * _st.norm.sf(abs(z)))


def recent_departure(ann: np.ndarray, k: int = 3, r_floor: float = 0.0, sd_floor: float = 0.0) -> dict:
    """Are the last `k` years unusual *beyond the long-term trend*? The trend is fitted on the earlier years and
    extrapolated; the departure's standard error includes carry-over between the recent years, the reduced
    information in the base period and the extrapolation error. (The old test compared the recent mean with the
    base mean as if years were independent: carry-over plus spatial coherence produced a 200-cell false family.)"""
    y = np.asarray(ann, float)
    n = len(y)
    t = np.arange(n, dtype=float)
    base_ok = ~np.isnan(y[:-k])
    if base_ok.sum() < 10 or np.isnan(y[-k:]).all():
        return {"valid": False}
    tb, yb = t[:-k][base_ok], y[:-k][base_ok]
    A = np.vstack([np.ones(len(tb)), tb]).T
    coef, *_ = np.linalg.lstsq(A, yb, rcond=None)
    res = yb - A @ coef
    sd = max(float(np.std(res, ddof=2)), sd_floor)
    r1 = ar1(res, r_floor)
    tr = t[-k:]
    dep = float(np.nanmean(y[-k:] - (coef[0] + coef[1] * tr)))
    # variance of the mean of k consecutive AR(1) values
    lags = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    v_recent = float((r1 ** lags).sum()) / k ** 2
    # extrapolation variance of the trend line at the mean recent time, inflated for carry-over
    xr = np.array([1.0, tr.mean()])
    v_line = float(xr @ np.linalg.inv(A.T @ A) @ xr) * inflation(r1)
    se = sd * math.sqrt(v_recent + v_line)
    z = dep / se if se > 0 else 0.0
    nb_eff = max(len(yb) / inflation(r1), 4.0)
    p = float(2 * _st.t.sf(abs(z), df=nb_eff - 2))
    return {"valid": True, "departure": dep, "z": float(z), "p": p, "sd": sd, "r1": r1, "trend": float(coef[1])}


def model_choice(x: np.ndarray, t: np.ndarray, cp: int | None) -> dict:
    """Compare constant / linear trend / step-at-cp / step+trend by BIC. The physics of a
    discovery differs: a trend is gradual forcing, a step is a regime shift (or an artifact)."""
    ok = ~np.isnan(x)
    x, t = x[ok], t[ok].astype(float)
    n = len(x)
    out = {}

    def bic(rss, k):
        return n * math.log(max(rss, 1e-12) / n) + k * math.log(n)

    out["const"] = bic(np.sum((x - x.mean()) ** 2), 1)
    A = np.vstack([np.ones(n), t - t.mean()]).T
    rss = np.linalg.lstsq(A, x, rcond=None)[1]
    out["trend"] = bic(float(rss[0]) if len(rss) else 0.0, 2)
    if cp is not None and 3 <= cp <= n - 4:
        step = (np.arange(n) > cp).astype(float)
        A2 = np.vstack([np.ones(n), step]).T
        r2 = np.linalg.lstsq(A2, x, rcond=None)[1]
        out["shift"] = bic(float(r2[0]) if len(r2) else 0.0, 3)  # +1 for the estimated break date
        A3 = np.vstack([np.ones(n), step, t - t.mean()]).T
        r3 = np.linalg.lstsq(A3, x, rcond=None)[1]
        out["shift+trend"] = bic(float(r3[0]) if len(r3) else 0.0, 4)
    best = min(out, key=out.get)
    ranked = sorted(out.values())
    return {"best": best, "bic": out, "margin": float(ranked[1] - ranked[0]) if len(ranked) > 1 else 0.0}


def seasonal_phase(clim: np.ndarray) -> tuple[float, float]:
    """Amplitude and phase (day of year of the peak) of the first harmonic of a 12-month climatology."""
    m = np.arange(12)
    c = np.nansum(clim * np.cos(2 * np.pi * m / 12))
    s = np.nansum(clim * np.sin(2 * np.pi * m / 12))
    amp = 2 * math.hypot(c, s) / 12
    phase_month = (math.atan2(s, c) / (2 * np.pi) * 12) % 12
    return amp, phase_month * 365.25 / 12 + 15


def analyze_series(monthly: np.ndarray, years: np.ndarray, months: list[int] | None = None,
                   recent_years: int = 3, n_boot: int = 200, rng: np.random.Generator | None = None,
                   r_floor: float = 0.0, sd_floor: float = 0.0) -> dict:
    """All tests for one location. `monthly` is [nyear, 12].

    Every p-value accounts for year-to-year carry-over (AR(1), bias-corrected, at least `r_floor`, which the sweep
    sets to the regional carry-over so a lucky low estimate in one cell cannot undo the correction)."""
    rng = rng or np.random.default_rng(0)
    idx = [m - 1 for m in (months or range(1, 13))]
    with np.errstate(all="ignore"):
        ann = np.nanmean(monthly[:, idx], axis=1)
    ok = ~np.isnan(ann)
    if ok.sum() < 15:
        return {"valid": False}
    yrs = years.astype(float)
    sd = float(np.nanstd(ann))
    mk = mann_kendall(ann, r_floor)
    slope = sen_slope(ann, yrs)
    pt = pettitt(ann, r_floor)
    res = {"valid": True, "n_years": int(ok.sum()), "mean": float(np.nanmean(ann)), "sd": sd,
           "mk_tau": mk["tau"], "mk_p": mk["p"], "sen_per_decade": slope * 10, "r1": mk["r1"],
           "trend_effect": slope * (np.nanmax(yrs) - np.nanmin(yrs)) / sd if sd else 0.0}
    if pt["cp"] is not None:
        cp = pt["cp"]
        a, b = ann[: cp + 1], ann[cp + 1:]
        pooled = math.sqrt((np.nanvar(a) + np.nanvar(b)) / 2) or sd or 1.0
        res.update({"cp_year": int(years[cp + 1]), "cp_p": pt["p"],
                    "shift": float(np.nanmean(b) - np.nanmean(a)),
                    "shift_effect": float((np.nanmean(b) - np.nanmean(a)) / pooled)})
        mc = model_choice(ann, yrs, cp)
        res.update({"best_model": mc["best"], "bic_margin": mc["margin"]})
        # variance change around the break (Brown-Forsythe); carry-over inflates the variance of a variance
        # estimate by (1 + r^2) / (1 - r^2), so the statistic is deflated by that factor
        if len(a[~np.isnan(a)]) >= 6 and len(b[~np.isnan(b)]) >= 6:
            ra = a - np.nanmean(a)
            rb = b - np.nanmean(b)
            na, nb = int((~np.isnan(ra)).sum()), int((~np.isnan(rb)).sum())
            va, vb = float(np.nanvar(ra, ddof=1)), float(np.nanvar(rb, ddof=1))
            res.update({"var_ratio": vb / max(va, 1e-12), "var_p": var_change_p(va, vb, na, nb, mk["r1"])})
    # seasonal-cycle phase shift: first half vs second half of the record; the null shuffles BLOCKS of 5 years
    # (shuffling single years would destroy carry-over and make the null too narrow)
    half = len(years) // 2
    if months is None and half >= 8:
        A, B = monthly[:half], monthly[half:]
        def ph(block):
            with np.errstate(all="ignore"):
                return seasonal_phase(np.nanmean(block, axis=0))
        ampA, phA = ph(A)
        ampB, phB = ph(B)
        dphase = ((phB - phA + 182.6) % 365.25) - 182.6
        boots = []
        allm = monthly[~np.isnan(monthly).any(axis=1)]
        if len(allm) >= 16:
            nb_ = int(math.ceil(len(allm) / 5))
            blocks = [np.arange(i * 5, min((i + 1) * 5, len(allm))) for i in range(nb_)]
            for _ in range(n_boot):
                order = np.concatenate([blocks[i] for i in rng.permutation(nb_)])
                _, p1 = ph(allm[order[:half]])
                _, p2 = ph(allm[order[half:]])
                boots.append(abs(((p2 - p1 + 182.6) % 365.25) - 182.6))
            res.update({"phase_shift_days": float(dphase), "amp_ratio": float(ampB / ampA) if ampA else np.nan,
                        "phase_p": float((np.sum(np.array(boots) >= abs(dphase)) + 1) / (len(boots) + 1))})
    # recent extreme: the last K years beyond the long-term trend, with carry-over accounted for
    if ok.sum() > recent_years + 10:
        rd = recent_departure(ann, recent_years, r_floor, sd_floor)
        if rd.get("valid"):
            res.update({"recent_diff": rd["departure"], "recent_base_sd": rd["sd"], "recent_z": rd["z"],
                        "recent_p": rd["p"], "recent_k": int(recent_years),
                        "recent_rank": int((ann[~np.isnan(ann)] > np.nanmean(ann[-recent_years:])).sum())})
    return res


# ---------------------------------------------------------------------------- multiple testing

def fdr_bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------- the sweep (stage 2)

def regional_carryover(cube: "Cube", months: list[int] | None = None) -> float:
    """Median bias-corrected lag-1 autocorrelation of detrended annual series over the field."""
    ann = cube.annual(months)
    rs = []
    for i in range(ann.shape[0]):
        for j in range(ann.shape[1]):
            y = ann[i, j]
            ok = ~np.isnan(y)
            if ok.sum() >= 15:
                t = np.arange(len(y), dtype=float)[ok]
                rs.append(ar1(y[ok] - np.polyval(np.polyfit(t, y[ok], 1), t)))
    return float(np.median(rs)) if rs else 0.0


def sweep(df: pd.DataFrame, variable: str | None = None, months: list[int] | None = None,
          n_boot: int = 120) -> pd.DataFrame:
    """Run every test at every grid cell. Returns one row per cell with p-values and FDR q-values
    computed across the whole field (field significance, not per-cell).

    Carry-over: each cell's correction uses at least the field's median carry-over. Year-to-year variability that
    is abnormally small (smoothed or clipped data) is floored at half the field's median before testing recent
    years (found live: 89% of "recent extremes" in the Horn of Africa disappeared once floored)."""
    cube = to_cube(df, variable)
    rng = np.random.default_rng(42)
    r_floor = regional_carryover(cube, months)
    ann = cube.annual(months)
    with np.errstate(all="ignore"):
        sds = np.nanstd(ann, axis=-1)
    sd_floor = 0.5 * float(np.nanmedian(sds)) if np.isfinite(sds).any() else 0.0
    rows = []
    for i, lat in enumerate(cube.lats):
        for j, lon in enumerate(cube.lons):
            r = analyze_series(cube.data[i, j], cube.years, months=months, n_boot=n_boot, rng=rng,
                               r_floor=r_floor, sd_floor=sd_floor)
            if r.get("valid"):
                rows.append({"variable": cube.variable, "lat": float(lat), "lon": float(lon), **r})
    out = pd.DataFrame(rows)
    if "recent_base_sd" in out:
        out["recent_sd_floored"] = out["recent_base_sd"].values <= sd_floor + 1e-12
    for col in ("mk_p", "cp_p", "var_p", "phase_p", "recent_p"):
        if col in out:
            out[col.replace("_p", "_q")] = fdr_bh(out[col].values)
    out.attrs["season"] = months
    out.attrs["carryover"] = r_floor
    return out


# ---------------------------------------------------------------------------- classify + cluster (stage 3)

CLASSES = ("trend", "shift", "variance", "seasonality", "recent_extreme")


def classify_cells(sw: pd.DataFrame, q_max: float = 0.05, p_form: float | None = 0.05) -> pd.DataFrame:
    """Assign each cell to its dominant signal class and sign.

    Default (cluster inference): a cell joins a candidate family when its own carry-over-adjusted p <= p_form;
    whether the FAMILY is real is then decided by its size against surrogate fields that keep carry-over and the
    correlation between neighbouring cells (synlib.cluster_null). Per-cell FDR alone treated correlated cells as
    independent and had no power for a coherent regional signal. With p_form=None the old per-cell FDR (q <= q_max)
    is used. Seasonality has no field test, so its cells are only taken at q <= q_max."""
    rows = []
    sel = (lambda r, c: r.get(c + "_p", 1) <= p_form) if p_form is not None else (lambda r, c: r.get(c + "_q", 1) <= q_max)
    for _, r in sw.iterrows():
        best = r.get("best_model", "trend")
        if sel(r, "cp") and best in ("shift", "shift+trend"):
            rows.append({**r, "cls": "shift", "sign": np.sign(r["shift"]), "q": r["cp_q"], "p": r["cp_p"],
                         "effect": r["shift_effect"], "onset": r["cp_year"]})
        elif sel(r, "mk"):
            rows.append({**r, "cls": "trend", "sign": np.sign(r["sen_per_decade"]), "q": r["mk_q"], "p": r["mk_p"],
                         "effect": r["trend_effect"], "onset": np.nan})
        if sel(r, "var"):
            rows.append({**r, "cls": "variance", "sign": np.sign(np.log(r["var_ratio"])), "q": r["var_q"],
                         "p": r["var_p"], "effect": float(np.log(r["var_ratio"])), "onset": r.get("cp_year", np.nan)})
        if r.get("phase_q", 1) <= q_max:
            rows.append({**r, "cls": "seasonality", "sign": np.sign(r["phase_shift_days"]), "q": r["phase_q"],
                         "p": r["phase_p"], "effect": r["phase_shift_days"], "onset": np.nan})
        if sel(r, "recent"):
            rows.append({**r, "cls": "recent_extreme", "sign": np.sign(r["recent_z"]), "q": r["recent_q"],
                         "p": r["recent_p"], "effect": r["recent_z"], "onset": np.nan})
    return pd.DataFrame(rows)


def families(cells: pd.DataFrame, onset_window: int = 3) -> pd.DataFrame:
    """Group significant cells into spatially connected 'anomaly families' (8-neighbour adjacency on
    the grid, same variable/class/sign, and for shifts an onset within +-onset_window years)."""
    if cells.empty:
        return pd.DataFrame()
    out = []
    for (var, cls, sign), g in cells.groupby(["variable", "cls", "sign"]):
        lats = np.sort(g["lat"].unique())
        lons = np.sort(g["lon"].unique())
        dlat = np.min(np.diff(lats)) if len(lats) > 1 else 0.5
        dlon = np.min(np.diff(lons)) if len(lons) > 1 else 0.625
        pts = g.reset_index(drop=True)
        n = len(pts)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        # grid-index lookup: each cell only checks its 8 neighbours (linear time instead of all pairs)
        gi = np.rint((pts["lat"].values - lats.min()) / dlat).astype(int)
        gj = np.rint((pts["lon"].values - lons.min()) / dlon).astype(int)
        where = {(a, b): k for k, (a, b) in enumerate(zip(gi, gj))}
        onset = pts["onset"].values
        for a in range(n):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    b = where.get((gi[a] + di, gj[a] + dj))
                    if b is None or b <= a:
                        continue
                    if cls == "shift" and abs(onset[a] - onset[b]) > onset_window:
                        continue
                    parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for a in range(n):
            groups.setdefault(find(a), []).append(a)
        for k, members in enumerate(sorted(groups.values(), key=len, reverse=True)):
            m = pts.iloc[members]
            fam = f"{var}:{cls}{'+' if sign > 0 else '-'}:{k + 1:02d}"
            onset = float(np.nanmedian(m["onset"])) if m["onset"].notna().any() else np.nan
            out.append({"family": fam, "variable": var, "cls": cls, "sign": int(sign), "n_cells": len(m),
                        "lat": float(m["lat"].mean()), "lon": float(m["lon"].mean()),
                        "lat_min": float(m["lat"].min()), "lat_max": float(m["lat"].max()),
                        "lon_min": float(m["lon"].min()), "lon_max": float(m["lon"].max()),
                        "onset": onset, "effect": float(m["effect"].median()), "q": float(m["q"].min()),
                        "p": float(m["p"].min()) if "p" in m else float(m["q"].min()),
                        "cells": json.dumps([[float(a), float(b)] for a, b in zip(m["lat"], m["lon"])])})
    return pd.DataFrame(out).sort_values(["n_cells", "q"], ascending=[False, True]).reset_index(drop=True)


def source_transition_flag(variable: str, onset: float, window: int = 1) -> str | None:
    for year, why in KNOWN_SOURCE_TRANSITIONS.get(variable, []):
        if not np.isnan(onset) and abs(onset - year) <= window:
            return f"onset {int(onset)} is within {window} yr of {year}: {why}"
    return None


def corroboration(fam: dict, sweeps: dict[str, pd.DataFrame], p_max: float = 0.1) -> dict:
    """Fraction of the family's cells where a physically linked variable also changes
    significantly (same class). sweeps maps variable -> sweep DataFrame."""
    cells = json.loads(fam["cells"]) if isinstance(fam["cells"], str) else fam["cells"]
    out = {}
    for lv in LINKED_VARIABLES.get(fam["variable"], []):
        sw = sweeps.get(lv)
        if sw is None or sw.empty:
            continue
        col = {"trend": "mk_p", "shift": "cp_p", "variance": "var_p", "seasonality": "phase_p",
               "recent_extreme": "recent_p"}[fam["cls"]]
        if col not in sw:
            continue
        hits = 0
        for la, lo in cells:
            d = (sw["lat"] - la) ** 2 + (sw["lon"] - lo) ** 2
            r = sw.loc[d.idxmin()]
            hits += int(r[col] <= p_max)
        out[lv] = hits / max(len(cells), 1)
    return out


# ---------------------------------------------------------------------------- figures

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    return plt


def plot_series(path: str, years, values, title: str, units: str = "", cp_year: int | None = None,
                months: np.ndarray | None = None) -> str:
    """Annual series with Sen trend and optional break; if `months` ([nyear,12]) is given also draws
    the monthly anomaly strip, which is often where an eye catches something tests miss."""
    plt = _style()
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    nrows = 2 if months is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 3.2 * nrows), squeeze=False)
    ax = axes[0, 0]
    ax.plot(years, values, "o-", ms=3.5, lw=1.2, color="#1f4e79")
    s = sen_slope(values, years)
    ok = ~np.isnan(values)
    mid = np.nanmedian(values - s * years)
    ax.plot(years, mid + s * years, "--", color="#c0392b", lw=1, label=f"Sen slope {s * 10:+.3g}/decade")
    if cp_year:
        ax.axvline(cp_year - 0.5, color="#8e44ad", lw=1.2, label=f"break {cp_year}")
        a = values[(years < cp_year) & ok]
        b = values[(years >= cp_year) & ok]
        ax.hlines([a.mean(), b.mean()], [years.min(), cp_year - 0.5], [cp_year - 0.5, years.max()],
                  colors="#8e44ad", linestyles=":", lw=1.2)
    ax.set_title(title)
    ax.set_ylabel(units)
    ax.legend(frameon=False, fontsize=8)
    if months is not None:
        ax2 = axes[1, 0]
        clim = np.nanmean(months, axis=0)
        anom = months - clim
        lim = np.nanpercentile(np.abs(anom), 98) or 1
        im = ax2.imshow(anom.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                        extent=[years.min() - 0.5, years.max() + 0.5, 0.5, 12.5])
        ax2.set_yticks([1, 4, 7, 10])
        ax2.set_yticklabels(["Jan", "Apr", "Jul", "Oct"])
        ax2.set_title("Monthly anomalies (vs. full-record climatology)")
        ax2.grid(False)
        fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.01, label=units)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_field(path: str, sw: pd.DataFrame, column: str, title: str, q_column: str | None = None,
               q_max: float = 0.05, cmap: str = "RdBu_r") -> str:
    """Map of one sweep statistic; significant cells (field FDR) are outlined."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    v = sw[column].values
    lim = np.nanpercentile(np.abs(v), 98) or 1
    sc = ax.scatter(sw["lon"], sw["lat"], c=v, cmap=cmap, vmin=-lim, vmax=lim, s=150, marker="s")
    if q_column and q_column in sw:
        sig = sw[sw[q_column] <= q_max]
        ax.scatter(sig["lon"], sig["lat"], facecolors="none", edgecolors="k", s=150, marker="s", lw=1.1)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    ax.set_title(title + (f"\n(outlined: FDR q <= {q_max})" if q_column else ""))
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label=column)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def series_text(years, values, per_line: int = 10, digits: int = 2) -> str:
    """Raw numbers as compact text, the Earth-data analogue of reading DNA into context."""
    pairs = [f"{int(y)}:{v:.{digits}f}" if not np.isnan(v) else f"{int(y)}:NA" for y, v in zip(years, values)]
    return "\n".join("  ".join(pairs[i:i + per_line]) for i in range(0, len(pairs), per_line))


# ---------------------------------------------------------------------------- robustness (senior-review checks)

# Calendar years dominated by strong ENSO events (ONI peak >= ~1.5 or <= ~-1.5, both years of the event).
ENSO_STRONG_YEARS = {
    "el_nino": [1982, 1983, 1987, 1988, 1991, 1992, 1997, 1998, 2009, 2010, 2015, 2016, 2023, 2024],
    "la_nina": [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2010, 2011, 2020, 2021, 2022],
}


def _trend_stats(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 1.0
    return sen_slope(y[ok], t[ok]), mann_kendall(y[ok])["p"]


def _step_stats(y: np.ndarray, t: np.ndarray, onset: int | None = None) -> tuple[float, float, int | None]:
    ok = ~np.isnan(y)
    yy, tt = y[ok], t[ok]
    if len(yy) < 10:
        return np.nan, 1.0, None
    pt = pettitt(yy)
    if pt["cp"] is None:
        return np.nan, 1.0, None
    cp_year = int(tt[pt["cp"] + 1]) if pt["cp"] + 1 < len(tt) else int(tt[-1])
    split = onset if onset is not None else cp_year
    a, b = yy[tt < split], yy[tt >= split]
    step = float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan
    return step, pt["p"], cp_year


def robustness(values, years, cls: str, onset: int | None = None, n_boot: int = 300, block: int = 4,
               seed: int = 0) -> dict:
    """Checks a senior reviewer runs before believing a trend or a step in an annual series.

    trend: leave-one-year-out (sign and p<0.1 hold for every year left out), endpoints (start 3 years later /
           end 3 years earlier), strong-ENSO years removed, moving-block bootstrap CI of the Sen slope.
    shift: leverage (break survives removing the most extreme year and the break year itself, and stays within
           2 years), strong-ENSO years removed, block-bootstrap CI of the step, and the step is not one event
           year (the El Nino change-point trap: one huge year can pull Pettitt to itself).
    Returns per-check booleans, 'passed' (count), 'n_checks' and 'robust' (all passed).
    """
    y = np.asarray(values, float)
    t = np.asarray(years, float)
    rng = np.random.default_rng(seed)
    out: dict = {"cls": cls}
    enso = set(ENSO_STRONG_YEARS["el_nino"]) | set(ENSO_STRONG_YEARS["la_nina"])
    keep = np.array([int(v) not in enso for v in t])
    if cls == "trend":
        s0, p0 = _trend_stats(y, t)
        sign = np.sign(s0)
        loo = []
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            s, p = _trend_stats(y[m], t[m])
            loo.append(np.sign(s) == sign and p < 0.1)
        out["leave_one_out"] = bool(all(loo))
        s1, p1 = _trend_stats(y[t >= t.min() + 3], t[t >= t.min() + 3])
        s2, p2 = _trend_stats(y[t <= t.max() - 3], t[t <= t.max() - 3])
        out["endpoints"] = bool(np.sign(s1) == sign and np.sign(s2) == sign and p1 < 0.1 and p2 < 0.1)
        s3, p3 = _trend_stats(y[keep], t[keep])
        out["without_enso_years"] = bool(np.sign(s3) == sign and p3 < 0.1)
        ok = ~np.isnan(y)
        fit = np.nanmedian(y[ok] - s0 * t[ok]) + s0 * t
        res = y - fit
        boots = []
        n = len(y)
        for _ in range(n_boot):
            idx = np.concatenate([np.arange(st, st + block) for st in rng.integers(0, max(n - block, 1), n // block + 1)])[:n]
            yb = fit + res[np.clip(idx, 0, n - 1)]
            boots.append(_trend_stats(yb, t)[0])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out["bootstrap_ci"] = [round(float(lo) * 10, 4), round(float(hi) * 10, 4)]  # per decade
        out["bootstrap"] = bool(lo > 0 or hi < 0)
        out["slope_per_decade"] = round(float(s0) * 10, 4)
        out["p"] = float(p0)
        checks = ["leave_one_out", "endpoints", "without_enso_years", "bootstrap"]
    elif cls == "shift":
        step0, p0, cp0 = _step_stats(y, t)
        onset = int(onset) if onset is not None and not np.isnan(onset) else cp0
        sign = np.sign(step0)
        a = np.where(t < onset, y - np.nanmean(y[t < onset]), y - np.nanmean(y[t >= onset]))
        worst = int(np.nanargmax(np.abs(a))) if np.isfinite(a).any() else 0
        m = np.ones(len(y), bool)
        m[worst] = False
        m[np.flatnonzero(t == onset)] = False
        s1, p1, cp1 = _step_stats(y[m], t[m])
        out["leverage"] = bool(cp1 is not None and abs(cp1 - onset) <= 2 and p1 < 0.05 and np.sign(s1) == sign)
        s2, p2, cp2 = _step_stats(y[keep], t[keep])
        out["without_enso_years"] = bool(cp2 is not None and abs(cp2 - onset) <= 3 and p2 < 0.1 and np.sign(s2) == sign)
        near = [v for v in t if abs(v - onset) <= 1]
        out["onset_on_strong_enso_year"] = bool(any(int(v) in enso for v in near))
        fit = np.where(t < onset, np.nanmean(y[t < onset]), np.nanmean(y[t >= onset]))
        res = y - fit
        n = len(y)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([np.arange(st, st + block) for st in rng.integers(0, max(n - block, 1), n // block + 1)])[:n]
            yb = fit + res[np.clip(idx, 0, n - 1)]
            boots.append(np.nanmean(yb[t >= onset]) - np.nanmean(yb[t < onset]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out["bootstrap_ci"] = [round(float(lo), 4), round(float(hi), 4)]
        out["bootstrap"] = bool(lo > 0 or hi < 0)
        # one event year carrying the step: the step without the 2 years after onset must keep half its size
        m2 = ~((t >= onset) & (t <= onset + 1))
        s4 = np.nanmean(y[m2 & (t >= onset)]) - np.nanmean(y[m2 & (t < onset)]) if (m2 & (t >= onset)).any() else np.nan
        out["not_single_event"] = bool(np.isfinite(s4) and np.sign(s4) == sign and abs(s4) >= 0.5 * abs(step0))
        out["step"] = round(float(step0), 4)
        out["onset"] = int(onset) if onset is not None else None
        out["p"] = float(p0)
        checks = ["leverage", "without_enso_years", "bootstrap", "not_single_event"]
    else:  # recent_extreme / variance / seasonality: the recent years must not all be one ENSO event
        recent = t >= t.max() - 2
        out["recent_years_all_enso"] = bool(all(int(v) in enso for v in t[recent]))
        out["without_enso_years"] = not out["recent_years_all_enso"]
        checks = ["without_enso_years"]
    out["passed"] = int(sum(bool(out[c]) for c in checks))
    out["n_checks"] = len(checks)
    out["robust"] = out["passed"] == len(checks)
    return out


def _row_detrend(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS trend per row of Y [ncell, nyear] (no NaNs): returns (trend part, residual)."""
    n = Y.shape[1]
    t = np.arange(n, dtype=float) - (n - 1) / 2
    b = (Y * t).sum(1) / (t * t).sum()
    tr = Y.mean(1, keepdims=True) + b[:, None] * t[None, :]
    return tr, Y - tr


def _row_ar1(R: np.ndarray, r_floor: float = 0.0) -> np.ndarray:
    """Bias-corrected lag-1 autocorrelation per row (same rule as ar1)."""
    n = R.shape[1]
    a, b = R[:, :-1] - R[:, :-1].mean(1, keepdims=True), R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    r = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)
    r = r + (1 + 4 * np.maximum(r, 0)) / n
    return np.clip(np.maximum(r, r_floor), 0.0, 0.9)


def _mk_p_matrix(Y: np.ndarray, r_floor: float = 0.0, adjust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Mann-Kendall p and sign for rows of Y [ncell, nyear] (no NaNs), carry-over adjusted."""
    n = Y.shape[1]
    iu = np.triu_indices(n, 1)
    S = np.sign(Y[:, None, :] - Y[:, :, None])[:, iu[0], iu[1]].sum(axis=1)
    var = np.full(len(Y), n * (n - 1) * (2 * n + 5) / 18)
    if adjust:
        r = _row_ar1(_row_detrend(Y)[1], r_floor)
        var = var * (1 + r) / (1 - r)
    z = (S - np.sign(S)) / np.sqrt(var)
    return 2 * _st.norm.sf(np.abs(z)), np.sign(S)


def _pettitt_p_matrix(Y: np.ndarray, r_floor: float = 0.0, adjust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Pettitt p and step sign for rows of Y (no NaNs), carry-over adjusted like `pettitt`."""
    n = Y.shape[1]
    V = np.sign(Y[:, :, None] - Y[:, None, :]).sum(axis=2)  # V_t = sum_j sign(x_t - x_j)
    U = np.cumsum(V, axis=1)[:, :-1]
    K = np.abs(U).max(axis=1)
    cp = np.abs(U).argmax(axis=1)
    f = np.ones(len(Y))
    idx = np.arange(n)[None, :]
    after = idx > cp[:, None]
    if adjust:
        A = _step_design(n, cp, True)
        c, _ = _batched_rss(A, Y)
        r = _row_ar1(Y - np.einsum("mnk,mk->mn", A, c), r_floor)
        f = (1 + r) / (1 - r)
    p = np.minimum(1.0, 2 * np.exp(-6 * K ** 2 / (f * (n ** 3 + n ** 2))))
    step = (Y * after).sum(1) / np.maximum(after.sum(1), 1) - (Y * ~after).sum(1) / np.maximum((~after).sum(1), 1)
    return p, np.sign(step)


def _recent_p_matrix(Y: np.ndarray, k: int = 3, r_floor: float = 0.0, sd_floor: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `recent_departure` for rows of Y (no NaNs)."""
    n = Y.shape[1]
    t = np.arange(n, dtype=float)
    tb = t[:-k]
    A = np.vstack([np.ones(len(tb)), tb]).T
    AtA_inv = np.linalg.inv(A.T @ A)
    coef = Y[:, :-k] @ A @ AtA_inv                      # [ncell, 2]
    res = Y[:, :-k] - coef @ A.T
    sd = np.maximum(np.sqrt((res ** 2).sum(1) / (len(tb) - 2)), sd_floor)
    r = _row_ar1(res, r_floor)
    tr = t[-k:]
    dep = (Y[:, -k:] - (coef[:, :1] + coef[:, 1:] * tr[None, :])).mean(1)
    lags = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    v_recent = (r[:, None, None] ** lags[None]).sum((1, 2)) / k ** 2
    xr = np.array([1.0, tr.mean()])
    v_line = float(xr @ AtA_inv @ xr) * (1 + r) / (1 - r)
    se = sd * np.sqrt(v_recent + v_line)
    z = np.where(se > 0, dep / np.where(se > 0, se, 1), 0.0)
    nb_eff = np.maximum(len(tb) / ((1 + r) / (1 - r)), 4.0)
    return 2 * _st.t.sf(np.abs(z), df=nb_eff - 2), np.sign(dep)


def _var_p_matrix(Y: np.ndarray, r_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised variance-change test around each row's Pettitt break (same rule as the per-cell var_p)."""
    m, n = Y.shape
    cp = _pettitt_cp_matrix(Y)
    after = np.arange(n)[None, :] > cp[:, None]
    na, nb = (~after).sum(1), after.sum(1)
    ma = (Y * ~after).sum(1) / np.maximum(na, 1)
    mb = (Y * after).sum(1) / np.maximum(nb, 1)
    va = (((Y - ma[:, None]) ** 2) * ~after).sum(1) / np.maximum(na - 1, 1)
    vb = (((Y - mb[:, None]) ** 2) * after).sum(1) / np.maximum(nb - 1, 1)
    r = _row_ar1(_row_detrend(Y)[1], r_floor)
    f2 = (1 + r * r) / (1 - r * r)
    ok = (na >= 6) & (nb >= 6) & (va > 0) & (vb > 0)
    z = np.where(ok, np.log(np.where(ok, vb / np.where(va > 0, va, 1), 1)) /
                 np.sqrt(f2 * (2 / np.maximum(na - 1, 1) + 2 / np.maximum(nb - 1, 1))), 0.0)
    return np.where(ok, 2 * _st.norm.sf(np.abs(z)), 1.0), np.sign(z)


def _shift_p_matrix(Y: np.ndarray, r_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pettitt p, but a cell only counts as a SHIFT when the BIC model choice prefers a step (the same rule
    classify_cells applies): Pettitt also fires on plain trends."""
    p, sg = _pettitt_p_matrix(Y, r_floor)
    stepish = _model_choice_matrix(Y, _pettitt_cp_matrix(Y))
    return np.where(stepish, p, 1.0), sg


def _cell_tests(Y: np.ndarray, cls: str, r_floor: float, sd_floor: float, chunk: int = 300):
    f = {"trend": lambda M: _mk_p_matrix(M, r_floor), "shift": lambda M: _shift_p_matrix(M, r_floor),
         "recent_extreme": lambda M: _recent_p_matrix(M, 3, r_floor, sd_floor),
         "variance": lambda M: _var_p_matrix(M, r_floor)}[cls]
    ps, sg = [], []
    for i in range(0, len(Y), chunk):
        p, s = f(Y[i:i + chunk])
        ps.append(p)
        sg.append(s)
    return np.concatenate(ps), np.concatenate(sg)


def _pettitt_cp_matrix(Y: np.ndarray) -> np.ndarray:
    V = np.sign(Y[:, :, None] - Y[:, None, :]).sum(axis=2)
    return np.abs(np.cumsum(V, axis=1)[:, :-1]).argmax(axis=1)


def _batched_rss(A: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares for a batch: A [m, n, k], Y [m, n] -> (coef [m, k], rss [m])."""
    G = np.einsum("mnk,mnl->mkl", A, A) + 1e-9 * np.eye(A.shape[2])[None]
    c = np.linalg.solve(G, np.einsum("mnk,mn->mk", A, Y)[..., None])[..., 0]
    r = Y - np.einsum("mnk,mk->mn", A, c)
    return c, (r * r).sum(1)


def _step_design(n: int, cp: np.ndarray, with_trend: bool) -> np.ndarray:
    tc = np.arange(n, dtype=float) - (n - 1) / 2
    step = (np.arange(n)[None, :] > cp[:, None]).astype(float)
    cols = [np.ones_like(step), step] + ([np.broadcast_to(tc, step.shape)] if with_trend else [])
    return np.stack(cols, axis=2)


def _step_trend_parts(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the linear-trend part of a step+trend fit at the row's Pettitt break, and the residual."""
    n = Y.shape[1]
    A = _step_design(n, _pettitt_cp_matrix(Y), True)
    c, _ = _batched_rss(A, Y)
    R = Y - np.einsum("mnk,mk->mn", A, c)
    tc = np.arange(n, dtype=float) - (n - 1) / 2
    return Y.mean(1, keepdims=True) + c[:, 2:3] * tc[None, :], R


def _model_choice_matrix(Y: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """Vectorised model_choice: True where a step model (shift or shift+trend) has the lowest BIC."""
    m, n = Y.shape
    tc = np.arange(n, dtype=float) - (n - 1) / 2
    def bic(rss, k):
        return n * np.log(np.maximum(rss, 1e-12) / n) + k * math.log(n)
    b_const = bic(((Y - Y.mean(1, keepdims=True)) ** 2).sum(1), 1)
    _, rt = _batched_rss(np.stack([np.ones((m, n)), np.broadcast_to(tc, (m, n))], 2), Y)
    _, rs = _batched_rss(_step_design(n, cp, False), Y)
    _, rst = _batched_rss(_step_design(n, cp, True), Y)
    valid = (cp >= 3) & (cp <= n - 4)
    best_step = np.minimum(bic(rs, 3), bic(rst, 4))
    return valid & (best_step < np.minimum(b_const, bic(rt, 2)))


def phase_surrogates(Y: np.ndarray, rng: np.random.Generator, keep_trend: bool, remove_step: bool = False,
                     parts: tuple | None = None) -> np.ndarray:
    """Surrogate field with the same autocorrelation in every cell AND the same correlation between cells:
    each cell's residual series is Fourier-transformed and ALL cells get the same random phases (Ebisuzaki 1997;
    Theiler et al. 1992). With keep_trend the cell's linear trend is added back, so the null is 'trend + red noise'
    (a step or recent extreme must beat what a trend alone produces). With remove_step the residual is taken around
    a fitted step + trend, so a real step does not leak into the null's red noise and make the test blind to steps."""
    if parts is None:
        parts = _step_trend_parts(Y) if remove_step else _row_detrend(Y)
    tr, R = parts
    n = Y.shape[1]
    F = np.fft.rfft(R, axis=1)
    ph = rng.uniform(0, 2 * np.pi, F.shape[1])
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    S = np.fft.irfft(F * np.exp(1j * ph)[None, :], n=n, axis=1)
    return (tr + S) if keep_trend else (Y.mean(1, keepdims=True) + S)


def grid_clusters(lat: np.ndarray, lon: np.ndarray, label: np.ndarray) -> list[tuple[int, list[int]]]:
    """Connected groups (8-neighbour on the regular grid) of cells sharing a nonzero label (e.g. +1/-1).
    Returns [(label, member indices)], largest first."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    sel = np.flatnonzero(label != 0)
    if len(sel) == 0:
        return []
    ula, ulo = np.unique(lat), np.unique(lon)
    dla = np.min(np.diff(ula)) if len(ula) > 1 else 1.0
    dlo = np.min(np.diff(ulo)) if len(ulo) > 1 else 1.0
    gi = np.rint((lat - ula.min()) / dla).astype(int)
    gj = np.rint((lon - ulo.min()) / dlo).astype(int)
    where = {(gi[k], gj[k]): k for k in sel}
    parent = {k: k for k in sel}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for k in sel:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                m = where.get((gi[k] + di, gj[k] + dj))
                if m is not None and m != k and label[m] == label[k]:
                    parent[find(k)] = find(m)
    groups: dict[int, list[int]] = {}
    for k in sel:
        groups.setdefault(find(k), []).append(int(k))
    return sorted(((int(label[g[0]]), g) for g in groups.values()), key=lambda x: -len(x[1]))


TESTABLE_CLASSES = ("trend", "shift", "recent_extreme", "variance")


def cluster_null(lat, lon, Y: np.ndarray, cls: str, n_sim: int = 400, p_form: float | None = 0.05, q: float = 0.05,
                 seed: int = 0) -> dict:
    """Null distribution of the LARGEST same-sign cluster of FDR-significant cells in surrogate fields that keep
    each cell's carry-over and the spatial correlation between cells (phase surrogates). A family is field
    significant when it is larger than the null's 95th percentile: this is what stops one coherent random
    excursion over a large area from becoming a "200-cell discovery".

    Fails closed: classes that cannot be tested this way, or fields too small to test, return valid=False and
    the caller must treat the family as NOT field significant."""
    Y = np.asarray(Y, float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if cls not in TESTABLE_CLASSES:
        return {"valid": False, "note": f"no field test for '{cls}' families; they cannot pass promotion"}
    good = np.isfinite(Y).sum(1) >= max(15, int(0.8 * Y.shape[1]))
    Y, lat, lon = Y[good], lat[good], lon[good]
    if len(Y) < 20 or Y.shape[1] < 15:
        return {"valid": False, "note": "too few cells or years for a field test"}
    mu = np.nanmean(Y, axis=1, keepdims=True)
    Y = np.where(np.isnan(Y), mu, Y)
    tr, R = _row_detrend(Y)
    r_floor = float(np.median(_row_ar1(R)))
    sd_floor = 0.5 * float(np.median(R.std(1)))
    rng = np.random.default_rng(seed)
    step_like = cls in ("shift", "variance")
    # H0 for steps: the cell's linear trend + red noise measured AROUND a fitted step (a real step must not leak
    # into the null's noise); for everything else: trend + red noise around the trend
    parts = (_row_detrend(Y)[0], _step_trend_parts(Y)[1]) if step_like else _row_detrend(Y)
    null = []
    for _ in range(n_sim):
        S = phase_surrogates(Y, rng, keep_trend=cls != "trend", parts=parts)
        p, sg = _cell_tests(S, cls, r_floor, sd_floor)
        lab = np.where((p <= p_form) if p_form is not None else (fdr_bh(p) <= q), sg, 0).astype(int)
        cl = grid_clusters(lat, lon, lab)
        null.append(len(cl[0][1]) if cl else 0)
    null = np.array(null)
    return {"valid": True, "cls": cls, "cells": int(len(Y)), "n_sim": n_sim, "null_max_cluster": null.tolist(),
            "null_p95": float(np.percentile(null, 95)), "carryover": r_floor}


FIELD_ALPHA = 0.05 / len(TESTABLE_CLASSES)  # one test per class and variable: keep the family-wise rate near 5%


def cluster_p(n_cells: int, null: dict, alpha: float = FIELD_ALPHA) -> dict:
    """Field significance of one family of `n_cells` against a cluster_null result (fails closed). The default
    alpha splits 5% over the four classes tested on each variable."""
    if not null or not null.get("valid"):
        return {"valid": False, "field_significant": False, "note": (null or {}).get("note", "not tested")}
    arr = np.asarray(null["null_max_cluster"])
    p = float((1 + (arr >= n_cells).sum()) / (len(arr) + 1))
    return {"valid": True, "field_significant": bool(p < alpha), "p_field": p, "alpha": alpha,
            "observed": int(n_cells), "null_p95": null["null_p95"]}


def field_significance(Y: np.ndarray, test: str = "trend", n_perm: int = 100, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """Livezey & Chen (1983) count test, now with phase surrogates (carry-over and spatial correlation kept;
    shuffling years destroyed the carry-over and made the null too narrow). Y: [ncell, nyear] annual values.
    Fails closed: when the test cannot run, field_significant is False."""
    Y = np.asarray(Y, float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    if len(Y) < 10 or test not in TESTABLE_CLASSES:
        return {"valid": False, "field_significant": False}
    tr, R = _row_detrend(Y)
    r_floor = float(np.median(_row_ar1(R)))
    sd_floor = 0.5 * float(np.median(R.std(1)))

    def count(M):
        return int((_cell_tests(M, test, r_floor, sd_floor)[0] < alpha).sum())

    obs = count(Y)
    rng = np.random.default_rng(seed)
    null = [count(phase_surrogates(Y, rng, keep_trend=test != "trend")) for _ in range(n_perm)]
    p = (1 + sum(v >= obs for v in null)) / (n_perm + 1)
    return {"valid": True, "test": test, "cells": int(len(Y)), "observed": obs,
            "null_p95": float(np.percentile(null, 95)), "null_mean": float(np.mean(null)), "p_field": float(p),
            "field_significant": bool(p < 0.05)}


# ---------------------------------------------------------------------------- honest p-values for a region

def effective_cells(R: np.ndarray) -> float:
    """Effective number of independent cells (Bretherton et al. 1999): N^2 / sum of squared correlations between
    the cells' anomaly series R [ncell, nyear]. Neighbouring grid cells are not independent evidence."""
    R = np.asarray(R, float)
    R = R[np.isfinite(R).all(1)]
    n = len(R)
    if n < 2:
        return float(n)
    C = np.corrcoef(R)
    C = np.nan_to_num(C, nan=0.0)
    return float(min(n, n * n / (C ** 2).sum()))


def sign_test_spatial(values: np.ndarray, R: np.ndarray) -> dict:
    """'k of N cells agree in sign' with the spatial correlation accounted for: the count is rescaled to the
    effective number of independent cells before the binomial test (counting 52 neighbouring cells as 52
    independent votes made p far too small)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"valid": False}
    k = int((v > 0).sum())
    neff = max(1.0, effective_cells(R))
    k_eff, n_eff = int(round(k / n * neff)), int(round(neff))
    p_naive = float(_st.binomtest(k, n, 0.5, alternative="greater").pvalue)
    p = float(_st.binomtest(k_eff, max(n_eff, 1), 0.5, alternative="greater").pvalue) if n_eff >= 1 else 1.0
    return {"valid": True, "cells": n, "positive": k, "effective_cells": round(neff, 1), "p": p, "p_naive": p_naive}


# ---------------------------------------------------------------------------- selection-aware change tests

def _ar1_noise(n: int, r: float, sd: float, rng: np.random.Generator, size: int) -> np.ndarray:
    e = rng.standard_normal((size, n))
    out = np.empty_like(e)
    out[:, 0] = e[:, 0]
    for t in range(1, n):
        out[:, t] = r * out[:, t - 1] + math.sqrt(1 - r * r) * e[:, t]
    return out * sd


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-300)


def correlation_change_test(y, x, years, split="scan", min_side: int = 8, n_sim: int = 2000, seed: int = 0,
                            detrend: bool = True) -> dict:
    """Has the link between a driver x and a target y changed between an early and a late period?

    split: a year (pre-registered: the late period starts that year) or "scan" (every split leaving min_side years
    each side is tried and the LARGEST change is kept). The p-value repeats exactly the same procedure on
    simulated data with a CONSTANT link (y = a + b x + AR(1) noise fitted to the whole record), so a split picked
    after looking at the data is not rewarded for being picked. Statistic: |r_late| - |r_early| (strengthening
    either sign) and its two-sided magnitude."""
    y, x, t = (np.asarray(v, float) for v in (y, x, years))
    ok = np.isfinite(y) & np.isfinite(x)
    y, x, t = y[ok], x[ok], t[ok]
    n = len(y)
    if n < 2 * min_side:
        return {"valid": False, "note": "record too short"}
    if detrend:
        y = y - np.polyval(np.polyfit(t, y, 1), t)
        x = x - np.polyval(np.polyfit(t, x, 1), t)
    if split == "scan":
        cands = [int(s) for s in t[min_side:n - min_side + 1]]
    else:
        cands = [int(split)]
    masks = np.array([t >= s for s in cands])                                   # [S, n]
    ok_m = (masks.sum(1) >= min_side) & ((~masks).sum(1) >= min_side)
    masks, cands = masks[ok_m], [c for c, o in zip(cands, ok_m) if o]
    if not len(cands):
        return {"valid": False, "note": "split leaves too few years on one side"}

    def stat(Y):                                                               # Y [m, n] -> best change per row
        out = np.full((Y.shape[0], len(cands)), np.nan)
        for j, m in enumerate(masks):
            out[:, j] = np.abs(_corr_rows(Y[:, m], np.broadcast_to(x[m], Y[:, m].shape))) - \
                np.abs(_corr_rows(Y[:, ~m], np.broadcast_to(x[~m], Y[:, ~m].shape)))
        return out
    obs_all = stat(y[None, :])[0]
    j = int(np.nanargmax(np.abs(obs_all)))
    obs = float(obs_all[j])
    b = float(np.polyfit(x, y, 1)[0])
    res = y - b * x
    r = ar1(res)
    rng = np.random.default_rng(seed)
    sims = b * x[None, :] + _ar1_noise(n, r, float(res.std()), rng, n_sim)
    null = np.nanmax(np.abs(stat(sims)), axis=1)
    p = float((1 + (null >= abs(obs)).sum()) / (n_sim + 1))
    m = masks[j]
    return {"valid": True, "split": cands[j], "scanned": split == "scan", "n_splits": len(cands),
            "r_early": float(_corr_rows(y[~m][None], x[~m][None])[0]), "r_late": float(_corr_rows(y[m][None], x[m][None])[0]),
            "change_abs_r": obs, "p_selection_aware": p, "carryover": r,
            "note": "p accounts for choosing the split after seeing the data" if split == "scan" else
                    "pre-registered split"}


def step_change_test(y, years, split="scan", min_side: int = 8, n_sim: int = 2000, seed: int = 0) -> dict:
    """Mean change between periods, selection-aware like correlation_change_test: the null is the series' own
    linear trend + AR(1) noise, and a scanned split pays for the scan."""
    y, t = np.asarray(y, float), np.asarray(years, float)
    ok = np.isfinite(y)
    y, t = y[ok], t[ok]
    n = len(y)
    if n < 2 * min_side:
        return {"valid": False, "note": "record too short"}
    cands = [int(s) for s in t[min_side:n - min_side + 1]] if split == "scan" else [int(split)]
    masks = np.array([t >= s for s in cands])
    ok_m = (masks.sum(1) >= min_side) & ((~masks).sum(1) >= min_side)
    masks, cands = masks[ok_m], [c for c, o in zip(cands, ok_m) if o]
    if not cands:
        return {"valid": False, "note": "split leaves too few years on one side"}
    tc = t - t.mean()

    def stat(Y):
        out = np.empty((Y.shape[0], len(cands)))
        for j, m in enumerate(masks):
            # step beyond the linear trend: regress on [1, trend, step]
            A = np.vstack([np.ones(n), tc, m.astype(float)]).T
            c = np.linalg.lstsq(A, Y.T, rcond=None)[0]
            out[:, j] = c[2]
        return out
    sd_scale = float(np.std(y))
    obs_all = stat(y[None])[0]
    j = int(np.argmax(np.abs(obs_all)))
    obs = float(obs_all[j])
    A0 = np.vstack([np.ones(n), tc]).T
    c0 = np.linalg.lstsq(A0, y, rcond=None)[0]
    res = y - A0 @ c0
    r = ar1(res)
    rng = np.random.default_rng(seed)
    sims = (A0 @ c0)[None, :] + _ar1_noise(n, r, float(res.std()), rng, n_sim)
    null = np.max(np.abs(stat(sims)), axis=1)
    p = float((1 + (null >= abs(obs)).sum()) / (n_sim + 1))
    return {"valid": True, "split": cands[j], "scanned": split == "scan", "step_beyond_trend": obs,
            "step_sd": obs / sd_scale if sd_scale else 0.0, "p_selection_aware": p, "carryover": r}


def sen_ci(x: np.ndarray, t: np.ndarray, alpha: float = 0.05, r_floor: float = 0.0) -> tuple[float, float]:
    """Confidence interval of the Sen slope from the Mann-Kendall variance (Gilbert 1987), with the same carry-over
    inflation as mann_kendall, so the interval and the p-value always agree."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    ok = ~np.isnan(x)
    x, t = x[ok], t[ok]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    i, j = np.triu_indices(n, 1)
    slopes = np.sort((x[j] - x[i]) / (t[j] - t[i]))
    var = n * (n - 1) * (2 * n + 5) / 18 * inflation(ar1(x - sen_slope(x, t) * t, r_floor))
    c = _st.norm.ppf(1 - alpha / 2) * math.sqrt(var)
    N = len(slopes)
    lo = int(np.floor((N - c) / 2)) - 1
    hi = int(np.ceil((N + c) / 2))
    return float(slopes[max(lo, 0)]), float(slopes[min(hi, N - 1)])


def correlation_test(y, x) -> dict:
    """Pearson r with a p-value whose degrees of freedom shrink for carry-over in BOTH series
    (n_eff = n (1 - r_x r_y) / (1 + r_x r_y), Bretherton et al. 1999)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 8:
        return {"valid": False}
    r = float(np.corrcoef(y, x)[0, 1])
    t = np.arange(n, dtype=float)
    ry = ar1(y - np.polyval(np.polyfit(t, y, 1), t))
    rx = ar1(x - np.polyval(np.polyfit(t, x, 1), t))
    neff = max(4.0, n * (1 - rx * ry) / (1 + rx * ry))
    tv = r * math.sqrt((neff - 2) / max(1 - r * r, 1e-12))
    return {"valid": True, "r": r, "n": n, "n_eff": round(neff, 1), "p": float(2 * _st.t.sf(abs(tv), neff - 2))}


def event_departure(y, years, event_years, n_sim: int = 0) -> dict:
    """How far are the given event years (e.g. [2023, 2024]) from the trend of all OTHER years, with carry-over
    accounted for (same construction as recent_departure, but the event can be anywhere in the record)."""
    y, t = np.asarray(y, float), np.asarray(years, float)
    ev = np.isin(t.astype(int), [int(v) for v in event_years])
    base = ~ev & np.isfinite(y)
    if base.sum() < 10 or not (ev & np.isfinite(y)).any():
        return {"valid": False}
    tb, yb = t[base], y[base]
    A = np.vstack([np.ones(len(tb)), tb - tb.mean()]).T
    c, *_ = np.linalg.lstsq(A, yb, rcond=None)
    res = yb - A @ c
    sd = float(np.std(res, ddof=2))
    r = ar1(res)
    te = t[ev & np.isfinite(y)]
    dep = float(np.mean(y[ev & np.isfinite(y)] - (c[0] + c[1] * (te - tb.mean()))))
    k = len(te)
    lags = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    v_ev = float((r ** lags).sum()) / k ** 2
    xr = np.array([1.0, te.mean() - tb.mean()])
    v_line = float(xr @ np.linalg.inv(A.T @ A) @ xr) * inflation(r)
    se = sd * math.sqrt(v_ev + v_line)
    z = dep / se if se > 0 else 0.0
    neff = max(len(yb) / inflation(r), 4.0)
    return {"valid": True, "departure": dep, "z": float(z), "p": float(2 * _st.t.sf(abs(z), neff - 2)),
            "sd": sd, "carryover": r, "event_years": [int(v) for v in te]}


def mediation_test(y, x, m, years, split=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ablation test for a mechanism claim 'x acts on y through m' (or 'the change in x->y comes from m').

    Without split: how much of the x->y association disappears when m is controlled for (the indirect share),
    with a block-bootstrap CI over years. With split: does the CHANGE of the x->y slope between periods disappear
    when m is controlled for? A mechanism is supported only when the effect shrinks significantly once m is in the
    model; otherwise the claim stays an association."""
    y, x, m, t = (np.asarray(v, float) for v in (y, x, m, years))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(m)
    y, x, m, t = y[ok], x[ok], m[ok], t[ok]
    n = len(y)
    if n < 12:
        return {"valid": False, "note": "too few years"}

    def z(v):
        v = v - np.polyval(np.polyfit(t, v, 1), t)
        return v / (v.std() or 1)
    y, x, m = z(y), z(x), z(m)

    def slopes(idx):
        yy, xx, mm = y[idx], x[idx], m[idx]
        b_tot = np.polyfit(xx, yy, 1)[0]
        A = np.vstack([np.ones(len(idx)), xx, mm]).T
        b_dir = np.linalg.lstsq(A, yy, rcond=None)[0][1]
        return b_tot, b_dir

    def stat(idx):
        if split is None:
            bt, bd = slopes(idx)
            return bt - bd, bt
        late = t[idx] >= split
        if late.sum() < 6 or (~late).sum() < 6:
            return np.nan, np.nan
        bt_e, bd_e = slopes(idx[~late])
        bt_l, bd_l = slopes(idx[late])
        return (bt_l - bt_e) - (bd_l - bd_e), bt_l - bt_e
    allidx = np.arange(n)
    mediated, total = stat(allidx)
    rng = np.random.default_rng(seed)
    block = 4
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = np.sort(idx)
        b = stat(idx)[0]
        if np.isfinite(b):
            boots.append(b)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    share = float(mediated / total) if total else float("nan")
    supported = bool(np.isfinite(lo) and (lo > 0 if total > 0 else hi < 0) and abs(share) >= 0.3)
    return {"valid": True, "split": split, "total_effect": float(total), "mediated": float(mediated),
            "mediated_share": share, "ci95_mediated": [float(lo), float(hi)], "mechanism_supported": supported,
            "note": ("the effect shrinks when the mediator is controlled for: consistent with the mechanism"
                     if supported else "controlling for the mediator does not remove the effect: association only")}
