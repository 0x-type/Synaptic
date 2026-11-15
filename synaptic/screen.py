"""Automatic data screening: the cheap, deterministic quality check that runs BEFORE the agents.

In live tests the agents spent most of a 12-hour campaign proving, one lead at a time, that NASA POWER had
data-production problems. The paper's agents worked on primary observations; reanalysis-based POWER is model
output, so its own breaks are the first "anomalies" any search finds. This module finds the common failure
types in minutes, with plain statistics against independent references, and writes a trust map. Families that
fall in untrusted data are labelled automatically and never promoted, so the agents' time goes to science.

Checks (per grid cell):
  rain vs GPCP v2.3 (satellite + gauge, independent of MERRA-2's CMAP correction): break in the log ratio
      (Pettitt), single bad years, single bad months
  temperature vs Berkeley Earth (station-based): break in the difference, single bad years
  any variable: runs of exactly-zero rain where rain is normal, runs of identical values (gap fill),
      suppressed year-to-year variability (inflates "recent extreme" scores)
  propagation: a rain failure in MERRA-2 drives its soil moisture, humidity and temperature, so the same
      cell-years are flagged for those variables
  known issues (known_issues.yaml): confirmed by earlier campaigns or documented by the producers
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

import synlib
from .config import RUNS_DIR

REF_DIR = RUNS_DIR / "cache" / "ref"
GPCP_URL = "https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc"
BERKELEY = {"T2M_MAX": "TMAX", "T2M_MIN": "TMIN", "T2M": "TAVG"}
BERKELEY_URL = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Complete_{}_LatLong1.nc"
RAIN_DRIVEN = {"GWETROOT", "GWETTOP", "GWETPROF", "RH2M", "QV2M", "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "TS",
               "EVPTRNS", "T2M_RANGE", "EVLAND"}
KNOWN = Path(__file__).with_name("known_issues.yaml")

# thresholds are fixed here, before any campaign data are seen
RAIN_BREAK = math.log(1.20)    # >=20% change in the POWER/reference ratio
RAIN_YEAR = math.log(2.0)      # a year off by >=2x relative to the cell's usual ratio
RAIN_MONTH = math.log(4.0)     # a month off by >=4x (single-month correction failures)
TEMP_BREAK = 0.5               # deg C step in POWER minus Berkeley
TEMP_YEAR = 1.5                # deg C single-year departure
P_BREAK = 0.01


def _download(url: str, name: str) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    p = REF_DIR / name
    if not p.exists() or p.stat().st_size < 1000:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = p.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(p)
    return p


def load_gpcp():
    import xarray as xr

    ds = xr.open_dataset(_download(GPCP_URL, "gpcp_precip.mon.mean.nc"), decode_times=False)
    t = pd.Timestamp("1800-01-01") + pd.to_timedelta(ds["time"].values, unit="D")
    da = ds["precip"].assign_coords(time=t)
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")


def load_berkeley(var: str):
    import xarray as xr

    name = BERKELEY[var]
    ds = xr.open_dataset(_download(BERKELEY_URL.format(name), f"berkeley_{name}.nc"), decode_times=False)
    t = ds["time"].values  # decimal years, mid-month
    years = np.floor(t).astype(int)
    months = np.clip(np.floor((t - years) * 12).astype(int) + 1, 1, 12)
    da = ds["temperature"]  # anomaly
    da = da.assign_coords(time=pd.to_datetime({"year": years, "month": months, "day": 15}))
    return da.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    return df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)


def _ref_monthly(da, cells: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Nearest reference cell for every POWER cell, monthly, long format (lat, lon, year, month, ref)."""
    sub = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    pts = sub.sel(lat=xr_points(cells.lat), lon=xr_points(cells.lon), method="nearest")
    arr = pts.values  # (time, n)
    times = pd.to_datetime(sub["time"].values)
    out = pd.DataFrame({"lat": np.repeat(cells.lat.values, len(times)), "lon": np.repeat(cells.lon.values, len(times)),
                        "year": np.tile(times.year, len(cells)), "month": np.tile(times.month, len(cells)),
                        "ref": arr.T.reshape(-1)})
    return out


def xr_points(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values, float), dims="pt")


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 1e-9 else np.nan)


def _break_and_outliers(years: np.ndarray, d: np.ndarray, step_min: float, year_min: float):
    """Pettitt break in a difference series + single-year outliers that are not part of the break."""
    out = []
    ok = ~np.isnan(d)
    if ok.sum() < 12:
        return out
    pt = synlib.pettitt(d)
    if pt["cp"] is not None and pt["p"] < P_BREAK:
        cp = pt["cp"]
        before, after = d[: cp + 1], d[cp + 1:]
        step = float(np.nanmedian(after) - np.nanmedian(before))
        if abs(step) >= step_min and np.isfinite(step):
            out.append(("break", int(years[cp + 1]), None, step, pt["p"]))
            d = d.copy()
            d[cp + 1:] -= step  # look for outliers on the de-stepped series
    z = _robust_z(d)
    dev = d - np.nanmedian(d)
    for i in np.flatnonzero((np.abs(z) > 4) & (np.abs(dev) >= year_min)):
        out.append(("bad_year", int(years[i]), int(years[i]), float(dev[i]), float(z[i])))
    return out


def screen_rain(pw: pd.DataFrame, gpcp) -> list[dict]:
    issues = []
    y0, y1 = int(pw.year.min()), int(pw.year.max())
    ref = _ref_monthly(gpcp, _cells(pw), y0, y1)
    m = pw.merge(ref, on=["lat", "lon", "year", "month"], how="inner")
    if m.empty:
        return issues
    for (la, lo), g in m.groupby(["lat", "lon"]):
        ann = g.groupby("year").agg(pw=("value", "mean"), ref=("ref", "mean"), n=("value", "size"))
        ann = ann[ann.n == 12]
        if len(ann) < 12 or ann.ref.mean() < 0.3:  # deserts: ratios are meaningless
            continue
        d = np.log((ann.pw.values + 0.1) / (ann.ref.values + 0.1))
        for kind, a, b, size, stat in _break_and_outliers(ann.index.values, d, RAIN_BREAK, RAIN_YEAR):
            issues.append({"variable": "PRECTOTCORR", "lat": la, "lon": lo, "kind": kind, "year_from": a,
                           "year_to": b, "size": round(size, 3), "stat": round(stat, 4),
                           "detail": f"POWER/GPCP log-ratio {'step' if kind == 'break' else 'departure'} {size:+.2f} "
                                     f"({(math.exp(size) - 1) * 100:+.0f}%)"})
        # single bad months (correction failures like one month of fake record rain)
        g = g[g.ref > 1.0]
        if len(g) >= 60:
            g = g.assign(lr=np.log((g.value + 0.1) / (g.ref + 0.1)))
            g["dev"] = g.lr - g.groupby("month").lr.transform("median")
            g["z"] = _robust_z(g.dev.values)
            bad = g[(np.abs(g.z) > 5) & (np.abs(g.dev) >= RAIN_MONTH)]
            for r in bad.itertuples():
                issues.append({"variable": "PRECTOTCORR", "lat": la, "lon": lo, "kind": "bad_month",
                               "year_from": int(r.year), "year_to": int(r.year), "month": int(r.month),
                               "size": round(float(r.dev), 3), "stat": round(float(r.z), 2),
                               "detail": f"{int(r.year)}-{int(r.month):02d}: POWER {r.value:.1f} vs GPCP {r.ref:.1f} mm/day"})
    # a real correction failure covers a block of cells (the input is a 2.5-degree product): drop isolated months
    bm = [i for i in issues if i["kind"] == "bad_month"]
    if bm:
        cnt = pd.Series([(i["year_from"], i["month"]) for i in bm]).value_counts()
        keep = {k for k, n in cnt.items() if n >= 4}
        issues = [i for i in issues if i["kind"] != "bad_month" or (i["year_from"], i["month"]) in keep]
    return issues


def screen_temperature(var: str, pw: pd.DataFrame, berk) -> list[dict]:
    issues = []
    y0, y1 = int(pw.year.min()), int(pw.year.max())
    ref = _ref_monthly(berk, _cells(pw), y0, y1)
    m = pw.merge(ref, on=["lat", "lon", "year", "month"], how="inner").dropna(subset=["ref"])
    if m.empty:
        return issues
    # POWER anomaly against its own monthly climatology, Berkeley is already an anomaly
    m["pa"] = m.value - m.groupby(["lat", "lon", "month"]).value.transform("mean")
    m["ra"] = m.ref - m.groupby(["lat", "lon", "month"]).ref.transform("mean")
    for (la, lo), g in m.groupby(["lat", "lon"]):
        ann = g.groupby("year").agg(pa=("pa", "mean"), ra=("ra", "mean"), n=("pa", "size"))
        ann = ann[ann.n >= 10]
        if len(ann) < 12:
            continue
        d = (ann.pa - ann.ra).values
        for kind, a, b, size, stat in _break_and_outliers(ann.index.values, d, TEMP_BREAK, TEMP_YEAR):
            issues.append({"variable": var, "lat": la, "lon": lo, "kind": kind, "year_from": a, "year_to": b,
                           "size": round(size, 3), "stat": round(stat, 4),
                           "detail": f"POWER minus Berkeley Earth {'step' if kind == 'break' else 'departure'} {size:+.2f} C"})
    return issues


def screen_runs(var: str, pw: pd.DataFrame) -> list[dict]:
    """Gap fills: runs of identical values, and runs of exactly-zero rain where rain is normal."""
    issues = []
    d = pw.sort_values(["lat", "lon", "year", "month"])
    clim = d.groupby(["lat", "lon", "month"]).value.transform("mean")
    d = d.assign(clim=clim)
    for (la, lo), g in d.groupby(["lat", "lon"]):
        v = g.value.values
        same = np.r_[False, np.isclose(v[1:], v[:-1], atol=0, rtol=0)]
        run_id = np.cumsum(~same)
        # runs that recur in many years are model behaviour (frozen or saturated soil every winter), not gap fills
        run_years = {int(y) for rid in np.unique(run_id) if (run_id == rid).sum() >= 4
                     for y in g.year.values[run_id == rid]}
        if len(run_years) > 0.3 * g.year.nunique():
            continue
        for rid in np.unique(run_id):
            idx = np.flatnonzero(run_id == rid)
            if len(idx) < 4:
                continue
            val = v[idx[0]]
            yrs = g.year.values[idx]
            if var == "PRECTOTCORR" and val == 0:
                if len(idx) >= 6 and g.clim.values[idx].mean() > 0.3:
                    issues.append({"variable": var, "lat": la, "lon": lo, "kind": "zero_run",
                                   "year_from": int(yrs.min()), "year_to": int(yrs.max()), "size": float(len(idx)),
                                   "stat": float(g.clim.values[idx].mean()),
                                   "detail": f"{len(idx)} consecutive months of exactly 0 where the normal is "
                                             f"{g.clim.values[idx].mean():.2f} mm/day"})
            elif val != 0:
                lo_q, hi_q = np.nanpercentile(v, [5, 95])
                if not (lo_q < val < hi_q) or len(idx) < 5:
                    continue  # values pinned at a physical bound (saturated or dry soil) are model behaviour
                issues.append({"variable": var, "lat": la, "lon": lo, "kind": "constant_run",
                               "year_from": int(yrs.min()), "year_to": int(yrs.max()), "size": float(len(idx)),
                               "stat": float(val), "detail": f"{len(idx)} consecutive identical values ({val})"})
    return issues


def screen_low_variability(var: str, pw: pd.DataFrame) -> list[dict]:
    if var not in ("T2M_MAX", "T2M_MIN", "T2M"):
        return []
    ann = pw.groupby(["lat", "lon", "year"]).value.mean().unstack()
    sd = ann.std(axis=1)
    floor = 0.5 * float(np.nanmedian(sd))
    low = sd[sd < floor]
    return [{"variable": var, "lat": la, "lon": lo, "kind": "low_variability", "year_from": None, "year_to": None,
             "size": round(float(s), 3), "stat": round(floor, 3),
             "detail": f"year-to-year SD {s:.2f} C < half the regional median ({2 * floor:.2f}): recent-extreme scores inflated"}
            for (la, lo), s in low.items()]


def known_issues(region: dict, variables: list[str], cells: pd.DataFrame) -> list[dict]:
    out = []
    try:
        entries = yaml.safe_load(KNOWN.read_text()) or []
    except Exception:  # noqa: BLE001
        return out
    for e in entries:
        la0, la1, lo0, lo1 = e["box"]
        if la1 < region["lat_min"] or la0 > region["lat_max"] or lo1 < region["lon_min"] or lo0 > region["lon_max"]:
            continue
        inside = cells[(cells.lat >= la0) & (cells.lat <= la1) & (cells.lon >= lo0) & (cells.lon <= lo1)]
        for v in e["variables"]:
            if v not in variables:
                continue
            for r in inside.itertuples():
                out.append({"variable": v, "lat": r.lat, "lon": r.lon, "kind": f"known_{e['kind']}",
                            "year_from": e.get("year_from"), "year_to": e.get("year_to"), "size": None, "stat": None,
                            "detail": f"{e['id']}: {' '.join(str(e['description']).split())[:220]}"})
    return out


def propagate(issues: list[dict], variables: list[str]) -> list[dict]:
    """A MERRA-2 rain failure drives its land model: soil, humidity and temperature inherit it."""
    out = []
    for i in issues:
        if i["variable"] != "PRECTOTCORR" or i["kind"] not in ("break", "bad_year"):
            continue
        for v in variables:
            if v in RAIN_DRIVEN:
                out.append({**i, "variable": v, "kind": f"propagated_{i['kind']}",
                            "detail": f"inherits a POWER rain {i['kind']} at {i['year_from']} ({i['detail']})"})
    return out


def run_screen(data: dict[str, pd.DataFrame], region: dict, log=print) -> pd.DataFrame:
    """data maps variable -> POWER long table (lat, lon, variable, year, month, value)."""
    issues: list[dict] = []
    variables = list(data)
    if "PRECTOTCORR" in data:
        try:
            issues += screen_rain(data["PRECTOTCORR"], load_gpcp())
        except Exception as e:  # noqa: BLE001
            log(f"screen: rain-vs-GPCP check skipped ({str(e)[:120]})")
    for v in variables:
        if v in BERKELEY:
            try:
                issues += screen_temperature(v, data[v], load_berkeley(v))
            except Exception as e:  # noqa: BLE001
                log(f"screen: {v}-vs-Berkeley check skipped ({str(e)[:120]})")
        issues += screen_runs(v, data[v])
        issues += screen_low_variability(v, data[v])
    issues += propagate(issues, variables)
    any_df = next(iter(data.values()))
    issues += known_issues(region, variables, _cells(any_df))
    cols = ["variable", "lat", "lon", "kind", "year_from", "year_to", "month", "size", "stat", "detail"]
    return pd.DataFrame(issues, columns=cols)


# ---------------------------------------------------------------------------- use by the pipeline

MASK_KINDS = {"bad_year", "propagated_bad_year", "known_period", "zero_run", "constant_run", "bad_month"}


def apply_trust(df: pd.DataFrame, trust: pd.DataFrame | None, variable: str) -> pd.DataFrame:
    """The same table with untrusted values removed: bad years / periods / gap fills set to NaN, and for a cell
    with a break only the longer homogeneous segment kept. What agents should analyse by default."""
    out = df.copy()
    if trust is None or trust.empty:
        return out
    t = trust[(trust.variable == variable) & (trust.kind != "known_caveat")]
    if t.empty:
        return out
    y0, y1 = int(out.year.min()), int(out.year.max())
    rows = out.groupby([out.lat.round(3), out.lon.round(3)]).indices  # cell -> positional row indices
    years, months = out.year.values, out.month.values
    drop = np.zeros(len(out), bool)
    for r in t.itertuples():
        idx = rows.get((round(r.lat, 3), round(r.lon, 3)))
        if idx is None or pd.isna(r.year_from):
            continue
        a = int(r.year_from)
        yy = years[idx]
        if "break" in str(r.kind):
            m = (yy < a) if (y1 - a) >= (a - y0) else (yy >= a)
        elif str(r.kind) in MASK_KINDS:
            b = int(r.year_to) if not pd.isna(r.year_to) else a
            m = (yy >= a) & (yy <= b)
            if str(r.kind) == "bad_month" and not pd.isna(getattr(r, "month", np.nan)):
                m &= months[idx] == int(r.month)
        else:
            continue
        drop[idx[m]] = True
    out.loc[drop, "value"] = np.nan
    return out


def family_issue(fam: dict, trust: pd.DataFrame, end_year: int, window: int = 2) -> tuple[float, str]:
    """Fraction of a family's cells whose signal is explained by untrusted data, and the main reason."""
    if trust is None or trust.empty:
        return 0.0, ""
    cells = json.loads(fam["cells"]) if isinstance(fam["cells"], str) else fam["cells"]
    t = trust[(trust.variable == fam["variable"]) & (trust.kind != "known_caveat")]  # caveats warn, not label
    if t.empty or not cells:
        return 0.0, ""
    key = {(round(a, 4), round(b, 4)) for a, b in cells}
    t = t[[(round(a, 4), round(b, 4)) in key for a, b in zip(t.lat, t.lon)]]
    if t.empty:
        return 0.0, ""
    cls, onset = fam["cls"], fam.get("onset")
    yf = pd.to_numeric(t.year_from, errors="coerce")
    yt = pd.to_numeric(t.year_to, errors="coerce").fillna(yf)
    kind = t.kind.astype(str)
    breakish = kind.str.contains("break")
    if cls == "shift" and onset is not None and not (isinstance(onset, float) and np.isnan(onset)):
        hit = breakish & ((yf - float(onset)).abs() <= window) | (~breakish & (yf <= onset + window) & (yt >= onset - window))
    elif cls == "recent_extreme":
        hit = (yt >= end_year - 3) | (kind == "low_variability") | breakish
    elif cls == "trend":
        hit = breakish | kind.isin(["zero_run", "constant_run"])
    else:
        hit = kind != "low_variability"
    bad = t[hit.values]
    if bad.empty:
        return 0.0, ""
    n_bad = bad[["lat", "lon"]].drop_duplicates().shape[0]
    frac = n_bad / len(cells)
    reason = bad.kind.value_counts().index[0]
    example = bad[bad.kind == reason].detail.iloc[0]
    return frac, f"{reason} in {n_bad}/{len(cells)} cells (e.g. {example[:160]})"


def plot_trust(trust: pd.DataFrame, path: Path, region: dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    vs = sorted(trust.variable.unique()) or ["none"]
    fig, axes = plt.subplots(1, len(vs), figsize=(4.2 * len(vs), 4.2), squeeze=False)
    for ax, v in zip(axes[0], vs):
        t = trust[(trust.variable == v) & (trust.kind != "known_caveat")]
        n = t.groupby(["lat", "lon"]).size().reset_index(name="n")
        sc = ax.scatter(n.lon, n.lat, c=n.n, s=10, marker="s", cmap="inferno_r")
        ax.set_xlim(region["lon_min"], region["lon_max"])
        ax.set_ylim(region["lat_min"], region["lat_max"])
        ax.set_aspect("equal")
        ax.set_title(f"{v}: {n.shape[0]} untrusted cells", fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle("Automatic screen: cells with data issues (count of issues)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- v4: satellite / multi-product screen

SATELLITE_KNOWN = {
    "MODIS": "MODIS Aqua: the satellite stopped orbit-inclination manoeuvres around 2021-2022 and its overpass time has "
             "been drifting later since; cloud and aerosol retrievals sample a different time of day, so changes "
             "after ~2022 can be partly sampling. Check against CERES (diurnally corrected) before trusting them.",
    "CERES": "CERES EBAF: built from Terra + Aqua; the afternoon instrument changes around 2022 (Aqua -> NOAA-20) as "
             "Aqua drifts. EBAF is calibrated across the change, but check the EBAF Data Quality Summary before "
             "trusting a step at that time.",
    "MERRA2": "MERRA-2: production streams change in 1992, 2001 and 2011; aerosol observations assimilated change "
              "around 2000-2002 (AVHRR -> MODIS/MISR) and again after 2014; land rain is replaced by gauge/satellite "
              "products. MERRA-2 aerosols are a model constrained by MODIS AOD: MODIS and MERRA-2 AOD are NOT "
              "independent.",
    "IMERG": "GPM IMERG V07: TRMM era before mid-2014, GPM afterwards; monthly values are gauge-adjusted.",
    "GLDAS": "GLDAS Noah 2.1: model driven by GDAS meteorology and observed rain; not an observation of soil water.",
    "GRACE": "GRACE / GRACE-FO: gap from mid-2017 to mid-2018 between the missions; months are missing elsewhere too.",
}


def satellite_known_issues(dataset_ids: list[str]) -> list[str]:
    out = []
    for key, txt in SATELLITE_KNOWN.items():
        if any(d.upper().startswith(key) for d in dataset_ids):
            out.append(txt)
    return out


def pair_breaks(a: pd.DataFrame, b: pd.DataFrame, alpha: float = 0.05) -> str:
    """Two products of the same quantity on the same grid: does their difference break? Area-weighted regional
    difference of annual anomalies, carry-over-adjusted Pettitt test, plus the share of cells whose own difference
    breaks."""
    import synlib

    ka = a.groupby(["lat", "lon", "year"]).value.mean().rename("a")
    kb = b.groupby(["lat", "lon", "year"]).value.mean().rename("b")
    m = pd.concat([ka, kb], axis=1).dropna().reset_index()
    if m.empty:
        return "no overlap"
    m["a"] -= m.groupby(["lat", "lon"]).a.transform("mean")
    m["b"] -= m.groupby(["lat", "lon"]).b.transform("mean")
    m["d"] = m.a - m.b
    m["w"] = np.cos(np.radians(m.lat))
    reg = m.groupby("year").apply(lambda g: np.average(g.d, weights=g.w))
    pt = synlib.pettitt(reg.values)
    yr = int(reg.index[pt["cp"] + 1]) if pt["cp"] is not None else None
    cells = []
    for _, g in m.groupby(["lat", "lon"]):
        if len(g) >= 12:
            p = synlib.pettitt(g.sort_values("year").d.values)
            if p["p"] < alpha and p["cp"] is not None:
                cells.append(int(g.sort_values("year").year.values[p["cp"] + 1]))
    n = m[["lat", "lon"]].drop_duplicates().shape[0]
    common = pd.Series(cells).value_counts().head(3).to_dict() if cells else {}
    verdict = "BREAK" if pt["p"] < alpha else "no break"
    return (f"{verdict} in the regional difference (year {yr}, p = {pt['p']:.3g}); {len(cells)}/{n} cells with their "
            f"own difference break, most common years {common}")
