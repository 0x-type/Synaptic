"""Automatic evidence engine: the checks a senior team runs on every candidate before anyone spends a day on it.

In the v2 global run, 13 of the first 20 follow-up tasks were agents re-checking one family at a time against
CHIRPS, GPCP or stations, and two families the census promoted were the same cells under different variables.
This module does that work deterministically for every family that passes the basic filters:

1. Independent recomputation. The same statistic (trend, step at the family's onset, recent anomaly, variance
   change) is recomputed on the family's cells in data that do not come from MERRA-2:
     rain         GPCP v2.3 (gauges + satellites, 1979-)   and GPM IMERG V07 (1998-)      observations
     temperature  Berkeley Earth TAVG / TMAX / TMIN (stations)                              observations
                  GLDAS forcing air temperature                                             model
     soil water   GLDAS Noah 2.0 (1981-2014) and 2.1 (2000-2025) root-zone soil moisture    independent model
     evaporation  GLDAS Noah EVAP                                                           independent model
     humidity     GLDAS forcing humidity (RH from Qair and Tair)                            model
   Each reference is compared with MERRA-2 over its own years (never spliced). For a step, the difference series
   (MERRA-2 minus reference, standardized) is tested for a break at the onset: a break there means the step is
   in one product only (inhomogeneity).
2. Robustness suite (synlib.robustness) on the family-mean series: leave-one-year-out, endpoints, strong-ENSO
   years removed, block bootstrap, one-extreme-year leverage.
3. Field significance (synlib.field_significance, Livezey-Chen with spatial correlation kept) per variable and
   kind of change.
4. Place names from offline reverse geocoding (GeoNames), so reports never call central Brazil "southern South
   America" or James Bay "Alaska".
5. Merging: families on the same cells (any variable) become one multi-variable signal; one deep dive, not three.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import synlib

OBS, MODEL = "observation", "independent model"
REFS = {
    "PRECTOTCORR": [("GPCP", OBS), ("IMERG", OBS)],
    "PRECTOT": [("GPCP", OBS), ("IMERG", OBS)],
    "PRECTOTLAND": [("GPCP", OBS), ("IMERG", OBS)],
    "T2M": [("BERKELEY_TAVG", OBS), ("GLDAS20_TAIR", MODEL), ("GLDAS21_TAIR", MODEL)],
    "T2M_MAX": [("BERKELEY_TMAX", OBS)],
    "T2M_MIN": [("BERKELEY_TMIN", OBS)],
    "GWETROOT": [("GLDAS20_ROOTMOIST", MODEL), ("GLDAS21_ROOTMOIST", MODEL)],
    "GWETPROF": [("GLDAS20_ROOTMOIST", MODEL), ("GLDAS21_ROOTMOIST", MODEL)],
    "GWETTOP": [("GLDAS20_SOILM_0_10", MODEL), ("GLDAS21_SOILM_0_10", MODEL)],
    "EVLAND": [("GLDAS20_EVAP", MODEL), ("GLDAS21_EVAP", MODEL)],
    "EVPTRNS": [("GLDAS20_EVAP", MODEL), ("GLDAS21_EVAP", MODEL)],
    "RH2M": [("GLDAS20_RH", MODEL), ("GLDAS21_RH", MODEL)],
    "QV2M": [("GLDAS20_QAIR", MODEL), ("GLDAS21_QAIR", MODEL)],
}
DOCUMENTED_BY_DEFAULT = {("T2M", "trend"), ("T2M_MAX", "trend"), ("T2M_MIN", "trend"), ("T2M", "recent_extreme"),
                         ("T2M_MAX", "recent_extreme"), ("T2M_MIN", "recent_extreme")}
_FRAMES: dict = {}


# ---------------------------------------------------------------------------- reference data at campaign cells

def _ref_frame(ref: str, cells: pd.DataFrame, y0: int, y1: int, glob: bool) -> pd.DataFrame:
    """Monthly reference values at the campaign cells, long format (lat, lon, year, month, ref); cached."""
    key = (ref, len(cells), float(cells.lat.sum()), float(cells.lon.sum()), y0, y1, glob)
    if key in _FRAMES:
        return _FRAMES[key]
    from . import screen
    from .tools import nasa_archive as na

    if ref == "GPCP":
        df = screen._ref_monthly(screen.load_gpcp(), cells, y0, y1)
    elif ref.startswith("BERKELEY_"):
        var = {"TAVG": "T2M", "TMAX": "T2M_MAX", "TMIN": "T2M_MIN"}[ref.split("_", 1)[1]]
        df = screen._ref_monthly(screen.load_berkeley(var), cells, y0, y1)
    elif ref == "IMERG":
        df = na.cells_frame("imerg", "PRECIP", cells, y0, y1, glob)
    elif ref.startswith("GLDAS"):
        prod, var = ref.split("_", 1)
        prod = prod.lower()
        if var == "RH":
            q = na.cells_frame(prod, "QAIR", cells, y0, y1, glob)
            t = na.cells_frame(prod, "TAIR", cells, y0, y1, glob)
            df = q.merge(t, on=["lat", "lon", "year", "month"], suffixes=("_q", "_t"))
            e = df.ref_q * 1000.0 / (0.622 + 0.378 * df.ref_q)  # hPa at ~1000 hPa (approximate; anomalies matter)
            es = 6.112 * np.exp(17.67 * df.ref_t / (df.ref_t + 243.5))
            df = df[["lat", "lon", "year", "month"]].assign(ref=np.clip(100 * e / es, 0, 100))
        else:
            df = na.cells_frame(prod, var, cells, y0, y1, glob)
    else:
        raise ValueError(ref)
    df = df.dropna(subset=["ref"])
    _FRAMES[key] = df
    return df


def _annual(df: pd.DataFrame, col: str, months: list[int] | None) -> pd.DataFrame:
    """Annual (or seasonal) means per cell; only complete years. Returns wide: index (lat, lon), columns years."""
    if months:
        df = df[df.month.isin(months)]
    need = len(months) if months else 12
    g = df.groupby(["lat", "lon", "year"])[col].agg(["mean", "size"])
    g = g[g["size"] >= need]["mean"].unstack("year")
    return g


def _z_family_mean(wide: pd.DataFrame, years: list[int]) -> np.ndarray:
    """Mean over cells of per-cell standardized anomalies over the given years."""
    w = wide.reindex(columns=years)
    mu = w.mean(axis=1)
    sd = w.std(axis=1).replace(0, np.nan)
    z = w.sub(mu, axis=0).div(sd, axis=0)
    return z.mean(axis=0, skipna=True).values


def _slope_p(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return np.nan, 1.0
    return synlib.sen_slope(y[ok], t[ok]) * 10, synlib.mann_kendall(y[ok])["p"]


def _compare(cls: str, onset, d: np.ndarray, r: np.ndarray, t: np.ndarray) -> dict:
    """Same statistic in MERRA-2 (d) and the reference (r), both family-mean z-series over common years t."""
    from scipy import stats as st

    ok = ~np.isnan(d) & ~np.isnan(r)
    d, r, t = d[ok], r[ok], t[ok]
    out = {"years": f"{int(t.min())}-{int(t.max())}" if len(t) else "", "n_years": int(len(t))}
    if len(t) < 10:
        out["verdict"] = "insufficient"
        return out
    dd = d - np.polyval(np.polyfit(t, d, 1), t)
    rr = r - np.polyval(np.polyfit(t, r, 1), t)
    out["r_interannual"] = round(float(np.corrcoef(dd, rr)[0, 1]), 2)
    if cls == "trend":
        sd_, pd_ = _slope_p(d, t)
        sr, pr = _slope_p(r, t)
        out.update({"merra2_slope_sd_per_decade": round(sd_, 3), "ref_slope_sd_per_decade": round(sr, 3),
                    "ref_p": round(pr, 4)})
        # a "trend" that is really a step in one product: break in MERRA-2 minus reference carrying most of the trend
        # (found live: central-Brazil rain decline confirmed in sign by GPCP/IMERG, but MERRA-2 minus CHIRPS/IMERG
        # breaks in ~2004 and the uncorrected model rain has no trend)
        diff = d - r
        pt = synlib.pettitt(diff)
        inhom = False
        if pt["cp"] is not None and pt["p"] < 0.01:
            cp = pt["cp"] + 1
            step = float(np.mean(diff[cp:]) - np.mean(diff[:cp]))
            total = sd_ / 10 * (t.max() - t.min())  # MERRA-2 change over the period, SD units
            inhom = abs(step) >= 0.5 * abs(total) and np.sign(step) == np.sign(total)
            out.update({"difference_break_year": int(t[cp]) if cp < len(t) else None,
                        "difference_break_p": round(float(pt["p"]), 5), "difference_step_sd": round(step, 3)})
        if np.sign(sr) == np.sign(sd_) and pr < 0.1 and abs(sr) >= 0.4 * abs(sd_) and not inhom:
            out["verdict"] = "agrees"
            if out.get("difference_break_p", 1) < 0.01 and np.sign(out.get("difference_step_sd", 0)) == np.sign(sd_):
                out["inflated"] = True
                out["note"] = (f"real in sign, but MERRA-2 inflates it: the difference series breaks in "
                               f"{out.get('difference_break_year')}; reference slope is {abs(sr) / max(abs(sd_), 1e-9):.0%} "
                               "of MERRA-2's")
        elif inhom or (np.sign(sr) != np.sign(sd_) and pr < 0.1) or (pd_ < 0.05 and abs(sr) < 0.2 * abs(sd_)):
            out["verdict"] = "contradicts"
            if inhom:
                out["note"] = (f"MERRA-2 minus reference breaks in {out.get('difference_break_year')} and the break "
                               "carries at least half of the MERRA-2 trend: the trend is largely an inhomogeneity")
        else:
            out["verdict"] = "weak"
    elif cls == "shift" and onset is not None and not (isinstance(onset, float) and np.isnan(onset)):
        onset = int(onset)
        a, b = t < onset, t >= onset
        if a.sum() < 6 or b.sum() < 6:
            out["verdict"] = "insufficient"
            out["note"] = f"reference period {out['years']} does not bracket the {onset} onset with 6+ years each side"
            return out
        step_d = float(d[b].mean() - d[a].mean())
        step_r = float(r[b].mean() - r[a].mean())
        pr = float(st.ttest_ind(r[b], r[a], equal_var=False).pvalue)
        diff = d - r
        pt = synlib.pettitt(diff)
        cp_year = int(t[pt["cp"] + 1]) if pt["cp"] is not None and pt["cp"] + 1 < len(t) else None
        diff_step = float(diff[b].mean() - diff[a].mean())
        out.update({"merra2_step_sd": round(step_d, 3), "ref_step_sd": round(step_r, 3), "ref_p": round(pr, 4),
                    "difference_break_year": cp_year, "difference_break_p": round(float(pt["p"]), 4),
                    "difference_step_sd": round(diff_step, 3)})
        inhom = (pt["p"] < 0.05 and cp_year is not None and abs(cp_year - onset) <= 3
                 and abs(diff_step) >= 0.5 * abs(step_d))
        if np.sign(step_r) == np.sign(step_d) and pr < 0.1 and abs(step_r) >= 0.4 * abs(step_d) and not inhom:
            out["verdict"] = "agrees"
        elif inhom or (np.sign(step_r) != np.sign(step_d) and pr < 0.1):
            out["verdict"] = "contradicts"
            if inhom:
                out["note"] = (f"MERRA-2 minus reference breaks in {cp_year} (p={pt['p']:.2g}): the step is in "
                               "one product only")
        else:
            out["verdict"] = "weak"
    elif cls == "recent_extreme":
        k = 3
        rd = float(d[-k:].mean() - d[:-k].mean())
        rr_ = float(r[-k:].mean() - r[:-k].mean())
        out.update({"merra2_recent_sd": round(rd, 3), "ref_recent_sd": round(rr_, 3),
                    "ref_last_year": int(t[-1])})
        if int(t[-1]) < int(t.max()):
            out["verdict"] = "insufficient"
        elif np.sign(rr_) == np.sign(rd) and abs(rr_) >= 0.5 * abs(rd):
            out["verdict"] = "agrees"
        elif np.sign(rr_) != np.sign(rd) or abs(rr_) < 0.2 * abs(rd):
            out["verdict"] = "contradicts"
        else:
            out["verdict"] = "weak"
    elif cls == "variance":
        h = len(t) // 2
        vd = math.log(np.var(dd[h:]) / max(np.var(dd[:h]), 1e-9))
        vr = math.log(np.var(rr[h:]) / max(np.var(rr[:h]), 1e-9))
        out.update({"merra2_log_var_ratio": round(vd, 3), "ref_log_var_ratio": round(vr, 3)})
        out["verdict"] = "agrees" if np.sign(vr) == np.sign(vd) and abs(vr) >= 0.4 * abs(vd) else \
            ("contradicts" if np.sign(vr) != np.sign(vd) else "weak")
    else:
        out["verdict"] = "not_tested"
    return out


# ---------------------------------------------------------------------------- place names

_GEO = None
_SHORT = {"CD": "DR Congo", "CG": "Congo", "BO": "Bolivia", "VE": "Venezuela", "IR": "Iran", "RU": "Russia",
          "TZ": "Tanzania", "KR": "South Korea", "KP": "North Korea", "SY": "Syria", "LA": "Laos", "VN": "Vietnam",
          "MD": "Moldova", "US": "United States", "GB": "United Kingdom", "CF": "Central African Republic"}


def _country(cc: str) -> str:
    if cc in _SHORT:
        return _SHORT[cc]
    try:
        import pycountry

        c = pycountry.countries.get(alpha_2=cc)
        return (getattr(c, "common_name", None) or c.name) if c else cc
    except Exception:  # noqa: BLE001
        return cc


def place_name(lat: float, lon: float) -> str:
    """'near Mozarlandia, Goias, Brazil' (nearest GeoNames town; distance shown when far)."""
    global _GEO
    try:
        import reverse_geocoder as rg
        import pycountry

        if _GEO is None:
            _GEO = rg.RGeocoder(mode=1, verbose=False)
        r = _GEO.query([(float(lat), float(lon))])[0]
        c = pycountry.countries.get(alpha_2=r["cc"])
        country = _country(r["cc"])
        dlat = math.radians(float(r["lat"]) - lat)
        dlon = math.radians(float(r["lon"]) - lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(float(r["lat"]))) * \
            math.sin(dlon / 2) ** 2
        km = 6371 * 2 * math.asin(math.sqrt(a))
        where = ", ".join(x for x in (r["name"], r["admin1"], country) if x)
        return f"near {where}" if km < 150 else f"{km:.0f} km from {where}"
    except Exception:  # noqa: BLE001
        return f"{lat:.1f}, {lon:.1f}"


def family_place(fam: dict) -> str:
    """Centroid town plus the countries the family covers, by number of cells."""
    cells = json.loads(fam["cells"]) if isinstance(fam.get("cells"), str) else (fam.get("cells") or [])
    head = place_name(fam["lat"], fam["lon"])
    if len(cells) < 4:
        return head
    try:
        import reverse_geocoder as rg
        import pycountry

        global _GEO
        if _GEO is None:
            _GEO = rg.RGeocoder(mode=1, verbose=False)
        res = _GEO.query([(float(a), float(b)) for a, b in cells])
        cnt = pd.Series([r["cc"] for r in res]).value_counts()
        names = []
        for cc, n in cnt.items():
            names.append(f"{_country(cc)} {n}")
        more = f", +{len(names) - 6} more" if len(names) > 6 else ""
        return f"{head}; cells by country: {', '.join(names[:6])}{more}"
    except Exception:  # noqa: BLE001
        return head


# ---------------------------------------------------------------------------- the engine

class Evidence:
    def __init__(self, campaign_dir: Path, data_index: dict, start_year: int, end_year: int, glob: bool,
                 months: list[int] | None = None, log=print):
        self.cdir, self.data_index, self.glob = Path(campaign_dir), data_index, glob
        self.y0, self.y1, self.months, self.log = start_year, end_year, months, log
        self._data: dict[str, pd.DataFrame] = {}
        self._ann: dict[str, pd.DataFrame] = {}
        self._refann: dict[tuple, pd.DataFrame] = {}
        self.cache_path = self.cdir / "evidence.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    # data
    def data(self, var: str) -> pd.DataFrame | None:
        if var not in self._data:
            p = self.data_index.get(var)
            if not p or not Path(p).exists():
                return None
            df = pd.read_csv(p)
            if "variable" in df:
                df = df[df.variable == var]
            self._data[var] = df
        return self._data[var]

    def ann(self, var: str) -> pd.DataFrame | None:
        if var not in self._ann:
            df = self.data(var)
            if df is None:
                return None
            self._ann[var] = _annual(df, "value", self.months)
        return self._ann[var]

    def ref_ann(self, ref: str, var: str) -> pd.DataFrame | None:
        key = (ref, var)
        if key not in self._refann:
            df = self.data(var)
            cells = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
            try:
                rf = _ref_frame(ref, cells, self.y0, self.y1, self.glob)
                self._refann[key] = _annual(rf, "ref", self.months) if len(rf) else None
            except Exception as e:  # noqa: BLE001
                self.log(f"evidence: reference {ref} unavailable ({type(e).__name__}: {e})")
                self._refann[key] = None
        return self._refann[key]

    # checks
    def field(self, var: str, cls: str) -> dict:
        """Null distribution of the largest same-sign cluster for this variable and class (phase surrogates keep
        carry-over and the correlation between neighbouring cells). Cached. Fails closed: when the test cannot
        run, families of this kind cannot pass promotion."""
        key = f"cluster_null::{var}::{cls}"
        if key in self.cache:
            return self.cache[key]
        a = self.ann(var)
        if a is None or cls not in synlib.TESTABLE_CLASSES:
            res = {"valid": False, "note": "no field test for this kind of change" if a is not None else "no data"}
        else:
            lat = np.array([i[0] for i in a.index], float)
            lon = np.array([i[1] for i in a.index], float)
            res = synlib.cluster_null(lat, lon, a.values.astype(float), cls)
        self.cache[key] = res
        return res

    def field_for(self, fam: dict) -> dict:
        """Is this family bigger than the largest cluster pure carry-over + spatially correlated noise produce?"""
        return synlib.cluster_p(int(fam["n_cells"]), self.field(fam["variable"], fam["cls"]))

    def family(self, fam: dict) -> dict:
        key = f"family::{fam['family']}"
        if key in self.cache:
            return self.cache[key]
        var, cls = fam["variable"], fam["cls"]
        cells = json.loads(fam["cells"]) if isinstance(fam["cells"], str) else fam["cells"]
        idx = pd.MultiIndex.from_tuples([(round(a, 3), round(b, 3)) for a, b in cells], names=["lat", "lon"])
        a = self.ann(var)
        out = {"place": family_place(fam), "independent": [], "robust": None}
        if a is None:
            self.cache[key] = out
            return out
        a = a.copy()
        a.index = pd.MultiIndex.from_tuples([(round(x, 3), round(y, 3)) for x, y in a.index], names=["lat", "lon"])
        fa = a.reindex(idx).dropna(how="all")
        years = [int(y) for y in fa.columns]
        z = _z_family_mean(fa, years)
        onset = fam.get("onset")
        onset = None if onset is None or (isinstance(onset, float) and np.isnan(onset)) else int(onset)
        out["robust"] = synlib.robustness(z, np.array(years, float), cls, onset)
        for ref, kind in REFS.get(var, []):
            ra = self.ref_ann(ref, var)
            if ra is None or ra.empty:
                continue
            ra = ra.copy()
            ra.index = pd.MultiIndex.from_tuples([(round(x, 3), round(y, 3)) for x, y in ra.index],
                                                 names=["lat", "lon"])
            fr = ra.reindex(idx).dropna(how="all")
            common_cells = fa.index.intersection(fr.index)
            yrs = sorted(set(years) & {int(y) for y in fr.columns})
            if len(common_cells) == 0 or len(yrs) < 10:
                continue
            dz = _z_family_mean(fa.loc[common_cells], yrs)
            rz = _z_family_mean(fr.loc[common_cells], yrs)
            cmp_ = _compare(cls, onset, dz, rz, np.array(yrs, float))
            cmp_.update({"reference": ref, "kind": kind, "cells": int(len(common_cells))})
            out["independent"].append(cmp_)
        out.update(_support(out["independent"]))
        self.cache[key] = out
        return out

    def save(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        except Exception:  # noqa: BLE001
            pass


def _support(comps: list[dict]) -> dict:
    """Combine reference comparisons: observations outrank models; a contradicting observation with no agreeing
    observation closes the family as a single-product signal."""
    v = [(c["kind"], c["verdict"]) for c in comps]
    obs_agree = any(k == OBS and x == "agrees" for k, x in v)
    obs_contra = any(k == OBS and x == "contradicts" for k, x in v)
    mod_agree = any(k == MODEL and x == "agrees" for k, x in v)
    mod_contra = any(k == MODEL and x == "contradicts" for k, x in v)
    if obs_agree:
        infl = all(c.get("inflated") for c in comps if c["kind"] == OBS and c["verdict"] == "agrees")
        return {"support": 0.75 if infl else 1.0,
                "status": "confirmed_by_observations" + (" (MERRA-2 inflates it)" if infl else "")
                          + (" (a model disagrees)" if mod_contra else "")}
    if obs_contra:
        return {"support": 0.0, "status": "contradicted_by_observations"}
    if mod_agree and not mod_contra:
        return {"support": 0.6, "status": "supported_by_independent_model"}
    if mod_contra and not mod_agree:
        return {"support": 0.2, "status": "contradicted_by_independent_model"}
    if mod_agree and mod_contra:
        return {"support": 0.4, "status": "models_disagree"}
    if any(x == "weak" for _, x in v):
        return {"support": 0.3, "status": "weak_independent_support"}
    return {"support": None, "status": "unverifiable (no independent data cover this family/period)"}


def score(fam: dict, ev: dict, th: dict) -> float:
    """Ranking for promotion: independent support first, then robustness, size and coherence; documented-by-default
    warming is down-weighted (it is known)."""
    sup = ev.get("support")
    sup = 0.3 if sup is None else sup
    rb = ev.get("robust") or {}
    rob = rb.get("passed", 0) / max(rb.get("n_checks", 1), 1)
    size_rule = {"trend": th["min_effect"], "shift": th["min_effect"], "seasonality": th["min_phase_days"],
                 "variance": th["min_log_var"], "recent_extreme": th["min_recent_z"]}[fam["cls"]]
    size = min(abs(fam["effect"]) / size_rule, 3) / 3
    coh = min(math.log1p(fam["n_cells"]) / math.log1p(100), 1)
    s = 0.45 * sup + 0.30 * rob + 0.15 * size + 0.10 * coh
    if (fam["variable"], fam["cls"]) in DOCUMENTED_BY_DEFAULT:
        s *= 0.5
    return round(s, 3)


def merge_groups(fams: list[dict], min_jaccard: float = 0.5) -> dict[str, str]:
    """Families on the same cells (any variable) form one signal. Greedy in score order against group
    representatives only (no chaining, so a continent-wide family cannot glue two regions together); overlap is
    Jaccard (shared / union cells), so a small family is not swallowed by a large one that merely contains it.
    Returns member -> representative."""
    sets = {f["family"]: {(round(a, 2), round(b, 2)) for a, b in
                          (json.loads(f["cells"]) if isinstance(f["cells"], str) else f["cells"])} for f in fams}
    reps: list[str] = []
    out: dict[str, str] = {}
    for f in sorted(fams, key=lambda f: -(f.get("score") or 0)):
        n, sa = f["family"], sets[f["family"]]
        best, bj = None, 0.0
        for r in reps:
            sb = sets[r]
            j = len(sa & sb) / max(len(sa | sb), 1)
            if j > bj:
                best, bj = r, j
        if best is not None and bj >= min_jaccard:
            out[n] = best
        else:
            reps.append(n)
            out[n] = n
    return out
