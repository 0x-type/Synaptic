"""NASA Earthdata access (Earthdata Login bearer token in EARTHDATA_TOKEN).

merra2_series reads native MERRA-2 monthly fields from GES DISC. MERRA-2 is on the same 0.5 x 0.625 degree grid as
NASA POWER's meteorology, so a POWER-minus-MERRA-2 difference tells whether a problem is in POWER's processing
(difference breaks) or already in the reanalysis (difference flat). Whole monthly files are downloaded (~50 MB,
about a second each), cut to the box, and deleted; the cut is cached for all campaigns.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import ToolContext, ToolResult

_NC_LOCK = threading.Lock()  # the netCDF/HDF5 C library is not thread-safe: parse one file at a time
GESDISC = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY"
# variable -> (collection, short name, file tag)
COLLECTIONS = {
    "flx": ("M2TMNXFLX.5.12.4", "tavgM_2d_flx_Nx", {"PRECTOTCORR", "PRECTOT", "PRECCON", "EVAP"}),
    "slv": ("M2TMNXSLV.5.12.4", "tavgM_2d_slv_Nx", {"T2M", "T2MDEW", "T2MWET", "QV2M", "PS", "SLP", "U10M", "V10M",
                                                    "U850", "V850", "TS", "TQV", "OMEGA500"}),
    "lnd": ("M2TMNXLND.5.12.4", "tavgM_2d_lnd_Nx", {"GWETROOT", "GWETTOP", "GWETPROF", "RZMC", "SFMC", "EVLAND",
                                                    "PRECTOTLAND"}),
    "rad": ("M2TMNXRAD.5.12.4", "tavgM_2d_rad_Nx", {"SWGDN", "SWGDNCLR", "LWGAB", "CLDTOT"}),
}
UNITS = {"PRECTOTCORR": ("mm/day", 86400.0), "PRECTOT": ("mm/day", 86400.0), "PRECCON": ("mm/day", 86400.0),
         "EVAP": ("mm/day", 86400.0), "PRECTOTLAND": ("mm/day", 86400.0), "EVLAND": ("mm/day", 86400.0),
         "T2M": ("C", None), "T2MDEW": ("C", None), "T2MWET": ("C", None),
         "T2MMEAN": ("C", None), "TS": ("C", None)}


def _token() -> str | None:
    return os.getenv("EARTHDATA_TOKEN") or None


def _stream(year: int, month: int) -> list[str]:
    base = 100 if year < 1992 else 200 if year < 2001 else 300 if year < 2011 else 400
    return [f"MERRA2_{base}", f"MERRA2_{base + 1}"]  # reprocessed months carry x01


def _collection(var: str):
    for key, (coll, tag, names) in COLLECTIONS.items():
        if var in names:
            return coll, tag
    return None


def _cache_dir() -> Path:
    from ..config import RUNS_DIR

    d = RUNS_DIR / "cache" / "merra2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_cut(var: str, coll: str, tag: str, y: int, m: int, box: tuple, tmp: Path) -> pd.DataFrame | None:
    import xarray as xr

    la0, la1, lo0, lo1 = box
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {_token()}"
    for stream in _stream(y, m):
        url = f"{GESDISC}/{coll}/{y}/{stream}.{tag}.{y}{m:02d}.nc4"
        f = tmp / f"{stream}.{tag}.{y}{m:02d}.nc4"
        try:
            r = s.get(url, timeout=300)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            f.write_bytes(r.content)
            with _NC_LOCK, xr.open_dataset(f) as ds:
                da = ds[var].isel(time=0).sel(lat=slice(la0 - 0.01, la1 + 0.01), lon=slice(lo0 - 0.01, lo1 + 0.01))
                df = da.load().to_dataframe(name="value").reset_index()[["lat", "lon", "value"]]
            unit, scale = UNITS.get(var, ("", None))
            if scale:
                df["value"] = df["value"] * scale
            elif unit == "C":
                df["value"] = df["value"] - 273.15
            df["year"], df["month"] = y, m
            return df
        except Exception:  # noqa: BLE001
            continue
        finally:
            f.unlink(missing_ok=True)
    return None


def merra2_series(ctx: ToolContext, variable: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                  start_year: int, end_year: int, months: list[int] | None = None) -> ToolResult:
    """Native MERRA-2 monthly values for a box (same grid as NASA POWER), saved in POWER's long format with
    variable name M2_<VAR>, so POWER minus MERRA-2 can be computed cell by cell."""
    if not _token():
        return ToolResult("No Earthdata token configured (EARTHDATA_TOKEN). MERRA-2 cannot be read in this run; "
                          "state the limitation.", is_error=True)
    var = variable.upper().strip()
    found = _collection(var)
    if not found:
        allv = sorted(set().union(*[c[2] for c in COLLECTIONS.values()]))
        return ToolResult(f"Unknown MERRA-2 variable {var}. Available: {allv}. Monthly MERRA-2 files have no daily "
                          "max/min temperature: use POWER (power_region) for T2M_MAX / T2M_MIN.", is_error=True)
    coll, tag = found
    months = sorted({int(m) for m in (months or range(1, 13))})
    y0, y1 = max(1980, int(start_year)), min(2026, int(end_year))
    box = (float(lat_min), float(lat_max), float(lon_min), float(lon_max))
    jobs = [(y, m) for y in range(y0, y1 + 1) for m in months]
    if len(jobs) > 600:
        return ToolResult(f"{len(jobs)} months requested; limit 600 per call (one variable, ~50 years of all months).",
                          is_error=True)
    key = hashlib.md5(json.dumps([var, box, y0, y1, months]).encode()).hexdigest()[:10]
    cache = _cache_dir() / f"m2_{var}_{key}.csv"
    if not cache.exists():
        tmp = _cache_dir() / "tmp"
        tmp.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            parts = list(ex.map(lambda ym: _month_cut(var, coll, tag, ym[0], ym[1], box, tmp), jobs))
        got = [p for p in parts if p is not None]
        if not got:
            return ToolResult("MERRA-2 download failed for every month (token expired or not authorized for "
                              "'NASA GESDISC DATA ARCHIVE'?).", is_error=True)
        df = pd.concat(got, ignore_index=True)
        df.insert(2, "variable", f"M2_{var}")
        df["lat"], df["lon"] = df.lat.round(4), df.lon.round(4)
        df.to_csv(cache, index=False)
    df = pd.read_csv(cache)
    out = ctx.data_dir / cache.name
    if not out.exists():
        out.write_bytes(cache.read_bytes())
    missing = len(jobs) - df.groupby(["year", "month"]).ngroups
    ann = df.groupby("year").value.mean()
    return ToolResult(
        f"Saved native MERRA-2 {var} ({coll}) for lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}, {y0}-{y1}, "
        f"months {months}: {len(df):,} cell-months, {df.groupby(['lat', 'lon']).ngroups} cells (same grid as POWER), "
        f"{missing} months missing, to {out.name} (variable M2_{var}).\nMerge with POWER on lat, lon, year, month: a "
        f"break in POWER minus MERRA-2 is POWER processing; no break means the problem is already in MERRA-2.\n"
        f"Box-mean by year: " + "  ".join(f"{int(y)}:{v:.2f}" for y, v in ann.items()), files=[str(out)])
