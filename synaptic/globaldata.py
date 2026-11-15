"""Whole-Earth dataset straight from NASA's MERRA-2 archive (Earthdata token), for global campaigns.

NASA POWER's meteorology is MERRA-2 passed through unchanged (verified cell by cell), but POWER's API serves at most
10 x 10 degrees per request. For the whole planet the harness reads the monthly MERRA-2 files directly, averages the
0.5 x 0.625 degree grid to 2 x 2.5 degrees, keeps land cells, and writes the same long format as POWER.

Variables (POWER names): PRECTOTCORR (mm/day, observation-corrected rain), T2M (C), RH2M (%, from QV2M, T2M, PS),
GWETROOT (root-zone soil wetness 0-1), EVLAND (land evaporation, mm/day).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GESDISC = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY"
FILES = {"flx": ("M2TMNXFLX.5.12.4", "tavgM_2d_flx_Nx"), "slv": ("M2TMNXSLV.5.12.4", "tavgM_2d_slv_Nx"),
         "lnd": ("M2TMNXLND.5.12.4", "tavgM_2d_lnd_Nx")}
VARIABLES = ["PRECTOTCORR", "T2M", "RH2M", "GWETROOT", "EVLAND"]
COARSEN = (4, 4)            # 0.5 x 0.625 -> 2 x 2.5 degrees
LAT_RANGE = (-60.0, 80.0)   # Antarctica excluded (no rain-fed land climate to study, sparse checks)


def cache_dir() -> Path:
    from .config import RUNS_DIR

    d = RUNS_DIR / "cache" / "global_m2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _streams(y: int) -> list[str]:
    base = 100 if y < 1992 else 200 if y < 2001 else 300 if y < 2011 else 400
    return [f"MERRA2_{base}", f"MERRA2_{base + 1}"]


def _get(session, key: str, y: int, m: int, tmp: Path) -> Path | None:
    coll, tag = FILES[key]
    for s in _streams(y):
        url = f"{GESDISC}/{coll}/{y}/{s}.{tag}.{y}{m:02d}.nc4"
        for attempt in range(3):
            try:
                r = session.get(url, timeout=600)
                if r.status_code == 404:
                    break
                r.raise_for_status()
                f = tmp / f"{key}_{y}{m:02d}_{os.getpid()}.nc4"
                f.write_bytes(r.content)
                return f
            except Exception:  # noqa: BLE001
                time.sleep(3 * (attempt + 1))
    return None


def _month(y: int, m: int, token: str) -> tuple[int, int, dict | None]:
    """Download the three monthly files, compute the variables on the native grid, coarsen. One process each."""
    import xarray as xr

    out = cache_dir() / f"m2_{y}{m:02d}.npz"
    if out.exists():
        return y, m, {"cached": True}
    tmp = cache_dir() / "tmp"
    tmp.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    paths = {k: _get(s, k, y, m, tmp) for k in FILES}
    try:
        if any(p is None for p in paths.values()):
            return y, m, None
        with xr.open_dataset(paths["flx"]) as fx, xr.open_dataset(paths["slv"]) as sv, xr.open_dataset(paths["lnd"]) as ld:
            pr = fx["PRECTOTCORR"].isel(time=0).values * 86400.0
            t2 = sv["T2M"].isel(time=0).values - 273.15
            q = sv["QV2M"].isel(time=0).values
            ps = sv["PS"].isel(time=0).values / 100.0
            e = q * ps / (0.622 + 0.378 * q)
            es = 6.112 * np.exp(17.67 * t2 / (t2 + 243.5))
            rh = np.clip(100.0 * e / es, 0, 100)
            gw = ld["GWETROOT"].isel(time=0).values.astype(float)
            ev = ld["EVLAND"].isel(time=0).values.astype(float) * 86400.0
            lat, lon = fx["lat"].values, fx["lon"].values
        land = np.isfinite(gw) & (gw < 1e10)
        gw[~land] = np.nan
        ev[~land | (np.abs(ev) > 1e5)] = np.nan
        a, b = COARSEN
        ny, nx = (len(lat) // a) * a, (len(lon) // b) * b

        def co(x):
            return np.nanmean(x[:ny, :nx].reshape(ny // a, a, nx // b, b), axis=(1, 3))

        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        landfrac = land[:ny, :nx].reshape(ny // a, a, nx // b, b).mean(axis=(1, 3))
        np.savez_compressed(out, PRECTOTCORR=co(pr), T2M=co(t2), RH2M=co(rh), GWETROOT=co(gw), EVLAND=co(ev),
                            landfrac=landfrac, lat=lat[:ny].reshape(-1, a).mean(1), lon=lon[:nx].reshape(-1, b).mean(1))
        return y, m, {"cached": False}
    finally:
        for p in paths.values():
            if p is not None:
                p.unlink(missing_ok=True)


def build(start_year: int, end_year: int, token: str | None = None, workers: int = 6, log=print) -> dict:
    """Fetch every month (resumable: finished months are cached), then write POWER-format CSVs per variable."""
    token = token or os.getenv("EARTHDATA_TOKEN")
    if not token:
        raise RuntimeError("EARTHDATA_TOKEN is required for global campaigns")
    jobs = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]
    todo = [j for j in jobs if not (cache_dir() / f"m2_{j[0]}{j[1]:02d}.npz").exists()]
    t0, missing = time.time(), []
    if todo:
        log(f"global data: fetching {len(todo)} of {len(jobs)} months of MERRA-2 (3 files each) with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_month, y, m, token) for y, m in todo]
            for i, f in enumerate(as_completed(futs), 1):
                y, m, info = f.result()
                if info is None:
                    missing.append(f"{y}-{m:02d}")
                if i % 60 == 0:
                    log(f"global data: {i}/{len(todo)} months ({(time.time() - t0) / 60:.1f} min)")
    return {"months": len(jobs), "missing": missing, "minutes": round((time.time() - t0) / 60, 1)}


def write_tables(start_year: int, end_year: int, out_dir: Path, min_landfrac: float = 0.5) -> dict[str, Path]:
    """Long-format CSV per variable (lat, lon, variable, year, month, value) over land cells."""
    files = sorted(cache_dir().glob("m2_*.npz"))
    files = [f for f in files if start_year <= int(f.stem[3:7]) <= end_year]
    if not files:
        raise RuntimeError("no cached MERRA-2 months: run build() first")
    first = np.load(files[0])
    lat, lon = first["lat"], first["lon"]
    keep = (first["landfrac"] >= min_landfrac) & (lat[:, None] >= LAT_RANGE[0]) & (lat[:, None] <= LAT_RANGE[1])
    ii, jj = np.nonzero(keep)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for v in VARIABLES:
        rows = []
        for f in files:
            d = np.load(f)
            y, m = int(f.stem[3:7]), int(f.stem[7:9])
            vals = d[v][ii, jj]
            rows.append(pd.DataFrame({"lat": np.round(lat[ii], 3), "lon": np.round(lon[jj], 3), "variable": v,
                                      "year": y, "month": m, "value": np.round(vals.astype(float), 4)}))
        df = pd.concat(rows, ignore_index=True)
        p = out_dir / f"global_{v}_{start_year}_{end_year}.csv"
        df.to_csv(p, index=False)
        paths[v] = p
    return paths
