"""Independent observational products for corroborating (or refuting) NASA POWER / MERRA-2 signals.

CHIRPS v2.0  monthly rainfall, 0.05 deg, 1981-present, 50S-50N, rain gauges + satellite infrared
             (Climate Hazards Center, UC Santa Barbara). Independent of the MERRA-2 reanalysis.
CHIRTS-max   monthly mean daily maximum temperature, 0.05 deg, 1983-2016, station + satellite infrared.

A difference series POWER minus independent product that shows a step = an inhomogeneity in one of
the two products (the classic homogeneity test used by climate-data scientists).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from .base import ToolContext, ToolResult

CHC = "https://data.chc.ucsb.edu/products"
PRODUCTS = {
    "chirps": {"years": (1981, 2025), "units": "mm/day", "what": "rainfall (CHIRPS v2.0)"},
    "chirts": {"years": (1983, 2016), "units": "deg C", "what": "daily maximum temperature (CHIRTS-max)"},
}
DAYS = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def _in_africa(lat_min, lat_max, lon_min, lon_max) -> bool:
    return lat_min >= -40 and lat_max <= 40 and lon_min >= -20 and lon_max <= 55


def _read_window(product: str, y: int, m: int, box: tuple):
    """Return (array, transform-like (lat_top, lon_left, res)) for the box, or None."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds

    la0, la1, lo0, lo1 = box
    for attempt in range(4):
        try:
            if product == "chirps":
                region = "africa_monthly" if _in_africa(*box) else "global_monthly"
                raw = requests.get(f"{CHC}/CHIRPS-2.0/{region}/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz", timeout=180)
                if raw.status_code == 404:
                    return None
                raw.raise_for_status()
                with MemoryFile(gzip.decompress(raw.content)) as mf, mf.open() as src:
                    w = from_bounds(lo0, la0, lo1, la1, src.transform)
                    a = src.read(1, window=w).astype(float)
                    t = src.window_transform(w)
                a[a < 0] = np.nan
                a = a / DAYS[m]
            else:
                url = f"/vsicurl/{CHC}/CHIRTSmonthly/CHIRTSmax.CDR/CHIRTSmax.{y}.{m:02d}.tif"
                with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                    with rasterio.open(url) as src:
                        w = from_bounds(lo0, la0, lo1, la1, src.transform)
                        a = src.read(1, window=w).astype(float)
                        t = src.window_transform(w)
                a[a < -100] = np.nan
            return a, (t.f, t.c, t.a)
        except Exception:  # noqa: BLE001
            continue
    return None


def _to_power_grid(a, geo, box) -> list[tuple[float, float, float]]:
    """Average 0.05-deg pixels onto NASA POWER / MERRA-2 cell centres (0.5 lat x 0.625 lon, lon = -180 + 0.625k)."""
    top, left, res = geo
    ny, nx = a.shape
    plat = top - (np.arange(ny) + 0.5) * res
    plon = left + (np.arange(nx) + 0.5) * res
    la0, la1, lo0, lo1 = box
    lats = np.arange(np.ceil(la0 / 0.5) * 0.5, la1 + 1e-9, 0.5)
    lons = -180 + 0.625 * np.arange(np.ceil((lo0 + 180) / 0.625), np.floor((lo1 + 180) / 0.625) + 1)
    out = []
    for la in lats:
        ri = np.abs(plat - la) <= 0.25
        for lo in lons:
            ci = np.abs(plon - lo) <= 0.3125
            blk = a[np.ix_(ri, ci)]
            if blk.size and np.isfinite(blk).any():
                out.append((round(float(la), 4), round(float(lo), 4), float(np.nanmean(blk))))
    return out


def _box_mean(product: str, y: int, m: int, box: tuple) -> float:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds

    la0, la1, lo0, lo1 = box
    for attempt in range(4):
        try:
            if product == "chirps":
                region = "africa_monthly" if _in_africa(*box) else "global_monthly"
                raw = requests.get(f"{CHC}/CHIRPS-2.0/{region}/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz", timeout=180)
                if raw.status_code == 404:
                    return np.nan
                raw.raise_for_status()
                with MemoryFile(gzip.decompress(raw.content)) as mf, mf.open() as src:
                    a = src.read(1, window=from_bounds(lo0, la0, lo1, la1, src.transform)).astype(float)
                a[a < 0] = np.nan
                return float(np.nanmean(a)) / DAYS[m]  # mm/month -> mm/day
            url = f"/vsicurl/{CHC}/CHIRTSmonthly/CHIRTSmax.CDR/CHIRTSmax.{y}.{m:02d}.tif"
            with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(url) as src:
                    a = src.read(1, window=from_bounds(lo0, la0, lo1, la1, src.transform)).astype(float)
            a[a < -100] = np.nan
            return float(np.nanmean(a))
        except Exception:  # noqa: BLE001
            continue
    return np.nan


def independent_series(ctx: ToolContext, product: str, lat_min: float, lat_max: float, lon_min: float,
                       lon_max: float, start_year: int, end_year: int, months: list[int] | None = None,
                       per_cell: bool = False) -> ToolResult:
    """Box-mean monthly series from an independent product ('chirps' rainfall or 'chirts' Tmax), saved as a CSV in
    the same long format as POWER data (variable CHIRPS_PRECIP or CHIRTS_TMAX), ready for a difference series."""
    product = product.lower().strip()
    if product not in PRODUCTS:
        return ToolResult(f"product must be one of {list(PRODUCTS)}", is_error=True)
    p = PRODUCTS[product]
    y0, y1 = max(int(start_year), p["years"][0]), min(int(end_year), p["years"][1])
    if y0 > y1:
        return ToolResult(f"{product} covers {p['years'][0]}-{p['years'][1]} only.", is_error=True)
    if (lat_max - lat_min) * (lon_max - lon_min) > 400:
        return ToolResult("Box too large (max 400 square degrees); corroborate a family's own box.", is_error=True)
    months = sorted({int(m) for m in (months or range(1, 13))})
    box = (float(lat_min), float(lat_max), float(lon_min), float(lon_max))
    key = hashlib.md5(json.dumps([product, box, y0, y1, months, bool(per_cell)]).encode()).hexdigest()[:10]
    path = ctx.data_dir / f"{product}_{'grid_' if per_cell else ''}{key}.csv"
    var = "CHIRPS_PRECIP" if product == "chirps" else "CHIRTS_TMAX"
    if per_cell and not path.exists():
        jobs = [(y, m) for y in range(y0, y1 + 1) for m in months]
        rows = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for (y, m), res in zip(jobs, ex.map(lambda ym: _read_window(product, ym[0], ym[1], box), jobs)):
                if res is None:
                    continue
                for la, lo, v in _to_power_grid(res[0], res[1], box):
                    rows.append({"lat": la, "lon": lo, "variable": var, "year": y, "month": m, "value": v})
        pd.DataFrame(rows).to_csv(path, index=False)
        df = pd.read_csv(path)
        return ToolResult(
            f"Saved {len(df):,} cell-month values of {p['what']} [{p['units']}] on the NASA POWER grid "
            f"({df.groupby(['lat', 'lon']).ngroups} cells, lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}, {y0}-{y1}, "
            f"months {months}) to {path.name}. Same lat/lon as POWER cells: merge on lat, lon, year, month and map "
            "the per-cell POWER/CHIRPS ratio (rain) or POWER-CHIRTS difference (Tmax) break.", files=[str(path)])
    if not path.exists():
        jobs = [(y, m) for y in range(y0, y1 + 1) for m in months]
        with ThreadPoolExecutor(max_workers=8) as ex:
            vals = list(ex.map(lambda ym: _box_mean(product, ym[0], ym[1], box), jobs))
        lat_c, lon_c = round((box[0] + box[1]) / 2, 3), round((box[2] + box[3]) / 2, 3)
        pd.DataFrame([{"lat": lat_c, "lon": lon_c, "variable": var, "year": y, "month": m, "value": v}
                      for (y, m), v in zip(jobs, vals)]).to_csv(path, index=False)
    df = pd.read_csv(path)
    miss = df.value.isna().mean()
    ann = df.groupby("year").value.mean()
    return ToolResult(
        f"Saved {len(df)} monthly box means of {p['what']} [{p['units']}] for lat {lat_min}..{lat_max}, lon "
        f"{lon_min}..{lon_max}, {y0}-{y1}, months {months} to {path.name} (single pseudo-cell at the box centre; "
        f"missing {miss:.0%}).\nIndependent of MERRA-2: build POWER-minus-{product} (or ratio for rain) over the same "
        f"box and months, then test that difference series for a break (synlib.pettitt). A break in the difference "
        f"means one product is inhomogeneous.\nAnnual means: {synlib_text(ann)}", files=[str(path)])


def synlib_text(ann: pd.Series) -> str:
    return "  ".join(f"{int(y)}:{v:.2f}" for y, v in ann.items() if np.isfinite(v))


# ---------------------------------------------------------------------------- GHCN-Daily weather stations
GHCN = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCN_VARS = {"TMAX": ("GHCN_TMAX", "deg C", 10.0), "TMIN": ("GHCN_TMIN", "deg C", 10.0),
             "PRCP": ("GHCN_PRCP", "mm/day", 10.0)}


def _ghcn_meta():
    """Station list and inventory (cached once for all campaigns)."""
    from ..config import RUNS_DIR

    cache = RUNS_DIR / "cache" / "ghcn"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("ghcnd-stations.txt", "ghcnd-inventory.txt"):
        p = cache / name
        if not p.exists():
            r = requests.get(f"{GHCN}/{name}", timeout=300)
            r.raise_for_status()
            p.write_bytes(r.content)
    st = pd.read_fwf(cache / "ghcnd-stations.txt", colspecs=[(0, 11), (12, 20), (21, 30), (41, 71)],
                     names=["id", "lat", "lon", "name"])
    inv = pd.read_fwf(cache / "ghcnd-inventory.txt", colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                      names=["id", "lat", "lon", "el", "y0", "y1"])
    return cache, st, inv


def _ghcn_station(cache, sid: str):
    p = cache / f"{sid}.csv.gz"
    if not p.exists():
        r = requests.get(f"{GHCN}/by_station/{sid}.csv.gz", timeout=300)
        if not r.ok:
            return None
        p.write_bytes(r.content)
    return pd.read_csv(p, header=None, usecols=[0, 1, 2, 3, 5], names=["id", "date", "el", "v", "q"],
                       dtype={"date": str, "q": str})


def station_series(ctx: ToolContext, element: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                   start_year: int, end_year: int, min_days: int = 20, max_stations: int = 40) -> ToolResult:
    """Monthly means from NOAA GHCN-Daily weather stations (ground truth, independent of MERRA-2), each station mapped
    to its NASA POWER grid cell so it merges directly with POWER data on lat, lon, year, month."""
    el = element.upper().strip()
    if el not in GHCN_VARS:
        return ToolResult(f"element must be one of {list(GHCN_VARS)}", is_error=True)
    if (lat_max - lat_min) * (lon_max - lon_min) > 900:
        return ToolResult("Box too large (max 900 square degrees).", is_error=True)
    var, units, scale = GHCN_VARS[el]
    try:
        cache, st, inv = _ghcn_meta()
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"GHCN station list unreachable: {str(e)[:120]}", is_error=True)
    y0, y1 = int(start_year), int(end_year)
    cand = inv[(inv.el == el) & inv.lat.between(lat_min, lat_max) & inv.lon.between(lon_min, lon_max)
               & (inv.y1 >= y0) & (inv.y0 <= y1)].copy()
    if cand.empty:
        return ToolResult(f"No GHCN {el} stations in that box for {y0}-{y1}.")
    cand["overlap"] = cand.y1.clip(upper=y1) - cand.y0.clip(lower=y0)
    cand = cand.sort_values("overlap", ascending=False).head(int(max_stations))
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(lambda s: _ghcn_station(cache, s), cand.id))
    rows = []
    for sid, d in zip(cand.id, frames):
        if d is None:
            continue
        d = d[(d.el == el) & d.q.isna()]  # drop values that failed NOAA quality control
        d = d.assign(year=d.date.str[:4].astype(int), month=d.date.str[4:6].astype(int))
        d = d[d.year.between(y0, y1)]
        g = d.groupby(["year", "month"]).v.agg(["mean", "count"]).reset_index()
        g = g[g["count"] >= int(min_days)]
        meta = st[st.id == sid].iloc[0]
        clat = round(round(meta.lat / 0.5) * 0.5, 4)
        clon = round(-180 + round((meta.lon + 180) / 0.625) * 0.625, 4)
        for r in g.itertuples():
            rows.append({"lat": clat, "lon": clon, "variable": var, "year": r.year, "month": r.month,
                         "value": r.mean / scale, "station": sid, "name": meta["name"], "st_lat": meta.lat,
                         "st_lon": meta.lon, "n_days": r.count})
    if not rows:
        return ToolResult(f"GHCN {el}: stations found but no month with >= {min_days} valid days in {y0}-{y1}.")
    df = pd.DataFrame(rows)
    key = hashlib.md5(json.dumps([el, lat_min, lat_max, lon_min, lon_max, y0, y1, min_days]).encode()).hexdigest()[:10]
    path = ctx.data_dir / f"ghcn_{el.lower()}_{key}.csv"
    df.to_csv(path, index=False)
    cov = df.groupby(["station", "name", "st_lat", "st_lon", "lat", "lon"]).agg(
        months=("value", "size"), first=("year", "min"), last=("year", "max")).reset_index()
    cov = cov.sort_values("months", ascending=False)
    lines = [f"{r.station} {r.name.strip()[:22]:22s} ({r.st_lat:.2f},{r.st_lon:.2f}) -> POWER cell ({r.lat},{r.lon}): "
             f"{r.months} months {r.first}-{r.last}" for r in cov.head(25).itertuples()]
    return ToolResult(
        f"Saved {len(df):,} station-months of GHCN-Daily {el} [{units}] from {len(cov)} stations to {path.name} "
        f"(months with >= {min_days} quality-controlled days only). Columns lat, lon are the POWER cell of each station, "
        f"so merge with POWER on lat, lon, year, month and test POWER minus station (Tmax/Tmin) or the ratio (rain) for "
        f"a break. Compare month by month: annual means over months with gaps are biased.\n" + "\n".join(lines)
        + (f"\n... {len(cov) - 25} more" if len(cov) > 25 else ""), files=[str(path)])
