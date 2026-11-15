"""More of NASA's Earth archive through the Earthdata token: satellite rain (GPM IMERG) and land-model soil
moisture / evaporation (GLDAS Noah), on the campaign's own grid.

GPM IMERG V07 monthly   0.1 deg, 1998-present. Satellites (microwave + infrared) calibrated to GPCC rain gauges.
                        Independent of MERRA-2's rain (MERRA-2 corrects with CPC/CMAP, not IMERG).
GLDAS Noah 2.0 monthly  1 deg, 1948-2014. Land model driven by the Princeton observation-based forcing.
GLDAS Noah 2.1 monthly  1 deg, 2000-present. Driven by GDAS analysis + GPCP rain. Not the MERRA-2 land model, so its
                        root-zone soil moisture and evaporation are an independent model estimate.
                        2.0 and 2.1 have different forcing: never splice them; compare each with MERRA-2 separately
                        (they overlap 2000-2014).

Each monthly file is downloaded once (IMERG ~18 MB, GLDAS ~2 MB, well under a second each), regridded to the NASA
POWER / MERRA-2 grid (0.5 x 0.625 deg) and cached for every campaign; global campaigns get the same 2 x 2.5 deg cells
as their data table (4 x 4 POWER cells, exactly as globaldata.py averages MERRA-2).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import ToolContext, ToolResult

GPM = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07"
HYDRO = "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS"
# GLDAS short names -> (file variable, scale, offset, units)
GLDAS_VARS = {
    "ROOTMOIST": ("RootMoist_inst", 1.0, 0.0, "kg/m2 root-zone soil water"),
    "SOILM_0_10": ("SoilMoi0_10cm_inst", 1.0, 0.0, "kg/m2"),
    "SOILM_10_40": ("SoilMoi10_40cm_inst", 1.0, 0.0, "kg/m2"),
    "SOILM_40_100": ("SoilMoi40_100cm_inst", 1.0, 0.0, "kg/m2"),
    "EVAP": ("Evap_tavg", 86400.0, 0.0, "mm/day"),
    "TRANSP": ("Tveg_tavg", 86400.0 / 2.501e6, 0.0, "mm/day (transpiration)"),
    "POTEVAP": ("PotEvap_tavg", 86400.0 / 2.501e6, 0.0, "mm/day (potential evaporation)"),
    "RAIN": ("Rainf_f_tavg", 86400.0, 0.0, "mm/day (forcing rain)"),
    "TAIR": ("Tair_f_inst", 1.0, -273.15, "C"),
    "QAIR": ("Qair_f_inst", 1.0, 0.0, "kg/kg"),
    "RUNOFF": ("Qs_acc", 8.0, 0.0, "mm/day (surface runoff)"),
    "LATENT": ("Qle_tavg", 1.0, 0.0, "W/m2"),
    "SENSIBLE": ("Qh_tavg", 1.0, 0.0, "W/m2"),
    "SWE": ("SWE_inst", 1.0, 0.0, "kg/m2 snow water"),
}
PRODUCTS = {
    "imerg": {"years": (1998, 2025), "vars": {"PRECIP": ("precipitation", 24.0, 0.0, "mm/day")},
              "what": "GPM IMERG V07 satellite+gauge rain"},
    "gldas20": {"years": (1948, 2014), "vars": GLDAS_VARS, "what": "GLDAS Noah 2.0 land model (Princeton forcing)"},
    "gldas21": {"years": (2000, 2025), "vars": GLDAS_VARS, "what": "GLDAS Noah 2.1 land model (GDAS + GPCP forcing)"},
}
# POWER / MERRA-2 grid
PLAT = -90.0 + 0.5 * np.arange(361)
PLON = -180.0 + 0.625 * np.arange(576)


def _cache_dir(product: str) -> Path:
    from ..config import RUNS_DIR

    d = RUNS_DIR / "cache" / "nasa" / product
    d.mkdir(parents=True, exist_ok=True)
    return d


def _url(product: str, y: int, m: int) -> list[str]:
    if product == "imerg":
        return [f"{GPM}/{y}/3B-MO.MS.MRG.3IMERG.{y}{m:02d}01-S000000-E235959.{m:02d}.V07{v}.HDF5" for v in "BA"]
    ver = "2.0" if product == "gldas20" else "2.1"
    return [f"{HYDRO}/GLDAS_NOAH10_M.{ver}/{y}/GLDAS_NOAH10_M.A{y}{m:02d}.0{ver.replace('.', '')}.nc4"]


def _to_power(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Average a regular lat/lon field (lat x lon) onto POWER cell centres; coarser sources are sampled (nearest)."""
    out = np.full((len(PLAT), len(PLON)), np.nan, dtype=np.float32)
    res = abs(float(lat[1] - lat[0]))
    if res < 0.5:  # finer than POWER: bin-average
        li = np.clip(np.rint((lat + 90.0) / 0.5).astype(int), 0, 360)
        lj = np.rint((lon + 180.0) / 0.625).astype(int) % 576
        idx = (li[:, None] * 576 + lj[None, :]).ravel()
        v = field.ravel().astype(float)
        ok = np.isfinite(v)
        s = np.bincount(idx[ok], v[ok], minlength=361 * 576)
        n = np.bincount(idx[ok], minlength=361 * 576)
        with np.errstate(invalid="ignore"):
            out = (s / np.where(n > 0, n, np.nan)).reshape(361, 576).astype(np.float32)
    else:  # coarser: each POWER cell takes the source pixel containing it
        ii = np.rint((PLAT - lat[0]) / res).astype(int)
        jj = np.rint((PLON - lon[0]) / res).astype(int)
        oki = (ii >= 0) & (ii < len(lat))
        okj = (jj >= 0) & (jj < len(lon))
        sub = field[np.clip(ii, 0, len(lat) - 1)][:, np.clip(jj, 0, len(lon) - 1)]
        out = np.where(oki[:, None] & okj[None, :], sub, np.nan).astype(np.float32)
    return out


def _month(product: str, y: int, m: int, token: str) -> tuple[int, int, bool]:
    """Download one month, regrid every variable of the product, cache as one npz. Runs in its own process."""
    out = _cache_dir(product) / f"{y}{m:02d}.npz"
    if out.exists():
        return y, m, True
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    tmp = _cache_dir(product) / f"tmp_{y}{m:02d}_{os.getpid()}"
    for url in _url(product, y, m):
        for attempt in range(3):
            try:
                r = s.get(url, timeout=300)
                if r.status_code == 404:
                    break
                r.raise_for_status()
                tmp.write_bytes(r.content)
                fields = {}
                if product == "imerg":
                    import h5py

                    with h5py.File(tmp) as f:
                        g = f["Grid"]
                        a = g["precipitation"][0].T.astype(float)  # (lon, lat) -> (lat, lon)
                        a[a < 0] = np.nan
                        fields["PRECIP"] = _to_power(a * 24.0, g["lat"][:], g["lon"][:])
                else:
                    import xarray as xr

                    with xr.open_dataset(tmp) as d:
                        lat, lon = d["lat"].values, d["lon"].values
                        for k, (name, scale, off, _) in GLDAS_VARS.items():
                            a = d[name].isel(time=0).values.astype(float)
                            a[np.abs(a) > 1e10] = np.nan
                            fields[k] = _to_power(a * scale + off, lat, lon)
                np.savez_compressed(out, **fields)
                return y, m, True
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
            finally:
                tmp.unlink(missing_ok=True)
    return y, m, False


def fetch(product: str, years: range, token: str, workers: int = 4) -> list[str]:
    todo = [(y, m) for y in years for m in range(1, 13) if not (_cache_dir(product) / f"{y}{m:02d}.npz").exists()]
    missing = []
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for y, m, ok in ex.map(_month, [product] * len(todo), [t[0] for t in todo], [t[1] for t in todo],
                                   [token] * len(todo)):
                if not ok:
                    missing.append(f"{y}-{m:02d}")
    return missing


def _is_global(ctx: ToolContext) -> bool:
    try:
        row = ctx.record.one("SELECT brief FROM campaigns WHERE id=?", (ctx.campaign_id,))
        return bool(json.loads(row["brief"]).get("region", {}).get("global"))
    except Exception:  # noqa: BLE001
        return False


def nasa_series(ctx: ToolContext, product: str, variable: str, lat_min: float, lat_max: float, lon_min: float,
                lon_max: float, start_year: int, end_year: int, months: list[int] | None = None) -> ToolResult:
    """Per-cell monthly series from GPM IMERG or GLDAS on the campaign's grid, saved in the POWER long format."""
    from .catalogdata import refuse_if_held_out

    no = refuse_if_held_out(ctx, ["IMERG_PRECIP"] if str(product).lower() == "imerg" else
                            ["GLDAS_SOILM", "GLDAS_EVAP", "GLDAS_SNOWDEPTH"])
    if no:
        return no
    token = os.getenv("EARTHDATA_TOKEN")
    if not token:
        return ToolResult("nasa_series needs an Earthdata token (EARTHDATA_TOKEN); not configured in this run.",
                          is_error=True)
    product = str(product).lower().replace(" ", "").replace("-", "").replace(".", "")
    product = {"gpm": "imerg", "gpmimerg": "imerg", "gldas": "gldas21"}.get(product, product)
    if product not in PRODUCTS:
        return ToolResult(f"product must be one of {list(PRODUCTS)}", is_error=True)
    p = PRODUCTS[product]
    var = str(variable).upper().replace("PRECIPITATION", "PRECIP")
    if product == "imerg":
        var = "PRECIP"
    if var not in p["vars"]:
        return ToolResult(f"{product} variables: {list(p['vars'])}", is_error=True)
    y0, y1 = max(int(start_year), p["years"][0]), min(int(end_year), p["years"][1])
    if y0 > y1:
        return ToolResult(f"{product} covers {p['years'][0]}-{p['years'][1]} only.", is_error=True)
    glob = _is_global(ctx)
    area = (lat_max - lat_min) * (lon_max - lon_min)
    if area > (20000 if glob else 400):
        return ToolResult(f"Box too large (max {20000 if glob else 400} square degrees).", is_error=True)
    months = sorted({int(m) for m in (months or range(1, 13))})
    key = hashlib.md5(json.dumps([product, var, lat_min, lat_max, lon_min, lon_max, y0, y1, months, glob]).encode()
                      ).hexdigest()[:10]
    from .catalogdata import held_out
    # found live (round 1): a referee's IMERG download landed in the SHARED data folder and explorers used it; data
    # of a held-out product stay in the requesting task's own folder
    priv = (product == "imerg" and "IMERG_PRECIP" in held_out(ctx)) or (product != "imerg" and held_out(ctx) & {
        "GLDAS_SOILM", "GLDAS_EVAP", "GLDAS_SNOWDEPTH"})
    path = (ctx.workdir if priv else ctx.data_dir) / f"{product}_{var}_{key}.csv"
    name = f"{product.upper()}_{var}"
    if not path.exists():
        missing = fetch(product, range(y0, y1 + 1), token)
        if glob:  # 2 x 2.5 deg cells, the same averaging as globaldata.py
            lat = PLAT[:360].reshape(-1, 4).mean(1)
            lon = PLON.reshape(-1, 4).mean(1)
        else:
            lat, lon = PLAT, PLON
        ii = np.nonzero((lat >= lat_min) & (lat <= lat_max))[0]
        jj = np.nonzero((lon >= lon_min) & (lon <= lon_max))[0]
        if not len(ii) or not len(jj):
            return ToolResult("No grid cells inside that box.", is_error=True)
        rows = []
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for y in range(y0, y1 + 1):
            for m in months:
                f = _cache_dir(product) / f"{y}{m:02d}.npz"
                if not f.exists():
                    continue
                a = np.load(f)[var].astype(float)
                if glob:
                    a = np.nanmean(a[:360].reshape(90, 4, 144, 4), axis=(1, 3))
                blk = a[np.ix_(ii, jj)]
                la, lo = np.meshgrid(lat[ii], lon[jj], indexing="ij")
                ok = np.isfinite(blk)
                rows.append(pd.DataFrame({"lat": np.round(la[ok], 3), "lon": np.round(lo[ok], 3), "variable": name,
                                          "year": y, "month": m, "value": np.round(blk[ok], 5)}))
        if not rows:
            return ToolResult(f"No {product} data could be read for {y0}-{y1} (missing months: {missing[:10]}).",
                              is_error=True)
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    df = pd.read_csv(path)
    ann = df.groupby("year").value.mean()
    units = p["vars"][var][3]
    note = ("IMERG = satellites + GPCC gauges, independent of MERRA-2's CPC/CMAP rain correction (coastal cells include ocean "
            "pixels, so compare inland cells or ratios). "
            if product == "imerg" else
            "GLDAS Noah = a different land model with different forcing from MERRA-2: an independent model estimate "
            "(not an observation). Never splice gldas20 and gldas21; compare each with MERRA-2 over its own years. ")
    return ToolResult(
        f"Saved {len(df):,} cell-month values of {p['what']} {var} [{units}] to {path.name}: "
        f"{df.groupby(['lat', 'lon']).ngroups} cells on the {'campaign 2 x 2.5 deg' if glob else 'NASA POWER'} grid "
        f"(same lat/lon as the campaign data: merge on lat, lon, year, month), {y0}-{y1}, months {months}.\n{note}"
        f"Annual means over the box: " + ", ".join(f"{int(k)}: {v:.3g}" for k, v in ann.items()),
        files=[str(path)])


def cells_frame(product: str, var: str, cells: pd.DataFrame, y0: int, y1: int, glob: bool,
                token: str | None = None) -> pd.DataFrame:
    """Monthly values at given campaign cells (lat, lon), long format (lat, lon, year, month, ref).
    Used by the automatic independent check; fetches missing months first when a token is available."""
    p = PRODUCTS[product]
    y0, y1 = max(y0, p["years"][0]), min(y1, p["years"][1])
    if y0 > y1:
        return pd.DataFrame(columns=["lat", "lon", "year", "month", "ref"])
    token = token or os.getenv("EARTHDATA_TOKEN")
    if token:
        fetch(product, range(y0, y1 + 1), token)
    if glob:
        lat, lon = PLAT[:360].reshape(-1, 4).mean(1), PLON.reshape(-1, 4).mean(1)
    else:
        lat, lon = PLAT, PLON
    ii = np.abs(lat[None, :] - cells.lat.values[:, None]).argmin(1)
    jj = np.abs(lon[None, :] - cells.lon.values[:, None]).argmin(1)
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    frames = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            f = _cache_dir(product) / f"{y}{m:02d}.npz"
            if not f.exists():
                continue
            a = np.load(f)[var].astype(float)
            if glob:
                a = np.nanmean(a[:360].reshape(90, 4, 144, 4), axis=(1, 3))
            frames.append(pd.DataFrame({"lat": cells.lat.values, "lon": cells.lon.values, "year": y, "month": m,
                                        "ref": a[ii, jj]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["lat", "lon", "year", "month", "ref"])
