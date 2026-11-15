"""NASA Earth data tools: POWER (point + regional), GIBS imagery, CMR dataset search."""
from __future__ import annotations

import hashlib
import json
import shutil
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import ToolContext, ToolResult

POWER = "https://power.larc.nasa.gov/api/temporal/monthly"
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
CMR = "https://cmr.earthdata.nasa.gov/search"
UA = {"User-Agent": "synaptic-ai-scientist/1.0 (NASA Space Apps)"}


def _cached(ctx: ToolContext, name: str) -> Path:
    """Downloads go to a cache shared by all campaigns (fast reruns, works offline once fetched),
    then get copied into the campaign's own data folder, which is the only place agents can read."""
    from ..config import RUNS_DIR

    cache = RUNS_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / name


def _publish(ctx: ToolContext, cached: Path) -> Path:
    dest = ctx.data_dir / cached.name
    for src, dst in ((cached, dest), (cached.with_suffix(".meta.json"), dest.with_suffix(".meta.json"))):
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
    return dest


def _get(url: str, params: dict, timeout: int = 180, tries: int = 4) -> requests.Response:
    delay = 3
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


SOLAR = {"ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "ALLSKY_KT", "CLOUD_AMT"}
SOLAR_START = 1984  # POWER solar/cloud fields (GEWEX SRB, then CERES SYN1deg) start in 1984


def _clamp(params: list[str], start_year: int) -> tuple[int, str]:
    if any(p in SOLAR for p in params) and int(start_year) < SOLAR_START:
        return SOLAR_START, f"(start year raised to {SOLAR_START}: solar/cloud fields begin then)\n"
    return int(start_year), ""


def _params(p) -> list[str]:
    if isinstance(p, str):
        p = re.split(r"[,\s]+", p)
    return [x.strip().upper() for x in p if x and x.strip()]


def _features_to_rows(features: list, var_meta: dict) -> list[dict]:
    rows = []
    for f in features:
        lon, lat = f["geometry"]["coordinates"][:2]
        for var, series in f["properties"]["parameter"].items():
            for key, v in series.items():
                y, m = int(key[:4]), int(key[4:])
                if m == 13:  # POWER's annual value; we recompute our own
                    continue
                rows.append({"lat": round(lat, 4), "lon": round(lon, 4), "variable": var, "year": y, "month": m,
                             "value": np.nan if v is None or v <= -998 else v})
    return rows


def _summary(df: pd.DataFrame, path: Path, meta: dict) -> str:
    lines = [f"Saved {len(df):,} rows to {path}",
             f"  columns: lat, lon, variable, year, month, value   (month 1-12)",
             f"  grid cells: {df.groupby(['lat', 'lon']).ngroups}   years: {df.year.min()}-{df.year.max()}"]
    for v, g in df.groupby("variable"):
        u = meta.get(v, {})
        lines.append(f"  {v}: {u.get('longname', '')} [{u.get('units', '?')}]  "
                     f"missing {g.value.isna().mean():.1%}")
    if meta.get("_sources"):
        lines.append(f"  production sources reported by POWER: {', '.join(meta['_sources'])}")
    lines.append("Raw values are not shown here. Inspect them with look(path=...) or python.")
    return "\n".join(lines)


def power_point(ctx: ToolContext, lat: float, lon: float, parameters, start_year: int = 1981,
                end_year: int = 2025) -> ToolResult:
    params = _params(parameters)
    start_year, note = _clamp(params, start_year)
    key = hashlib.md5(json.dumps([round(float(lat), 3), round(float(lon), 3), params, start_year, end_year]).encode()).hexdigest()[:10]
    path = _cached(ctx, f"power_point_{key}.csv")
    meta_path = path.with_suffix(".meta.json")
    if not path.exists():
        r = _get(f"{POWER}/point", {"parameters": ",".join(params), "community": "AG", "latitude": lat,
                                    "longitude": lon, "start": start_year, "end": end_year, "format": "JSON"})
        d = r.json()
        if r.status_code != 200:
            return ToolResult(f"POWER error {r.status_code}: {d.get('messages') or d}", is_error=True)
        meta = dict(d.get("parameters", {}))
        meta["_sources"] = d.get("header", {}).get("sources", [])
        pd.DataFrame(_features_to_rows([d], meta)).to_csv(path, index=False)
        meta_path.write_text(json.dumps(meta))
    df = pd.read_csv(path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    path = _publish(ctx, path)
    return ToolResult(note + _summary(df, path, meta), files=[str(path)])


def _tiles(lat_min, lat_max, lon_min, lon_max, step=10.0):
    la = np.arange(lat_min, lat_max, step)
    lo = np.arange(lon_min, lon_max, step)
    for a in la:
        for b in lo:
            yield a, min(a + step, lat_max), b, min(b + step, lon_max)


def power_region(ctx: ToolContext, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                 parameters, start_year: int = 1981, end_year: int = 2025) -> ToolResult:
    params = _params(parameters)
    start_year, note = _clamp(params, start_year)
    area = (lat_max - lat_min) * (lon_max - lon_min)
    if area > 1600:
        return ToolResult("Region too large for one call (max 40x40 degrees). Split it.", is_error=True)
    if lat_max - lat_min < 2 or lon_max - lon_min < 2:
        return ToolResult("POWER regional needs a box of at least 2x2 degrees; use power_point for a site.",
                          is_error=True)
    key = hashlib.md5(json.dumps([lat_min, lat_max, lon_min, lon_max, params, start_year, end_year]).encode()).hexdigest()[:10]
    path = _cached(ctx, f"power_region_{key}.csv")
    meta_path = path.with_suffix(".meta.json")
    if not path.exists():
        rows, meta = [], {"_sources": []}
        for p in params:  # POWER regional allows one parameter and <=10x10 degrees per request
            for a, b, c, d_ in _tiles(lat_min, lat_max, lon_min, lon_max):
                if b - a < 2 or d_ - c < 2:  # widen thin edge tiles to the minimum box
                    a, c = min(a, b - 2), min(c, d_ - 2)
                r = _get(f"{POWER}/regional", {"parameters": p, "community": "AG", "latitude-min": a,
                                               "latitude-max": b, "longitude-min": c, "longitude-max": d_,
                                               "start": start_year, "end": end_year, "format": "JSON"})
                d = r.json()
                if r.status_code != 200:
                    return ToolResult(f"POWER error {r.status_code} for {p}: {d.get('messages') or d}",
                                      is_error=True)
                meta.update(d.get("parameters", {}))
                meta["_sources"] = sorted(set(meta["_sources"]) | set(d.get("header", {}).get("sources", [])))
                rows += _features_to_rows(d.get("features", []), meta)
        df = pd.DataFrame(rows).drop_duplicates(["lat", "lon", "variable", "year", "month"])
        df = df[(df.lat >= lat_min) & (df.lat <= lat_max) & (df.lon >= lon_min) & (df.lon <= lon_max)]
        df.to_csv(path, index=False)
        meta_path.write_text(json.dumps(meta))
    df = pd.read_csv(path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    path = _publish(ctx, path)
    return ToolResult(note + _summary(df, path, meta), files=[str(path)])


# ---------------------------------------------------------------------------- GIBS imagery

_GIBS_LAYERS: list[str] = []


def gibs_layers(ctx: ToolContext, keyword: str) -> ToolResult:
    global _GIBS_LAYERS
    if not _GIBS_LAYERS:
        cache = ctx.data_dir / "gibs_layers.json"
        if cache.exists():
            _GIBS_LAYERS = json.loads(cache.read_text())
        else:
            r = _get(GIBS_WMS, {"SERVICE": "WMS", "REQUEST": "GetCapabilities", "VERSION": "1.3.0"}, timeout=120)
            _GIBS_LAYERS = sorted(set(re.findall(r"<Layer[^>]*>\s*<Name>([^<]+)</Name>", r.text)))
            cache.write_text(json.dumps(_GIBS_LAYERS))
    words = [w.lower() for w in keyword.split()]
    hits = [l for l in _GIBS_LAYERS if all(w in l.lower() for w in words)]
    return ToolResult(f"{len(hits)} GIBS layers match '{keyword}':\n" + "\n".join(hits[:40]))


def gibs_snapshot(ctx: ToolContext, layer: str, date: str, lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float, width: int = 900) -> ToolResult:
    """Satellite image of a box on a date (YYYY-MM-DD). Saved as PNG; use look(path) to see it."""
    h = max(200, int(width * (lat_max - lat_min) / max(lon_max - lon_min, 1e-6)))
    r = _get(GIBS_WMS, {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layer,
                        "CRS": "EPSG:4326", "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
                        "WIDTH": width, "HEIGHT": min(h, 1400), "FORMAT": "image/png", "TIME": date}, timeout=120)
    if "image" not in r.headers.get("Content-Type", ""):
        return ToolResult(f"GIBS returned no image: {r.text[:300]}", is_error=True)
    path = ctx.workdir / f"gibs_{re.sub(r'[^A-Za-z0-9]+', '_', layer)[:40]}_{date}.png"
    path.write_bytes(r.content)
    return ToolResult(f"Saved {layer} on {date} to {path}. Use look(path='{path.name}') to view it.",
                      files=[str(path)])


# ---------------------------------------------------------------------------- CMR

def cmr_search(ctx: ToolContext, keyword: str, limit: int = 10) -> ToolResult:
    r = _get(f"{CMR}/collections.json", {"keyword": keyword, "page_size": min(int(limit), 25),
                                         "sort_key": "-usage_score"}, timeout=60)
    entries = r.json().get("feed", {}).get("entry", [])
    lines = []
    for e in entries:
        lines.append(f"- {e.get('short_name')} v{e.get('version_id')}: {e.get('title')}\n"
                     f"  time {e.get('time_start', '?')[:10]} .. {(e.get('time_end') or 'present')[:10]}"
                     f"  | {e.get('data_center', '')}")
    return ToolResult(f"CMR collections for '{keyword}':\n" + ("\n".join(lines) or "(none)"))
