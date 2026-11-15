"""NASA Earthdata Cloud access for the catalogue (synaptic/catalog.py): CMR granule search, OPeNDAP subsetting,
Harmony variable subsetting and whole-record files, with a disk cache. One Earthdata token (EARTHDATA_TOKEN) is
all that is needed.

Main entry points
  field(var_id, bbox, y0, y1)            -> Field (times, lat, lon, data[t, y, x]) in catalogue units
  regional_series(var_id, region, y0, y1) -> DataFrame year, month, value, coverage (area-weighted mean)
  table(var_id, bbox, y0, y1, res)        -> long table lat, lon, variable, year, month, value (campaign format)
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import Var, get

CMR = "https://cmr.earthdata.nasa.gov/search"
HARMONY = "https://harmony.earthdata.nasa.gov"
CACHE = Path(os.environ.get("SYNAPTIC_CACHE", Path(__file__).resolve().parents[1] / "runs" / "cache")) / "cloud"
try:  # whole-record files of held-out products live here: agent code (unprivileged user) must not read them
    CACHE.mkdir(parents=True, exist_ok=True)
    CACHE.parent.chmod(0o700)
except OSError:
    pass


class DataError(RuntimeError):
    pass


def _token() -> str:
    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not tok:
        raise DataError("EARTHDATA_TOKEN is not set. Create one at https://urs.earthdata.nasa.gov (Generate Token) and "
                        "put it in .env.")
    return tok


def _get(url: str, auth: bool = True, timeout: int = 120, data: bytes | None = None, tries: int = 4) -> bytes:
    hdr = {"User-Agent": "synaptic/4"}
    if auth:
        hdr["Authorization"] = "Bearer " + _token()
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdr, data=data)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            last = e
            if e.code in (400, 401, 403, 404):
                body = e.read()[:300].decode("utf-8", "ignore")
                raise DataError(f"HTTP {e.code} for {url[:160]}: {body}") from e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 * (k + 1))
    raise DataError(f"request failed after {tries} tries: {url[:160]} ({last})")


# ---------------------------------------------------------------------------- granules

def granules(var: Var, y0: int, y1: int) -> list[dict]:
    """[{'ym': 'YYYY-MM', 'title', 'opendap', 'https'}] one per month (first granule wins), cached for a day."""
    key = CACHE / "cmr" / f"{var.collection}{'_' + var.id if var.granule else ''}_{y0}_{y1}.json"
    if key.exists() and time.time() - key.stat().st_mtime < 86400:
        return json.loads(key.read_text())
    out: dict[str, dict] = {}
    page = 1
    while True:
        params = {"collection_concept_id": var.collection, "page_size": 2000, "page_num": page,
                  "temporal": f"{y0}-01-01T00:00:00Z,{y1}-12-31T23:59:59Z", "sort_key": "start_date"}
        if var.granule:  # e.g. only the monthly 9 km files of a collection that also holds daily and 4 km ones
            params.update({"readable_granule_name": var.granule, "options[readable_granule_name][pattern]": "true"})
        q = urllib.parse.urlencode(params)
        ent = json.loads(_get(f"{CMR}/granules.json?{q}", auth=False))["feed"]["entry"]
        for e in ent:
            ym = e.get("time_start", "")[:7]
            title = e.get("title", "")
            m = re.search(r"\.(\d{4})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})\.L3m\.MO", title)
            if m:  # MODIS L3m monthly names carry the month exactly
                ym = f"{m.group(1)}-{m.group(2)}"
            if not ym or ym in out or not (y0 <= int(ym[:4]) <= y1):
                continue
            links = [lk["href"] for lk in e.get("links", [])]
            out[ym] = {"ym": ym, "title": title,
                       "opendap": next((h for h in links if "opendap.earthdata" in h), None),
                       "https": next((h for h in links if h.startswith("https") and "/opendap/" not in h and
                                      h.split("?")[0].endswith((".nc", ".nc4", ".hdf", ".HDF5", ".h5"))), None)}
        if len(ent) < 2000:
            break
        page += 1
    res = [out[k] for k in sorted(out)]
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------------------- grids and boxes

@dataclass
class Field:
    var: Var
    times: list[str]            # 'YYYY-MM'
    lat: np.ndarray
    lon: np.ndarray             # always -180..180, ascending
    data: np.ndarray            # [t, lat, lon] in catalogue units, NaN = missing
    missing: list = None        # months that could not be fetched (left as NaN)

    def frame(self) -> pd.DataFrame:
        t, y, x = np.meshgrid(np.arange(len(self.times)), self.lat, self.lon, indexing="ij")
        df = pd.DataFrame({"ym": np.array(self.times)[t.ravel()], "lat": y.ravel(), "lon": x.ravel(),
                           "value": self.data.ravel()})
        df["year"] = df.ym.str[:4].astype(int)
        df["month"] = df.ym.str[5:7].astype(int)
        return df.drop(columns="ym").dropna(subset=["value"])


import threading

_NC_LOCK = threading.Lock()  # netCDF4/HDF5 is not thread-safe: downloads run in parallel, decoding one at a time


def _dap_raw(url: str, ce: str) -> bytes:
    ce = ";".join(c if c.startswith("/") else "/" + c for c in ce.split(";"))
    return _get(f"{url}.dap.nc4?dap4.ce={urllib.parse.quote(ce, safe='/:;,')}", timeout=60, tries=2)


def _dap(url: str, ce: str) -> "netCDF4.Dataset":  # noqa: F821
    import netCDF4

    ce = ";".join(c if c.startswith("/") else "/" + c for c in ce.split(";"))
    raw = _get(f"{url}.dap.nc4?dap4.ce={urllib.parse.quote(ce, safe='/:;,')}", timeout=60, tries=2)  # brackets encoded
    return netCDF4.Dataset("mem.nc", memory=raw)


def grid(var: Var, g: dict) -> tuple[np.ndarray, np.ndarray]:
    if var.grid_spec:  # regular grid whose coordinates are not 1-D variables in the file (AIRS, OMI L3)
        la0, dla, nla, lo0, dlo, nlo = var.grid_spec
        return la0 + dla * np.arange(nla), lo0 + dlo * np.arange(nlo)
    key = CACHE / "grids" / f"{var.collection}.npz"
    if key.exists():
        z = np.load(key)
        return z["lat"], z["lon"]
    raw = _dap_raw(g["opendap"], f"{var.lat};{var.lon}")
    import netCDF4

    with _NC_LOCK:
        ds = netCDF4.Dataset("mem.nc", memory=raw)
        lat = np.asarray(ds[var.lat][:], float)
        lon = np.asarray(ds[var.lon][:], float)
        ds.close()
    key.parent.mkdir(parents=True, exist_ok=True)
    np.savez(key, lat=lat, lon=lon)
    return lat, lon


def _index_range(coord: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    idx = np.flatnonzero((coord >= lo) & (coord <= hi))
    if len(idx) == 0:  # a box smaller than one grid cell: take the nearest cell
        idx = np.array([int(np.abs(coord - (lo + hi) / 2).argmin())])
    return int(idx.min()), int(idx.max())


def _norm_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def _crop(lat, lon, arr, bbox):
    """Crop a global [.., lat, lon] array (any lon convention, any lat order) to bbox = (S, N, W, E); returns
    ascending lat and lon in -180..180."""
    S, N, W, E = bbox
    lon2 = _norm_lon(lon)
    oi = np.argsort(lon2)
    lon2, arr = lon2[oi], arr[..., oi]
    li = np.argsort(lat)
    lat2, arr = lat[li], arr[..., li, :]
    ym = (lat2 >= S) & (lat2 <= N)
    xm = (lon2 >= W) & (lon2 <= E) if W <= E else (lon2 >= W) | (lon2 <= E)
    if not ym.any():
        ym[np.abs(lat2 - (S + N) / 2).argmin()] = True
    if not xm.any():
        xm[np.abs(lon2 - (W + E) / 2).argmin()] = True
    return lat2[ym], lon2[xm], arr[..., ym, :][..., xm]


def _apply(var: Var, a) -> np.ndarray:
    a = np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)
    a[np.abs(a) > 1e14] = np.nan
    return a * var.convert[0] + var.convert[1]


# ---------------------------------------------------------------------------- one month, per backend

def _month_opendap(var: Var, g: dict, bbox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = grid(var, g)
    S, N, W, E = bbox
    i0, i1 = _index_range(lat, S, N)
    lon2 = _norm_lon(lon)
    if W <= E and np.all(np.diff(lon2) > 0):
        j0, j1 = _index_range(lon2, W, E)
    else:
        j0, j1 = 0, len(lon) - 1
    key = CACHE / "months" / var.id / f"{g['ym']}_{i0}_{i1}_{j0}_{j1}.npy"
    if key.exists():
        a = np.load(key)
    else:
        sl = {"t": "[0:0]", "y": f"[{i0}:{i1}]", "x": f"[{j0}:{j1}]", "z": f"[{var.level}:{var.level}]"}
        ce = var.variable + "".join(sl[d] for d in var.dims)
        raw = _dap_raw(g["opendap"], ce)
        name = var.variable.split("/")[-1]
        import netCDF4

        with _NC_LOCK:
            ds = netCDF4.Dataset("mem.nc", memory=raw)
            try:
                try:
                    v = ds[var.variable]  # full path, also through nested groups (AIRS: /ascending/Data_Fields/..)
                except (IndexError, KeyError):
                    v = ds.variables[name] if name in ds.variables else \
                        next(iter(ds.groups.values())).variables[name]
                a = _apply(var, v[:])
            finally:
                ds.close()
        a = np.squeeze(a, axis=var.dims.index("t")) if "t" in var.dims else a
        a = np.squeeze(a, axis=var.dims.index("z")) if "z" in var.dims else a
        if var.dims.replace("t", "") == "xy":
            a = a.T
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, a.astype(np.float32))
    la, lo = lat[i0:i1 + 1], lon[j0:j1 + 1]
    return _crop(la, lo, a, bbox)


def _harmony_months(var: Var, y0: int, y1: int, progress=None) -> dict[str, Path]:
    """Global monthly files of one variable through Harmony (one job per call, cached per month)."""
    d = CACHE / "harmony" / var.id
    d.mkdir(parents=True, exist_ok=True)
    have = {p.stem: p for p in d.glob("*.npy")}
    avail = {g["ym"] for g in granules(var, max(y0, var.start), y1)}  # months that exist in the archive
    missing = sorted(w for w in avail if w not in have)
    if missing:
        t0, t1 = missing[0], missing[-1]
        q = (f"{HARMONY}/{var.collection}/ogc-api-coverages/1.0.0/collections/parameter_vars/coverage/rangeset?"
             + urllib.parse.urlencode({"subset": f'time("{t0}-01T00:00:00Z":"{t1}-28T23:59:59Z")',
                                       "variable": var.variable, "forceAsync": "true"}))
        req = urllib.request.Request(q, headers={"Authorization": "Bearer " + _token()})
        job = None
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                job = json.loads(body).get("jobID")
                if job is None and r.geturl().rstrip("/").split("/")[-2] == "jobs":
                    job = r.geturl().rstrip("/").split("/")[-1]
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            loc = e.headers.get("Location", "")
            if "/jobs/" in loc:
                job = loc.rstrip("/").split("/")[-1]
            else:
                raise DataError(f"Harmony refused the request: {e}") from e
        if not job:
            raise DataError("Harmony returned no job id")
        status = {}
        for _ in range(360):
            status = json.loads(_get(f"{HARMONY}/jobs/{job}"))
            if progress:
                progress(f"NASA Harmony subsetting {var.source}: {status.get('progress', 0)}%")
            if status.get("status") in ("successful", "complete_with_errors", "failed", "canceled"):
                break
            time.sleep(5)
        if status.get("status") in ("failed", "canceled"):
            raise DataError(f"Harmony job {job} {status.get('status')}: {status.get('message')}")
        links = [lk["href"] for lk in status.get("links", []) if lk.get("rel") == "data"]

        def one(url):
            m = re.search(r"\.A(\d{4})(\d{3})\.", url)
            if not m:
                return
            ym = time.strftime("%Y-%m", time.strptime(f"{m.group(1)} {m.group(2)}", "%Y %j"))
            if ym in have:
                return
            raw = _get(url)
            tmp = d / f"{ym}.hdf"
            tmp.write_bytes(raw)
            try:
                from pyhdf.SD import SD, SDC

                with _NC_LOCK:  # HDF4 is not thread-safe either
                    f = SD(str(tmp), SDC.READ)
                    sds = f.select(var.variable)
                    arr = sds.get().astype(float)
                    at = sds.attributes()
                    f.end()
                fill = at.get("_FillValue")
                if fill is not None:
                    arr[arr == fill] = np.nan
                arr = (arr - at.get("add_offset", 0.0)) * at.get("scale_factor", 1.0)
                np.save(d / f"{ym}.npy", (arr * var.convert[0] + var.convert[1]).astype(np.float32))
            finally:
                tmp.unlink(missing_ok=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, links))
    return {p.stem: p for p in d.glob("*.npy")}


def _file(var: Var, progress=None):
    import netCDF4

    path = CACHE / "files" / Path(urllib.parse.urlparse(var.file_url).path).name if var.file_url else None
    if var.file_url and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"downloading {var.source} (whole record, once)")
        host = urllib.parse.urlparse(var.file_url).hostname or ""
        # the Earthdata token goes only to Earthdata hosts (GSFC's own mascon site needs no login)
        hdr = {"Authorization": "Bearer " + _token()} if host.endswith((".earthdata.nasa.gov", ".eosdis.nasa.gov")) else {}
        req = urllib.request.Request(var.file_url, headers=hdr)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        tmp.rename(path)
    if path is None:  # GRACE-like: the collection holds one granule with the whole record
        g = granules(var, var.start, 2100)
        if not g or not g[-1]["https"]:
            raise DataError(f"no downloadable file for {var.id}")
        var2 = Var(**{**var.__dict__, "file_url": g[-1]["https"]})
        return _file(var2, progress)
    return netCDF4.Dataset(str(path))


def _file_field(var: Var, bbox, y0, y1, progress=None):
    ds = _file(var, progress)
    t = ds["time"]
    import netCDF4

    dates = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    yms = [f"{d.year}-{d.month:02d}" for d in dates]
    lat = np.asarray(ds["lat"][:], float)
    lon = np.asarray(ds["lon"][:], float)
    sel = [i for i, ym in enumerate(yms) if y0 <= int(ym[:4]) <= y1]
    a = _apply(var, ds[var.variable][sel, :, :])
    la, lo, a = _crop(lat, lon, a, bbox)
    times = [yms[i] for i in sel]
    # GRACE: two solutions can fall in one calendar month; keep the first
    keep, seen = [], set()
    for k, ym in enumerate(times):
        if ym not in seen:
            seen.add(ym)
            keep.append(k)
    return [times[k] for k in keep], la, lo, a[keep]


def _monthfile_one(var: Var, g: dict, block: int = 3) -> Path | None:
    """One global file per month (no remote subsetting offered): download once, average block x block cells
    (NaN-aware), keep the reduced global grid. 9 km chlorophyll with block 3 gives 0.25 degrees."""
    import netCDF4

    out = CACHE / "monthfile" / var.id / f"{g['ym']}.npz"
    if out.exists():
        return out
    if not g.get("https"):
        return None
    req = urllib.request.Request(g["https"], headers={"Authorization": "Bearer " + _token()})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    with _NC_LOCK:
        ds = netCDF4.Dataset("m.nc", memory=raw)
        a = _apply(var, ds[var.variable][:])
        lat = np.asarray(ds[var.lat][:], float)
        lon = np.asarray(ds[var.lon][:], float)
        ds.close()
    ny, nx = a.shape[0] // block, a.shape[1] // block
    b = a[:ny * block, :nx * block].reshape(ny, block, nx, block)
    n = np.isfinite(b).sum(axis=(1, 3))
    m = np.where(n > 0, np.nansum(b, axis=(1, 3)) / np.maximum(n, 1), np.nan).astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.npz")
    np.savez_compressed(tmp, data=m, lat=lat[:ny * block].reshape(ny, block).mean(1),
                        lon=lon[:nx * block].reshape(nx, block).mean(1))
    tmp.rename(out)
    return out


def _monthfile_field(var: Var, bbox, y0: int, y1: int, progress=None, workers: int = 8) -> Field:
    gs = [g for g in granules(var, y0, y1) if g.get("https")]
    if not gs:
        raise DataError(f"no downloadable monthly files for {var.id} {y0}-{y1}")
    done = [0]

    def one(g):
        r = None
        for k in range(3):
            try:
                r = _monthfile_one(var, g)
                break
            except Exception:  # noqa: BLE001
                time.sleep(3 * (k + 1))
        done[0] += 1
        if progress and done[0] % 12 == 0:
            progress(f"{var.source.split(',')[0]}: {done[0]}/{len(gs)} months")
        return r
    with ThreadPoolExecutor(workers) as ex:
        files = list(ex.map(one, gs))
    times, arrs, grid_ = [], [], None
    for g, f in zip(gs, files):
        if f is None:
            continue
        z = np.load(f)
        la, lo, a = _crop(z["lat"], z["lon"], z["data"].astype(float), bbox)
        times.append(g["ym"]), arrs.append(a)
        grid_ = (la, lo)
    if not arrs:
        raise DataError(f"every month failed for {var.id}")
    f = Field(var, times, grid_[0], grid_[1], np.stack(arrs))
    f.missing = [g["ym"] for g, x in zip(gs, files) if x is None]
    return f


def _daily_granules(var: Var, year: int, month: int) -> list[str]:
    """OPeNDAP links of every daily file of one month (one per day; a reprocessed day replaces the earlier one)."""
    import calendar

    key = CACHE / "cmr" / f"daily_{var.collection}_{year}-{month:02d}.json"
    now = time.gmtime()
    fresh = key.exists() and ((year, month) < (now.tm_year, now.tm_mon - 1) or time.time() - key.stat().st_mtime < 86400)
    if fresh:
        return json.loads(key.read_text())
    last = calendar.monthrange(year, month)[1]
    q = urllib.parse.urlencode({"collection_concept_id": var.collection, "page_size": 200, "sort_key": "start_date",
                                "temporal": f"{year}-{month:02d}-01T00:00:00Z,{year}-{month:02d}-{last}T23:59:59Z"})
    days = {}
    for e in json.loads(_get(f"{CMR}/granules.json?{q}", auth=False))["feed"]["entry"]:
        d = e.get("time_start", "")[:10]
        od = next((lk["href"] for lk in e.get("links", []) if "opendap.earthdata" in lk["href"]), None)
        if od and d.startswith(f"{year}-{month:02d}"):
            days[d] = od
    out = [days[d] for d in sorted(days)]
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(out))
    return out


def _dailymean_field(var: Var, bbox, y0: int, y1: int, progress=None, workers: int = 16) -> Field:
    """Monthly means of a product NASA publishes daily (OMI L3): each day's box is fetched through OPeNDAP and the
    valid days are averaged. The monthly box means are cached, so only the first request for a box is slow."""
    import netCDF4

    lat, lon = grid(var, {})
    S, N, W, E = bbox
    i0, i1 = _index_range(lat, S, N)
    j0, j1 = _index_range(lon, W, E) if W <= E else (0, len(lon) - 1)
    now = time.gmtime()
    months = [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)
              if (var.start, 10) <= (y, m) and (y, m) < (now.tm_year, now.tm_mon)]
    ce = f"{var.variable}[{i0}:{i1}][{j0}:{j1}]"
    name = var.variable.split("/")[-1]

    def keyf(y, m):
        return CACHE / "months" / var.id / f"{y}-{m:02d}_{i0}_{i1}_{j0}_{j1}.npy"

    def one_day(url):
        for k in range(3):
            try:
                raw = _dap_raw(url, ce)
                with _NC_LOCK:
                    ds = netCDF4.Dataset("mem.nc", memory=raw)
                    try:
                        g = ds
                        for part in var.variable.strip("/").split("/")[:-1]:
                            g = g.groups[part]
                        # not _apply: it treats |value| > 1e14 as missing, and NO2 columns are ~1e15-1e16
                        a = np.ma.filled(np.ma.asarray(g.variables[name][:], dtype=float), np.nan)
                        a[np.abs(a) > 1e29] = np.nan
                        return a * var.convert[0] + var.convert[1]
                    finally:
                        ds.close()
            except Exception:  # noqa: BLE001
                time.sleep(2 * (k + 1))
        return None
    todo = [(y, m) for (y, m) in months if not keyf(y, m).exists()]
    done = [0]
    with ThreadPoolExecutor(workers) as ex:
        for (y, m) in todo:
            try:
                urls = _daily_granules(var, y, m)
            except Exception:  # noqa: BLE001
                urls = []
            arrs = [a for a in ex.map(one_day, urls) if a is not None]
            if arrs:
                st = np.stack(arrs)
                st[st < 0] = np.nan  # fill values and unphysical negatives
                n = np.isfinite(st).sum(axis=0)
                mean = np.where(n > 0, np.nansum(st, axis=0) / np.maximum(n, 1), np.nan)
                keyf(y, m).parent.mkdir(parents=True, exist_ok=True)
                np.save(keyf(y, m), mean.astype(np.float32))
            done[0] += 1
            if progress and done[0] % 6 == 0:
                progress(f"{var.source.split(',')[0]}: {done[0]}/{len(todo)} months (daily files averaged)")
    times, out, grid_ = [], [], None
    for (y, m) in months:
        if keyf(y, m).exists():
            la, lo, a = _crop(lat[i0:i1 + 1], lon[j0:j1 + 1], np.load(keyf(y, m)).astype(float), bbox)
            times.append(f"{y}-{m:02d}"), out.append(a)
            grid_ = (la, lo)
    if not out:
        raise DataError(f"no data for {var.id} {y0}-{y1}")
    f = Field(var, times, grid_[0], grid_[1], np.stack(out))
    f.missing = [f"{y}-{m:02d}" for (y, m) in months if not keyf(y, m).exists()]
    return f


# ---------------------------------------------------------------------------- public API

def field(var_id: str, bbox, y0: int, y1: int, progress=None, workers: int = 8) -> Field:
    """Monthly field over bbox = (S, N, W, E) in degrees, years y0..y1 inclusive."""
    var = get(var_id)
    y0 = max(y0, var.start)
    if var.backend == "file":
        times, la, lo, a = _file_field(var, bbox, y0, y1, progress)
        return Field(var, times, la, lo, a)
    if var.backend == "monthfile":
        return _monthfile_field(var, bbox, y0, y1, progress, workers)
    if var.backend == "dailymean":
        return _dailymean_field(var, bbox, y0, y1, progress)
    if var.backend == "harmony":
        files = _harmony_months(var, y0, y1, progress)
        times = sorted(k for k in files if y0 <= int(k[:4]) <= y1)
        lat = np.linspace(89.5, -89.5, 180)
        lon = np.linspace(-179.5, 179.5, 360)
        out = None
        arrs = []
        for k in times:
            la, lo, a = _crop(lat, lon, np.load(files[k]), bbox)
            arrs.append(a)
            out = (la, lo)
        if not arrs:
            raise DataError(f"no data for {var_id} {y0}-{y1}")
        return Field(var, times, out[0], out[1], np.stack(arrs))
    gs = [g for g in granules(var, y0, y1) if g["opendap"]]
    if not gs:
        raise DataError(f"no granules with OPeNDAP access for {var_id} {y0}-{y1}")
    done = [0]

    def one(g):
        err = None
        for k in range(2):  # the cloud OPeNDAP server sometimes answers 400/5xx under load: retry, then skip
            try:
                r = _month_opendap(var, g, bbox)
                break
            except Exception as e:  # noqa: BLE001
                err, r = e, None
                time.sleep(3 * (k + 1))
        done[0] += 1
        if progress and done[0] % 12 == 0:
            progress(f"{var.source}: {done[0]}/{len(gs)} months")
        return r if r is not None else err
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, gs))
    good = [r for r in res if isinstance(r, tuple)]
    if not good:
        raise DataError(f"every month failed for {var_id}: {res[0]}")
    la, lo = good[0][0], good[0][1]
    shape = good[0][2].shape
    data = np.stack([r[2] if isinstance(r, tuple) and r[2].shape == shape else np.full(shape, np.nan) for r in res])
    f = Field(var, [g["ym"] for g in gs], la, lo, data)
    f.missing = [g["ym"] for g, r in zip(gs, res) if not isinstance(r, tuple)]
    return f


def area_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.repeat(np.cos(np.radians(lat))[:, None], len(lon), axis=1)


def regional_series(var_id: str, bbox, y0: int, y1: int, mask=None, progress=None) -> pd.DataFrame:
    """Area-weighted regional mean per month. `mask(lat, lon) -> bool[lat, lon]` restricts it to a country or
    state; coverage = share of the region's area with data that month."""
    f = field(var_id, bbox, y0, y1, progress)
    w = area_weights(f.lat, f.lon)
    if mask is not None:
        w = w * mask(f.lat, f.lon)
    rows = []
    tot = w.sum()
    for k, ym in enumerate(f.times):
        a = f.data[k]
        ok = np.isfinite(a) & (w > 0)
        cov = float(w[ok].sum() / tot) if tot else 0.0
        val = float((a[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        rows.append({"year": int(ym[:4]), "month": int(ym[5:7]), "value": val, "coverage": round(cov, 3)})
    return pd.DataFrame(rows)


def table(var_id: str, bbox, y0: int, y1: int, res: float = 1.0, progress=None) -> pd.DataFrame:
    """Campaign table (lat, lon, variable, year, month, value) block-averaged to a res-degree grid (NaN-aware,
    done in numpy month by month so a global 0.5-degree record never becomes a 60-million-row frame)."""
    f = field(var_id, bbox, y0, y1, progress)
    yi = np.floor(f.lat / res).astype(int)
    xi = np.floor(f.lon / res).astype(int)
    uy, ys = np.unique(yi, return_inverse=True)
    ux, xs = np.unique(xi, return_inverse=True)
    cell = (ys[:, None] * len(ux) + xs[None, :]).ravel()
    ncell = len(uy) * len(ux)
    clat = np.repeat((uy + 0.5) * res, len(ux))
    clon = np.tile((ux + 0.5) * res, len(uy))
    out = []
    for k, ym in enumerate(f.times):
        a = f.data[k].ravel()
        ok = np.isfinite(a)
        s_ = np.bincount(cell[ok], weights=a[ok], minlength=ncell)
        n_ = np.bincount(cell[ok], minlength=ncell)
        good = n_ > 0
        out.append(pd.DataFrame({"lat": clat[good], "lon": clon[good], "variable": var_id, "year": int(ym[:4]),
                                 "month": int(ym[5:7]), "value": (s_[good] / n_[good]).astype(np.float32)}))
    return pd.concat(out, ignore_index=True)
