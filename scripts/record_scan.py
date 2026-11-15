"""Global record scan (step 1 of docs/preregistrations/2_record_scan.md): where did the latest seasons break the GRACE record?

For every 4-degree land block between 45S and 50N, on BOTH GRACE solutions (JPL RL06.3 mascons, NASA GSFC RL06v2.0
mascons): the wet-season storage gain (from the block's climatological low to its high) and the dry-season drawdown
(high to low), for every year. An event season is one that ends between Nov 2024 and Jun 2026; the baseline is every
earlier season since 2003. A block is a candidate when, on both solutions, the event season is the record (highest
or lowest) and at least 3 baseline standard deviations from the baseline mean, in the same direction.
Reads only the GRACE files cached by the harness. Writes evidence/record_scan/scan_blocks.csv and candidates.json.
Usage: python scripts/record_scan.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FILES = ROOT / "runs" / "cache" / "cloud" / "files"
OUT = ROOT / "evidence" / "record_scan"
BLOCK, LAT0, LAT1 = 4.0, -45.0, 50.0
EVENT_FROM, EVENT_TO = (2024, 11), (2026, 6)     # the 'to' period of an event season ends in this range
EXCLUDE = [("Sahel-Sahara (rounds 1-4)", (8, 23, -18, 30)), ("Morocco (rounds 3-4)", (27, 37, -14, 2))]
Z_MIN = 3.0


def load(pattern: str):
    f = sorted(FILES.glob(pattern))[-1]
    ds = netCDF4.Dataset(f)
    t = ds["time"]
    dates = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    lat, lon = np.asarray(ds["lat"][:], float), np.asarray(ds["lon"][:], float)
    a = np.ma.filled(ds["lwe_thickness"][:].astype("float32"), np.nan)
    land = np.asarray(ds["land_mask"][:]) > 0.5
    lon = np.where(lon > 180, lon - 360, lon)
    o = np.argsort(lon)
    lon, a, land = lon[o], a[:, :, o], land[:, o]
    if lat[0] > lat[-1]:
        lat, a, land = lat[::-1], a[:, ::-1], land[::-1]
    ym, keep, seen = [], [], set()
    for k, d in enumerate(dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            ym.append(key)
            keep.append(k)
    return f.name, ym, lat, lon, a[keep], land


def blocks(ym, lat, lon, a, land):
    """{(blat, blon): Series indexed (year, month)} of area-weighted land means; blocks with >= 60% land."""
    out = {}
    w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))[None, :]
    for bl in np.arange(LAT0, LAT1, BLOCK):
        yi = (lat >= bl) & (lat < bl + BLOCK)
        for bo in np.arange(-180.0, 180.0, BLOCK):
            xi = (lon >= bo) & (lon < bo + BLOCK)
            m = land[np.ix_(yi, xi)]
            if m.mean() < 0.6:
                continue
            ww = (w[np.ix_(yi, xi)] * m)
            sub = a[:, yi][:, :, xi]
            ok = np.isfinite(sub) & (ww > 0)[None]
            val = np.where(ok, sub, 0) * ww[None]
            s = val.sum(axis=(1, 2)) / np.maximum((ok * ww[None]).sum(axis=(1, 2)), 1e-9)
            s[ok.sum(axis=(1, 2)) == 0] = np.nan
            out[(bl + BLOCK / 2, bo + BLOCK / 2)] = pd.Series(s, index=pd.MultiIndex.from_tuples(ym))
    return out


def nxt(m):
    return 1 if m == 12 else m + 1


def season_table(s: pd.Series, frm: int, to: int):
    """Gain from the 2-month mean starting at `frm` to the 2-month mean starting at `to`, for each season.
    Returns [(season_end (year, month), gain)]. The from period precedes the to period by less than 12 months."""
    rows = []
    for y in range(2002, 2027):
        fy = y
        f1, f2 = (fy, frm), (fy + (frm == 12), nxt(frm))
        ty = fy if to > frm else fy + 1
        t1, t2 = (ty, to), (ty + (to == 12), nxt(to))
        a = np.nanmean([s.get(f1, np.nan), s.get(f2, np.nan)]) if (f1 in s.index or f2 in s.index) else np.nan
        b = np.nanmean([s.get(t1, np.nan), s.get(t2, np.nan)]) if (t1 in s.index or t2 in s.index) else np.nan
        if np.isfinite(a) and np.isfinite(b):
            rows.append((t2, float(b - a)))
    return rows


def scan_solution(bs: dict, windows: dict):
    rows = []
    for key, s in bs.items():
        for kind, (frm, to) in windows.get(key, {}).items():
            tab = season_table(s, frm, to)
            base = [g for (e, g) in tab if e < EVENT_FROM]
            ev = [(e, g) for (e, g) in tab if EVENT_FROM <= e <= EVENT_TO]
            if len(base) < 12 or not ev:
                continue
            mu, sd = float(np.mean(base)), float(np.std(base, ddof=1))
            for e, g in ev:
                rows.append({"lat": key[0], "lon": key[1], "kind": kind, "from_month": frm, "to_month": to,
                             "season_end": f"{e[0]}-{e[1]:02d}", "gain": g, "base_mean": mu, "base_sd": sd,
                             "z": (g - mu) / sd if sd > 0 else np.nan, "n_base": len(base),
                             "record_high": g > max(base), "record_low": g < min(base)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jn, jym, lat, lon, ja, jland = load("GRCTellus.JPL.*.nc")
    gn, gym, glat, glon, ga, gland = load("GSFC.glb_*.nc")
    jb, gb = blocks(jym, lat, lon, ja, jland), blocks(gym, glat, glon, ga, gland)
    # each block's seasonal windows from the JPL climatology of the baseline years only
    windows = {}
    for key, s in jb.items():
        base = s[[i for i in s.index if i < EVENT_FROM and i[0] >= 2003]]
        clim = base.groupby(level=1).mean()
        if len(clim) < 12 or clim.max() - clim.min() < 2.0:   # needs a seasonal cycle of at least 2 cm
            continue
        mx, mn = int(clim.idxmax()), int(clim.idxmin())
        if min((mx - mn) % 12, (mn - mx) % 12) < 3:
            continue
        windows[key] = {"wet-season gain": (mn, mx), "dry-season drawdown": (mx, mn)}
    J, G = scan_solution(jb, windows), scan_solution(gb, windows)
    m = J.merge(G, on=["lat", "lon", "kind", "from_month", "to_month", "season_end"], suffixes=("_jpl", "_gsfc"))
    m["excluded"] = ""
    for name, (S, N, W, E) in EXCLUDE:
        hit = (m.lat + BLOCK / 2 > S) & (m.lat - BLOCK / 2 < N) & (m.lon + BLOCK / 2 > W) & (m.lon - BLOCK / 2 < E)
        m.loc[hit, "excluded"] = name
    m["direction"] = np.where(m.z_jpl > 0, "+", "-")
    same = np.sign(m.z_jpl) == np.sign(m.z_gsfc)
    rec = np.where(m.z_jpl > 0, m.record_high_jpl & m.record_high_gsfc, m.record_low_jpl & m.record_low_gsfc)
    m["zmin"] = np.minimum(m.z_jpl.abs(), m.z_gsfc.abs())
    m["candidate"] = same & rec & (m.zmin >= Z_MIN) & (m.excluded == "")
    m.sort_values("zmin", ascending=False).to_csv(OUT / "scan_blocks.csv", index=False)
    c = m[m.candidate].sort_values("zmin", ascending=False)
    summary = {"jpl_file": jn, "gsfc_file": gn, "blocks_with_windows": len(windows), "block_seasons_tested": int(len(m)),
               "records_on_both": int((same & rec).sum()), "candidates": int(len(c)),
               # by chance alone (exchangeable years): a season is the highest or lowest of n+1 with probability
               # 2/(n+1); it is also >= 3 baseline sd out with the Student-t probability below (one solution; the two
               # solutions are strongly correlated, so "both" removes little of this)
               "expected_records_by_chance": round(float((2.0 / (m.n_base_jpl + 1)).sum()), 1),
               "expected_candidates_by_chance": round(float(sum(
                   2 * _st.t.sf(Z_MIN / np.sqrt(1 + 1 / n), n - 1) for n in m.n_base_jpl)), 1)}
    print(json.dumps(summary, indent=1))
    try:
        from synaptic.verify import place_name
        c = c.assign(place=[place_name(a, b) for a, b in zip(c.lat, c.lon)])
    except Exception:  # noqa: BLE001
        c = c.assign(place="")
    cols = ["lat", "lon", "place", "kind", "from_month", "to_month", "season_end", "direction", "z_jpl", "z_gsfc",
            "gain_jpl", "base_mean_jpl", "gain_gsfc", "base_mean_gsfc"]
    print(c[cols].round(2).to_string(index=False))
    (OUT / "candidates.json").write_text(json.dumps({"summary": summary, "blocks": json.loads(
        c[cols + ["base_sd_jpl", "base_sd_gsfc", "n_base_jpl"]].to_json(orient="records"))}, indent=1))


if __name__ == "__main__":
    main()
