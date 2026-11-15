"""Pre-registered SMAP confirmation (see docs/preregistrations/1_water_gains_smap.md). Usage:
    python scripts/smap_confirm.py fetch     # download daily box subsets (cached in runs/cache/smap)
    python scripts/smap_confirm.py test      # run the three registered tests, write evidence/smap_confirmation/
Needs EARTHDATA_TOKEN. Nothing here reads GRACE."""
from __future__ import annotations

import calendar
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "runs" / "cache" / "smap"
OUT = ROOT / "evidence" / "smap_confirmation"
COLL = "C2938664585-NSIDC_CPRD"  # SPL3SMP v009
NROW, NCOL = 406, 964            # EASE-Grid 2.0 global 36 km

LEADS = {
    "A_lake_chad": {"box": [10, 17, 12, 18], "months": [8, 9, 10], "years": list(range(2015, 2027)), "event": 2024},
    "B_central_sahara": {"box": [15, 23, 10, 25], "months": [8, 9, 10], "years": list(range(2015, 2027)),
                         "event": 2024},
    "C_north_morocco": {"box": [31, 36, -10, -1], "months": [12, 1, 2, 3, 4], "years": list(range(2016, 2027)),
                        "event": 2026},
}
# one download window per season group (union of the boxes, padded)
WINDOWS = {"sahel": {"box": [9, 24, 9, 26], "months": [8, 9, 10], "years": list(range(2015, 2027))},
           "morocco": {"box": [30, 37, -11, 0], "months": [12, 1, 2, 3, 4], "years": list(range(2016, 2027))}}


def ease2_lat():
    """Row-centre latitudes of the 36 km global EASE-Grid 2.0 (cylindrical equal area, WGS84, std parallel 30)."""
    a, e = 6378137.0, 0.0818191908426
    k0 = np.cos(np.radians(30)) / np.sqrt(1 - (e * np.sin(np.radians(30))) ** 2)
    qp = (1 - e * e) * (1 / (1 - e * e) - np.log((1 - e) / (1 + e)) / (2 * e))
    y = (NROW / 2 - 0.5 - np.arange(NROW)) * 36032.220840584
    beta = np.arcsin(np.clip(2 * y * k0 / (a * qp), -1, 1))
    e2, e4, e6 = e ** 2, e ** 4, e ** 6
    phi = (beta + (e2 / 3 + 31 * e4 / 180 + 517 * e6 / 5040) * np.sin(2 * beta)
           + (23 * e4 / 360 + 251 * e6 / 3780) * np.sin(4 * beta) + (761 * e6 / 45360) * np.sin(6 * beta))
    return np.degrees(phi)


LAT = ease2_lat()
LON = -180 + (np.arange(NCOL) + 0.5) * 360.0 / NCOL


def idx(box):
    S, N, W, E = box
    r = np.where((LAT >= S) & (LAT <= N))[0]
    c = np.where((LON >= W) & (LON <= E))[0]
    return int(r.min()), int(r.max()), int(c.min()), int(c.max())


def token():
    return os.environ["EARTHDATA_TOKEN"].strip()


def granules(year, month):
    f = CACHE / "granules" / f"{year}-{month:02d}.json"
    if f.exists():
        return json.loads(f.read_text())
    last = calendar.monthrange(year, month)[1]
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", params={
        "collection_concept_id": COLL, "temporal": f"{year}-{month:02d}-01T00:00:00Z,{year}-{month:02d}-{last}T23:59:59Z",
        "page_size": 100}, timeout=90)
    r.raise_for_status()
    out = {}
    for e in r.json()["feed"]["entry"]:
        day = e["title"].split("_")[4]
        if not day.startswith(f"{year}{month:02d}"):
            continue
        od = [l["href"] for l in e["links"] if "opendap.earthdata" in l["href"]]
        if od:
            out[day] = od[0]  # a later reprocessing of the same day replaces an earlier one
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


_nc = threading.Lock()


def fetch_day(win, day, url):
    import netCDF4

    f = CACHE / win / f"{day}.npz"
    if f.exists():
        return True
    r0, r1, c0, c1 = idx(WINDOWS[win]["box"])
    g = "/Soil_Moisture_Retrieval_Data_AM/"
    ce = ";".join(f"{g}{v}[{r0}:1:{r1}][{c0}:1:{c1}]" for v in ("soil_moisture", "retrieval_qual_flag"))
    for attempt in range(4):
        try:
            r = requests.get(url + ".dap.nc4", params={"dap4.ce": ce}, headers={"Authorization": "Bearer " + token()},
                             timeout=180)
            if r.status_code != 200:
                continue
            with _nc:
                ds = netCDF4.Dataset("m", memory=r.content)
                ds.set_auto_maskandscale(False)
                grp = ds["Soil_Moisture_Retrieval_Data_AM"]
                sm = np.array(grp["soil_moisture"][:], dtype="float32")
                q = np.array(grp["retrieval_qual_flag"][:], dtype="int32")
                ds.close()
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, sm=sm, q=q)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def cmd_fetch():
    jobs = []
    only = sys.argv[2] if len(sys.argv) > 2 else None   # optional: fetch one window (parallel downloaders)
    for win, w in WINDOWS.items():
        if only and win != only:
            continue
        for y in w["years"]:
            for m in w["months"]:
                yy = y - 1 if (m == 12 and 1 in w["months"]) else y
                try:
                    for day, url in granules(yy, m).items():
                        jobs.append((win, day, url))
                except Exception as e:  # noqa: BLE001
                    print("granule list failed", yy, m, type(e).__name__, flush=True)
    print(len(jobs), "daily granules", flush=True)
    done = fail = 0
    with ThreadPoolExecutor(12) as ex:
        for k, ok in enumerate(ex.map(lambda j: fetch_day(*j), jobs)):
            done += ok
            fail += (not ok)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(jobs)} fetched ok={done} failed={fail}", flush=True)
    print(f"FETCH DONE ok={done} failed={fail}", flush=True)


def daily_series(lead, quality_only=False):
    L = LEADS[lead]
    win = "sahel" if 8 in L["months"] else "morocco"
    r0, r1, c0, c1 = idx(WINDOWS[win]["box"])
    lat, lon = LAT[r0:r1 + 1], LON[c0:c1 + 1]
    S, N, W, E = L["box"]
    m = ((lat >= S) & (lat <= N))[:, None] & ((lon >= W) & (lon <= E))[None, :]
    rows = []
    for f in sorted((CACHE / win).glob("*.npz")):
        d = np.load(f)
        sm = d["sm"].astype(float)
        ok = (sm > 0) & (sm <= 0.6) & m
        if quality_only:
            ok &= (d["q"] & 1) == 0
        if ok.sum() >= 0.3 * m.sum():
            rows.append({"date": f.stem, "value": float(sm[ok].mean()), "cover": float(ok.sum() / m.sum())})
    df = pd.DataFrame(rows)
    df["year"] = df.date.str[:4].astype(int)
    df["month"] = df.date.str[4:6].astype(int)
    return df


def season_means(lead, quality_only=False):
    L = LEADS[lead]
    d = daily_series(lead, quality_only)
    mo = d.groupby(["year", "month"]).value.agg(["mean", "count"]).reset_index()
    mo = mo[mo["count"] >= 8]
    wraps = 12 in L["months"] and 1 in L["months"]
    mo["season"] = np.where(wraps & (mo.month == 12), mo.year + 1, mo.year)
    mo = mo[mo.month.isin(L["months"])]
    g = mo.groupby("season")["mean"].agg(["mean", "count"]).reset_index()
    g = g[(g["count"] == len(L["months"])) & g.season.isin(L["years"])]
    return g.rename(columns={"season": "year", "mean": "value"})[["year", "value"]], mo


def cmd_test():
    import synlib

    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for lead, L in LEADS.items():
        out = {"box": L["box"], "months": L["months"], "event_year": L["event"]}
        for label, q in (("primary", False), ("quality_only", True)):
            s, mo = season_means(lead, q)
            r = synlib.event_departure(s.value.values, s.year.values, [L["event"]])
            if label == "primary":
                s.to_csv(OUT / f"{lead}_season_means.csv", index=False)
                mo.to_csv(OUT / f"{lead}_monthly.csv", index=False)
            if not r.get("valid"):
                out[label] = {"verdict": "could not be tested", "n_seasons": int(len(s))}
                continue
            p, dep = float(r["p"]), float(r["departure"])
            verdict = ("confirmed" if dep > 0 and p < 0.05 / 3 else "supported, not confirmed" if dep > 0 and p < 0.05
                       else "not shown")
            rank = int((s.value > s.value[s.year == L["event"]].iloc[0]).sum()) + 1
            out[label] = {"n_seasons": int(len(s)), "event_value": float(s.value[s.year == L["event"]].iloc[0]),
                          "mean_other_years": float(s.value[s.year != L["event"]].mean()),
                          "departure_from_trend": dep, "z": float(r["z"]), "p": p, "carryover": float(r["carryover"]),
                          "rank_of_event_year": rank, "verdict": verdict}
        res[lead] = out
        print(lead, json.dumps(out, indent=1))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "test": cmd_test}[sys.argv[1]]()
