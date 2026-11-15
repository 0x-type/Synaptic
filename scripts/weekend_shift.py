"""Did the air notice when a country moved its weekend? (docs/preregistrations/3_weekend_shift.md)

    python scripts/weekend_shift.py selftest   # synthetic data only: false alarms and power of the test
    python scripts/weekend_shift.py fetch      # OMI daily NO2 over the study window (cached in runs/cache/omi_daily)
    python scripts/weekend_shift.py noise      # blind check: valid days and day-to-day noise, pre-change data only
    python scripts/weekend_shift.py test       # the registered tests -> evidence/weekend_shift/

Data: OMI/Aura tropospheric NO2, cloud-screened, daily L3 0.25 degree (OMNO2d v4), NASA GES DISC.
Nothing here is used by the app."""
from __future__ import annotations

import datetime as dt
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
CACHE = ROOT / "runs" / "cache" / "omi_daily"
OUT = ROOT / "evidence" / "weekend_shift"
COLL = "C3333493715-GES_DISC"
VAR = "/HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2TropCloudScreened"
WIN = (20.0, 38.0, 2.0, 60.0)                       # S, N, W, E: covers every city below
LAT = -89.875 + 0.25 * np.arange(720)
LON = -179.875 + 0.25 * np.arange(1440)
I0, I1 = int(np.where(LAT >= WIN[0])[0].min()), int(np.where(LAT <= WIN[1])[0].max())
J0, J1 = int(np.where(LON >= WIN[2])[0].min()), int(np.where(LON <= WIN[3])[0].max())
HALF = 0.375                                        # city box: centre +- 0.375 degree (3 x 3 cells, about 80 km)
MON, TUE, WED, THU, FRI, SAT, SUN = range(7)

CITIES = {"Dubai": (25.20, 55.30), "Abu Dhabi": (24.45, 54.40), "Riyadh": (24.70, 46.70), "Jeddah": (21.50, 39.20),
          "Dammam": (26.40, 50.10), "Muscat": (23.60, 58.40), "Algiers": (36.75, 3.05),
          "Kuwait City": (29.37, 47.98), "Doha": (25.29, 51.53), "Cairo": (30.05, 31.25), "Tehran": (35.70, 51.40)}

# A test: on `date` the weekend moved. `up` = the day that became a working day (or half day), `down` = the day that
# became a rest day, `ref` = days that were working days before and after. Statistic: D = (R_up - R_down) after
# minus (R_up - R_down) before, with R_d = mean NO2 on weekday d / mean NO2 on the reference days. Predicted D > 0.
TESTS = {
    "UAE_2022": {"cities": ["Dubai", "Abu Dhabi"], "date": "2022-01-01", "before": "2017-01-01", "after": "2025-12-31",
                 "up": FRI, "down": SUN, "ref": [MON, TUE, WED, THU],
                 "what": "UAE: Friday-Saturday weekend became Saturday-Sunday (Friday a half day), 1 Jan 2022"},
    "SAUDI_2013": {"cities": ["Riyadh", "Jeddah", "Dammam"], "date": "2013-06-29", "before": "2005-01-01",
                   "after": "2021-12-31", "up": THU, "down": SAT, "ref": [SUN, MON, TUE, WED],
                   "what": "Saudi Arabia: Thursday-Friday weekend became Friday-Saturday, 29 Jun 2013"},
    "OMAN_2013": {"cities": ["Muscat"], "date": "2013-05-01", "before": "2005-01-01", "after": "2021-12-31",
                  "up": THU, "down": SAT, "ref": [SUN, MON, TUE, WED],
                  "what": "Oman: Thursday-Friday weekend became Friday-Saturday, 1 May 2013"},
    "ALGERIA_2009": {"cities": ["Algiers"], "date": "2009-08-14", "before": "2005-01-01", "after": "2014-12-31",
                     "up": THU, "down": SAT, "ref": [SUN, MON, TUE, WED],
                     "what": "Algeria: Thursday-Friday weekend became Friday-Saturday, 14 Aug 2009"},
}
# Placebos: the same test, same dates and days, in cities whose weekend did NOT change then.
PLACEBO_CITIES = ["Kuwait City", "Doha", "Cairo", "Tehran"]
N_PERM = 20000


# ----------------------------------------------------------------------------- the registered statistic and test
def ratios(d: pd.DataFrame, up: int, down: int, ref: list[int]) -> float:
    """R_up - R_down for one city and one period; d has columns dow, value (valid days only)."""
    r = d[d.dow.isin(ref)].value.mean()
    return float((d[d.dow == up].value.mean() - d[d.dow == down].value.mean()) / r)


def shift_test(series: dict, spec: dict, n_perm: int = N_PERM, seed: int = 1) -> dict:
    """series: {city: DataFrame(date, value)} daily valid values. Weeks are the unit: each 7-day block (counted from
    the change date) is wholly before or after. D = mean over cities of [(R_up - R_down)_after - (..)_before].
    Null: the weekly cycle did not change, so before/after labels of weeks are exchangeable; the labels are permuted
    (the same permutation for all cities, keeping the numbers of before and after weeks). Two-sided p."""
    t0 = pd.Timestamp(spec["date"])
    lo, hi = pd.Timestamp(spec["before"]), pd.Timestamp(spec["after"])
    per_city = []
    for c in spec["cities"]:
        d = series[c]
        d = d[(d.date >= lo) & (d.date <= hi)].copy()
        d["dow"] = d.date.dt.dayofweek
        d["week"] = np.floor((d.date - t0).dt.days / 7.0).astype(int)
        per_city.append(d[d.dow.isin([spec["up"], spec["down"]] + list(spec["ref"]))])
    weeks = np.array(sorted(set().union(*[set(d.week) for d in per_city])))
    after = weeks >= 0
    # per city: sums and counts per week for up / down / ref days
    packs = []
    for d in per_city:
        def sc(mask):
            g = d[mask].groupby("week").value.agg(["sum", "count"]).reindex(weeks).fillna(0.0)
            return g["sum"].values, g["count"].values
        packs.append((sc(d.dow == spec["up"]), sc(d.dow == spec["down"]), sc(d.dow.isin(spec["ref"]))))

    def stat(a):
        out = []
        for (us, un), (ds_, dn), (rs, rn) in packs:
            v = []
            for m in (a, ~a):
                ref = rs[m].sum() / max(rn[m].sum(), 1)
                v.append((us[m].sum() / max(un[m].sum(), 1) - ds_[m].sum() / max(dn[m].sum(), 1)) / ref)
            out.append(v[0] - v[1])
        return float(np.mean(out))
    D = stat(after)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = stat(rng.permutation(after))
    p = float((np.sum(np.abs(null) >= abs(D)) + 1) / (n_perm + 1))
    detail = {}
    for c, d in zip(spec["cities"], per_city):
        b, a = d[d.week < 0], d[d.week >= 0]
        rb, ra = b[b.dow.isin(spec["ref"])].value.mean(), a[a.dow.isin(spec["ref"])].value.mean()
        detail[c] = {"R_up_before": float(b[b.dow == spec["up"]].value.mean() / rb),
                     "R_up_after": float(a[a.dow == spec["up"]].value.mean() / ra),
                     "R_down_before": float(b[b.dow == spec["down"]].value.mean() / rb),
                     "R_down_after": float(a[a.dow == spec["down"]].value.mean() / ra),
                     "valid_days_before": int(len(b)), "valid_days_after": int(len(a))}
    return {"D": D, "p": p, "null_sd": float(null.std()), "z": float(D / null.std()) if null.std() > 0 else 0.0,
            "weeks_before": int((~after).sum()), "weeks_after": int(after.sum()), "cities": detail}


# ----------------------------------------------------------------------------- synthetic self-test
def synth_city(rng, start, end, change, shift, noise=0.45, valid=0.45, drop=0.2):
    days = pd.date_range(start, end, freq="D")
    dow = days.dayofweek.values
    base = 5.0 * (1 + 0.3 * np.sin(2 * np.pi * days.dayofyear.values / 365.25)) * (1 + 0.02 * (days.year.values - 2010))
    rest_before, rest_after = (THU, FRI), ((FRI, SAT) if shift else (THU, FRI))
    f = np.ones(len(days))
    aft = days >= pd.Timestamp(change)
    for r in rest_before:
        f[(dow == r) & ~aft] -= drop
    for r in rest_after:
        f[(dow == r) & aft] -= drop
    v = base * f * (1 + rng.normal(0, noise, len(days)))
    keep = rng.random(len(days)) < valid
    return pd.DataFrame({"date": days[keep], "value": v[keep]})


def cmd_selftest():
    rng = np.random.default_rng(7)
    spec = {**TESTS["SAUDI_2013"], "cities": ["A", "B"]}
    out = {}
    for label, shift, drop in (("no change", False, 0.2), ("weekend moved, 20% rest-day drop", True, 0.2),
                               ("weekend moved, 10% rest-day drop", True, 0.1)):
        hits, n = 0, 200
        for k in range(n):
            s = {c: synth_city(rng, spec["before"], spec["after"], spec["date"], shift, drop=drop) for c in "AB"}
            r = shift_test(s, spec, n_perm=400, seed=k)
            hits += r["p"] < 0.0125 and (r["D"] > 0 or not shift)
        out[label] = hits / n
        print(f"{label}: p < 0.0125 in {hits / n:.3f} of runs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selftest.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- OMI daily
def granules(year, month):
    import calendar

    f = CACHE / "granules" / f"{year}-{month:02d}.json"
    if f.exists():
        return json.loads(f.read_text())
    last = calendar.monthrange(year, month)[1]
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", params={
        "collection_concept_id": COLL, "page_size": 200, "sort_key": "start_date",
        "temporal": f"{year}-{month:02d}-01T00:00:00Z,{year}-{month:02d}-{last}T23:59:59Z"}, timeout=90)
    r.raise_for_status()
    out = {}
    for e in r.json()["feed"]["entry"]:
        d = e.get("time_start", "")[:10]
        od = [l["href"] for l in e["links"] if "opendap.earthdata" in l["href"]]
        if od and d.startswith(f"{year}-{month:02d}"):
            out[d] = od[0]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


_nc = threading.Lock()


def fetch_day(day, url):
    import netCDF4

    f = CACHE / "daily" / f"{day}.npy"
    if f.exists():
        return True
    ce = f"{VAR}[{I0}:1:{I1}][{J0}:1:{J1}]"
    for _ in range(4):
        try:
            r = requests.get(url + ".dap.nc4", params={"dap4.ce": ce},
                             headers={"Authorization": "Bearer " + os.environ["EARTHDATA_TOKEN"].strip()}, timeout=240)
            if r.status_code != 200:
                continue
            with _nc:
                ds = netCDF4.Dataset("m", memory=r.content)
                ds.set_auto_maskandscale(False)
                g = ds
                for part in VAR.strip("/").split("/")[:-1]:
                    g = g.groups[part]
                a = np.array(g.variables[VAR.split("/")[-1]][:], dtype="float64")
                ds.close()
            a[np.abs(a) > 1e29] = np.nan
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, (a * 1e-15).astype("float32"))
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def cmd_fetch():
    jobs = []
    for y in range(2004, 2027):
        for m in range(1, 13):
            if (y, m) < (2004, 10) or (y, m) > (2026, 10):
                continue
            try:
                jobs += sorted(granules(y, m).items())
            except Exception as e:  # noqa: BLE001
                print("granule list failed", y, m, type(e).__name__, flush=True)
    print(len(jobs), "daily granules", flush=True)
    ok = bad = 0
    with ThreadPoolExecutor(16) as ex:
        for k, r in enumerate(ex.map(lambda j: fetch_day(*j), jobs)):
            ok += r
            bad += (not r)
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(jobs)} ok={ok} failed={bad}", flush=True)
    print(f"FETCH DONE ok={ok} failed={bad}", flush=True)


def city_series() -> dict:
    """{city: DataFrame(date, value)}: mean of the valid cells of the 3 x 3 box, days with at least 3 valid cells.
    Units 10^15 molecules/cm2. Negative retrievals are kept (noise around zero must not be truncated)."""
    f = OUT / "city_daily.csv"
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"])
        return {c: g[["date", "value"]].reset_index(drop=True) for c, g in d.groupby("city")}
    lat, lon = LAT[I0:I1 + 1], LON[J0:J1 + 1]
    idx = {c: (np.where(np.abs(lat - la) <= HALF)[0], np.where(np.abs(lon - lo) <= HALF)[0])
           for c, (la, lo) in CITIES.items()}
    rows = []
    for p in sorted((CACHE / "daily").glob("*.npy")):
        a = np.load(p)
        for c, (ii, jj) in idx.items():
            sub = a[np.ix_(ii, jj)]
            ok = np.isfinite(sub)
            if ok.sum() >= 3:
                rows.append((c, p.stem, float(sub[ok].mean()), int(ok.sum())))
    d = pd.DataFrame(rows, columns=["city", "date", "value", "cells"])
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(f, index=False)
    d["date"] = pd.to_datetime(d.date)
    return {c: g[["date", "value"]].reset_index(drop=True) for c, g in d.groupby("city")}


def cmd_noise():
    """Blind check, registered: uses ONLY days before each test's change date and ONLY the reference weekdays, so it
    cannot reveal whether the weekly cycle moved. Reports valid days per weekday and the day-to-day spread."""
    s = city_series()
    out = {}
    for name, spec in TESTS.items():
        for c in spec["cities"]:
            d = s[c]
            d = d[(d.date >= spec["before"]) & (d.date < spec["date"])]
            d = d[d.date.dt.dayofweek.isin(spec["ref"])]
            n_wd = len(d) / len(spec["ref"])
            cv = float(d.value.std() / d.value.mean())
            out[f"{name}:{c}"] = {"valid_ref_days_before": int(len(d)), "per_weekday": round(n_wd, 1),
                                  "mean": round(float(d.value.mean()), 2), "cv": round(cv, 2),
                                  "approx_se_of_one_ratio": round(cv / np.sqrt(max(n_wd, 1)), 3)}
            print(name, c, out[f"{name}:{c}"], flush=True)
    (OUT / "noise_check.json").write_text(json.dumps(out, indent=1))


def cmd_test():
    s = city_series()
    res = {"tests": {}, "placebos": {}}
    for name, spec in TESTS.items():
        r = shift_test(s, spec)
        res["tests"][name] = {"what": spec["what"], **r}
        print(name, json.dumps({k: v for k, v in r.items() if k != "cities"}), flush=True)
        for c in PLACEBO_CITIES:
            if name == "ALGERIA_2009":
                continue  # placebos are Gulf-region and Fri-Sat or Thu-Fri cities; used for the three Gulf tests
            pr = shift_test(s, {**spec, "cities": [c]}, n_perm=5000)
            res["placebos"][f"{name}:{c}"] = {k: v for k, v in pr.items() if k != "cities"}
    # Holm over the four primary tests
    order = sorted(res["tests"], key=lambda k: res["tests"][k]["p"])
    worst = 0.0
    for i, k in enumerate(order):
        t = res["tests"][k]
        adj = max(worst, min(1.0, (len(order) - i) * t["p"]))
        worst = adj
        t["holm_p"] = adj
        t["verdict"] = ("confirmed" if t["D"] > 0 and adj < 0.05 else "supported, not confirmed"
                        if t["D"] > 0 and t["p"] < 0.05 else "not shown")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    for k, t in res["tests"].items():
        print(k, t["verdict"], "D", round(t["D"], 3), "p", t["p"], "holm", t["holm_p"])
    pl = res["placebos"]
    print("placebos with p < 0.05:", sum(v["p"] < 0.05 for v in pl.values()), "of", len(pl))


if __name__ == "__main__":
    {"selftest": cmd_selftest, "fetch": cmd_fetch, "noise": cmd_noise, "test": cmd_test}[sys.argv[1]]()
