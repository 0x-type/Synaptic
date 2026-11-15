"""Confirmation of the record-scan candidates (docs/preregistrations/2_record_scan.md).

    python scripts/record_confirm.py calibrate   # synthetic data only: false-alarm rate and power of the test
    python scripts/record_confirm.py fetch       # SMAP daily global soil moisture (cached in runs/cache/smap_global)
    python scripts/record_confirm.py test        # the registered tests -> evidence/record_scan/

The test (`monthly_event_test`) uses MONTHLY values instead of one value per year: about 130 months of SMAP instead of
11 seasons, with the month-to-month carry-over modelled (AR(1)). That is what gives it the power the season-mean
test of docs/preregistrations/1_water_gains_smap.md lacked."""
from __future__ import annotations

import calendar
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "runs" / "cache" / "smap_global"
OUT = ROOT / "evidence" / "record_scan"
COLL = "C2938664585-NSIDC_CPRD"  # SPL3SMP v009
NROW, NCOL = 406, 964


# ----------------------------------------------------------------------------- the registered test
def monthly_event_test(df: pd.DataFrame, event: list[tuple[int, int]]) -> dict:
    """df: columns year, month, value (one row per valid month). event: [(year, month), ...].
    1. anomalies = value minus the calendar-month mean of the NON-event months;
    2. a linear trend fitted on the non-event anomalies is removed;
    3. carry-over r = lag-1 autocorrelation of consecutive non-event residuals (0 <= r <= 0.9);
    4. D = mean residual of the event months; its variance combines the AR(1) covariance of the k event months,
       the uncertainty of the trend line and of the calendar-month means; t-test with the effective number of
       non-event months, n_eff = n (1 - r) / (1 + r), minus 2 degrees of freedom."""
    d = df.dropna(subset=["value"]).copy()
    d["t"] = d.year + (d.month - 0.5) / 12.0
    ev = set(event)
    d["ev"] = [(int(y), int(m)) in ev for y, m in zip(d.year, d.month)]
    base, e = d[~d.ev], d[d.ev]
    if len(base) < 60 or len(e) < max(2, math.ceil(0.6 * len(event))):
        return {"valid": False, "n_base": int(len(base)), "n_event": int(len(e))}
    clim = base.groupby("month").value.mean()
    ncal = base.groupby("month").value.count()
    if not set(e.month) <= set(clim.index) or (ncal.reindex(sorted(set(e.month))) < 5).any():
        return {"valid": False, "n_base": int(len(base)), "n_event": int(len(e)), "note": "no climatology"}
    ab = base.value.values - clim.reindex(base.month).values
    tb = base.t.values
    A = np.vstack([np.ones(len(tb)), tb - tb.mean()]).T
    coef, *_ = np.linalg.lstsq(A, ab, rcond=None)
    res = ab - A @ coef
    idx = (base.year * 12 + base.month).values
    pairs = np.where(np.diff(idx) == 1)[0]
    r = float(np.corrcoef(res[pairs], res[pairs + 1])[0, 1]) if len(pairs) > 10 else 0.0
    r = min(0.9, max(0.0, r if np.isfinite(r) else 0.0))
    sd = float(np.std(res, ddof=2 + 12))
    ae = e.value.values - clim.reindex(e.month).values - (coef[0] + coef[1] * (e.t.values - tb.mean()))
    D = float(ae.mean())
    k = len(ae)
    ie = (e.year * 12 + e.month).values
    v_ev = float((r ** np.abs(np.subtract.outer(ie, ie))).sum()) / k ** 2
    infl = (1 + r) / (1 - r)
    xr = np.array([1.0, e.t.values.mean() - tb.mean()])
    v_line = float(xr @ np.linalg.inv(A.T @ A) @ xr) * infl
    v_clim = float(np.mean(1.0 / ncal.reindex(e.month).values)) * infl / k
    se = sd * math.sqrt(v_ev + v_line + v_clim)
    neff = max(len(res) / infl, 6.0)
    z = D / se if se > 0 else 0.0
    return {"valid": True, "departure": D, "se": se, "t": float(z), "p": float(2 * stats.t.sf(abs(z), neff - 2)),
            "carryover": r, "sd_monthly": sd, "n_base": int(len(base)), "n_event": int(k), "n_eff": float(neff)}


# ----------------------------------------------------------------------------- calibration on synthetic data
def synth(rng, r, n_years=12, shift=0.0, k=5, gaps=0.08):
    rows, e = [], 0.0
    innov = math.sqrt(1 - r * r)
    start = (n_years - 2) * 12 + int(rng.integers(0, 12))     # an event near the end of the record, as in the scan
    event = []
    for i in range(n_years * 12):
        y, m = 2015 + i // 12, i % 12 + 1
        e = r * e + rng.normal(0, innov)
        v = 0.2 + 0.05 * math.sin(2 * math.pi * m / 12) + 0.02 * e + 0.0003 * i
        if start <= i < start + k:
            event.append((y, m))
            v += 0.02 * shift
        if rng.random() > gaps:
            rows.append({"year": y, "month": m, "value": v})
    return pd.DataFrame(rows), event


def cmd_calibrate():
    rng = np.random.default_rng(20261114)
    out = {}
    for r in (0.0, 0.3, 0.6, 0.8):
        for k in (4, 6, 8):
            fp, pw15, pw25, n = 0, 0, 0, 1500
            for _ in range(n):
                d, ev = synth(rng, r, k=k)
                t = monthly_event_test(d, ev)
                fp += t["valid"] and t["p"] < 0.05
                d, ev = synth(rng, r, k=k, shift=1.5)
                t = monthly_event_test(d, ev)
                pw15 += t["valid"] and t["p"] < 0.01 and t["departure"] > 0
                d, ev = synth(rng, r, k=k, shift=2.5)
                t = monthly_event_test(d, ev)
                pw25 += t["valid"] and t["p"] < 0.01 and t["departure"] > 0
            out[f"carryover_{r}_months_{k}"] = {"false_alarm_at_0.05": round(fp / n, 3),
                                                "power_p<0.01_shift_1.5sd": round(pw15 / n, 2),
                                                "power_p<0.01_shift_2.5sd": round(pw25 / n, 2)}
            print(f"carry-over {r}, {k} event months: false alarms {fp / n:.3f} | power at p<0.01: "
                  f"1.5 sd shift {pw15 / n:.2f}, 2.5 sd shift {pw25 / n:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- SMAP global daily
def ease2_lat():
    a, e = 6378137.0, 0.0818191908426
    k0 = np.cos(np.radians(30)) / np.sqrt(1 - (e * np.sin(np.radians(30))) ** 2)
    qp = (1 - e * e) * (1 / (1 - e * e) - np.log((1 - e) / (1 + e)) / (2 * e))
    y = (NROW / 2 - 0.5 - np.arange(NROW)) * 36032.220840584
    beta = np.arcsin(np.clip(2 * y * k0 / (a * qp), -1, 1))
    e2, e4, e6 = e ** 2, e ** 4, e ** 6
    return np.degrees(beta + (e2 / 3 + 31 * e4 / 180 + 517 * e6 / 5040) * np.sin(2 * beta)
                      + (23 * e4 / 360 + 251 * e6 / 3780) * np.sin(4 * beta) + (761 * e6 / 45360) * np.sin(6 * beta))


LAT = ease2_lat()
LON = -180 + (np.arange(NCOL) + 0.5) * 360.0 / NCOL
R0, R1 = int(np.where(LAT <= 52)[0].min()), int(np.where(LAT >= -47)[0].max())   # rows covering 47S-52N


def granules(year, month):
    f = CACHE / "granules" / f"{year}-{month:02d}.json"
    if f.exists():
        return json.loads(f.read_text())
    last = calendar.monthrange(year, month)[1]
    r = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", params={
        "collection_concept_id": COLL, "page_size": 100,
        "temporal": f"{year}-{month:02d}-01T00:00:00Z,{year}-{month:02d}-{last}T23:59:59Z"}, timeout=90)
    r.raise_for_status()
    out = {}
    for e in r.json()["feed"]["entry"]:
        day = e["title"].split("_")[4]
        od = [l["href"] for l in e["links"] if "opendap.earthdata" in l["href"]]
        if day.startswith(f"{year}{month:02d}") and od:
            out[day] = od[0]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


_nc = threading.Lock()


def fetch_day(day, url):
    import netCDF4

    f = CACHE / "daily" / f"{day}.npz"
    if f.exists():
        return True
    ce = f"/Soil_Moisture_Retrieval_Data_AM/soil_moisture[{R0}:1:{R1}][0:1:{NCOL - 1}]"
    for _ in range(4):
        try:
            r = requests.get(url + ".dap.nc4", params={"dap4.ce": ce},
                             headers={"Authorization": "Bearer " + os.environ["EARTHDATA_TOKEN"].strip()}, timeout=240)
            if r.status_code != 200:
                continue
            with _nc:
                ds = netCDF4.Dataset("m", memory=r.content)
                ds.set_auto_maskandscale(False)
                sm = np.array(ds["Soil_Moisture_Retrieval_Data_AM"]["soil_moisture"][:], dtype="float32")
                ds.close()
            sm[(sm <= 0) | (sm > 0.6)] = np.nan
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".part.npz")
            np.savez_compressed(tmp, sm=(sm * 10000).round().astype("int16", copy=False) if False else sm)
            tmp.rename(f)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def cmd_fetch():
    jobs = []
    for y in range(2015, 2027):
        for m in range(1, 13):
            if (y, m) < (2015, 4) or (y, m) > (2026, 10):
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
            if (k + 1) % 300 == 0:
                print(f"{k + 1}/{len(jobs)} ok={ok} failed={bad}", flush=True)
    print(f"FETCH DONE ok={ok} failed={bad}", flush=True)


def box_monthly(box) -> pd.DataFrame:
    """Monthly SMAP box mean: a day counts if >= 30% of the box's cells have a valid retrieval; a month needs 8
    valid days. Cells = 36 km cells whose centre is inside the box."""
    S, N, W, E = box
    lat = LAT[R0:R1 + 1]
    m = ((lat >= S) & (lat <= N))[:, None] & ((LON >= W) & (LON <= E))[None, :]
    rr, cc = np.where(m.any(1))[0], np.where(m.any(0))[0]
    sub = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    rows = []
    for f in sorted((CACHE / "daily").glob("*.npz")):
        sm = np.load(f)["sm"][rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        ok = np.isfinite(sm) & sub
        if ok.sum() >= 0.3 * sub.sum():
            rows.append((int(f.stem[:4]), int(f.stem[4:6]), float(sm[ok].mean())))
    d = pd.DataFrame(rows, columns=["year", "month", "value"])
    g = d.groupby(["year", "month"]).value.agg(["mean", "count"]).reset_index()
    return g[g["count"] >= 8].rename(columns={"mean": "value", "count": "days"})


def cmd_test():
    import synlib
    from synaptic import cloud
    from synaptic.regions import land_mask

    reg = json.loads((OUT / "registered_candidates.json").read_text())
    res = []
    for c in reg["candidates"]:
        ev = [tuple(x) for x in c["event_months"]]
        mo = box_monthly(c["box"])
        mo.to_csv(OUT / f"{c['id']}_smap_monthly.csv", index=False)
        t = monthly_event_test(mo, ev)
        sign = 1 if c["direction"] == "+" else -1
        out = {"id": c["id"], "name": c["name"], "box": c["box"], "direction": c["direction"],
               "event_months": c["event_months"], "smap": t}
        # placebo: the same window shifted to every other year (calibration on real data; not a test of the claim)
        pl = []
        for dy in range(-12, 3):
            if dy == 0:
                continue
            ev2 = [(y + dy, m) for y, m in ev]
            if min(ev2) < (2015, 4) or max(ev2) > (2026, 10):
                continue
            t2 = monthly_event_test(mo, ev2)
            if t2.get("valid"):
                pl.append({"shift_years": dy, "p": t2["p"], "t": t2["t"]})
        out["placebo"] = pl
        # secondary: precipitation over the same months, MERRA-2 (gauge-corrected over land), 1981-2026
        try:
            months = sorted({m for _, m in ev})
            p = cloud.regional_series("MERRA2_PRECIP", tuple(c["box"]), 1981, 2026, mask=land_mask)
            p["season"] = np.where((p.month >= ev[0][1]) if ev[0][0] != ev[-1][0] else False, p.year + 1, p.year)
            p = p[p.month.isin(months)]
            g = p.groupby("season").value.agg(["mean", "count"]).reset_index()
            g = g[g["count"] == len(months)]
            g.rename(columns={"season": "year", "mean": "value"})[["year", "value"]].to_csv(
                OUT / f"{c['id']}_rain_seasons.csv", index=False)
            r = synlib.event_departure(g["mean"].values, g.season.values, [ev[-1][0]])
            evv = float(g["mean"][g.season == ev[-1][0]].iloc[0])
            out["rain"] = {"valid": bool(r.get("valid")), "departure_mm_day": float(r.get("departure", np.nan)),
                           "p": float(r.get("p", np.nan)), "rank_wettest": int((g["mean"] > evv).sum()) + 1,
                           "n_seasons": int(len(g))}
        except Exception as e:  # noqa: BLE001
            out["rain"] = {"valid": False, "note": f"{type(e).__name__}: {str(e)[:120]}"}
        res.append(out)
        print(json.dumps({k: v for k, v in out.items() if k != "placebo"}, default=str)[:600], flush=True)
    # verdicts: Holm over the SMAP tests that could run
    tested = sorted([r for r in res if r["smap"].get("valid")], key=lambda r: r["smap"]["p"])
    K = len(tested)
    worst = 0.0
    for i, r in enumerate(tested):
        sign = 1 if r["direction"] == "+" else -1
        adj = max(worst, min(1.0, (K - i) * r["smap"]["p"]))
        worst = adj
        ok_dir = sign * r["smap"]["departure"] > 0
        r["holm_p"] = adj
        r["verdict"] = ("confirmed" if ok_dir and adj < 0.05 else "supported, not confirmed"
                        if ok_dir and r["smap"]["p"] < 0.05 else "not shown")
    for r in res:
        if not r["smap"].get("valid"):
            r["verdict"] = "could not be tested"
    # pooled test: one z per independent event (candidates sharing an event group are averaged), Stouffer, one-sided
    groups = {}
    for c, r in zip(reg["candidates"], res):
        if r["smap"].get("valid"):
            sign = 1 if r["direction"] == "+" else -1
            p1 = stats.t.sf(sign * r["smap"]["t"], r["smap"]["n_eff"] - 2)
            groups.setdefault(c["event_group"], []).append(float(stats.norm.isf(min(max(p1, 1e-12), 1 - 1e-12))))
    zs = [float(np.mean(v)) for v in groups.values()]
    pooled = {"n_events": len(zs), "z": float(np.sum(zs) / math.sqrt(len(zs))) if zs else None}
    pooled["p_one_sided"] = float(stats.norm.sf(pooled["z"])) if zs else None
    (OUT / "results.json").write_text(json.dumps({"candidates": res, "pooled": pooled}, indent=1, default=str))
    for r in res:
        print(r["id"], r["verdict"], r["smap"].get("p"), r.get("holm_p"), "rain:", r["rain"])
    print("pooled:", pooled)


if __name__ == "__main__":
    {"calibrate": cmd_calibrate, "fetch": cmd_fetch, "test": cmd_test}[sys.argv[1]]()
