"""Descriptive estimates registered in docs/preregistrations/1_water_gains_smap.md: seasonal storage gains of the three leads as volumes of
water, on both GRACE solutions, and (leads A, B) the share still present in April-May 2025.
Writes evidence/water_gains/. Usage: python scripts/measure_gains.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synaptic import confirm  # noqa: E402
from synaptic.regions import land_mask  # noqa: E402

OUT = ROOT / "evidence" / "water_gains"
LEADS = {
    "A_lake_chad": {"name": "Lake Chad basin", "box": [10, 17, 12, 18], "frm": [6, 7], "to": [9, 10], "event": 2024,
                    "after": [4, 5]},
    "B_central_sahara": {"name": "Central Sahara", "box": [15, 23, 10, 25], "frm": [6, 7], "to": [9, 10],
                         "event": 2024, "after": [4, 5]},
    "C_north_morocco": {"name": "Northern Morocco", "box": [31, 36, -10, -1], "frm": [9, 10], "to": [3, 4],
                        "event": 2026, "after": None},
}
SOLUTIONS = {"JPL": "GRACE_TWS", "GSFC": "GRACE_GSFC_TWS"}


def land_area_km2(box) -> float:
    S, N, W, E = box
    lat = np.arange(S + 0.125, N, 0.25)
    lon = np.arange(W + 0.125, E, 0.25)
    m = land_mask(lat, lon)
    R = 6371.0
    cell = (R ** 2) * np.radians(0.25) * (np.sin(np.radians(lat + 0.125)) - np.sin(np.radians(lat - 0.125)))
    return float((m * cell[:, None]).sum())


def mean_of(s, year, months):
    v = [s.get((year, m), np.nan) for m in months]
    return np.nan if np.isnan(v).all() else float(np.nanmean(v))


def gains(s, frm, to, years):
    prev = min(frm) > max(to)
    return pd.Series({y: mean_of(s, y, to) - mean_of(s, y - 1 if prev else y, frm) for y in years}).dropna()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for key, L in LEADS.items():
        area = land_area_km2(L["box"])
        out = {"name": L["name"], "box": L["box"], "land_area_km2": round(area), "event_year": L["event"],
               "gain_window_months": [L["frm"], L["to"]], "solutions": {}}
        for sol, ds in SOLUTIONS.items():
            s = confirm._monthly(ds, L["box"], 2002, 2026)
            s.rename("lwe_cm").reset_index().to_csv(OUT / f"{key}_{sol}_monthly.csv", index=False)
            g = gains(s, L["frm"], L["to"], range(2003, 2027))
            g.rename("gain_cm").rename_axis("year").reset_index().to_csv(OUT / f"{key}_{sol}_gains.csv", index=False)
            if L["event"] not in g.index:
                out["solutions"][sol] = {"note": "event year not available"}
                continue
            other = g.drop(L["event"])
            excess = float(g[L["event"]] - other.mean())
            sd = float(other.std(ddof=1))
            k = area * 1e-5  # cm of water over the land area -> km3
            d = {"n_other_years": int(len(other)), "event_gain_cm": round(float(g[L["event"]]), 2),
                 "mean_gain_other_years_cm": round(float(other.mean()), 2), "sd_other_years_cm": round(sd, 2),
                 "excess_cm": round(excess, 2), "excess_cm_90pct": [round(excess - 1.645 * sd, 2),
                                                                    round(excess + 1.645 * sd, 2)],
                 "excess_km3": round(excess * k, 1),
                 "excess_km3_90pct": [round((excess - 1.645 * sd) * k, 1), round((excess + 1.645 * sd) * k, 1)],
                 "rank_of_event_gain": int((g > g[L["event"]]).sum()) + 1, "n_years": int(len(g))}
            if L["after"]:
                # change from before the rains to the following dry season, event year vs other years
                span = pd.Series({y: mean_of(s, y + 1, L["after"]) - mean_of(s, y, L["frm"])
                                  for y in range(2003, 2026)}).dropna()
                if L["event"] in span.index:
                    o2 = span.drop(L["event"])
                    left = float(span[L["event"]] - o2.mean())
                    d["excess_left_next_dry_season_cm"] = round(left, 2)
                    d["excess_left_sd_cm"] = round(float(o2.std(ddof=1)), 2)
                    d["share_left"] = round(left / excess, 2) if excess > 0 else None
            out["solutions"][sol] = d
        res[key] = out
        print(key, json.dumps(out, indent=1))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
