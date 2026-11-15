"""Seasonal water-storage gains of the three leads, recomputed offline from the bundled monthly regional series
(GRACE/GRACE-FO, JPL RL06.3 and NASA GSFC RL06v2.0 mascons, land cells of each box).
For each lead and solution: the event year's gain, its excess over the mean of the other years (cm and km3), and its
departure from the TREND of the other years with the carry-over-aware p-value (synlib.event_departure)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
sys.path.insert(0, str(here.parents[1]))
import synlib  # noqa: E402

LEADS = {"A_lake_chad": ([6, 7], [9, 10], 2024, 504639), "B_central_sahara": ([6, 7], [9, 10], 2024, 1401743),
         "C_north_morocco": ([9, 10], [3, 4], 2026, 330460)}
out = {}
for key, (frm, to, ev, area) in LEADS.items():
    for sol in ("JPL", "GSFC"):
        s = pd.read_csv(here / f"{key}_{sol}_monthly.csv").set_index(["year", "month"]).lwe_cm
        prev = min(frm) > max(to)
        g = {}
        for y in range(2003, 2027):
            a = [s.get(((y - 1) if prev else y, m), np.nan) for m in frm]
            b = [s.get((y, m), np.nan) for m in to]
            if not (np.isnan(a).all() or np.isnan(b).all()):
                g[y] = float(np.nanmean(b) - np.nanmean(a))
        g = pd.Series(g)
        other = g.drop(ev)
        r = synlib.event_departure(g.values, g.index.values.astype(float), [ev])
        k = f"{key}_{sol}"
        out[k + "_excess_cm"] = float(g[ev] - other.mean())
        out[k + "_excess_km3"] = float((g[ev] - other.mean()) * area * 1e-5)
        out[k + "_departure_from_trend_cm"] = float(r["departure"])
        out[k + "_departure_from_trend_km3"] = float(r["departure"] * area * 1e-5)
        out[k + "_p"] = float(r["p"])
print(json.dumps(out))
