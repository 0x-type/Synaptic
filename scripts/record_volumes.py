"""Descriptive GRACE numbers for the registered candidates (docs/preregistrations/2_record_scan.md): each season's gain on both
solutions, the event season's excess over the baseline mean as cm and km3, with the baseline year-to-year range.
Usage: python scripts/record_volumes.py  -> evidence/record_scan/volumes.json and <id>_grace_gains.csv"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_scan as rs  # noqa: E402

OUT = rs.OUT


def box_series(ym, lat, lon, a, land, box):
    S, N, W, E = box
    yi, xi = (lat >= S) & (lat < N), (lon >= W) & (lon < E)
    m = land[np.ix_(yi, xi)]
    w = np.cos(np.radians(lat[yi]))[:, None] * m
    sub = a[:, yi][:, :, xi]
    ok = np.isfinite(sub) & (w > 0)[None]
    s = (np.where(ok, sub, 0) * w[None]).sum(axis=(1, 2)) / np.maximum((ok * w[None]).sum(axis=(1, 2)), 1e-9)
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    cell = 6371.0 ** 2 * np.radians(dlon) * (np.sin(np.radians(lat[yi] + dlat / 2)) - np.sin(np.radians(lat[yi] - dlat / 2)))
    return pd.Series(s, index=pd.MultiIndex.from_tuples(ym)), float((m * cell[:, None]).sum())


def main():
    reg = json.loads((OUT / "registered_candidates.json").read_text())
    sols = {"JPL": rs.load("GRCTellus.JPL.*.nc"), "GSFC": rs.load("GSFC.glb_*.nc")}
    out = {}
    for c in reg["candidates"]:
        frm, to = c["gain_window_months"]
        ey, em = map(int, c["season_end"].split("-"))
        d = {"name": c["name"], "box": c["box"]}
        tabs = {}
        for sol, (_, ym, lat, lon, a, land) in sols.items():
            s, area = box_series(ym, lat, lon, a, land, c["box"])
            tab = rs.season_table(s, frm, to)
            base = [g for (e, g) in tab if e < rs.EVENT_FROM]
            ev = [g for (e, g) in tab if e == (ey, em)][0]
            mu, sd = float(np.mean(base)), float(np.std(base, ddof=1))
            k = area * 1e-5
            d["land_area_km2"] = round(area)
            d[sol] = {"event_gain_cm": round(ev, 1), "normal_gain_cm": round(mu, 1), "sd_cm": round(sd, 1),
                      "excess_cm": round(ev - mu, 1), "excess_km3": round((ev - mu) * k, 1),
                      "excess_km3_90pct": [round((ev - mu - 1.645 * sd) * k, 1), round((ev - mu + 1.645 * sd) * k, 1)],
                      "rank": int(sum(g > ev for g in base)) + 1, "n_seasons": len(base) + 1}
            tabs[sol] = pd.Series({e[0]: g for e, g in tab})
        pd.DataFrame(tabs).rename_axis("season_end_year").reset_index().to_csv(OUT / f"{c['id']}_grace_gains.csv", index=False)
        out[c["id"]] = d
        print(c["id"], json.dumps(d))
    (OUT / "volumes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
