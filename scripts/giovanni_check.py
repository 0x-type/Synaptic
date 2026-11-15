"""Cross-check the app's data path against NASA Giovanni (the tool the challenge page recommends).
For each variable and point: the monthly series of the grid cell containing the point, fetched (a) by the app
(synaptic.cloud, Earthdata OPeNDAP) and (b) by Giovanni's time-series API. Writes evidence/giovanni_crosscheck/.
Usage: python scripts/giovanni_check.py     (needs EARTHDATA_TOKEN)"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synaptic import cloud  # noqa: E402

OUT = ROOT / "evidence" / "giovanni_crosscheck"
API = "https://api.giovanni.earthdata.nasa.gov/timeseries"
POINTS = {"Marrakech": (31.65, -7.95), "Wisconsin": (44.55, -89.55), "Lake_Chad": (13.55, 14.55)}
# app id, Giovanni id, factor turning Giovanni units into the app's, offset, years, hourly?
VARS = [("IMERG_PRECIP", "GPM_3IMERGM_07_precipitation", 24.0, 0.0, (2001, 2024), False),
        ("GLDAS_SOILM", "GLDAS_NOAH025_M_2_1_SoilMoi0_10cm_inst", 1.0, 0.0, (2001, 2024), False),
        ("GLDAS_EVAP", "GLDAS_NOAH025_M_2_1_Evap_tavg", 86400.0, 0.0, (2001, 2024), False),
        ("MERRA2_T2M", "M2T1NXSLV_5_12_4_T2M", 1.0, -273.15, (2020, 2020), True)]


def giovanni(gid, lat, lon, y0, y1):
    h = {"Authorization": "Bearer " + os.environ["EARTHDATA_TOKEN"].strip()}
    parts = []
    for y in range(y0, y1 + 1, 6):
        ye = min(y + 5, y1)
        for attempt in range(4):
            r = requests.get(API, params={"data": gid, "location": f"[{lat},{lon}]",
                                          "time": f"{y}-01-01T00:00:00/{ye}-12-31T23:59:59"}, headers=h, timeout=300)
            if r.status_code == 200:
                break
        r.raise_for_status()
        head, body = r.text.split("Timestamp (UTC),Data")
        meta = dict(l.split(",", 1) for l in head.strip().splitlines() if "," in l)
        d = pd.read_csv(io.StringIO("t,v" + body))
        parts.append(d)
    d = pd.concat(parts)
    d["t"] = pd.to_datetime(d.t)
    undef = float(meta.get("undef", -9999))
    d = d[np.isfinite(d.v) & ~np.isclose(d.v, undef, rtol=1e-6) & (d.v.abs() < 1e14)]
    return d, meta


def app_series(var, lat, lon, y0, y1):
    f = cloud.field(var, (lat - 0.3, lat + 0.3, lon - 0.4, lon + 0.4), y0, y1)
    i, j = int(np.abs(f.lat - lat).argmin()), int(np.abs(f.lon - lon).argmin())
    return pd.DataFrame({"year": [int(t[:4]) for t in f.times], "month": [int(t[5:7]) for t in f.times],
                         "app": f.data[:, i, j]}), float(f.lat[i]), float(f.lon[j])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for var, gid, fac, off, (y0, y1), hourly in VARS:
        for name, (lat, lon) in POINTS.items():
            try:
                g, meta = giovanni(gid, lat, lon, y0, y1)
                g["year"], g["month"] = g.t.dt.year, g.t.dt.month
                gm = g.groupby(["year", "month"]).v.mean().reset_index()
                gm["giovanni"] = gm.v * fac + off
                a, alat, alon = app_series(var, float(meta["lat"]), float(meta["lon"]), y0, y1)
                m = a.merge(gm[["year", "month", "giovanni"]], on=["year", "month"]).dropna()
                m.to_csv(OUT / f"{var}_{name}.csv", index=False)
                diff = m.app - m.giovanni
                rows.append({"variable": var, "place": name, "cell_lat": alat, "cell_lon": alon, "months": int(len(m)),
                             "correlation": float(np.corrcoef(m.app, m.giovanni)[0, 1]),
                             "mean_app": float(m.app.mean()), "mean_giovanni": float(m.giovanni.mean()),
                             "max_abs_diff": float(diff.abs().max()),
                             "max_rel_diff_pct": float(100 * diff.abs().max() / max(m.giovanni.abs().mean(), 1e-12))})
                print(rows[-1], flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"variable": var, "place": name, "error": f"{type(e).__name__}: {str(e)[:150]}"})
                print(rows[-1], flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
