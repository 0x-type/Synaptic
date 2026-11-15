"""Independent re-check of the river-mouth chlorophyll study (our teammate's Synaptic campaign).

The campaign re-tested the trends of the NASA study "Time-Series Analysis of Remotely-Sensed SeaWiFS Chlorophyll in
River-Influenced Coastal Regions" (NTRS 20090042747, in the challenge's resource list) on MODIS Aqua 2002-2025.
This script repeats the MODIS part with THIS branch's own code path (the app's chlorophyll reader and
`synaptic.trend`), for the same boxes. It does not use VIIRS (not in this branch's catalogue).

For each box, monthly log10 chlorophyll, July 2002 - December 2025:
  - the least-squares trend with the p-value the original paper's method gives (no correction for carry-over);
  - the same trend with the standard error corrected for lag-1 carry-over (`trend.ols_ar1`);
  - the same two numbers for the 2002-2007 overlap with the paper's period.
Usage: python scripts/river_recheck.py  -> evidence/river_chlorophyll/"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synaptic import cloud, trend  # noqa: E402

OUT = ROOT / "evidence" / "river_chlorophyll"
# box [S, N, W, E], months (None = all), what the 1998-2007 SeaWiFS paper reported (as quoted in the campaign)
RIVERS = {
    "Mississippi West": ([28, 29, -91.5, -90.5], None, "decreasing, p ~ 0.002"),
    "Mississippi East": ([29, 30, -88.5, -87.5], None, "decreasing, p ~ 0.017"),
    "Eel (summer)": ([40, 41, -125.5, -124.5], [6, 7, 8], "increasing, p ~ 0.035"),
    "Po": ([44.3, 45.3, 12.5, 13.5], None, "decreasing, p ~ 0.065 without the 2000 flood"),
    "Pearl": ([21, 22, 113.5, 114.5], None, "increasing, not significant (p ~ 0.068)"),
    "Guadalquivir": ([36, 37, -7.5, -6.5], None, "decreasing, not significant (p ~ 0.07)"),
    "Ganges": ([19.5, 20.5, 89, 90], None, "decreasing"),
    "Amazon": ([0, 4, -52, -48], None, "no trend"),
}


def fit(d: pd.DataFrame) -> dict:
    """Monthly log10 chlorophyll with the mean seasonal cycle removed; trend per decade, naive and corrected p."""
    d = d.dropna(subset=["value"]).copy()
    d = d[d.value > 0]
    y = np.log10(d.value.values)
    y = y - pd.Series(y).groupby(d.month.values).transform("mean").values
    t = (d.year + (d.month - 0.5) / 12.0).values
    r = stats.linregress(t, y)
    c = trend.ols_ar1(y, t)
    se = c["se"]
    return {"months": int(len(y)), "slope_log10_per_decade": float(r.slope * 10),
            "percent_per_decade": float((10 ** (r.slope * 10) - 1) * 100),
            "p_uncorrected": float(r.pvalue), "p_corrected": float(c.get("p", np.nan)),
            "ci95_log10_per_decade": [float(c["ci95"][0] * 10), float(c["ci95"][1] * 10)],
            "carryover": float(c.get("carryover", np.nan)), "se_ratio": float(se / r.stderr) if r.stderr else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (box, months, paper) in RIVERS.items():
        s = cloud.regional_series("MODIS_CHL", tuple(box), 2002, 2025)
        s = s[s.coverage >= 0.5]
        if months:
            s = s[s.month.isin(months)]
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        s[["year", "month", "value", "coverage"]].to_csv(OUT / f"{key}_monthly.csv", index=False)
        res[name] = {"box": box, "months": months, "paper_1998_2007": paper, "modis_2002_2025": fit(s),
                     "modis_2002_2007": fit(s[s.year <= 2007]), "mean_mg_m3": float(s.value.mean())}
        a, b = res[name]["modis_2002_2025"], res[name]["modis_2002_2007"]
        print(f"{name:18s} 2002-2025: {a['percent_per_decade']:+6.1f}%/decade  p naive {a['p_uncorrected']:.3g}  "
              f"corrected {a['p_corrected']:.3g} (carry-over {a['carryover']:.2f}) | 2002-2007: p naive "
              f"{b['p_uncorrected']:.3g} corrected {b['p_corrected']:.3g}", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
