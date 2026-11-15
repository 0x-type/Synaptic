"""v3 headline H50424603 recomputed with the v4 statistics: DJF NINO3.4 -> JJA 2 m temperature over western
southern Africa (MERRA-2, 35-15S, 15-25E). v3 reported '44 of 52 cells, binomial p = 2e-7'."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import synlib

here = Path(__file__).parent
c = pd.read_csv(here / "t2m_jja_cells.csv")
x = pd.read_csv(here / "nino34_djf.csv").set_index("year").nino34_djf
J = c.pivot_table(index=["lat", "lon"], columns="year", values="value")
years = [y for y in J.columns if y in x.index]
J = J[years]
t = np.array(years, float)
X = x.loc[years].values
w = np.cos(np.radians(J.index.get_level_values(0).values))
V = J.values
reg = np.nansum(V * w[:, None], 0) / np.nansum(np.isfinite(V) * w[:, None], 0)
out = {}
for split in (2003, 2008):
    out[f"p_split_{split}"] = synlib.correlation_change_test(reg, X, t, split=split, n_sim=5000)["p_selection_aware"]
out["p_scan"] = synlib.correlation_change_test(reg, X, t, split="scan", n_sim=5000)["p_selection_aware"]
full = V[np.isfinite(V).all(1)]


def dr(y):
    m = t >= 2008
    yy = y - np.polyval(np.polyfit(t, y, 1), t)
    xx = X - np.polyval(np.polyfit(t, X, 1), t)
    return abs(np.corrcoef(yy[m], xx[m])[0, 1]) - abs(np.corrcoef(yy[~m], xx[~m])[0, 1])


R = np.array([r - np.polyval(np.polyfit(t, r, 1), t) for r in full])
st = synlib.sign_test_spatial(np.array([dr(r) for r in full]), R)
out.update({"cells": st["cells"], "positive": st["positive"], "effective_cells": st["effective_cells"],
            "p_sign_spatial": st["p"], "p_sign_naive": st["p_naive"]})
print(json.dumps(out))
