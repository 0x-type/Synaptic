"""River-mouth chlorophyll trends, MODIS Aqua 2002-2025, recomputed offline from the bundled monthly box means:
least-squares trend of deseasonalised monthly log10 chlorophyll, with the uncorrected p-value (the 2009 paper's
method) and the p-value corrected for lag-1 carry-over (synaptic.trend.ols_ar1)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

here = Path(__file__).parent
sys.path.insert(0, str(here.parents[1]))
from synaptic import trend  # noqa: E402

out = {}
for f in sorted(here.glob("*_monthly.csv")):
    d = pd.read_csv(f).dropna(subset=["value"])
    d = d[d.value > 0]
    for tag, part in (("2002_2025", d), ("2002_2007", d[d.year <= 2007])):
        y = np.log10(part.value.values)
        y = y - pd.Series(y).groupby(part.month.values).transform("mean").values
        t = (part.year + (part.month - 0.5) / 12.0).values
        k = f.stem.replace("_monthly", "") + "_" + tag
        out[k + "_pct_per_decade"] = float((10 ** (stats.linregress(t, y).slope * 10) - 1) * 100)
        out[k + "_p_uncorrected"] = float(stats.linregress(t, y).pvalue)
        out[k + "_p_corrected"] = float(trend.ols_ar1(y, t)["p"])
print(json.dumps(out))
