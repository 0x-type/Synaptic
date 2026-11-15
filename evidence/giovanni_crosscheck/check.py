"""The app's monthly grid-cell series against NASA Giovanni's, recomputed offline from the bundled pairs
(app column: synaptic.cloud through Earthdata OPeNDAP; giovanni column: Giovanni time-series API, same cell)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
worst_r, worst_rel, n = 1.0, 0.0, 0
for f in sorted(here.glob("*_*.csv")):
    m = pd.read_csv(f)
    worst_r = min(worst_r, float(np.corrcoef(m.app, m.giovanni)[0, 1]))
    worst_rel = max(worst_rel, float(100 * (m.app - m.giovanni).abs().max() / m.giovanni.abs().mean()))
    n += 1
print(json.dumps({"series_compared": n, "lowest_correlation": worst_r, "largest_difference_pct": worst_rel}))
