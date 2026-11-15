"""The pre-registered SMAP confirmation (docs/preregistrations/1_water_gains_smap.md), recomputed offline from the bundled season means."""
import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).parent
sys.path.insert(0, str(here.parents[1]))
import synlib  # noqa: E402

out = {}
for key, ev in (("A_lake_chad", 2024), ("B_central_sahara", 2024), ("C_north_morocco", 2026)):
    s = pd.read_csv(here / f"{key}_season_means.csv")
    r = synlib.event_departure(s.value.values, s.year.values, [ev])
    out[key + "_departure"] = float(r["departure"])
    out[key + "_p"] = float(r["p"])
print(json.dumps(out))
