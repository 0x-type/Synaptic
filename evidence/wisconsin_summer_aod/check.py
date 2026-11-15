"""The challenge video's example: summer (JJA) aerosol optical depth over Wisconsin from MODIS Aqua (MYD08_M3 C6.1),
area-weighted inside the state outline, 2002-2025, analysed by the Trend Detective."""
import json
from pathlib import Path

import pandas as pd

from synaptic import trend

s = pd.read_csv(Path(__file__).parent / "modis_aqua_aod_wisconsin_monthly.csv")
r = trend.analyse(s, [6, 7, 8], "", "MODIS Aqua AOD")
print(json.dumps({"sen_per_decade": r["trend"]["sen_per_decade"], "mk_p": r["trend"]["mk_p"],
                  "recent_departure": r["recent"]["departure"], "recent_p": r["recent"]["p"]}))
