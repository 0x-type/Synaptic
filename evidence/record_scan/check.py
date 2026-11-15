"""The registered SMAP tests of the record-scan candidates (docs/preregistrations/2_record_scan.md), recomputed offline from the
bundled monthly box means with the registered function."""
import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).parent
sys.path.insert(0, str(here.parents[1] / "scripts"))
from record_confirm import monthly_event_test  # noqa: E402

reg = json.loads((here / "registered_candidates.json").read_text())
out = {}
for c in reg["candidates"]:
    f = here / f"{c['id']}_smap_monthly.csv"
    if not f.exists():
        continue
    t = monthly_event_test(pd.read_csv(f), [tuple(x) for x in c["event_months"]])
    if t.get("valid"):
        out[c["id"] + "_departure"] = t["departure"]
        out[c["id"] + "_p"] = t["p"]
print(json.dumps(out))
