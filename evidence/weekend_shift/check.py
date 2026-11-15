"""The registered weekend-shift tests (docs/preregistrations/3_weekend_shift.md), recomputed offline from the bundled daily city
values with the registered function (2,000 permutations: p-values agree with the report to about +-0.01)."""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here.parents[1] / "scripts"))
import weekend_shift as ws  # noqa: E402

s = ws.city_series()
out = {}
for name, spec in ws.TESTS.items():
    r = ws.shift_test(s, spec, n_perm=2000)
    out[name + "_D"] = r["D"]
    out[name + "_p"] = r["p"]
print(json.dumps(out))
