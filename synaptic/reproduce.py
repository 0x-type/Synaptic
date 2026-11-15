"""Evidence bundles: every reported claim can be re-checked on a laptop without the campaign database.

A bundle is a folder with the data slice the claim rests on (CSV), the script that computes the claim's numbers
from it (check.py, offline), and expected.json. `python -m synaptic reproduce [folder ...]` re-runs every bundle
under evidence/ (and results/evidence/ of campaigns) and compares the numbers.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECK_CONFIRM = '''"""Recompute a held-out confirmation from the bundled annual series (offline)."""
import json, sys
from pathlib import Path
import pandas as pd
here = Path(__file__).parent
sys.path.insert(0, str(here.parents[2])) if (here.parents[2] / "synlib").exists() else None
import synlib
spec = json.loads((here / "spec.json").read_text())
t = pd.read_csv(here / "target_annual.csv")
y, yr = t.value.values, t.year.values.astype(float)
k = spec["kind"]
if k in ("correlation", "correlation_change"):
    d = pd.read_csv(here / "driver_annual.csv")
    m = pd.merge(t, d, on="year", suffixes=("", "_d"))
    y, yr, x = m.value.values, m.year.values.astype(float), m.value_d.values
if k == "trend":
    r = synlib.mann_kendall(y); out = {"effect": synlib.sen_slope(y, yr) * 10, "p": r["p"]}
elif k == "step":
    r = synlib.step_change_test(y, yr, split=spec["split"], n_sim=4000); out = {"effect": r["step_beyond_trend"], "p": r["p_selection_aware"]}
elif k == "event":
    r = synlib.event_departure(y, yr, spec["event_years"]); out = {"effect": r["departure"], "p": r["p"]}
elif k == "correlation":
    r = synlib.correlation_test(y, x); out = {"effect": r["r"], "p": r["p"]}
else:
    r = synlib.correlation_change_test(y, x, yr, split=spec["split"], n_sim=4000); out = {"effect": r["change_abs_r"], "p": r["p_selection_aware"]}
print(json.dumps(out))
'''


def bundle_hypothesis(record, cid: str, hid: str, out_root: Path) -> Path | None:
    """Write the held-out confirmation of one hypothesis as a bundle (re-fetches the held-out series once)."""
    from . import confirm, trend

    h = record.hypotheses(cid, id=hid)
    if not h or not h[0].get("confirm") or h[0]["confirm"] in ("{}", ""):
        return None
    spec, problem = confirm.validate(h[0]["confirm"])
    if spec is None:
        return None
    d = out_root / hid
    d.mkdir(parents=True, exist_ok=True)
    y0, y1 = spec["years"]
    tgt = confirm._annual_dataset(spec["dataset"], spec["box"], spec.get("months"), y0, y1)
    tgt.to_csv(d / "target_annual.csv", index=False)
    if spec["kind"] in ("correlation", "correlation_change"):
        from . import catalog

        dm = spec.get("driver_months") or spec.get("months")
        drv = confirm._annual_dataset(spec["driver"], spec.get("driver_box") or spec["box"], dm, y0 - 1, y1) \
            if spec["driver"] in catalog.BY_ID else confirm._annual_index(spec["driver"], dm, y0, y1)
        drv.assign(year=drv.year + int(spec.get("driver_lag_years") or 0)).to_csv(d / "driver_annual.csv", index=False)
    (d / "spec.json").write_text(json.dumps(spec, indent=1))
    (d / "check.py").write_text(CHECK_CONFIRM)
    res = confirm.run(spec)
    (d / "expected.json").write_text(json.dumps({"effect": res.get("effect"), "p": res.get("p"),
                                                  "outcome": res.get("outcome"), "claim": h[0]["statement"],
                                                  "tolerance": {"effect": 1e-6, "p": 0.02}}, indent=1))
    (d / "README.md").write_text(f"# {hid}: {h[0]['title']}\n\n{h[0]['statement']}\n\nHeld-out confirmation: "
                                 f"{res.get('detail')}\n\nRe-check: `python check.py` (prints effect and p).\n")
    return d


def check(folder: Path) -> tuple[bool, str]:
    exp = json.loads((folder / "expected.json").read_text())
    tol = exp.get("tolerance", {})
    p = subprocess.run([sys.executable, "check.py"], cwd=folder, capture_output=True, text=True, timeout=900,
                       env={"PYTHONPATH": f"{ROOT}:{ROOT / 'synlib'}", "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"})
    if p.returncode != 0:
        return False, f"check.py failed: {p.stderr[-500:]}"
    got = json.loads(p.stdout.strip().splitlines()[-1])
    bad = []
    for k, v in got.items():
        e = exp.get(k)
        if isinstance(e, (int, float)) and isinstance(v, (int, float)):
            t = tol.get(k, 1e-6 + 1e-6 * abs(e))
            if not (math.isfinite(v) and abs(v - e) <= t):
                bad.append(f"{k}: expected {e}, got {v}")
    return (not bad), ("; ".join(bad) or json.dumps(got))


def run_all(paths: list[str] | None = None) -> int:
    roots = [Path(p) for p in paths] if paths else [ROOT / "evidence"]
    folders = sorted({f.parent for r in roots for f in r.rglob("expected.json")})
    fails = 0
    for f in folders:
        ok, msg = check(f)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}: {msg[:200]}")
    print(f"{len(folders) - fails}/{len(folders)} evidence bundles reproduce")
    return 1 if fails else 0
