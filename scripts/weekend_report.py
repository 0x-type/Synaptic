"""Figures, an offline check and a plain-language report for the weekend-shift tests (docs/preregistrations/3_weekend_shift.md).
Usage: python scripts/weekend_report.py   (after weekend_shift.py test)"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weekend_shift as ws  # noqa: E402

OUT = ws.OUT
FIG = OUT / "figures"
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekly_cycle(s, spec):
    """Mean over the test's cities of NO2 on each weekday relative to the reference days, before and after."""
    t0 = pd.Timestamp(spec["date"])
    out = {"before": np.zeros(7), "after": np.zeros(7)}
    for c in spec["cities"]:
        d = s[c]
        d = d[(d.date >= spec["before"]) & (d.date <= spec["after"])].copy()
        d["dow"] = d.date.dt.dayofweek
        for k, part in (("before", d[d.date < t0]), ("after", d[d.date >= t0])):
            ref = part[part.dow.isin(spec["ref"])].value.mean()
            out[k] += part.groupby("dow").value.mean().reindex(range(7)).values / ref / len(spec["cities"])
    return out


def figure(name, spec, s, t):
    cyc = weekly_cycle(s, spec)
    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    x, w = np.arange(7), 0.38
    ax.bar(x - (w + 0.04) / 2, cyc["before"], width=w, color=BLUE, label=f"before {spec['date']}", linewidth=0)
    ax.bar(x + (w + 0.04) / 2, cyc["after"], width=w, color=ORANGE, label=f"after {spec['date']}", linewidth=0)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    lo = min(np.nanmin(cyc["before"]), np.nanmin(cyc["after"]))
    ax.set_ylim(max(0.0, lo - 0.25), max(np.nanmax(cyc["before"]), np.nanmax(cyc["after"])) + 0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    for d, lab in ((spec["up"], "predicted to rise"), (spec["down"], "predicted to fall")):
        top = max(cyc["before"][d], cyc["after"][d])
        ax.annotate(lab, (d, top), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, color=INK,
                    fontweight="bold")
    ax.set_ylabel("NO2 relative to working days", color=MUTED, fontsize=9)
    fig.text(0.07, 0.95, f"{spec['what'].split(':')[0]}: nitrogen dioxide by day of the week, seen by OMI", fontsize=12,
             color=INK, fontweight="bold", va="top")
    fig.text(0.07, 0.885, f"{', '.join(spec['cities'])}. Shift statistic D = {t['D']:+.3f}, p = {t['p']:.2g}. "
             f"Registered verdict: {t['verdict']}", fontsize=9, color=MUTED, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=MUTED, ncol=2, bbox_to_anchor=(0, 1.14))
    fig.subplots_adjust(top=0.74, left=0.08, right=0.98, bottom=0.1)
    fig.savefig(FIG / f"{name}.png", facecolor=SURF)
    plt.close(fig)
    return cyc


def yearly_figure(name, spec, s):
    """Descriptive, not registered: the two weekday ratios year by year (mean over the test's cities), 2005-2026."""
    rows = []
    for y in range(2005, 2027):
        v = []
        for c in spec["cities"]:
            d = s[c]
            d = d[d.date.dt.year == y]
            dow = d.date.dt.dayofweek
            rf = d[dow.isin(spec["ref"])].value.mean()
            v.append((d[dow == spec["up"]].value.mean() / rf, d[dow == spec["down"]].value.mean() / rf))
        rows.append((y, *np.mean(v, axis=0)))
    t = pd.DataFrame(rows, columns=["year", "rise_day", "fall_day"])
    t.to_csv(OUT / f"{name}_yearly.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    cy = pd.Timestamp(spec["date"]).year + (pd.Timestamp(spec["date"]).dayofyear - 1) / 365.0 - 0.5
    ax.axvline(cy, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(t.year, t.rise_day, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURF,
            label=DAYS[spec["up"]] + "day")
    ax.plot(t.year, t.fall_day, color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURF,
            label=DAYS[spec["down"]] + "day")
    ax.annotate(f"weekend moved {spec['date']}", (cy, ax.get_ylim()[1]), xytext=(-6, -2), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=INK)
    ax.annotate(DAYS[spec["up"]] + "day", (t.year.iloc[-1], t.rise_day.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.annotate(DAYS[spec["down"]] + "day", (t.year.iloc[-1], t.fall_day.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlim(2004.5, 2028.2)
    ax.set_xticks(range(2006, 2027, 2))
    ax.set_ylabel("NO2 relative to working days", color=MUTED, fontsize=9)
    fig.text(0.07, 0.95, f"{spec['what'].split(':')[0]}: Fridays and Sundays in the air, year by year (OMI)"
             if name == "UAE_2022" else f"{spec['what'].split(':')[0]}: the two days, year by year (OMI)",
             fontsize=12, color=INK, fontweight="bold", va="top")
    fig.text(0.07, 0.885, f"{', '.join(spec['cities'])}. 1.0 = a normal working day (Monday to Thursday). "
             "2026 is January to October.", fontsize=9, color=MUTED, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=MUTED, ncol=2, bbox_to_anchor=(0, 1.14))
    fig.subplots_adjust(top=0.74, left=0.08, right=0.98, bottom=0.1)
    fig.savefig(FIG / f"{name}_yearly.png", facecolor=SURF)
    plt.close(fig)


CHECK = '''"""The registered weekend-shift tests (docs/preregistrations/3_weekend_shift.md), recomputed offline from the bundled daily city
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
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "results.json").read_text())
    s = ws.city_series()
    lines = ["# Weekend-shift tests: results", "",
             "Registered in `docs/preregistrations/3_weekend_shift.md` before any OMI value for these cities was downloaded. "
             "\"Not shown\" means not significant; it is not evidence against.", "",
             "| Test | Day that should rise: before -> after | Day that should fall: before -> after | Shift D | p "
             "(Holm) | Verdict |", "|---|---|---|---|---|---|"]
    cycles = {}
    for name, spec in ws.TESTS.items():
        t = res["tests"][name]
        cyc = figure(name, spec, s, t)
        yearly_figure(name, spec, s)
        cycles[name] = {k: [None if not np.isfinite(v) else round(float(v), 3) for v in a] for k, a in cyc.items()}
        u, d = spec["up"], spec["down"]
        lines.append(f"| {spec['what']} | {DAYS[u]}: {cyc['before'][u]:.2f} -> {cyc['after'][u]:.2f} | "
                     f"{DAYS[d]}: {cyc['before'][d]:.2f} -> {cyc['after'][d]:.2f} | {t['D']:+.3f} | "
                     f"{t['p']:.2g} ({t['holm_p']:.2g}) | **{t['verdict']}** |")
    pl = res["placebos"]
    fired = [k for k, v in pl.items() if v["p"] < 0.05]
    lines += ["", f"Values are NO2 on that weekday relative to working days (1.00 = a normal working day), mean over "
              "the test's cities.", "",
              f"**Placebos** (same test in cities whose weekend did not change): {len(fired)} of {len(pl)} have "
              f"p < 0.05" + (f" ({', '.join(fired)})." if fired else ".")]
    (OUT / "weekly_cycles.json").write_text(json.dumps(cycles, indent=1))
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "check.py").write_text(CHECK)
    o = json.loads(subprocess.run([sys.executable, str(OUT / "check.py")], capture_output=True, text=True).stdout)
    e = dict(o)
    e["tolerance"] = {k: (1e-9 if k.endswith("_D") else 0.02) for k in o}
    e["claim"] = "Registered weekend-shift tests on OMI NO2; verdicts in REPORT.md."
    (OUT / "expected.json").write_text(json.dumps(e, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
