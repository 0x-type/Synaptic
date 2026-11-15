"""Figures, an offline check and a plain-language report for the record scan (docs/preregistrations/2_record_scan.md).
Usage: python scripts/record_report.py   (after record_confirm.py test and record_volumes.py)
Writes evidence/record_scan/figures/*.png, REPORT.md, check.py inputs and expected.json."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "record_scan"
FIG = OUT / "figures"
BLUE, ORANGE, DARK, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#104281", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def window_label(ev):
    (y0, m0), (y1, m1) = ev[0], ev[-1]
    return f"{MN[m0 - 1]} {y0} to {MN[m1 - 1]} {y1}"


def fig_grace(c, vol):
    g = pd.read_csv(OUT / f"{c['id']}_grace_gains.csv")
    ey = int(c["season_end"][:4])
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200, facecolor=SURF)
    style(ax)
    w = 0.38
    for k, (col, colr, lab) in enumerate((("JPL", BLUE, "JPL mascons"), ("GSFC", ORANGE, "NASA Goddard mascons"))):
        d = g.dropna(subset=[col])
        ax.bar(d.season_end_year + (k - 0.5) * (w + 0.04), d[col], width=w, color=colr, label=lab, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    top = np.nanmax(g[g.season_end_year == ey][["JPL", "GSFC"]].values)
    ax.annotate(str(ey), (ey, top), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10, color=INK,
                fontweight="bold")
    ax.set_xticks([y for y in g.season_end_year if y % 2 == 0])
    ax.set_ylabel("storage gain (cm of water)", color=MUTED, fontsize=9)
    fm, tm = c["gain_window_months"]
    fig.text(0.07, 0.95, f"{c['name']}: water gained each wet season, weighed by GRACE", fontsize=12, color=INK,
             fontweight="bold", va="top")
    fig.text(0.07, 0.885, f"{MN[fm - 1]} to {MN[tm - 1]}. Extra water in the season ending {ey}: about "
             f"{vol['JPL']['excess_km3']:.0f} km3 (JPL) and {vol['GSFC']['excess_km3']:.0f} km3 (Goddard)", fontsize=9,
             color=MUTED, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=MUTED, ncol=2, bbox_to_anchor=(0, 1.14))
    fig.subplots_adjust(top=0.74, left=0.07, right=0.98, bottom=0.1)
    fig.savefig(FIG / f"{c['id']}_grace.png", facecolor=SURF)
    plt.close(fig)


def fig_smap(c, r):
    f = OUT / f"{c['id']}_smap_monthly.csv"
    if not f.exists() or not r["smap"].get("valid"):
        return
    d = pd.read_csv(f)
    ev = {tuple(x) for x in c["event_months"]}
    d["ev"] = [(int(y), int(m)) in ev for y, m in zip(d.year, d.month)]
    clim = d[~d.ev].groupby("month").value.mean()
    d["anom"] = d.value - clim.reindex(d.month).values
    d["t"] = d.year + (d.month - 0.5) / 12
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200, facecolor=SURF)
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.plot(d.t, d.anom, color=BLUE, linewidth=1.4)
    e = d[d.ev]
    ax.plot(e.t, e.anom, "o", color=DARK, markersize=5, markeredgecolor=SURF, markeredgewidth=1)
    ax.annotate("event months", (e.t.mean(), e.anom.max()), xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_ylabel("soil moisture anomaly (m3/m3)", color=MUTED, fontsize=9)
    s = r["smap"]
    fig.text(0.07, 0.95, f"{c['name']}: SMAP soil moisture, monthly departure from normal (independent check)",
             fontsize=11.5, color=INK, fontweight="bold", va="top")
    fig.text(0.07, 0.875, f"Event months {window_label(c['event_months'])}: {s['departure']:+.3f} m3/m3, "
             f"p = {s['p']:.2g}. Registered verdict: {r['verdict']}", fontsize=9, color=MUTED, va="top")
    fig.subplots_adjust(top=0.78, left=0.08, right=0.98, bottom=0.1)
    fig.savefig(FIG / f"{c['id']}_smap.png", facecolor=SURF)
    plt.close(fig)


CHECK = '''"""The registered SMAP tests of the record-scan candidates (docs/preregistrations/2_record_scan.md), recomputed offline from the
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
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    reg = json.loads((OUT / "registered_candidates.json").read_text())
    res = json.loads((OUT / "results.json").read_text())
    vol = json.loads((OUT / "volumes.json").read_text())
    by = {r["id"]: r for r in res["candidates"]}
    lines = ["# Record scan: results (14 Nov 2026)", "",
             "Registered in `docs/preregistrations/2_record_scan.md` before any confirming data were fetched. "
             "\"Not shown\" means not significant; it is not evidence against.", "",
             "| Region | Season | Extra water, JPL / Goddard (km3) | SMAP check: departure, p (Holm-adjusted) | Verdict | "
             "Rain rank (MERRA-2) | Placebo years firing |", "|---|---|---|---|---|---|---|"]
    for c in reg["candidates"]:
        r, v = by[c["id"]], vol[c["id"]]
        fig_grace(c, v)
        fig_smap(c, r)
        s = r["smap"]
        smap = (f"{s['departure']:+.3f} m3/m3, p = {s['p']:.2g} ({r.get('holm_p', float('nan')):.2g})"
                if s.get("valid") else "could not be tested")
        rain = r.get("rain", {})
        rr = f"{rain['rank_wettest']} of {rain['n_seasons']} (p = {rain['p']:.2g})" if rain.get("valid") else "n/a"
        pl = r.get("placebo", [])
        lines.append(f"| {c['name']} | {window_label(c['event_months'])} | {v['JPL']['excess_km3']:.0f} / "
                     f"{v['GSFC']['excess_km3']:.0f} | {smap} | **{r['verdict']}** | {rr} | "
                     f"{sum(p['p'] < 0.05 for p in pl)} of {len(pl)} |")
    p = res["pooled"]
    lines += ["", f"**Pooled test** (does the scan find real events at all?): {p['n_events']} independent events, "
              f"z = {p['z']:.2f}, one-sided p = {p['p_one_sided']:.2g}." if p.get("z") is not None else "",
              "", "Scan-level facts: 1,534 block-seasons examined; 97 records on both GRACE solutions where about 148 "
              "are expected by chance; 5 candidates where about 13 are expected by chance on one solution. The GRACE "
              "scan alone shows no excess of records; only the independent test separates real events from noise."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "check.py").write_text(CHECK)
    o = json.loads(subprocess.run([sys.executable, str(OUT / "check.py")], capture_output=True, text=True).stdout)
    e = dict(o)
    e["tolerance"] = {k: 1e-9 for k in o}
    e["claim"] = "Registered SMAP tests of the record-scan candidates; verdicts in REPORT.md."
    (OUT / "expected.json").write_text(json.dumps(e, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
