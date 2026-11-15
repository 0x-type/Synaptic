"""Figures for the findings page and the pitch. Reads evidence/water_gains and evidence/smap_confirmation.
Usage: python scripts/make_figures.py   -> evidence/figures/*.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G, S, OUT = ROOT / "evidence" / "water_gains", ROOT / "evidence" / "smap_confirmation", ROOT / "evidence" / "figures"
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TITLES = {"A_lake_chad": ("Lake Chad basin", "June-July to September-October"),
          "B_central_sahara": ("Central Sahara", "June-July to September-October"),
          "C_north_morocco": ("Northern Morocco", "September-October to March-April")}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def gains_figure(key, res):
    name, window = TITLES[key]
    j = pd.read_csv(G / f"{key}_JPL_gains.csv").set_index("year").gain_cm
    g = pd.read_csv(G / f"{key}_GSFC_gains.csv").set_index("year").gain_cm
    years = sorted(set(j.index) | set(g.index))
    ev = res[key]["event_year"]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200, facecolor=SURF)
    style(ax)
    w = 0.38
    for k, (ser, col, lab) in enumerate(((j, BLUE, "JPL mascons"), (g, ORANGE, "NASA Goddard mascons"))):
        xs = [y + (k - 0.5) * (w + 0.04) for y in years if y in ser.index]
        ax.bar(xs, [ser[y] for y in years if y in ser.index], width=w, color=col, label=lab, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    top = max(j.get(ev, 0), g.get(ev, 0))
    ax.annotate(f"{ev}", (ev, top), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_xticks([y for y in years if y % 2 == 0])
    ax.set_ylabel("storage gain (cm of water)", color=MUTED, fontsize=9)
    sol = res[key]["solutions"]
    sub = (f"Extra water in {ev} versus a normal year: about {sol['JPL']['excess_km3']:.0f} km3 (JPL) and "
           f"{sol['GSFC']['excess_km3']:.0f} km3 (Goddard)")
    fig.text(0.07, 0.95, f"{name}: water gained each season, weighed by GRACE", fontsize=12, color=INK,
             fontweight="bold", va="top")
    fig.text(0.07, 0.885, f"{window}. {sub}", fontsize=9, color=MUTED, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=MUTED, ncol=2, bbox_to_anchor=(0, 1.14))
    fig.subplots_adjust(top=0.74, left=0.07, right=0.98, bottom=0.1)
    fig.savefig(OUT / f"{key}_grace_gains.png", facecolor=SURF)
    plt.close(fig)


def smap_figure(key, sres):
    f = S / f"{key}_season_means.csv"
    if not f.exists() or "p" not in sres.get(key, {}).get("primary", {}):
        return
    name, _ = TITLES[key]
    s = pd.read_csv(f)
    r = sres[key]["primary"]
    ev = sres[key]["event_year"]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200, facecolor=SURF)
    style(ax)
    ax.bar(s.year, s.value, width=0.6, color=[BLUE if y != ev else "#104281" for y in s.year], linewidth=0)
    ax.annotate(f"{ev}", (ev, float(s.value[s.year == ev].iloc[0])), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(list(s.year))
    ax.set_ylim(0, s.value.max() * 1.18)
    ax.set_ylabel("soil moisture (m3/m3)", color=MUTED, fontsize=9)
    months = "August-October" if key != "C_north_morocco" else "December-April"
    fig.text(0.07, 0.95, f"{name}: season-mean surface soil moisture, SMAP (independent check)", fontsize=12,
             color=INK, fontweight="bold", va="top")
    fig.text(0.07, 0.875, f"{months}. {ev} versus the trend of the other years: {r['departure_from_trend']:+.3f} "
             f"m3/m3, p = {r['p']:.3g}. Pre-registered verdict: {r['verdict']}", fontsize=9, color=MUTED, va="top")
    fig.subplots_adjust(top=0.78, left=0.08, right=0.98, bottom=0.1)
    fig.savefig(OUT / f"{key}_smap.png", facecolor=SURF)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((G / "results.json").read_text())
    sres = json.loads((S / "results.json").read_text()) if (S / "results.json").exists() else {}
    for key in TITLES:
        gains_figure(key, res)
        smap_figure(key, sres)
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
