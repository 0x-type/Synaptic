"""Figure for the river-mouth chlorophyll re-test: the p-value the 1998-2007 SeaWiFS paper reported for each river
next to the p-value of the 2002-2025 MODIS trend, corrected for month-to-month carry-over.
Usage: python scripts/river_figure.py   (after river_recheck.py) -> evidence/river_chlorophyll/figures/"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "evidence" / "river_chlorophyll"
BLUE, GREY, INK, MUTED, GRID, SURF = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# p-values as reported by the SeaWiFS paper (NTRS 20090042747), as quoted in the campaign report
PAPER_P = {"Mississippi West": (0.002, "down"), "Mississippi East": (0.017, "down"), "Eel (summer)": (0.035, "up"),
           "Po": (0.065, "down"), "Pearl": (0.068, "up"), "Guadalquivir": (0.07, "down")}


def main():
    res = json.loads((DIR / "results.json").read_text())
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    names = list(PAPER_P)[::-1]
    for i, n in enumerate(names):
        p_paper, d_paper = PAPER_P[n]
        m = res[n]["modis_2002_2025"]
        p_new, pct = m["p_corrected"], m["percent_per_decade"]
        ax.plot([p_paper, p_new], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([p_paper], [i], s=70, facecolor=SURF, edgecolor=GREY, lw=2, zorder=3)
        ax.scatter([p_new], [i], s=70, color=BLUE, edgecolor=SURF, lw=2, zorder=4)
        side = 1 if p_new >= p_paper else -1
        ax.annotate(f"{pct:+.0f}% per decade", (p_new, i), xytext=(10 * side, 0), textcoords="offset points",
                    ha="left" if side > 0 else "right", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.05, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.05, len(names) - 0.35, "  p = 0.05", fontsize=8.5, color=INK, va="bottom")
    ax.axvspan(1e-7, 0.05, color="#2a78d6", alpha=0.05, lw=0)
    ax.text(8e-6, -0.85, "significant", fontsize=8.5, color=MUTED)
    ax.text(0.09, -0.85, "not significant", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(4e-6, 8)
    ax.minorticks_off()
    ax.set_ylim(-1.1, len(names) - 0.2)
    ax.set_yticks(range(len(names)), [""] * len(names))
    for i, n in enumerate(names):
        ax.text(-0.02, i + 0.1, n, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9.5,
                color=INK)
        ax.text(-0.02, i - 0.05, f"paper: trend {PAPER_P[n][1]}", transform=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=8, color=MUTED)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"], color=MUTED)
    ax.set_xlabel("p-value of the trend (log scale)", color=MUTED)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.scatter([], [], s=60, facecolor=SURF, edgecolor=GREY, lw=2, label="SeaWiFS 1998-2007, as the paper reported")
    ax.scatter([], [], s=60, color=BLUE, label="MODIS Aqua 2002-2025, carry-over corrected (ours)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.13), ncol=2, fontsize=8.5,
              labelcolor=MUTED)
    fig.text(0.02, 0.97, "River-mouth chlorophyll: NASA's reference trends re-tested with 24 years of MODIS",
             fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, "The three trends the paper found significant (Mississippi West and East, Eel) are not "
             "detected. Only the Guadalquivir shows one; VIIRS does not yet confirm it.", fontsize=8.5, color=MUTED,
             va="top")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.12)
    out = DIR / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "river_pvalues.png", facecolor=SURF)
    print(out / "river_pvalues.png")


if __name__ == "__main__":
    main()
