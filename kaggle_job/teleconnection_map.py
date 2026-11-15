# Synaptic GPU template: "teleconnection" -- has a driver's grip on a variable changed, cell by cell, and is the
# change real or a sample effect?  Heavy resampling runs on the GPU (tens of thousands of bootstrap / permutation
# draws for every cell at once).
#
# PARAMS: target (variable name in the input), driver (index name), season (target months, default all),
#   lag (months the driver leads the season, default 3), split_year (default 2003), n_boot (default 2000),
#   n_perm (default 500), window (sliding window years, default 15)
# Input: inputs/analysis_input.npz  (X [cells, vars, months] standardized anomalies, NaN = untrusted; I [drivers,
#   months]; lat, lon, var_names, driver_names, years, months) -- prepared automatically by gpu_submit.
# Output: out/tele_map.csv, out/tele_windows.csv, out/tele_summary.json, out/tele_map.png
import json
import os

import numpy as np

P = {"target": None, "driver": "NINO34", "season": None, "lag": 3, "split_year": 2003, "n_boot": 2000, "n_perm": 500,
     "window": 15, "seed": 0}
P.update({k: v for k, v in globals().get("PARAMS", {}).items() if v is not None})
if not isinstance(P.get("season"), (list, tuple)) or not all(str(m).isdigit() for m in P.get("season") or []):
    P["season"] = None  # "all", "annual" or anything else that is not a month list means every month
else:
    P["season"] = [int(m) for m in P["season"]]
os.makedirs("out", exist_ok=True)
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    P["n_boot"], P["n_perm"] = min(P["n_boot"], 300), min(P["n_perm"], 100)
D = np.load("inputs/analysis_input.npz", allow_pickle=True)
X, I = D["X"], D["I"]
vn, dn = [str(v) for v in D["var_names"]], [str(d) for d in D["driver_names"]]
lat, lon, years, months = D["lat"], D["lon"], D["years"].astype(int), D["months"].astype(int)
tv = vn.index(P["target"]) if P["target"] in vn else 0
dv = dn.index(P["driver"]) if P["driver"] in dn else 0
season = P["season"] or list(range(1, 13))
uy = np.unique(years)
# seasonal target per year and the driver averaged over the same months shifted back by `lag`
Ty, Dy = [], []
for y in uy:
    sel = (years == y) & np.isin(months, season)
    idx = np.flatnonzero(sel)
    with np.errstate(all="ignore"):
        Ty.append(np.nanmean(X[:, tv, idx], 1))
    li = idx - int(P["lag"])
    Dy.append(np.nanmean(I[dv, li[li >= 0]]) if (li >= 0).any() else np.nan)
Ty, Dy = np.stack(Ty, 1), np.array(Dy)                       # [C, Y], [Y]
okY = np.isfinite(Dy)
Ty, Dy, uy = Ty[:, okY], Dy[okY], uy[okY]
late = uy >= P["split_year"]
STRONG = {1982, 1983, 1987, 1988, 1991, 1992, 1997, 1998, 2009, 2010, 2015, 2016, 2023, 2024,
          1989, 1999, 2000, 2007, 2008, 2011, 2020, 2021, 2022}
T_ = torch.tensor(np.nan_to_num(Ty), device=dev, dtype=torch.float32)
M_ = torch.tensor(np.isfinite(Ty), device=dev, dtype=torch.float32)
D_ = torch.tensor(Dy, device=dev, dtype=torch.float32)


def corr(w):
    """Weighted Pearson r per cell; w [B, Y] year weights (counts), returns [B, C]."""
    w = w[:, None, :] * M_[None]                                   # [B, C, Y]
    sw = w.sum(-1).clamp(min=1e-6)
    mt = (w * T_[None]).sum(-1) / sw
    md = (w * D_[None, None]).sum(-1) / sw
    dt = T_[None] - mt[..., None]
    dd = D_[None, None] - md[..., None]
    cov = (w * dt * dd).sum(-1)
    return cov / torch.sqrt((w * dt * dt).sum(-1) * (w * dd * dd).sum(-1)).clamp(min=1e-9)


Y = len(uy)
e_w = torch.tensor(~late, device=dev, dtype=torch.float32)[None]
l_w = torch.tensor(late, device=dev, dtype=torch.float32)[None]
r_e, r_l = corr(e_w)[0], corr(l_w)[0]
change = (torch.abs(r_l) - torch.abs(r_e)).cpu().numpy()            # strengthening of the link (either sign)
enough = (((M_ * e_w).sum(-1) >= 8) & ((M_ * l_w).sum(-1) >= 8)).cpu().numpy()  # trust-cut cells are not comparable
change = np.where(enough, change, np.nan)
g = torch.Generator(device=dev).manual_seed(P["seed"])
ei, li = torch.tensor(np.flatnonzero(~late), device=dev), torch.tensor(np.flatnonzero(late), device=dev)
bs = []
for _ in range(0, P["n_boot"], 250):
    B = min(250, P["n_boot"] - len(bs) * 250)
    we = torch.zeros(B, Y, device=dev).scatter_add_(1, ei[torch.randint(len(ei), (B, len(ei)), generator=g, device=dev)],
                                                  torch.ones(B, len(ei), device=dev))
    wl = torch.zeros(B, Y, device=dev).scatter_add_(1, li[torch.randint(len(li), (B, len(li)), generator=g, device=dev)],
                                                  torch.ones(B, len(li), device=dev))
    bs.append((torch.abs(corr(wl)) - torch.abs(corr(we))).cpu().numpy())
bs = np.concatenate(bs, 0)
lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
# sample-effect controls: (1) without strong ENSO years; (2) amplitude-matched subsamples of the two periods
keep = torch.tensor([int(y) not in STRONG for y in uy], device=dev, dtype=torch.float32)[None]
ch_nostrong = (torch.abs(corr(l_w * keep)) - torch.abs(corr(e_w * keep)))[0].cpu().numpy()
amp = np.abs(Dy)
bins = np.quantile(amp, [0, 1 / 3, 2 / 3, 1.0])
bins[-1] += 1e-6
rngn = np.random.default_rng(P["seed"])
mt = []
for _ in range(200):
    we, wl = np.zeros(Y), np.zeros(Y)
    for b in range(3):
        inb = (amp >= bins[b]) & (amp < bins[b + 1])
        e_b, l_b = np.flatnonzero(inb & ~late), np.flatnonzero(inb & late)
        k = min(len(e_b), len(l_b))
        if k:
            we[rngn.choice(e_b, k, replace=False)] = 1
            wl[rngn.choice(l_b, k, replace=False)] = 1
    mt.append(torch.tensor(np.stack([we, wl]), device=dev, dtype=torch.float32))
W2 = torch.stack(mt)                                                # [200, 2, Y]
r_me = corr(W2[:, 0])
r_ml = corr(W2[:, 1])
ch_matched = torch.nanmean(torch.abs(r_ml) - torch.abs(r_me), 0).cpu().numpy()
sig = ((lo > 0) | (hi < 0)) & enough
robust = sig & (np.sign(ch_nostrong) == np.sign(change)) & (np.sign(ch_matched) == np.sign(change)) & \
    (np.abs(ch_matched) >= 0.5 * np.abs(change))
# field significance: shuffle the period labels across years (keeps spatial correlation), count robust-looking cells
null = []
for _ in range(P["n_perm"]):
    perm = rngn.permutation(late)
    pe = torch.tensor(~perm, device=dev, dtype=torch.float32)[None]
    pl = torch.tensor(perm, device=dev, dtype=torch.float32)[None]
    c_ = np.where(enough, (torch.abs(corr(pl)) - torch.abs(corr(pe)))[0].cpu().numpy(), np.nan)
    null.append(int(np.sum(np.abs(c_) >= np.nanpercentile(np.abs(change), 90))))
obs = int(np.sum(np.abs(change) >= np.nanpercentile(np.abs(change), 90)))
# sliding windows
Wn = int(P["window"])
win_rows = []
for s0 in range(0, Y - Wn + 1):
    w = torch.zeros(1, Y, device=dev)
    w[0, s0:s0 + Wn] = 1
    r = corr(w)[0].cpu().numpy()
    win_rows.append(r)
WR = np.stack(win_rows, 1)                                           # [C, nW]
import pandas as pd

mp = pd.DataFrame({"lat": lat, "lon": lon, "r_early": r_e.cpu().numpy(), "r_late": r_l.cpu().numpy(),
                   "change_abs_r": change, "ci_lo": lo, "ci_hi": hi, "change_without_strong_enso": ch_nostrong,
                   "change_amplitude_matched": ch_matched, "significant": sig, "robust": robust})
mp.round(4).to_csv("out/tele_map.csv", index=False)
centres = uy[: Y - Wn + 1] + Wn // 2
pd.DataFrame(WR.round(3), columns=[f"w{int(c)}" for c in centres]).assign(lat=lat, lon=lon).to_csv(
    "out/tele_windows.csv", index=False)
blk = pd.Series(np.floor(lat / 10) * 10).astype(int).astype(str) + "/" + pd.Series(np.floor(lon / 10) * 10).astype(int).astype(str)
bt = mp.assign(block=blk.values).groupby("block").agg(cells=("lat", "size"), mean_change=("change_abs_r", "mean"),
                                                       robust_frac=("robust", "mean")).reset_index()
bt = bt[bt.cells >= 4].sort_values("robust_frac", ascending=False)
summary = {"params": P, "cells": int(len(mp)), "years": [int(uy[0]), int(uy[-1])],
           "mean_abs_r_early": round(float(np.nanmean(np.abs(mp.r_early))), 3),
           "mean_abs_r_late": round(float(np.nanmean(np.abs(mp.r_late))), 3),
           "frac_significant_stronger": round(float(np.mean(sig & (change > 0))), 3),
           "frac_significant_weaker": round(float(np.mean(sig & (change < 0))), 3),
           "frac_robust_stronger": round(float(np.mean(robust & (change > 0))), 3),
           "frac_robust_weaker": round(float(np.mean(robust & (change < 0))), 3),
           "field_test_top_decile_cells": {"observed": obs, "null_p95": float(np.percentile(null, 95))},
           "regional_mean_window_r": dict(zip([int(c) for c in centres], np.round(np.nanmean(WR, 0), 3).tolist())),
           "top_blocks": bt.head(8).round(3).values.tolist(),
           "note": "robust = bootstrap CI excludes 0 AND same sign without strong ENSO years AND with amplitude-matched "
                   "periods (keeps >= half the change). Field test: a random split of the years produces this many "
                   "top-decile cells at the 95th percentile."}
json.dump(summary, open("out/tele_summary.json", "w"), indent=1, default=float)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col, t in ((ax[0], "r_early", f"r early (<{P['split_year']})"), (ax[1], "r_late", "r late")):
        s = a.scatter(mp.lon, mp.lat, c=mp[col], s=7, marker="s", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        a.set_title(f"{P['driver']} (lag {P['lag']}) vs {vn[tv]} {season}: {t}", fontsize=9); a.set_aspect("equal")
        fig.colorbar(s, ax=a, shrink=0.7)
    lim = float(np.nanpercentile(np.abs(mp.change_abs_r), 98)) or 0.1
    s = ax[2].scatter(mp.lon, mp.lat, c=mp.change_abs_r, s=7, marker="s", cmap="PuOr_r", vmin=-lim, vmax=lim)
    rb = mp[mp.robust]
    ax[2].scatter(rb.lon, rb.lat, s=2, c="k")
    ax[2].set_title("change in |r| (black dots: robust)", fontsize=9); ax[2].set_aspect("equal")
    fig.colorbar(s, ax=ax[2], shrink=0.7)
    fig.tight_layout(); fig.savefig("out/tele_map.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("plot failed", e)
print(json.dumps({k: v for k, v in summary.items() if k != "regional_mean_window_r"}, indent=1, default=float))
