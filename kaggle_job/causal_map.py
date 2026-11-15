# Synaptic GPU template: "causal" -- lagged causal discovery (conditional Granger / PCMCI-style screening) between
# drivers, land variables and a target, in every cell at once, early vs late in the record.
# For each cell and candidate source S, compare a model of target(t) on its own past (and the past of the other
# sources) with and without S's past (lags 1..max_lag). The F statistic and partial R^2 of S measure its direct,
# lagged influence given everything else; computed separately before and after split_year, with an FDR over cells.
# Batched least squares for all cells on the GPU.
#
# PARAMS: target, sources (list of variable or index names; default: all other variables + all drivers),
#   max_lag (default 3), split_year (2003), season (target months, default all)
# Input: inputs/analysis_input.npz (prepared automatically by gpu_submit)
# Output: out/causal_map.csv, out/causal_summary.json, out/causal_map.png
import json
import math
import os

import numpy as np

P = {"target": None, "sources": None, "max_lag": 3, "split_year": 2003, "season": None}
P.update({k: v for k, v in globals().get("PARAMS", {}).items() if v is not None})
if not isinstance(P.get("season"), (list, tuple)) or not all(str(m).isdigit() for m in P.get("season") or []):
    P["season"] = None  # "all", "annual" or anything else that is not a month list means every month
else:
    P["season"] = [int(m) for m in P["season"]]
os.makedirs("out", exist_ok=True)
import torch
from scipy import stats

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load("inputs/analysis_input.npz", allow_pickle=True)
X, I = D["X"], D["I"]
vn, dn = [str(v) for v in D["var_names"]], [str(d) for d in D["driver_names"]]
lat, lon, years, months = D["lat"], D["lon"], D["years"].astype(int), D["months"].astype(int)
tv = vn.index(P["target"]) if P["target"] in vn else 0
srcs = P["sources"] or [v for v in vn if v != vn[tv]] + dn
C, V, T = X.shape
L = int(P["max_lag"])
season = P["season"] or list(range(1, 13))


def series(name):
    if name in vn:
        return X[:, vn.index(name), :]
    return np.repeat(I[dn.index(name)][None, :], C, 0)


S = [series(s) for s in srcs]
Yt = X[:, tv, :]
ts = np.array([t for t in range(L, T) if months[t] in season])
rows = []


def design(tsel, drop=None):
    cols = [Yt[:, tsel - k] for k in range(1, L + 1)]
    for j, s in enumerate(S):
        if j == drop:
            continue
        cols += [s[:, tsel - k] for k in range(1, L + 1)]
    cols.append(np.ones((C, len(tsel))))
    return np.stack(cols, -1)                                            # [C, n, p]


def rss(A, y):
    """Residual sum of squares per cell by normal equations in float64 (masked months contribute nothing)."""
    m = np.isfinite(y) & np.isfinite(A).all(-1)
    A_ = torch.tensor(np.where(m[..., None], np.nan_to_num(A), 0.0), device=dev, dtype=torch.float64)
    y_ = torch.tensor(np.where(m, np.nan_to_num(y), 0.0), device=dev, dtype=torch.float64)
    G = A_.transpose(1, 2) @ A_
    lam = 1e-8 * torch.diagonal(G, dim1=1, dim2=2).mean(-1).clamp(min=1e-12)
    G = G + lam[:, None, None] * torch.eye(G.shape[-1], device=dev, dtype=torch.float64)
    beta = torch.linalg.solve(G, (A_.transpose(1, 2) @ y_[..., None]))
    r = (y_[..., None] - A_ @ beta)[..., 0]
    return (r ** 2).sum(-1).cpu().numpy(), m.sum(-1).astype(float)


out = {}
for per, tsel in (("early", ts[years[ts] < P["split_year"]]), ("late", ts[years[ts] >= P["split_year"]])):
    y = Yt[:, tsel]
    full = design(tsel)
    rss_f, n = rss(full, y)
    pf = full.shape[-1]
    for j, s in enumerate(srcs):
        rss_r, _ = rss(design(tsel, drop=j), y)
        df2 = np.maximum(n - pf, 1)
        F = ((rss_r - rss_f) / L) / np.maximum(rss_f / df2, 1e-12)
        p = stats.f.sf(F, L, df2)
        pr2 = (rss_r - rss_f) / np.maximum(rss_r, 1e-12)
        few = n < 0.6 * len(tsel)  # cells whose record was cut by the trust mask in this period: not comparable
        F, p, pr2 = np.where(few, np.nan, F), np.where(few, 1.0, p), np.where(few, np.nan, pr2)
        out[(per, s)] = (F, p, pr2)


def fdr(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r = np.empty_like(q)
    r[o] = np.minimum(q, 1)
    return r


import pandas as pd

recs, summ = [], []
for s in srcs:
    Fe, pe, re_ = out[("early", s)]
    Fl, pl, rl = out[("late", s)]
    qe, ql = fdr(np.nan_to_num(pe, nan=1)), fdr(np.nan_to_num(pl, nan=1))
    recs.append(pd.DataFrame({"lat": lat, "lon": lon, "source": s, "partial_r2_early": re_, "partial_r2_late": rl,
                              "q_early": qe, "q_late": ql}))
    summ.append({"source": s, "frac_cells_significant_early": round(float(np.mean(qe < 0.05)), 3),
                 "frac_cells_significant_late": round(float(np.mean(ql < 0.05)), 3),
                 "median_partial_r2_early": round(float(np.nanmedian(re_)), 4),
                 "median_partial_r2_late": round(float(np.nanmedian(rl)), 4),
                 "cells_compared": int(np.sum(np.isfinite(re_) & np.isfinite(rl))),
                 "cells_new_link": int(np.sum((qe >= 0.05) & (ql < 0.05))),
                 "cells_lost_link": int(np.sum((qe < 0.05) & (ql >= 0.05)))})
mp = pd.concat(recs, ignore_index=True)
mp.round(5).to_csv("out/causal_map.csv", index=False)
summ.sort(key=lambda d: -(d["cells_new_link"] + d["cells_lost_link"]))
res = {"params": P, "target": vn[tv], "sources": srcs, "cells": int(C), "summary": summ,
       "note": "Conditional Granger: each source's lags 1..max_lag added to a model with the target's own past and all "
               "other sources' past. q = BH-FDR over cells. Linear and lagged: a screening for direct influences, not "
               "proof of causation; confirm the strongest links physically."}
json.dump(res, open("out/causal_summary.json", "w"), indent=1, default=float)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = [d["source"] for d in summ[:4]]
    fig, ax = plt.subplots(2, len(top), figsize=(4.4 * len(top), 7), squeeze=False)
    for k, s in enumerate(top):
        g = mp[mp.source == s]
        for r, (col, q) in enumerate((("partial_r2_early", "q_early"), ("partial_r2_late", "q_late"))):
            a = ax[r, k]
            sc = a.scatter(g.lon, g.lat, c=g[col], s=6, marker="s", cmap="viridis", vmin=0,
                           vmax=float(np.nanpercentile(mp.partial_r2_late, 98)) or 0.1)
            sg = g[g[q] < 0.05]
            a.scatter(sg.lon, sg.lat, s=1, c="r")
            a.set_title(f"{s} -> {vn[tv]}: {col.split('_')[-1]} (red: q<0.05)", fontsize=8); a.set_aspect("equal")
            fig.colorbar(sc, ax=a, shrink=0.7)
    fig.tight_layout(); fig.savefig("out/causal_map.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("plot failed", e)
print(json.dumps(res, indent=1, default=float)[:4000])
