# Synaptic GPU template: "predictability" -- how far ahead can a season be forecast from ocean drivers and land
# memory, where, and has that changed?  For every cell: predict the target season's mean anomaly from predictors
# observed `lead` months before the season starts (driver indices + land-memory variables in that cell). Two models,
# both scored strictly out of sample (leave-one-year-out cross-validation): per-cell ridge regression (closed form,
# batched on the GPU) and one shared neural network across all cells (cell embedding), cross-fitted by year folds.
# Skill = correlation between forecast and outcome, for the early and the late years separately.
#
# PARAMS: target, season (target months, e.g. [3,4,5]), lead (months, default 3), predictors (land variables,
#   default: target + root-zone soil moisture if present), drivers (default all), split_year (2003), ridge (1.0)
# Input: inputs/analysis_input.npz (prepared automatically by gpu_submit)
# Output: out/pred_map.csv, out/pred_summary.json, out/pred_map.png
import json
import os

import numpy as np

P = {"target": None, "season": [3, 4, 5], "lead": 3, "predictors": None, "drivers": None, "split_year": 2003,
     "ridge": 1.0, "epochs": 200, "seed": 0}
P.update({k: v for k, v in globals().get("PARAMS", {}).items() if v is not None})
if not isinstance(P.get("season"), (list, tuple)) or not all(str(m).isdigit() for m in P.get("season") or []):
    P["season"] = None  # "all", "annual" or anything else that is not a month list means every month
else:
    P["season"] = [int(m) for m in P["season"]]
os.makedirs("out", exist_ok=True)
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(P["seed"])
D = np.load("inputs/analysis_input.npz", allow_pickle=True)
X, I = D["X"], D["I"]
vn, dn = [str(v) for v in D["var_names"]], [str(d) for d in D["driver_names"]]
lat, lon, years, months = D["lat"], D["lon"], D["years"].astype(int), D["months"].astype(int)
tv = vn.index(P["target"]) if P["target"] in vn else 0
preds = [p for p in (P["predictors"] or []) if p in vn] or list(dict.fromkeys([vn[tv]] + [v for v in vn if v.startswith("GWET")]))
preds = [p for p in preds if p in vn]
drv = [d for d in (P["drivers"] or [x for x in (P["predictors"] or []) if x in dn] or dn) if d in dn]
season = sorted(P["season"] or list(range(1, 13)))
C = X.shape[0]
uy = np.unique(years)
rows_y, feats, targ = [], [], []
for y in uy:
    sidx = np.flatnonzero((years == y) & np.isin(months, season))
    if len(sidx) < len(season):
        continue
    init = sidx[0] - int(P["lead"])
    if init < 0:
        continue
    with np.errstate(all="ignore"):
        targ.append(np.nanmean(X[:, tv, sidx], 1))
    f_land = np.stack([X[:, vn.index(p), init] for p in preds], 1)              # [C, nl]
    f_drv = np.repeat(np.array([I[dn.index(d), init] for d in drv])[None], C, 0)  # [C, nd]
    feats.append(np.concatenate([f_land, f_drv], 1))
    rows_y.append(y)
Fy = np.stack(feats, 1)        # [C, Y, F]
Ty = np.stack(targ, 1)         # [C, Y]
yy = np.array(rows_y)
Y, Fn = len(yy), Fy.shape[-1]
late = yy >= P["split_year"]
ok = np.isfinite(Ty) & np.isfinite(Fy).all(-1)
F_ = torch.tensor(np.nan_to_num(Fy), device=dev, dtype=torch.float32)
T_ = torch.tensor(np.nan_to_num(Ty), device=dev, dtype=torch.float32)
M_ = torch.tensor(ok, device=dev, dtype=torch.float32)

# per-cell ridge, leave-one-year-out (batched over cells)
pred_r = torch.zeros(C, Y, device=dev)
eye = torch.eye(Fn + 1, device=dev)
A = torch.cat([F_, torch.ones(C, Y, 1, device=dev)], -1)
for k in range(Y):
    w = M_.clone()
    w[:, k] = 0
    Aw = A * w[..., None]
    G = Aw.transpose(1, 2) @ A + P["ridge"] * eye
    b = Aw.transpose(1, 2) @ T_[..., None]
    beta = torch.linalg.solve(G, b)[..., 0]
    pred_r[:, k] = (A[:, k] * beta).sum(-1)
pred_r = pred_r.cpu().numpy()

# shared network across cells, cross-fitted over 5 year folds
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(C, 8)
        self.mlp = nn.Sequential(nn.Linear(Fn + 8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, f, c):
        return self.mlp(torch.cat([f, self.emb(c)], -1))[..., 0]


pred_n = np.full((C, Y), np.nan, np.float32)
cid = torch.arange(C, device=dev)[:, None].expand(C, Y)
for fold in range(5):
    test = torch.tensor((np.arange(Y) % 5) == fold, device=dev)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    wtr = M_ * (~test)[None].float()
    for ep in range(int(P["epochs"]) if dev == "cuda" else 30):
        out = net(F_, cid)
        loss = (((out - T_) ** 2) * wtr).sum() / wtr.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = net(F_, cid).cpu().numpy()
    pred_n[:, test.cpu().numpy()] = out[:, test.cpu().numpy()]


def skill(pred, sel):
    r = np.full(C, np.nan)
    for c in range(C):
        m = ok[c] & sel & np.isfinite(pred[c])
        if m.sum() >= 8:
            a, b = pred[c, m], Ty[c, m]
            if a.std() > 1e-9 and b.std() > 1e-9:
                r[c] = np.corrcoef(a, b)[0, 1]
    return r


import pandas as pd

mp = pd.DataFrame({"lat": lat, "lon": lon, "ridge_skill_early": skill(pred_r, ~late),
                   "ridge_skill_late": skill(pred_r, late), "nn_skill_early": skill(pred_n, ~late),
                   "nn_skill_late": skill(pred_n, late), "ridge_skill_all": skill(pred_r, np.ones(Y, bool))})
mp["skill_change"] = mp.ridge_skill_late - mp.ridge_skill_early
mp.round(4).to_csv("out/pred_map.csv", index=False)
blk = (np.floor(lat / 10) * 10).astype(int).astype(str).astype(object) + "/" + \
      (np.floor(lon / 10) * 10).astype(int).astype(str).astype(object)
bt = mp.assign(block=blk).groupby("block").agg(cells=("lat", "size"), skill_all=("ridge_skill_all", "mean"),
                                                change=("skill_change", "mean")).reset_index()
bt = bt[bt.cells >= 4]
res = {"params": P, "target": vn[tv], "season": season, "lead_months": P["lead"], "predictors": preds, "drivers": drv,
       "cells": int(C), "years": [int(yy[0]), int(yy[-1])],
       "mean_skill": {k: round(float(np.nanmean(mp[k])), 3) for k in mp.columns if "skill" in k},
       "frac_cells_skill_above_0.5": round(float(np.mean(mp.ridge_skill_all > 0.5)), 3),
       "most_predictable_blocks": bt.sort_values("skill_all", ascending=False).head(8).round(3).values.tolist(),
       "largest_skill_gain_blocks": bt.sort_values("change", ascending=False).head(5).round(3).values.tolist(),
       "largest_skill_loss_blocks": bt.sort_values("change").head(5).round(3).values.tolist(),
       "note": "Leave-one-year-out ridge per cell and a 5-fold cross-fitted shared network; skill = r(forecast, "
               "outcome) out of sample. With ~20 years per period, |r| < ~0.45 is within sampling noise for one cell; "
               "trust coherent blocks, not single cells."}
json.dump(res, open("out/pred_summary.json", "w"), indent=1, default=float)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, col in zip(ax, ("ridge_skill_all", "ridge_skill_early", "ridge_skill_late")):
        s = a.scatter(mp.lon, mp.lat, c=mp[col], s=7, marker="s", cmap="viridis", vmin=0, vmax=0.8)
        a.set_title(f"{vn[tv]} {season} at lead {P['lead']}: {col}", fontsize=9); a.set_aspect("equal")
        fig.colorbar(s, ax=a, shrink=0.7)
    fig.tight_layout(); fig.savefig("out/pred_map.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("plot failed", e)
print(json.dumps(res, indent=1, default=float)[:4000])
