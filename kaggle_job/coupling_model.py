# Synaptic GPU template: "coupling" model, v2 (cross-fitted, with significance and the sample-effect control).
# One neural network learns how every grid cell's climate evolves month to month: each variable's next-month
# anomaly from the recent history of ALL variables in the cell and of ocean/atmosphere drivers (ENSO, Indian Ocean
# Dipole, Atlantic, PDO, NAO). The model is CROSS-FITTED: the years are split into K folds, K models are trained,
# each scored only on the years it never saw, so every year gets an out-of-sample estimate. For each year, cell,
# target and input, the influence of the input = the loss of skill when that input is scrambled.
#
# What v2 adds over a single early/late split:
#   * influence for EVERY year (not 9 held-out years) -> sliding 15-year windows and thirds of the record;
#   * block bootstrap over years -> confidence intervals on the change of each driver -> target influence;
#   * the sample-effect control: a driver looks more influential in a period with more strong events. Each yearly
#     influence is regressed on [late period, |driver amplitude that year|]; the "late" coefficient is the change
#     AFTER accounting for how strong the events were. A change that vanishes under this control is a sample
#     effect, not a changed teleconnection;
#   * 10 x 10 degree block statistics (where the change is concentrated) and out-of-fold surprise.
#
# Input: inputs/coupling_input.npz with X [cells, vars, months] standardized monthly anomalies (NaN = untrusted or
# missing), I [drivers, months] standardized indices, lat, lon, var_names, driver_names, years, months.
import json
import os
import time

import numpy as np

P = {"split_year": 2003, "lags": 6, "epochs": 30, "hidden": 128, "emb": 8, "batch": 4096, "lr": 1e-3,
     "weight_decay": 1e-3, "patience": 4, "folds": 5, "n_boot": 300, "window": 15, "seed": 0}
P.update(globals().get("PARAMS", {}))
os.makedirs("out", exist_ok=True)

import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()
if dev == "cpu":
    P["epochs"], P["folds"] = min(P["epochs"], 3), min(P["folds"], 2)
print(f"device={dev} gpus={n_gpu} params={P}", flush=True)
torch.manual_seed(P["seed"])
rng = np.random.default_rng(P["seed"])

D = np.load("inputs/coupling_input.npz", allow_pickle=True)
X, I = D["X"].astype(np.float32), D["I"].astype(np.float32)
lat, lon = D["lat"], D["lon"]
vnames, dnames = [str(v) for v in D["var_names"]], [str(d) for d in D["driver_names"]]
years, months = D["years"].astype(int), D["months"].astype(int)
C, V, T = X.shape
K, L = I.shape[0], int(P["lags"])
uy = np.unique(years)
Y = len(uy)
print(f"cells={C} vars={V} months={T} drivers={K} years={Y}", flush=True)

Xt = torch.tensor(np.nan_to_num(X), device=dev)
Mt = torch.tensor(~np.isnan(X), device=dev)
It = torch.tensor(np.nan_to_num(I), device=dev)
mon = torch.tensor(months - 1, device=dev)
t_all = np.arange(L, T)
yr = years[t_all]


def features(cells, ts, perm_var=None, perm_drv=None, perm_src=None, perm_ts=None):
    lag_idx = ts[:, None] - torch.arange(1, L + 1, device=dev)[None, :]
    hist = Xt[cells[:, None], :, lag_idx]
    drv = It[:, lag_idx].permute(1, 2, 0)
    if perm_var is not None:
        hist = hist.clone()
        hist[:, :, perm_var] = hist[perm_src][:, :, perm_var]
    if perm_drv is not None:
        drv = drv.clone()
        lag_p = perm_ts[:, None] - torch.arange(1, L + 1, device=dev)[None, :]
        drv[:, :, perm_drv] = It[perm_drv][lag_p]
    onehot = torch.nn.functional.one_hot(mon[ts], 12).float()
    return torch.cat([hist.reshape(len(ts), -1), drv.reshape(len(ts), -1), onehot], 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(C, P["emb"])
        nin = V * L + K * L + 12 + P["emb"]
        h = P["hidden"]
        self.mlp = nn.Sequential(nn.Linear(nin, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, V))

    def forward(self, f, cells):
        return self.mlp(torch.cat([f, self.emb(cells)], 1))


inputs = [(f"hist:{v}", ("var", j)) for j, v in enumerate(vnames)] + [(d, ("drv", j)) for j, d in enumerate(dnames)]
NI = len(inputs)
SEb = np.zeros((C, V, Y), np.float32)          # out-of-fold squared error, base model, per year
SEp = np.zeros((NI, C, V, Y), np.float32)      # ... with one input scrambled
NN = np.zeros((C, V, Y), np.float32)
RES = np.full((C, V, T), np.nan, np.float32)   # out-of-fold absolute residuals
t0 = time.time()
yidx = {y: i for i, y in enumerate(uy)}

for fold in range(P["folds"]):
    test = (yr % P["folds"]) == fold
    val = ((yr % 7) == 3) & ~test
    train_t, val_t, test_t = t_all[~test & ~val], t_all[val], t_all[test]
    net = Net().to(dev)
    model = nn.DataParallel(net) if n_gpu > 1 else net
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])

    @torch.no_grad()
    def val_mse():
        model.eval()
        se, n = 0.0, 0.0
        for t in val_t:
            cells = torch.arange(C, device=dev)
            ts = torch.full((C,), int(t), device=dev)
            pred = model(features(cells, ts), cells)
            m = Mt[:, :, t]
            se += float((((pred - Xt[:, :, t]) ** 2) * m).sum())
            n += float(m.sum())
        model.train()
        return se / max(n, 1)

    best, best_state, bad = 1e9, None, 0
    pairs = np.array([(c, t) for t in train_t for c in range(C)], dtype=np.int64)
    for ep in range(P["epochs"]):
        rng.shuffle(pairs)
        tot, nb = 0.0, 0
        for k in range(0, len(pairs), P["batch"]):
            b = torch.tensor(pairs[k:k + P["batch"]], device=dev)
            cells, ts = b[:, 0], b[:, 1]
            y, m = Xt[cells, :, ts], Mt[cells, :, ts]
            pred = model(features(cells, ts), cells)
            loss = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        v = val_mse()
        print(f"fold {fold} epoch {ep} train {tot / max(nb, 1):.4f} val {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= P["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(P["seed"] + fold)
    with torch.no_grad():
        for t in test_t:
            yi = yidx[int(years[t])]
            cells = torch.arange(C, device=dev)
            ts = torch.full((C,), int(t), device=dev)
            y_, m_ = Xt[:, :, t], Mt[:, :, t]
            pred = model(features(cells, ts), cells)
            SEb[:, :, yi] += (((pred - y_) ** 2) * m_).cpu().numpy()
            NN[:, :, yi] += m_.float().cpu().numpy()
            RES[:, :, t] = np.where(m_.cpu().numpy(), (pred - y_).abs().cpu().numpy(), np.nan)
            src = torch.randperm(C, generator=g).to(dev)
            same_month = t_all[(t_all % 12 == t % 12) & (np.abs(t_all - t) > 24)]
            tp = int(same_month[torch.randint(len(same_month), (1,), generator=g).item()])
            pts = torch.full((C,), tp, device=dev)
            for ii, (name, (kind, j)) in enumerate(inputs):
                pr = model(features(cells, ts, j if kind == "var" else None, j if kind == "drv" else None, src, pts),
                           cells)
                SEp[ii, :, :, yi] += (((pr - y_) ** 2) * m_).cpu().numpy()
    print(f"fold {fold} scored {len(test_t)} months ({time.time() - t0:.0f}s)", flush=True)

import pandas as pd

with np.errstate(all="ignore"):
    ok = NN >= 3
    base_y = np.where(ok, SEb / NN, np.nan)                                   # [C, V, Y]
    imp_y = np.where(ok[None], SEp / NN[None], np.nan) - base_y[None]          # [NI, C, V, Y]
late = uy >= P["split_year"]
third = np.digitize(uy, np.quantile(uy, [1 / 3, 2 / 3]))


def nanmean(a, axis):
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=axis)


early_imp, late_imp = nanmean(imp_y[..., ~late], -1), nanmean(imp_y[..., late], -1)
# a cell whose record the trust mask cut in one period is not comparable across periods: need 8+ years in each
n_e, n_l = np.isfinite(imp_y[..., ~late]).sum(-1), np.isfinite(imp_y[..., late]).sum(-1)
early_imp = np.where(n_e >= 8, early_imp, np.nan)
late_imp = np.where(n_l >= 8, late_imp, np.nan)
thirds = [nanmean(imp_y[..., third == k], -1) for k in range(3)]
skill = np.stack([1 - nanmean(base_y[..., ~late], -1), 1 - nanmean(base_y[..., late], -1)], -1)  # [C, V, 2]

# sample-effect control: regress yearly influence on [1, late, amplitude of the driver that year]
amp = np.zeros((NI, Y), np.float32)
for ii, (name, (kind, j)) in enumerate(inputs):
    if kind == "drv":
        amp[ii] = [np.abs(I[j, years == y]).mean() for y in uy]
ctrl = np.full((NI, C, V), np.nan, np.float32)
for ii, (name, (kind, j)) in enumerate(inputs):
    if kind != "drv":
        continue
    Xd = np.stack([np.ones(Y), late.astype(float), (amp[ii] - amp[ii].mean()) / (amp[ii].std() + 1e-6)], 1)  # [Y,3]
    y_ = imp_y[ii]                                                                   # [C, V, Y]
    w = np.isfinite(y_).astype(float)
    y0 = np.nan_to_num(y_)
    XtWX = np.einsum("cvy,ya,yb->cvab", w, Xd, Xd) + 1e-6 * np.eye(3)
    XtWy = np.einsum("cvy,ya,cvy->cva", w, Xd, y0)
    beta = np.linalg.solve(XtWX, XtWy[..., None])[..., 0]
    ctrl[ii] = np.where(w.sum(-1) >= 12, beta[..., 1], np.nan)

rows = []
for ii, (name, _) in enumerate(inputs):
    for v in range(V):
        e, l_, c_ = early_imp[ii, :, v], late_imp[ii, :, v], ctrl[ii, :, v]
        t1, t2, t3 = thirds[0][ii, :, v], thirds[1][ii, :, v], thirds[2][ii, :, v]
        good = np.isfinite(e) & np.isfinite(l_)
        for c in np.flatnonzero(good):
            rows.append((float(lat[c]), float(lon[c]), vnames[v], name, round(float(e[c]), 4), round(float(l_[c]), 4),
                         None if not np.isfinite(c_[c]) else round(float(c_[c]), 4),
                         round(float(t1[c]), 4), round(float(t2[c]), 4), round(float(t3[c]), 4)))
imp = pd.DataFrame(rows, columns=["lat", "lon", "target", "driver", "early", "late", "change_ctrl", "third1",
                                  "third2", "third3"])
imp["change"] = imp["late"] - imp["early"]
imp.to_csv("out/importance.csv", index=False)
sk = pd.DataFrame([(float(lat[c]), float(lon[c]), vnames[v], float(skill[c, v, 0]), float(skill[c, v, 1]))
                   for c in range(C) for v in range(V)], columns=["lat", "lon", "variable", "skill_early", "skill_late"])
sk.to_csv("out/skill.csv", index=False)

# surprise per cell-year from out-of-fold residuals
top = []
for v in range(V):
    for y in uy:
        with np.errstate(all="ignore"):
            e = np.nanmean(RES[:, v, years == y], 1)
        z = (e - np.nanmean(e)) / (np.nanstd(e) + 1e-6)
        for c in np.flatnonzero(z > 4):
            top.append((float(lat[c]), float(lon[c]), vnames[v], int(y), float(e[c]), float(z[c])))
pd.DataFrame(top, columns=["lat", "lon", "variable", "year", "mean_abs_error", "z"]).to_csv(
    "out/surprise_top.csv", index=False)

# region-level summary with bootstrap confidence and the sample-effect control
blk = (np.floor(lat / 10) * 10).astype(int).astype(str).astype(object) + "/" + \
      (np.floor(lon / 10) * 10).astype(int).astype(str).astype(object)
summ, block_rows = [], []
W = int(P["window"])
for ii, (name, (kind, j)) in enumerate(inputs):
    for v in range(V):
        e, l_ = early_imp[ii, :, v], late_imp[ii, :, v]
        strong = np.isfinite(e) & np.isfinite(l_) & ((np.abs(e) > 0.02) | (np.abs(l_) > 0.02))
        cs = np.flatnonzero(strong)
        if len(cs) < 10:
            continue
        ch = l_[cs] - e[cs]
        Yc = imp_y[ii, cs, v, :]                                       # [n, Y]
        ei, li = np.flatnonzero(~late), np.flatnonzero(late)
        boots_mean, boots_frac = [], []
        for _ in range(P["n_boot"]):
            be = rng.choice(ei, len(ei)); bl = rng.choice(li, len(li))
            with np.errstate(all="ignore"):
                d = np.nanmean(Yc[:, bl], 1) - np.nanmean(Yc[:, be], 1)
            boots_mean.append(np.nanmean(d)); boots_frac.append(np.nanmean(d > 0.02) - np.nanmean(d < -0.02))
        lo, hi = np.nanpercentile(boots_mean, [2.5, 97.5])
        cc = ctrl[ii, cs, v] if kind == "drv" else np.full(len(cs), np.nan)
        win = []
        for s0 in range(0, Y - W + 1):
            with np.errstate(all="ignore"):
                win.append(round(float(np.nanmean(Yc[:, s0:s0 + W])), 4))
        rec = {"target": vnames[v], "driver": name, "cells": int(len(cs)),
               "early": round(float(np.nanmean(e[cs])), 4), "late": round(float(np.nanmean(l_[cs])), 4),
               "frac_stronger": round(float(np.mean(ch > 0.02)), 2), "frac_weaker": round(float(np.mean(ch < -0.02)), 2),
               "mean_change": round(float(np.nanmean(ch)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
               "significant": bool(lo > 0 or hi < 0),
               "change_ctrl_mean": None if kind != "drv" else round(float(np.nanmean(cc)), 4),
               "ctrl_same_sign_frac": None if kind != "drv" else round(float(np.nanmean(np.sign(cc) == np.sign(ch))), 2),
               "window_years": [int(uy[0]), int(uy[-1]), W], "window_mean": win}
        rec["survives_sample_control"] = None if kind != "drv" else bool(
            rec["change_ctrl_mean"] is not None and np.sign(rec["change_ctrl_mean"]) == np.sign(rec["mean_change"])
            and abs(rec["change_ctrl_mean"]) >= 0.5 * abs(rec["mean_change"]))
        summ.append(rec)
        df = pd.DataFrame({"block": blk[cs], "change": ch})
        for b, gb in df.groupby("block"):
            if len(gb) >= 4:
                block_rows.append((vnames[v], name, b, len(gb), round(float(gb.change.mean()), 4),
                                   round(float(gb.change.mean() / (gb.change.std(ddof=1) / np.sqrt(len(gb)) + 1e-9)), 2)))
blocks = pd.DataFrame(block_rows, columns=["target", "driver", "block_lat/lon", "cells", "mean_change", "t"])
blocks.to_csv("out/blocks.csv", index=False)
rank = lambda s: 0 if not s["driver"].startswith("hist:") else (2 if s["driver"] == "hist:" + s["target"] else 1)
summ = sorted(summ, key=lambda s: (rank(s), not s["significant"], not bool(s.get("survives_sample_control", True)),
                                   -max(s["frac_stronger"], s["frac_weaker"]) * s["cells"]))
for s in summ[:12]:
    b = blocks[(blocks.target == s["target"]) & (blocks.driver == s["driver"])]
    s["top_blocks"] = b.reindex(b.t.abs().sort_values(ascending=False).index).head(5)[
        ["block_lat/lon", "cells", "mean_change", "t"]].values.tolist()
out = {"params": P, "cells": C, "variables": vnames, "drivers": dnames, "split_year": P["split_year"],
       "folds": P["folds"], "years": [int(uy[0]), int(uy[-1])],
       "skill_mean": {v: [round(float(np.nanmean(skill[:, j, 0])), 3), round(float(np.nanmean(skill[:, j, 1])), 3)]
                      for j, v in enumerate(vnames)},
       "top_coupling_changes": summ[:12], "seconds": round(time.time() - t0, 1)}
json.dump(out, open("out/summary.json", "w"), indent=1, default=float)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [s for s in summ if not s["driver"].startswith("hist:")][:4] or summ[:4]
    if shown:
        fig, axes = plt.subplots(2, len(shown), figsize=(4.6 * len(shown), 7.6), squeeze=False)
        for k, s in enumerate(shown):
            g = imp[(imp.target == s["target"]) & (imp.driver == s["driver"])]
            ax = axes[0, k]
            lim = float(np.nanpercentile(np.abs(g.change), 98)) or 0.05
            sc = ax.scatter(g.lon, g.lat, c=g.change, s=8, marker="s", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_aspect("equal")
            ax.set_title(f"{s['driver']} -> {s['target']}\nchange in influence (late - early)", fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.7)
            ax = axes[1, k]
            x = np.arange(len(s["window_mean"])) + uy[0] + W // 2
            ax.plot(x, s["window_mean"], color="k")
            ax.set_title(f"{W}-yr window mean influence; 95% CI of change {s['ci95']}"
                         + ("" if s["survives_sample_control"] in (None, True) else "\n(does NOT survive event control)"),
                         fontsize=8)
            ax.set_xlabel("window centre")
        fig.tight_layout()
        fig.savefig("out/coupling_change.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("plot failed", e)
print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=1, default=float)[:5000], flush=True)
