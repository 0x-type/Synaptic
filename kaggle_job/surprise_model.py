# Synaptic GPU template: "surprise" model.
# Learns the normal month-to-month behaviour of every grid cell (a small causal GRU predicting the
# next monthly anomaly with its own uncertainty), trained on the early record only. Then scores the
# later record: high negative log-likelihood = the cell is doing something its own past says is unlikely.
# This is the Earth-data analogue of the genomic language-model likelihood profiles in the paper (Fig. 5).
#
# PARAMS (injected above by the harness): train_end_year (int), context (months), epochs, hidden
import glob
import json
import os
import time

import numpy as np
import pandas as pd

P = {"train_end_year": 2005, "context": 24, "epochs": 40, "hidden": 64, "batch": 512, "lr": 2e-3}
P.update(globals().get("PARAMS", {}))
os.makedirs("out", exist_ok=True)

import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu = torch.cuda.device_count()
print(f"device={dev} gpus={n_gpu} params={P}", flush=True)
if dev == "cpu":
    P["epochs"] = min(P["epochs"], 4)


class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gru = nn.GRU(1, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, 2)

    def forward(self, x):
        y, _ = self.gru(x)
        mu, logv = self.head(y[:, -1]).unbind(-1)
        return mu, logv.clamp(-6, 4)


summary = {}
for path in sorted(glob.glob("inputs/*.csv")):
    df = pd.read_csv(path)
    df.loc[df["value"] <= -998, "value"] = np.nan
    for var, d in df.groupby("variable"):
        t0 = time.time()
        piv = d.pivot_table(index=["lat", "lon"], columns=["year", "month"], values="value")
        piv = piv.T.sort_index().T
        X = piv.values  # [cells, time]
        years = np.array([c[0] for c in piv.columns])
        months = np.array([c[1] for c in piv.columns])
        train = years <= P["train_end_year"]
        # deseasonalize + standardize per cell using the TRAINING period only (no leakage)
        clim = np.stack([np.nanmean(X[:, train & (months == m)], axis=1) for m in range(1, 13)], 1)
        A = X - clim[:, months - 1]
        sd = np.nanstd(A[:, train], axis=1, keepdims=True) + 1e-6
        Z = np.nan_to_num(A / sd)
        C = P["context"]
        idx_train = [t for t in range(C, Z.shape[1]) if train[t]]
        Zt = torch.tensor(Z, dtype=torch.float32, device=dev)

        def batch(ts, cells):
            xs = torch.stack([Zt[cells, t - C:t] for t in ts], 0).reshape(-1, C, 1)
            ys = torch.stack([Zt[cells, t] for t in ts], 0).reshape(-1)
            return xs, ys

        net = Net(P["hidden"]).to(dev)
        model = nn.DataParallel(net) if n_gpu > 1 else net
        opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=1e-4)
        cells = torch.arange(Z.shape[0], device=dev)
        nb = max(1, len(idx_train) * Z.shape[0] // P["batch"])
        for ep in range(P["epochs"]):
            tot = 0.0
            for _ in range(nb):
                ts = np.random.choice(idx_train, size=max(1, P["batch"] // Z.shape[0] + 1))
                xs, ys = batch(ts, cells)
                mu, logv = model(xs)
                loss = (0.5 * (logv + (ys - mu) ** 2 / logv.exp())).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            if ep % 10 == 0 or ep == P["epochs"] - 1:
                print(f"{var} epoch {ep} nll {tot / nb:.3f}", flush=True)
        # score every month after the context window
        model.eval()
        nll = np.full(Z.shape, np.nan)
        with torch.no_grad():
            for t in range(C, Z.shape[1]):
                xs, ys = batch([t], cells)
                mu, logv = model(xs)
                nll[:, t] = (0.5 * (logv + (ys - mu) ** 2 / logv.exp()) + 0.9189).cpu().numpy()
        base = np.nanmean(nll[:, train & (np.arange(len(years)) >= C)], axis=1, keepdims=True)
        excess = nll - base  # nats above the cell's own training-period surprise
        out = []
        for i, (la, lo) in enumerate(piv.index):
            for y in np.unique(years[~train]):
                out.append({"lat": la, "lon": lo, "year": int(y),
                            "surprise": float(np.nanmean(excess[i, years == y]))})
        o = pd.DataFrame(out)
        o.to_csv(f"out/surprise_{var}.csv", index=False)
        top = o.groupby(["lat", "lon"]).surprise.mean().sort_values(ascending=False).head(10)
        summary[var] = {"cells": int(Z.shape[0]), "seconds": round(time.time() - t0, 1),
                        "top_cells": [[float(a), float(b), round(float(s), 3)] for (a, b), s in top.items()],
                        "yearly_mean": o.groupby("year").surprise.mean().round(3).to_dict()}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = o.groupby(["lat", "lon"]).surprise.mean().reset_index()
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            sc = ax[0].scatter(m.lon, m.lat, c=m.surprise, cmap="magma", s=120, marker="s")
            ax[0].set_aspect("equal")
            ax[0].set_title(f"{var}: mean surprise after {P['train_end_year']} (nats)")
            fig.colorbar(sc, ax=ax[0])
            yr = o.groupby("year").surprise.mean()
            ax[1].plot(yr.index, yr.values, "o-")
            ax[1].axhline(0, color="k", lw=0.6)
            ax[1].set_title("Region-mean surprise by year")
            fig.tight_layout()
            fig.savefig(f"out/surprise_{var}.png", dpi=110)
        except Exception as e:  # noqa: BLE001
            print("plot failed", e)
json.dump(summary, open("out/summary.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "yearly_mean"} for k, v in summary.items()}, indent=1))
