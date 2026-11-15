"""The GPU "learn" stage, run by the harness itself (not left to the agents' choice).

A neural network learns every cell's month-to-month climate from all variables and from ocean/atmosphere
drivers (kaggle_job/coupling_model.py), on trusted data only. It returns, per cell, how much each driver matters
early and late in the record. Coherent changes in a driver's influence become science leads.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

DRIVERS = ["NINO34", "DMI", "TNA", "PDO", "NAO"]
MASK_YEARS = {"bad_year", "propagated_bad_year", "known_period", "zero_run", "constant_run", "bad_month"}


def build_input(data: dict[str, pd.DataFrame], trust: pd.DataFrame | None, out: Path,
                drivers: list[str] | None = None) -> dict:
    from .tools.climate import load_index

    variables = sorted(data)
    base = data[variables[0]]
    y0, y1 = int(base.year.min()), int(base.year.max())
    years = np.repeat(np.arange(y0, y1 + 1), 12)
    months = np.tile(np.arange(1, 13), y1 - y0 + 1)
    cells = base[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    cidx = {(round(a, 4), round(b, 4)): i for i, (a, b) in enumerate(zip(cells.lat, cells.lon))}
    C, V, T = len(cells), len(variables), len(years)
    X = np.full((C, V, T), np.nan, dtype=np.float32)
    masked = {"cell_years": 0, "breaks_cut": 0}
    for j, v in enumerate(variables):
        d = data[v]
        ci = np.array([cidx.get((round(a, 4), round(b, 4)), -1) for a, b in zip(d.lat, d.lon)])
        ti = (d.year.values - y0) * 12 + d.month.values - 1
        ok = (ci >= 0) & (ti >= 0) & (ti < T)
        X[ci[ok], j, ti[ok]] = d.value.values[ok]
        if trust is not None and not trust.empty:
            t = trust[(trust.variable == v) & (trust.kind != "known_caveat")]
            for r in t.itertuples():
                c = cidx.get((round(r.lat, 4), round(r.lon, 4)))
                if c is None:
                    continue
                if "break" in str(r.kind) and not pd.isna(r.year_from):
                    yb = int(r.year_from)  # keep the longer homogeneous segment, drop the other side of the break
                    X[c, j, years < yb] = np.nan if (y1 - yb) >= (yb - y0) else X[c, j, years < yb]
                    if (y1 - yb) < (yb - y0):
                        X[c, j, years >= yb] = np.nan
                    masked["breaks_cut"] += 1
                elif str(r.kind) in MASK_YEARS and not pd.isna(r.year_from):
                    a = int(r.year_from)
                    b = int(r.year_to) if not pd.isna(r.year_to) else a
                    X[c, j, (years >= a) & (years <= b)] = np.nan
                    masked["cell_years"] += b - a + 1
        # standardized anomalies per cell and calendar month
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for m in range(1, 13):
            sel = months == m
            mu = np.nanmean(X[:, j, sel], axis=1, keepdims=True)
            sd = np.nanstd(X[:, j, sel], axis=1, keepdims=True)
            X[:, j, sel] = (X[:, j, sel] - mu) / np.where(sd > 1e-6, sd, np.nan)
    drivers = drivers or DRIVERS
    I = np.zeros((len(drivers), T), dtype=np.float32)
    used = []
    for k, n in enumerate(drivers):
        try:
            d = load_index(n).set_index(["year", "month"])["value"]
            s = np.array([d.get((y, m), np.nan) for y, m in zip(years, months)], dtype=float)
            s = (s - np.nanmean(s)) / (np.nanstd(s) or 1)
            I[k] = np.nan_to_num(s)
            used.append(n)
        except Exception:  # noqa: BLE001
            pass
    np.savez_compressed(out, X=X, I=I, lat=cells.lat.values, lon=cells.lon.values, var_names=np.array(variables),
                        driver_names=np.array(drivers), years=years, months=months)
    return {"cells": C, "variables": variables, "drivers": used, "months": T, **masked,
            "mb": round(out.stat().st_size / 1e6, 1)}


def run_learn(campaign, data: dict[str, pd.DataFrame], timeout_s: int = 3600) -> dict | None:
    """Build inputs, run the coupling model on the Kaggle GPU (local CPU fallback), fetch and summarise."""
    from .tools import gpu
    from .tools.base import ToolContext

    ldir = campaign.cdir / "learn"
    ldir.mkdir(exist_ok=True)
    trust = pd.read_csv(campaign.cdir / "trust.csv") if (campaign.cdir / "trust.csv").exists() else None
    inp = ldir / "coupling_input.npz"
    info = build_input(data, trust, inp)
    campaign.log(f"learn: input {info['cells']} cells x {len(info['variables'])} variables x {info['months']} months, "
                 f"drivers {info['drivers']}, {info['mb']} MB; submitting to the GPU")
    ctx = ToolContext(campaign.record, campaign.cid, None, None, ldir, campaign.cdir, campaign.mock)
    res = gpu.gpu_submit(ctx, template="coupling", inputs=[str(inp)],
                         params={"epochs": 3 if campaign.mock else 30})
    if res.is_error:
        campaign.log(f"learn: submit failed: {res.text[:200]}")
        return None
    job = res.meta["job"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = gpu.gpu_status(ctx, job, wait_seconds=60).text
        if "complete" in st or "error" in st or "cancel" in st:
            break
    fetched = gpu.gpu_fetch(ctx, job)
    outdir = ldir / f"gpu_{job}"
    summ = next(outdir.rglob("summary.json"), None)
    if not summ:
        campaign.log(f"learn: no results ({st}); log tail: {fetched.text[-300:]}")
        return None
    s = json.loads(summ.read_text())
    for f in outdir.rglob("*"):
        if f.is_file() and f.suffix in (".csv", ".png", ".json"):
            shutil.copy(f, ldir / f.name)
    s["job"] = job
    s["backend"] = res.meta.get("backend")
    s["minutes"] = round((time.time() - t0) / 60, 1)
    return s
