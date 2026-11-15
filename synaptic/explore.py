"""Code exploration for time-holdout campaigns: what the data up to the cut year show, before any hypothesis.

water_balance_sweep: for every block of Africa (or the campaign region), GRACE seasonal storage gains and drawdowns
compared with rain minus evaporation over the same months, on the EXPLORATION years only. For each block and season
it fits gain = a + b*rain + c*evaporation and reports the trend of what the climate does not explain (Sen slope,
Mann-Kendall with carry-over). A residual that was already drifting before the cut is a prediction for the unseen
years: the drift should continue, so the later years should depart from the early model in the same direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import synlib


def _block_monthly(df: pd.DataFrame, block: float) -> pd.DataFrame:
    d = df[["lat", "lon", "year", "month", "value"]].copy()
    d["bl"] = (np.floor(d.lat / block) * block + block / 2).round(2)
    d["bo"] = (np.floor(d.lon / block) * block + block / 2).round(2)
    w = np.cos(np.radians(d.lat))
    d["wv"] = d.value * w
    d["w"] = w.where(d.value.notna(), 0)
    g = d.groupby(["bl", "bo", "year", "month"])[["wv", "w"]].sum()
    return (g.wv / g.w.replace(0, np.nan)).rename("value").reset_index()


def _window(fm: int, tm: int, y: int):
    frm = ((y - 1) if fm > tm else y, fm)
    to = (y, tm)
    win, cur = [], frm
    while cur < to:
        cur = (cur[0] + 1, 1) if cur[1] == 12 else (cur[0], cur[1] + 1)
        win.append(cur)
    return frm, to, win


def water_balance_sweep(data: dict, explore_end: int, block: float = 4.0, top: int = 25, log=None) -> str:
    need = ("GRACE_TWS", "MERRA2_PRECIP", "GLDAS_EVAP")
    if not all(k in data for k in need):
        return f"Exploration sweep not run: needs {need}"
    tabs = {}
    for k in need:
        d = data[k]
        d = d[d.year <= explore_end] if k == "GRACE_TWS" else d
        tabs[k] = _block_monthly(d, block).set_index(["bl", "bo", "year", "month"]).value
    g = tabs["GRACE_TWS"]
    rows = []
    for (bl, bo), s in g.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        if s.notna().sum() < 100:
            continue
        clim = s.groupby(level=1).mean()
        mx, mn = int(clim.idxmax()), int(clim.idxmin())
        if clim.max() - clim.min() < 1.0:  # no seasonal cycle worth testing (hyper-arid)
            continue
        for season, (fm, tm) in (("wet-season gain", (mn, mx)), ("dry-season drawdown", (mx, mn))):
            ys, gains, P, E = [], [], [], []
            for y in range(2003, explore_end + 1):
                frm, to, win = _window(fm, tm, y)
                a, b = s.get(frm, np.nan), s.get(to, np.nan)
                try:
                    p = np.mean([tabs["MERRA2_PRECIP"].get((bl, bo) + k, np.nan) for k in win])
                    e = np.mean([tabs["GLDAS_EVAP"].get((bl, bo) + k, np.nan) for k in win])
                except Exception:  # noqa: BLE001
                    p = e = np.nan
                if np.isfinite([a, b, p, e]).all():
                    ys.append(y), gains.append(b - a), P.append(p), E.append(e)
            if len(ys) < 9:
                continue
            ys, gains = np.array(ys, float), np.array(gains)
            A = np.column_stack([np.ones(len(ys)), P, E])
            coef, *_ = np.linalg.lstsq(A, gains, rcond=None)
            res = gains - A @ coef
            mk = synlib.mann_kendall(res)
            slope = synlib.sen_slope(res, ys) * 10
            rows.append({"lat": bl, "lon": bo, "season": season, "from_month": fm, "to_month": tm, "n": len(ys),
                         "slope_cm_per_decade": round(float(slope), 2), "p": float(mk["p"]),
                         "mean_gain_cm": round(float(np.mean(gains)), 1)})
    if not rows:
        return "Exploration sweep: no blocks with enough data"
    t = pd.DataFrame(rows).sort_values("p")
    n_tests = len(t)
    try:
        from .verify import place_name
        t["place"] = [place_name(a, b) for a, b in zip(t.lat, t.lon)]
    except Exception:  # noqa: BLE001
        t["place"] = ""
    head = (f"{n_tests} block-season tests ({block:g}-degree blocks, 2003-{explore_end}); with no real drifts about "
            f"{n_tests * 0.05:.0f} would have p < 0.05 by chance; observed {int((t.p < 0.05).sum())}, "
            f"{int((t.p < 0.01).sum())} with p < 0.01.")
    lines = [head, "Strongest drifts in storage gain beyond rain minus evaporation (exploration years only). "
             "box = [lat-2, lat+2, lon-2, lon+2]; a drift should CONTINUE after the cut, so its direction is the "
             "prediction for the later years:"]
    for _, r in t.head(top).iterrows():
        lines.append(f"- block {r.lat:+.0f}N {r.lon:+.0f}E ({r.place}): {r.season} from month {r.from_month} to "
                     f"{r.to_month}, mean gain {r.mean_gain_cm:+.1f} cm, residual drift {r.slope_cm_per_decade:+.2f} "
                     f"cm/decade, p = {r.p:.3g} ({r.n} seasons)")
    if log:
        log(f"exploration sweep: {n_tests} block-seasons, {int((t.p < 0.01).sum())} drifts with p < 0.01")
    return "\n".join(lines) + "\n\n" + level_trend_sweep(tabs, explore_end, top=top, log=log)


def level_trend_sweep(tabs: dict, explore_end: int, top: int = 25, log=None) -> str:
    """Annual storage LEVEL minus what accumulated rain minus evaporation explains: a trend in what is left is
    water gained or lost by something the climate does not contain (pumping, reservoirs, lakes, irrigation)."""
    def annual(k):  # tabs hold monthly block series indexed (bl, bo, year, month)
        x = tabs[k]
        if isinstance(x, pd.DataFrame):
            x = x.set_index(["bl", "bo", "year", "month"]).value
        return x.groupby(level=[0, 1, 2]).mean()
    g, P, E = annual("GRACE_TWS").rename("g"), annual("MERRA2_PRECIP").rename("p"), annual("GLDAS_EVAP").rename("e")
    m = pd.concat([g, P, E], axis=1).dropna().reset_index()
    rows = []
    for (bl, bo), s in m.groupby(["bl", "bo"]):
        s = s[(s.year >= 2003) & (s.year <= explore_end)].sort_values("year")
        if len(s) < 10:
            continue
        x = (s.p - s.e).values
        acc = np.cumsum(x - x.mean())
        A = np.column_stack([np.ones(len(s)), acc])
        c, *_ = np.linalg.lstsq(A, s.g.values, rcond=None)
        r = s.g.values - A @ c
        mk = synlib.mann_kendall(r)
        rows.append({"lat": bl, "lon": bo, "slope": float(synlib.sen_slope(r, s.year.values.astype(float)) * 10),
                     "p": float(mk["p"]), "n": len(s)})
    if not rows:
        return "Level-trend sweep: no blocks"
    t = pd.DataFrame(rows).sort_values("p")
    try:
        from .verify import place_name
        t["place"] = [place_name(a, b) for a, b in zip(t.lat, t.lon)]
    except Exception:  # noqa: BLE001
        t["place"] = ""
    n = len(t)
    lines = [f"Storage-level trend beyond accumulated rain minus evaporation, {n} land blocks, 2003-{explore_end}: "
             f"{int((t.p < 0.05).sum())} with p < 0.05 (about {n * 0.05:.0f} expected by chance), "
             f"{int((t.p < 0.001).sum())} with p < 0.001. Confirm a block with kind 'residual', driver "
             f"['MERRA2_PRECIP', 'GLDAS_EVAP'], accumulate true, test 'later', after {explore_end}, direction = the "
             "sign of the trend (a trend that continues puts the later years beyond the early model):"]
    for _, r in t.head(top).iterrows():
        lines.append(f"- block {r.lat:+.0f}N {r.lon:+.0f}E ({r.place}): {r.slope:+.2f} cm/decade beyond the climate, "
                     f"p = {r.p:.2g} ({r.n} years)")
    if log:
        log(f"level-trend sweep: {n} blocks, {int((t.p < 0.001).sum())} with p < 0.001")
    return "\n".join(lines)
