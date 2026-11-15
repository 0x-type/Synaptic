"""Climate drivers and science-stage tools: ocean/atmosphere indices (ENSO, IOD, ...) and recording findings."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .base import ToolContext, ToolResult

INDICES = {
    "NINO34": ("https://psl.noaa.gov/data/correlation/nina34.anom.data", "El Nino-Southern Oscillation: Nino-3.4 SST anomaly (C)"),
    "SOI": ("https://psl.noaa.gov/data/correlation/soi.data", "Southern Oscillation Index (pressure)"),
    "DMI": ("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data", "Indian Ocean Dipole Mode Index (C)"),
    "AMO": ("https://psl.noaa.gov/data/correlation/amon.us.data", "Atlantic Multidecadal Oscillation (C, detrended)"),
    "PDO": ("https://psl.noaa.gov/data/correlation/pdo.data", "Pacific Decadal Oscillation"),
    "NAO": ("https://psl.noaa.gov/data/correlation/nao.data", "North Atlantic Oscillation"),
    "TNA": ("https://psl.noaa.gov/data/correlation/tna.data", "Tropical North Atlantic SST anomaly (C)"),
    "QBO": ("https://psl.noaa.gov/data/correlation/qbo.data", "Quasi-Biennial Oscillation, 30 hPa equatorial wind (m/s)"),
}
_CACHE: dict[str, pd.DataFrame] = {}


def _parse_psl(text: str, name: str) -> pd.DataFrame:
    lines = text.strip().splitlines()
    y0, y1 = (int(float(x)) for x in lines[0].split()[:2])
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue
        try:
            y = int(parts[0])
        except ValueError:
            continue
        if not y0 <= y <= y1:
            continue
        for m, v in enumerate(parts[1:], 1):
            val = float(v)
            rows.append({"index": name, "year": y, "month": m, "value": np.nan if val <= -99 else val})
    return pd.DataFrame(rows)


def load_index(name: str) -> pd.DataFrame:
    name = name.upper()
    if name not in _CACHE:
        url = INDICES[name][0]
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                _CACHE[name] = _parse_psl(r.text, name)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        else:
            raise RuntimeError(f"could not download {name}")
    return _CACHE[name]


def climate_index(ctx: ToolContext, names: list[str] | str, start_year: int = 1981, end_year: int = 2025) -> ToolResult:
    """Monthly climate-driver indices (long format: index, year, month, value) for correlation, lag and
    sliding-window tests against regional series."""
    names = [names] if isinstance(names, str) else list(names)
    bad = [n for n in names if n.upper() not in INDICES]
    if bad:
        return ToolResult(f"Unknown index {bad}. Available: " + "; ".join(f"{k} ({v[1]})" for k, v in INDICES.items()),
                          is_error=True)
    frames, info = [], []
    for n in names:
        try:
            d = load_index(n)
        except Exception as e:  # noqa: BLE001
            info.append(f"{n.upper()}: download failed ({str(e)[:80]})")
            continue
        d = d[(d.year >= int(start_year)) & (d.year <= int(end_year))]
        frames.append(d)
        last = d.dropna().iloc[-1] if d["value"].notna().any() else None
        info.append(f"{n.upper()}: {INDICES[n.upper()][1]}; {d['value'].notna().sum()} months"
                    + (f", last {int(last.year)}-{int(last.month):02d}" if last is not None else ""))
    if not frames:
        return ToolResult("No index could be downloaded.\n" + "\n".join(info), is_error=True)
    df = pd.concat(frames, ignore_index=True)
    path = ctx.data_dir / f"indices_{'_'.join(sorted(n.upper() for n in names))}_{start_year}_{end_year}.csv"
    df.to_csv(path, index=False)
    return ToolResult(f"Saved {len(df):,} index-months to {path.name}.\n" + "\n".join(info) +
                      "\nTypical tests: seasonal correlation with lags (e.g. Sep-Nov DMI vs Oct-Dec rain), "
                      "sliding-window correlation to see whether a teleconnection changed, partial correlation "
                      "controlling for another index.", files=[str(path)])


def record_finding(ctx: ToolContext, family: str, statement: str, evidence: str, confidence: float,
                   novelty: str = "") -> ToolResult:
    """Science-stage result: one clear scientific statement about a real signal, with its evidence."""
    if len(statement.strip()) < 30 or len(evidence.strip()) < 60:
        return ToolResult("A finding needs a clear statement (one or two sentences) and the evidence (numbers, "
                          "tests, data used).", is_error=True)
    c = ctx.record.candidates(ctx.campaign_id, id=family)
    if not c:
        return ToolResult(f"Unknown family {family}", is_error=True)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.5
    body = f"{statement.strip()}\nEvidence: {evidence.strip()}\nConfidence: {conf:.2f}" + (
        f"\nNovelty: {novelty.strip()}" if novelty.strip() else "")
    ctx.record.kb_add(ctx.campaign_id, ctx.task_id, "finding", f"Science finding on {family}: {statement[:80]}",
                      body, ["science", family])
    note = (c[0]["note"] or "") + f" [science: {statement.strip()[:400]} (confidence {conf:.2f})]"
    ctx.record.update_candidate(ctx.campaign_id, family, note=note)
    ctx.state.setdefault("findings", []).append({"family": family, "statement": statement, "confidence": conf})
    return ToolResult(f"Finding recorded for {family}.")
