"""Agent access to the NASA catalogue (synaptic/catalog.py) through Earthdata Cloud, with the held-out rule:
datasets reserved for the pre-registered confirmation cannot be fetched by exploring agents."""
from __future__ import annotations

import json

from .base import ToolContext, ToolResult

EXPLORER_KINDS = ("hypothesis", "followup", "deep_dive", "science", "sweep", "classify", "census")


def held_out(ctx: ToolContext) -> set:
    row = ctx.record.one("SELECT brief FROM campaigns WHERE id=?", (ctx.campaign_id,))
    try:
        return set((json.loads(row["brief"]) if row else {}).get("holdout") or [])
    except Exception:  # noqa: BLE001
        return set()


def time_holdout(ctx: ToolContext) -> dict:
    """{"datasets": [...], "explore_end": year}: explorers see these datasets only up to explore_end; the later years
    are held out for the pre-registered confirmation."""
    row = ctx.record.one("SELECT brief FROM campaigns WHERE id=?", (ctx.campaign_id,))
    try:
        return (json.loads(row["brief"]) if row else {}).get("time_holdout") or {}
    except Exception:  # noqa: BLE001
        return {}


def refuse_if_held_out(ctx: ToolContext, dataset_ids: list[str]) -> ToolResult | None:
    task = ctx.record.task(ctx.task_id) if ctx.task_id else None
    if not task or task["kind"] not in EXPLORER_KINDS:
        return None  # referees and reports may use everything
    hit = sorted(set(dataset_ids) & held_out(ctx))
    if hit:
        return ToolResult(f"{', '.join(hit)} is HELD OUT for this campaign's pre-registered confirmation: exploring "
                          "agents cannot use it (that is what makes a confirmation independent). Use the campaign "
                          "datasets; if your claim holds, the harness tests it on the held-out data itself.",
                          is_error=True)
    return None


def nasa_data(ctx: ToolContext, dataset: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
              start_year: int, end_year: int, resolution: float = 1.0) -> ToolResult:
    from .. import catalog, cloud

    try:
        var = catalog.get(dataset)
    except KeyError:
        return ToolResult(f"Unknown dataset {dataset}. Catalogue: " +
                          ", ".join(f"{v.id} ({v.label})" for v in catalog.CATALOG), is_error=True)
    no = refuse_if_held_out(ctx, [dataset])
    if no:
        return no
    th = time_holdout(ctx)
    task = ctx.record.task(ctx.task_id) if ctx.task_id else None
    if dataset in (th.get("datasets") or []) and task and task["kind"] in EXPLORER_KINDS:
        cut = int(th["explore_end"])
        if int(start_year) > cut:
            return ToolResult(f"{dataset} after {cut} is HELD OUT for this campaign's confirmation (time holdout): "
                              f"explore {var.start}-{cut}; the harness tests your claim on the later years.",
                              is_error=True)
        end_year = min(int(end_year), cut)
    try:
        df = cloud.table(dataset, (float(lat_min), float(lat_max), float(lon_min), float(lon_max)), int(start_year),
                         int(end_year), float(resolution))
    except Exception as e:  # noqa: BLE001
        return ToolResult(f"Could not fetch {dataset}: {type(e).__name__}: {str(e)[:400]}", is_error=True)
    out = ctx.workdir / f"{dataset}_{int(start_year)}_{int(end_year)}.csv"
    df.to_csv(out, index=False)
    ctx.state.setdefault("data_files", {})
    n = df[["lat", "lon"]].drop_duplicates().shape[0]
    return ToolResult(f"{var.label} [{var.units}] from {var.source}: {len(df)} rows, {n} cells at {resolution} deg, "
                      f"{df.year.min()}-{df.year.max()} -> {out.name} (columns lat, lon, variable, year, month, value)."
                      + (f" Note: {var.note}" if var.note else ""), files=[str(out)])
