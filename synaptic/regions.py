"""Regions for the Trend Detective: the whole world, a box, or a country / US state outline (Natural Earth
1:110m, public domain)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_FILE = Path(__file__).with_name("data") / "regions.json"
WORLD = (-90.0, 90.0, -180.0, 180.0)


@lru_cache(maxsize=1)
def _all() -> dict:
    return json.loads(_FILE.read_text())


def names() -> list[dict]:
    return sorted(({"name": k, "kind": v["kind"]} for k, v in _all().items()), key=lambda d: (d["kind"], d["name"]))


def bbox(name: str, pad: float = 0.5) -> tuple[float, float, float, float]:
    rings = _all()[name]["rings"]
    xs = np.concatenate([np.array(r)[:, 0] for r in rings])
    ys = np.concatenate([np.array(r)[:, 1] for r in rings])
    return (max(-90, ys.min() - pad), min(90, ys.max() + pad), max(-180, xs.min() - pad), min(180, xs.max() + pad))


def mask_fn(name: str):
    """mask(lat, lon) -> float array [lat, lon], 1 inside the outline. Falls back to the box when the grid is too
    coarse to put any cell centre inside (a small state on a 1-degree grid)."""
    from matplotlib.path import Path as MPath

    paths = [MPath(np.array(r)) for r in _all()[name]["rings"]]

    def mask(lat, lon):
        X, Y = np.meshgrid(lon, lat)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = np.zeros(len(pts), bool)
        for p in paths:
            inside |= p.contains_points(pts)
        m = inside.reshape(len(lat), len(lon)).astype(float)
        return m if m.any() else np.ones_like(m)
    return mask


@lru_cache(maxsize=1)
def _land_paths():
    from matplotlib.path import Path as MPath

    return [MPath(np.array(r)) for v in _all().values() if v.get("kind") == "country" for r in v["rings"]]


def land_mask(lat, lon):
    """1 over land (union of all country outlines), 0 over sea. GRACE mascons and CERES also have values over the
    ocean: a land-storage box that touches a coast must not average in sea mass."""
    X, Y = np.meshgrid(lon, lat)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for p in _land_paths():
        bb = p.get_extents()
        sel = (pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)
        if sel.any():
            inside[sel] |= p.contains_points(pts[sel])
    m = inside.reshape(len(lat), len(lon)).astype(float)
    return m if m.any() else np.ones_like(m)


def resolve(region: dict):
    """region = {"type": "world"} | {"type": "box", "bbox": [S, N, W, E]} | {"type": "named", "name": ...}
    -> (bbox, mask or None, label)."""
    kind = (region or {}).get("type", "world")
    if kind == "box":
        S, N, W, E = (float(v) for v in region["bbox"])
        if S > N:
            S, N = N, S
        return (S, N, W, E), None, f"box {S:g} to {N:g} N, {W:g} to {E:g} E"
    if kind == "named":
        name = region["name"]
        return bbox(name), mask_fn(name), name
    return WORLD, None, "whole world"
