"""Synthetic fields for calibrating the statistics: no planted signal unless asked."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def field(nlat=15, nlon=15, years=range(1981, 2026), phi=0.5, spatial=1.5, seed=0, month_noise=0.5,
          plant=None):
    """Long table (lat, lon, variable, year, month, value). Annual anomalies are AR(1) in time (carry-over phi)
    and share innovations with neighbours (Gaussian smoothing, `spatial` cells). `plant(lat_i, lon_i, years)`
    may return an array added to the annual anomalies of a cell."""
    rng = np.random.default_rng(seed)
    years = np.array(list(years))
    ny = len(years)
    eps = rng.standard_normal((ny, nlat, nlon))
    if spatial:
        eps = np.stack([gaussian_filter(e, spatial, mode="wrap") for e in eps])
        eps /= eps.std()
    ann = np.zeros_like(eps)
    ann[0] = eps[0]
    for t in range(1, ny):
        ann[t] = phi * ann[t - 1] + np.sqrt(1 - phi ** 2) * eps[t]
    lats = np.arange(nlat) * 2.0
    lons = np.arange(nlon) * 2.5
    if plant is not None:
        for i in range(nlat):
            for j in range(nlon):
                add = plant(i, j, years)
                if add is not None:
                    ann[:, i, j] += add
    clim = 10 + 5 * np.sin(2 * np.pi * (np.arange(12) - 3) / 12)
    rows = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            v = ann[:, i, j][:, None] + clim[None, :] + month_noise * rng.standard_normal((ny, 12))
            yy, mm = np.meshgrid(years, np.arange(1, 13), indexing="ij")
            rows.append(pd.DataFrame({"lat": la, "lon": lo, "variable": "X", "year": yy.ravel(),
                                      "month": mm.ravel(), "value": v.ravel()}))
    return pd.concat(rows, ignore_index=True)
