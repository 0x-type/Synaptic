"""Calibration of the change tests on synthetic noise (the expert review's failure cases) and power on planted
signals. Seeds are fixed, so the counts are deterministic."""
import numpy as np
import pandas as pd
import pytest

import synlib
from synth import field


def _ar(rng, n, phi):
    e = rng.standard_normal(n)
    y = np.zeros(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return y


@pytest.mark.parametrize("phi", [0.0, 0.5])
def test_single_series_tests_hold_their_size_under_carryover(phi):
    rng = np.random.default_rng(1)
    n, N = 45, 400
    rates = {"mk": 0, "pettitt": 0, "recent": 0}
    for _ in range(N):
        y = _ar(rng, n, phi)
        rates["mk"] += synlib.mann_kendall(y)["p"] < 0.05
        rates["pettitt"] += synlib.pettitt(y)["p"] < 0.05
        rates["recent"] += synlib.recent_departure(y)["p"] < 0.05
    for k, v in rates.items():
        assert v / N <= 0.10, (k, v / N)


def test_old_failure_false_shift_cells_are_gone():
    """Expert review: carry-over alone gave ~17 false 'shift' cells per run of 225 (25 here before the fix)."""
    false_cells = []
    for seed in range(4):
        sw = synlib.sweep(field(seed=seed), "X", n_boot=10)
        cells = synlib.classify_cells(sw, p_form=None)  # per-cell FDR, the old selection rule
        false_cells.append(int((cells.cls == "shift").sum()) if len(cells) else 0)
    assert np.mean(false_cells) <= 1.0, false_cells


def _families_and_nulls(df):
    sw = synlib.sweep(df, "X", n_boot=10)
    cells = synlib.classify_cells(sw)
    fams = synlib.families(cells) if len(cells) else pd.DataFrame()
    cube = synlib.to_cube(df, "X")
    ann = cube.annual()
    la, lo = np.meshgrid(cube.lats, cube.lons, indexing="ij")
    return fams, la.ravel(), lo.ravel(), ann.reshape(-1, ann.shape[-1])


def test_no_false_field_significant_families_on_noise():
    """Carry-over + neighbours moving together made a 200-cell false 'recent extreme' family. Now families must
    beat surrogate fields that keep both."""
    passing = 0
    for seed in range(100, 106):
        fams, la, lo, Y = _families_and_nulls(field(seed=seed, phi=0.5, spatial=2.0))
        for cls in synlib.TESTABLE_CLASSES:
            fs = fams[fams.cls == cls] if len(fams) else fams
            if len(fs):
                nl = synlib.cluster_null(la, lo, Y, cls, n_sim=200, seed=seed)
                passing += any(synlib.cluster_p(int(n), nl)["field_significant"] for n in fs.n_cells)
    assert passing <= 1


def test_planted_regional_step_is_found():
    def plant(i, j, years):
        return 1.5 * (years >= 2003) if (3 <= i <= 9 and 3 <= j <= 9) else None
    fams, la, lo, Y = _families_and_nulls(field(seed=11, plant=plant))
    big = fams[fams.cls == "shift"].n_cells.max()
    nl = synlib.cluster_null(la, lo, Y, "shift", n_sim=300)
    assert synlib.cluster_p(int(big), nl)["field_significant"]


def test_field_test_fails_closed():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((50, 45))
    la, lo = np.repeat(np.arange(5.0), 10), np.tile(np.arange(10.0), 5)
    nl = synlib.cluster_null(la, lo, Y, "seasonality")
    assert not nl["valid"] and not synlib.cluster_p(30, nl)["field_significant"]
    nl2 = synlib.cluster_null(la[:5], lo[:5], Y[:5], "trend")
    assert not synlib.cluster_p(5, nl2)["field_significant"]
    assert not synlib.field_significance(Y[:5], "trend")["field_significant"]


def test_spatial_sign_test_is_honest():
    rng = np.random.default_rng(3)
    common = rng.standard_normal(45)
    R = common[None, :] + 0.5 * rng.standard_normal((52, 45))  # strongly correlated neighbours
    v = rng.standard_normal(52) + 0.8
    out = synlib.sign_test_spatial(v, R)
    assert out["effective_cells"] < 10 and out["p"] > out["p_naive"] * 100


def test_scanned_split_pays_for_the_scan():
    rng = np.random.default_rng(5)
    t = np.arange(1981, 2026)
    fp = sum(synlib.correlation_change_test(0.4 * x + rng.standard_normal(45), x, t, n_sim=300, seed=k)
             ["p_selection_aware"] < 0.05 for k, x in enumerate(rng.standard_normal((80, 45))))
    assert fp / 80 <= 0.10
    x = rng.standard_normal(45)
    y = np.where(t >= 2005, 0.9, 0.0) * x + 0.4 * rng.standard_normal(45)
    assert synlib.correlation_change_test(y, x, t, n_sim=500)["p_selection_aware"] < 0.05
