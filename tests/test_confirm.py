"""Held-out confirmation runs the pre-registered test in code; agents cannot alter it."""
import numpy as np
import pandas as pd

from synaptic import confirm


def _fake(values):
    years = np.arange(2001, 2001 + len(values))
    return lambda *a, **k: pd.DataFrame({"year": years, "value": values, "count": 12})


def test_validate():
    assert confirm.validate({"dataset": "nope", "kind": "trend", "box": [0, 1, 0, 1]})[0] is None
    s, _ = confirm.validate({"dataset": "CERES_SW_UP", "kind": "event", "box": [0, 10, -40, -20],
                             "event_years": [2023], "direction": "-"})
    assert s and s["alpha"] == 0.05
    assert confirm.validate({"dataset": "CERES_SW_UP", "kind": "step", "box": [0, 10, -40, -20]})[0] is None


def test_event_supported_and_wrong_direction(monkeypatch):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(25) * 0.3
    v[-3] -= 3.0  # 2023 strongly below the trend
    monkeypatch.setattr(confirm, "_annual_dataset", _fake(v))
    spec = {"dataset": "CERES_SW_UP", "kind": "event", "box": [0, 10, -40, -20], "years": [2001, 2025],
            "event_years": [2023]}
    assert confirm.run({**spec, "direction": "-"})["outcome"] == "supported"
    assert confirm.run({**spec, "direction": "+"})["outcome"] == "refuted"


def test_noise_is_not_shown(monkeypatch):
    monkeypatch.setattr(confirm, "_annual_dataset", _fake(np.random.default_rng(1).standard_normal(25)))
    r = confirm.run({"dataset": "CERES_SW_UP", "kind": "trend", "box": [0, 10, -40, -20], "years": [2001, 2025],
                     "direction": "+"})
    assert r["outcome"] == "not_shown" and not r["passed"]


def test_residual_beyond_driver(monkeypatch):
    """Storage that follows rain is explained; a decline the rain does not explain is a supported residual."""
    rng = np.random.default_rng(3)
    rain = rng.standard_normal(25)
    store = np.cumsum(rain - rain.mean()) + rng.standard_normal(25) * 0.2
    pump = np.r_[np.zeros(15), -np.arange(1, 11) * 0.8]  # depletion from 2016 that rain does not explain

    def fake(var_id, *a, **k):
        v = rain if var_id == "MERRA2_PRECIP" else store + (pump if var_id == "GRACE_TWS" else 0)
        return pd.DataFrame({"year": np.arange(2001, 2026), "value": v, "count": 12})

    monkeypatch.setattr(confirm, "_annual_dataset", fake)
    spec = {"dataset": "GRACE_TWS", "kind": "residual", "driver": "MERRA2_PRECIP", "accumulate": True,
            "fit_years": [2001, 2014], "test": "event",
            "event_years": list(range(2016, 2026)), "box": [27, 37, -14, 2], "years": [2001, 2025],
            "direction": "-"}
    r = confirm.run(spec)
    assert r["outcome"] == "supported", r["detail"]
    monkeypatch.setattr(confirm, "_annual_dataset", lambda var_id, *a, **k: pd.DataFrame(
        {"year": np.arange(2001, 2026), "value": rain if var_id == "MERRA2_PRECIP" else store, "count": 12}))
    assert confirm.run(spec)["outcome"] != "supported"
    assert confirm.validate({**spec, "test": "nope"})[0] is None


def test_residual_two_drivers(monkeypatch):
    rng = np.random.default_rng(6)
    p, e = rng.standard_normal(25), rng.standard_normal(25)
    store = np.cumsum((p - p.mean()) - 0.5 * (e - e.mean())) + rng.standard_normal(25) * 0.2
    pump = np.r_[np.zeros(15), -np.arange(1, 11) * 0.8]
    vals = {"MERRA2_PRECIP": p, "GLDAS_EVAP": e, "GRACE_TWS": store + pump}
    monkeypatch.setattr(confirm, "_annual_dataset", lambda v, *a, **k: pd.DataFrame(
        {"year": np.arange(2001, 2026), "value": vals[v], "count": 12}))
    spec = {"dataset": "GRACE_TWS", "kind": "residual", "driver": ["MERRA2_PRECIP", "GLDAS_EVAP"],
            "accumulate": True, "fit_years": [2001, 2014], "test": "event", "event_years": list(range(2016, 2026)),
            "box": [27, 37, -14, 2], "years": [2001, 2025], "direction": "-"}
    assert confirm.run(spec)["outcome"] == "supported"
    vals["GRACE_TWS"] = store
    assert confirm.run(spec)["outcome"] != "supported"


def _synthetic_monthly(seed, planted=0.0, year=2026, carry=0.9):
    """Storage as a running total of noisy inputs (a level with strong carry-over), optionally with an extra gain
    planted in the winter before `year`."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    idx, vals, level = [], [], 0.0
    for y in range(2001, 2027):
        for m in range(1, 13):
            inp = rng.normal(0, 1.0) + (3.0 if m in (11, 12, 1, 2, 3) else -1.8)
            if planted and ((y == year - 1 and m in (11, 12)) or (y == year and m in (1, 2, 3))):
                inp += planted / 5
            level = carry * level + inp
            idx.append((y, m))
            vals.append(level)
    return pd.Series(vals, index=pd.MultiIndex.from_tuples(idx, names=["year", "month"]))


def test_gain_detects_planted_winter_gain_and_not_noise(monkeypatch):
    from synaptic import confirm
    spec = {"dataset": "GRACE_TWS", "box": [32, 37, -10, -2], "kind": "gain", "from_months": [9, 10],
            "to_months": [4, 5], "years": [2003, 2026], "test": "event", "event_years": [2026], "direction": "+"}
    monkeypatch.setattr(confirm, "_monthly", lambda *a, **k: _synthetic_monthly(1, planted=15.0))
    r = confirm.run(spec)
    assert r["outcome"] == "supported", r["detail"]
    false = 0
    for seed in range(40):
        monkeypatch.setattr(confirm, "_monthly", lambda *a, seed=seed, **k: _synthetic_monthly(100 + seed))
        false += confirm.run(spec)["passed"]
    assert false <= 5   # about 5% expected at alpha 0.05 (one-sided direction)


def test_later_test_finds_a_shift_after_the_exploration_period_and_not_noise(monkeypatch):
    import numpy as np
    import pandas as pd
    from synaptic import confirm

    def series(seed, shift=0.0):
        rng = np.random.default_rng(seed)
        idx, vals = [], []
        for y in range(2001, 2027):
            for m in range(1, 13):
                rain = max(0.0, rng.normal(2.0, 1.0))
                idx.append((y, m))
                vals.append(rain)
        return pd.Series(vals, index=pd.MultiIndex.from_tuples(idx))

    def storage(seed, shift):
        rng = np.random.default_rng(seed + 1000)
        r = series(seed)
        out = r.copy() * 0
        lvl = 0.0
        for (y, m), v in r.items():
            lvl = 0.6 * lvl + 3 * v + rng.normal(0, 1.5) - (shift if (y > 2015 and m in (1, 2, 3)) else 0)
            out[(y, m)] = lvl
        return out
    spec = {"dataset": "GRACE_TWS", "box": [0, 10, 0, 10], "kind": "gain", "from_months": [9, 10], "to_months": [4, 5],
            "years": [2003, 2025], "test": "later", "after": 2015, "driver": ["MERRA2_PRECIP"], "direction": "-"}
    monkeypatch.setattr(confirm, "_monthly", lambda v, *a, **k: series(7) if v == "MERRA2_PRECIP" else storage(7, 4.0))
    assert confirm.run(spec)["outcome"] == "supported"
    false = 0
    for seed in range(30):
        monkeypatch.setattr(confirm, "_monthly", lambda v, *a, s=seed, **k: series(s) if v == "MERRA2_PRECIP"
                            else storage(s, 0.0))
        false += confirm.run(spec)["passed"]
    assert false <= 4


def test_time_holdout_specs_must_test_only_later_years():
    from synaptic.harness import Campaign
    th = {"datasets": ["GRACE_TWS", "GRACE_GSFC_TWS"], "explore_end": 2015}
    base = {"dataset": "GRACE_TWS", "box": [0, 10, 0, 10], "kind": "gain", "from_months": [9, 10], "to_months": [4, 5],
            "years": [2003, 2025], "driver": ["MERRA2_PRECIP"]}
    ok = {"title": "t", "_time_holdout": th, "confirm": {**base, "test": "later", "after": 2015}}
    assert Campaign.spec_problem(ok) == ""
    seen = {"title": "t", "_time_holdout": th, "confirm": {**base, "test": "trend"}}
    assert "explorers saw" in Campaign.spec_problem(seen)
    later_only = {"title": "t", "_time_holdout": th, "confirm": {**base, "test": "trend", "years": [2016, 2025]}}
    assert Campaign.spec_problem(later_only) == ""
