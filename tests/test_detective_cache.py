"""Saved Trend Detective results: the same question maps to the same key, and every shipped example loads."""
from synaptic.dashboard import detective as d


def test_same_question_same_key():
    a = {"var": "IMERG_PRECIP", "region": {"type": "named", "name": "Morocco"}, "y0": 2001, "y1": 2025,
         "months": [1, 12, 2]}
    b = {"var": "IMERG_PRECIP", "region": {"type": "named", "name": "Morocco", "extra": 1}, "y0": "2001", "y1": 2025,
         "months": "12,1,2", "fresh": True}
    assert d._key(a) == d._key(b)
    assert d._key(a) != d._key({**a, "y1": 2024})


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "_LOCAL", tmp_path)
    spec = {"var": "MODIS_CHL", "region": {"type": "box", "bbox": [0, 1, 0, 1]}, "y0": 2003, "y1": 2004, "months": None}
    d.save_result(spec, {"verdict": "v", "trend": {"p": float("nan")}})
    out = d.load_saved(spec)
    assert out["verdict"] == "v" and out["trend"]["p"] is None and out["saved"]


def test_every_example_ships_a_saved_result():
    for demo in d.DEMOS:
        out = d.load_saved(demo["spec"])
        assert out is not None, demo["id"]
        assert out.get("verdict") and out.get("series") and out.get("map_png")
