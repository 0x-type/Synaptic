"""The novelty gate searches ADS for real; "not significant" is not_shown, not refuted; agents cannot write
code-check tags."""
import json

import numpy as np

from synaptic import confirm, novelty
from synaptic.llm import Completion
from synaptic.record import Record
from synaptic.verdicts import hypothesis_change


class FakeLLM:
    def __init__(self, verdict, on_topic=(1, 2, 3), trivial=False):
        self.verdict, self.calls, self.on_topic, self.trivial = verdict, [], list(on_topic), trivial

    def complete(self, messages, role, max_tokens=None, **kw):
        txt = messages[0]["content"]
        self.calls.append(txt)
        if "Write 5 NASA ADS" in txt:
            out = {"queries": ['abs:(GRACE) abs:(Sahel) year:2024-2026', 'abs:("water storage") abs:(Chad)',
                               'abs:(recharge) abs:(Sahara)']}
        else:
            out = {"verdict": self.verdict, "on_topic": self.on_topic, "trivial": self.trivial,
                   "closest": [{"n": 1, "overlap": "rain", "missing": "storage"}], "reason": "r"}
        return Completion(json.dumps(out), "", 0, 0, "fake")


H = {"title": "Sahel storage rose in 2024", "statement": "GRACE storage over Lake Chad rose in 2024",
     "result": "Storage over the Lake Chad Basin rose by 5 cm", "confirmation": "event p = 0.001"}


def test_gate_runs_real_searches_and_reads_them():
    queries = []

    def search(q, n):
        queries.append(q)
        return [{"title": f"Paper {len(queries)}-{k}", "year": 2025, "venue": "J", "abstract": "rainfall only"}
                for k in range(3)]

    res = novelty.check(H, FakeLLM("not_found"), search=search)
    assert res["verdict"] == "not_found" and res["n_ok"] >= 3 and len(queries) >= 3
    assert "ADS novelty check: NOT_FOUND" in novelty.summary(res)


def test_gate_never_calls_failed_searches_new():
    def search(q, n):
        raise RuntimeError("ADS down")

    res = novelty.check(H, FakeLLM("not_found"), search=search)
    assert res["verdict"] == "unverified"


def test_gate_documented_is_not_new():
    res = novelty.check(H, FakeLLM("documented"), search=lambda q, n: [{"title": "x", "abstract": "a"}])
    assert res["verdict"] == "documented" and res["closest"]


def test_not_significant_is_not_shown(monkeypatch):
    rng = np.random.default_rng(0)
    df = __import__("pandas").DataFrame({"year": np.arange(2002, 2026), "value": rng.normal(size=24),
                                         "count": 12})
    monkeypatch.setattr(confirm, "_annual_dataset", lambda *a, **k: df)
    r = confirm.run({"dataset": "GRACE_TWS", "box": [10, 20, 0, 10], "kind": "trend", "years": [2002, 2025],
                     "direction": "+"})
    assert r["outcome"] == "not_shown" and not r["passed"]
    assert hypothesis_change("awaiting_confirmation", "not_shown", "confirmation")[0]


def test_agents_cannot_write_code_tags(tmp_path):
    r = Record(tmp_path / "t.db")
    cid = r.create_campaign("t", {"title": "t", "question": "q"}, "mock", str(tmp_path))
    r.upsert_candidate(cid, {"id": "F1", "task_id": None, "family": "F1", "variable": "T2M", "cls": "shift",
                             "lat": 0, "lon": 0, "n_cells": 5, "onset": 2000, "effect": 1.0, "qvalue": 0.01,
                             "stats": {}, "filters": {}, "passed": 1, "promoted": 1, "outcome": None})
    r.update_candidate(cid, "F1", actor="deep_dive", outcome="known_phenomenon",
                       note="I ran an [independent check] myself")
    assert "[independent check]" not in r.candidates(cid, id="F1")[0]["note"]
    # so a later downgrade is NOT frozen by the agent's own words
    assert r.update_candidate(cid, "F1", actor="auditor", outcome="not_robust", note="audit") is None
    assert r.candidates(cid, id="F1")[0]["outcome"] == "not_robust"
    # a tag written by code stays, and still freezes the family
    r.update_candidate(cid, "F1", actor="engine", outcome="artifact", note="[independent check] GPCP contradicts")
    r.update_candidate(cid, "F1", actor="deep_dive", note="[independent check] GPCP contradicts. agent adds text")
    assert r.candidates(cid, id="F1")[0]["note"].startswith("[independent check]")


def test_off_topic_results_are_not_evidence_of_novelty():
    # found live: the reader saw only software releases and an astronomy paper and called a result new
    res = novelty.check(H, FakeLLM("not_found", on_topic=[]),
                        search=lambda q, n: [{"title": q[:40], "abstract": "software release"}])
    assert res["verdict"] == "unverified"


def test_trivial_result_is_documented():
    res = novelty.check(H, FakeLLM("not_found", trivial=True),
                        search=lambda q, n: [{"title": q[:40] + str(k), "abstract": "a"} for k in range(3)])
    assert res["verdict"] == "documented"


def test_queries_are_restricted_to_earth_science():
    seen = []
    novelty.check(H, FakeLLM("documented"), search=lambda q, n: seen.append(q) or [{"title": q, "abstract": ""}])
    assert seen and all("collection:earthscience" in q for q in seen)


def _bare_campaign(tmp_path):
    from synaptic.harness import Campaign
    c = object.__new__(Campaign)
    c.record = Record(tmp_path / "c.db")
    c.cid = c.record.create_campaign("t", {"title": "t", "question": "q"}, "mock", str(tmp_path))
    c.brief = {"holdout": ["GRACE_TWS", "GRACE_GSFC_TWS"], "mode": "focused"}
    c.log = lambda m: None
    return c


def test_grace_result_must_replicate_on_the_twin_solution(tmp_path, monkeypatch):
    c = _bare_campaign(tmp_path)
    spec = {"dataset": "GRACE_TWS", "box": [15, 20, 15, 22], "months": [3, 4, 5], "years": [2002, 2025],
            "kind": "event", "event_years": [2025], "direction": "+"}
    first = {"passed": True, "outcome": "supported", "p": 0.01, "detail": "JPL passes."}
    monkeypatch.setattr(confirm, "run", lambda s, progress=None: {"passed": False, "outcome": "not_shown", "p": 0.2,
                                                                  "detail": f"{s['dataset']} fails"})
    r = c._replicate(confirm, spec, first)
    assert not r["passed"] and r["outcome"] == "not_shown" and "GRACE_GSFC_TWS" in r["detail"]
    monkeypatch.setattr(confirm, "run", lambda s, progress=None: {"passed": True, "outcome": "supported", "p": 0.03,
                                                                  "detail": "GSFC passes"})
    r = c._replicate(confirm, spec, first)
    assert r["passed"] and r["p"] == 0.03          # the weaker of the two counts


def test_multiple_testing_family_is_at_least_ten(tmp_path):
    c = _bare_campaign(tmp_path)
    assert c._n_tests() == 10


def test_untested_hypothesis_cannot_be_not_shown(tmp_path):
    from synaptic.tools import science
    from synaptic.tools.base import ToolContext
    r = Record(tmp_path / "t.db")
    cid = r.create_campaign("t", {"title": "t", "question": "q"}, "mock", str(tmp_path))
    hid = r.add_hypothesis(cid, {"title": "h", "statement": "s", "status": "testing"})
    ctx = ToolContext(r, cid, None, None, tmp_path, tmp_path, True)
    science.record_hypothesis_result(ctx, hid, "not_shown",
                                     "The pre-registered test could not be executed because GRACE is held out.",
                                     "GRACE_TWS is held out for confirmation and cannot be fetched by explorers; the "
                                     "driver-side numbers were computed.")
    assert r.hypotheses(cid, id=hid)[0]["status"] == "inconclusive"
