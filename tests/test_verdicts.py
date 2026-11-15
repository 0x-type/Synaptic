"""Verdicts can only go down; only code gates move a claim up."""
import json

from synaptic.record import Record
from synaptic.verdicts import family_change, hypothesis_change


def test_family_rules():
    assert family_change(None, "new_signal", "deep_dive")[0]
    assert not family_change(None, "new_signal", "auditor")[0]
    assert not family_change("known_phenomenon", "new_signal", "auditor")[0]   # the live failure (Libya/Chad)
    assert not family_change("not_robust", "known_phenomenon", "referee")[0]
    assert family_change("new_signal", "known_phenomenon", "auditor")[0]
    assert not family_change("artifact", "not_robust", "deep_dive", "[independent check] GPCP contradicts")[0]
    assert family_change("artifact", "known_phenomenon", "operator")[0]


def test_hypothesis_rules():
    assert hypothesis_change("testing", "awaiting_confirmation", "agent")[0]
    assert not hypothesis_change("awaiting_confirmation", "supported", "agent")[0]
    assert hypothesis_change("awaiting_confirmation", "supported", "confirmation")[0]
    assert not hypothesis_change("supported", "supported_new", "auditor")[0]
    assert hypothesis_change("supported", "supported_new", "novelty_gate")[0]
    assert not hypothesis_change("refuted", "testing", "pi")[0]            # a lab meeting cannot reopen a result
    assert not hypothesis_change("inconclusive", "known", "referee")[0]
    assert hypothesis_change("supported", "refuted", "referee")[0]


def test_record_enforces_rules(tmp_path):
    r = Record(tmp_path / "t.db")
    cid = r.create_campaign("t", {"title": "t", "question": "q"}, "mock", str(tmp_path))
    r.upsert_candidate(cid, {"id": "F1", "task_id": None, "family": "F1", "variable": "T2M", "cls": "shift",
                             "lat": 0, "lon": 0, "n_cells": 5, "onset": 2000, "effect": 1.0, "qvalue": 0.01,
                             "stats": {}, "filters": {}, "passed": 1, "promoted": 1, "outcome": None})
    assert r.update_candidate(cid, "F1", actor="deep_dive", outcome="known_phenomenon", note="dd") is None
    why = r.update_candidate(cid, "F1", actor="auditor", outcome="new_signal", note="audit says new")
    assert why and r.candidates(cid, id="F1")[0]["outcome"] == "known_phenomenon"
    hid = r.add_hypothesis(cid, {"title": "h", "statement": "s"})
    r.update_hypothesis(cid, hid, actor="agent", status="awaiting_confirmation")
    assert r.update_hypothesis(cid, hid, actor="agent", status="supported")
    assert r.hypotheses(cid, id=hid)[0]["status"] == "awaiting_confirmation"


def test_confirmation_of_untestable_and_protection():
    from synaptic.verdicts import hypothesis_change
    assert hypothesis_change("inconclusive", "supported", "confirmation")[0]
    assert not hypothesis_change("inconclusive", "supported", "pi")[0]
    assert not hypothesis_change("supported", "inconclusive", "agent")[0]
    assert hypothesis_change("supported", "inconclusive", "referee")[0]
