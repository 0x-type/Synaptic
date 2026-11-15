"""Both orders of every judged pair are played; a judge that always prefers the first report produces ties only."""
import json

from synaptic.record import Record
from synaptic.tournament import balanced_pairs, run_tournament


class FirstShownJudge:
    def judge(self, a, b):
        return {"A": {"impact": 5, "novelty": 5, "soundness": 5, "actionability": 5},
                "B": {"impact": 1, "novelty": 1, "soundness": 4, "actionability": 1}}


class QualityJudge:
    def judge(self, a, b):
        qa, qb = int(a.split()[-1]), int(b.split()[-1])
        s = lambda q: {"impact": q, "novelty": q, "soundness": 4, "actionability": q}
        return {"A": s(qa), "B": s(qb)}


def _setup(tmp_path, n=6):
    r = Record(tmp_path / "t.db")
    cid = r.create_campaign("t", {"title": "t"}, "mock", str(tmp_path))
    for i in range(n):
        r.add_report(cid, None, f"F{i}", f"R{i}", f"report quality {i % 5 + 1}", [])
    return r, cid


def test_balanced_pairs_cover_reports():
    p = balanced_pairs(15, 30)
    assert len(p) == 30 and len(set(p)) == 30
    counts = [sum(i in x for x in p) for i in range(15)]
    assert min(counts) >= 3


def test_position_bias_cancels(tmp_path):
    r, cid = _setup(tmp_path)
    run_tournament(r, FirstShownJudge(), cid, max_games=30)
    games = r.q("SELECT a, b FROM games WHERE campaign_id=?", (cid,))
    pairs = {(g["a"], g["b"]) for g in games}
    assert all((b, a) in pairs for a, b in pairs)            # every pair in both orders
    bias = json.loads(r.q("SELECT body FROM kb WHERE campaign_id=? AND title='Tournament position bias'", (cid,))[0]["body"])
    assert bias["first_shown_share"] == 1.0 and bias["pairs_consistent"] == 0
    ratings = [x["rating"] for x in r.q("SELECT rating FROM ratings WHERE campaign_id=?", (cid,))]
    assert max(ratings) - min(ratings) < 0.05                 # pure position bias ranks nobody above anybody


def test_quality_wins(tmp_path):
    r, cid = _setup(tmp_path, 5)
    order = run_tournament(r, QualityJudge(), cid, max_games=40)
    assert order[0]["body"].endswith("5")
