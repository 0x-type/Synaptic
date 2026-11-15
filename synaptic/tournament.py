"""Pairwise tournament + Bradley-Terry ranking of reports (paper Methods, 'Ranking of reports').

Position bias is cancelled by design: every pair that is judged is judged in BOTH orders. A report wins the pair
only if it wins both games; when the two orders disagree the judge preferred whatever it saw first (or second), so
the pair counts as a tie (half a win each). The share of games won by the report shown first is reported as the
judge's position bias. (v3 played 60 of the n(n-1) ordered pairs, so most pairs were seen in one order only.)
"""
from __future__ import annotations

import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor

WEIGHTS = {"impact": 0.35, "novelty": 0.30, "soundness": 0.25, "actionability": 0.10}


def weighted(s: dict) -> float:
    return sum(WEIGHTS[k] * float(s.get(k, 0) or 0) for k in WEIGHTS)


def bradley_terry(n: int, games: list[tuple[int, int, float]], iters: int = 300) -> list[float]:
    """MM algorithm (Hunter 2004). games = list of (i, j, score of i) with score 1 (i won), 0.5 (tie), 0 (j won)."""
    w = [0.0] * n
    m = [[0.0] * n for _ in range(n)]
    for a, b, sa in games:
        w[a] += sa
        w[b] += 1 - sa
        m[a][b] += 1
        m[b][a] += 1
    p = [1.0] * n
    for _ in range(iters):
        new = []
        for i in range(n):
            den = sum(m[i][j] / (p[i] + p[j]) for j in range(n) if j != i and m[i][j])
            new.append((w[i] + 0.1) / den if den else p[i])  # +0.1 pseudo-win keeps undefeated finite
        s = sum(new)
        p = [x * n / s for x in new]
    return p


def balanced_pairs(n: int, max_pairs: int, seed: int = 7) -> list[tuple[int, int]]:
    """Unordered pairs chosen round by round (random perfect matchings) so every report meets about the same
    number of opponents; all pairs when the budget allows."""
    allp = list(itertools.combinations(range(n), 2))
    if len(allp) <= max_pairs:
        return allp
    rng = random.Random(seed)
    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(chosen) < max_pairs:
        order = list(range(n))
        rng.shuffle(order)
        added = 0
        for k in range(0, n - 1, 2):
            a, b = sorted((order[k], order[k + 1]))
            if (a, b) not in seen:
                seen.add((a, b))
                chosen.append((a, b))
                added += 1
                if len(chosen) >= max_pairs:
                    break
        if not added and len(seen) >= len(allp):
            break
    return chosen


def _game_winner(out: dict) -> int:
    """0 if report A (shown first) wins, 1 if B wins. A report judged unsound (soundness <= 2) loses."""
    sa, sb = out.get("A", {}) or {}, out.get("B", {}) or {}
    lose_a = float(sa.get("soundness", 3) or 3) <= 2
    lose_b = float(sb.get("soundness", 3) or 3) <= 2
    if lose_a != lose_b:
        return 1 if lose_a else 0
    return 0 if weighted(sa) >= weighted(sb) else 1


def run_tournament(record, runner, cid: str, max_games: int = 60, workers: int = 4) -> list[dict]:
    reports = record.q("SELECT * FROM reports WHERE campaign_id=? AND family != 'DATA_SCREEN' ORDER BY created", (cid,))
    if not reports:
        return []
    n = len(reports)
    if n == 1:
        record.x("INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?,?)", (cid, reports[0]["id"], 1.0, 0, 0, 1))
        return reports
    idx = {r["id"]: i for i, r in enumerate(reports)}
    pairs = balanced_pairs(n, max(1, max_games // 2))
    # games kept from an interrupted run: (first shown, second shown) -> winner index
    played: dict[tuple[int, int], int] = {}
    for g in record.q("SELECT a, b, winner FROM games WHERE campaign_id=?", (cid,)):
        if g["a"] in idx and g["b"] in idx and g["winner"] in idx:
            played[(idx[g["a"]], idx[g["b"]])] = idx[g["winner"]]
    todo = [(a, b) for p in pairs for (a, b) in (p, p[::-1]) if (a, b) not in played]

    def play(ab):
        a, b = ab
        try:
            out = runner.judge(reports[a]["body"][:12000], reports[b]["body"][:12000])
        except Exception:  # noqa: BLE001 - a failed comparison (e.g. a server error) must not sink the ranking
            return ab, None
        return ab, out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (a, b), out in ex.map(play, todo):
            if out is None:
                continue
            win = a if _game_winner(out) == 0 else b
            played[(a, b)] = win
            record.x("INSERT INTO games(campaign_id,a,b,scores,winner) VALUES(?,?,?,?,?)",
                     (cid, reports[a]["id"], reports[b]["id"], json.dumps(out), reports[win]["id"]))
    # combine the two orders of each pair
    results: list[tuple[int, int, float]] = []
    first_wins = total = consistent = 0
    for a, b in pairs:
        w1, w2 = played.get((a, b)), played.get((b, a))
        for (x, y), w in (((a, b), w1), ((b, a), w2)):
            if w is not None:
                total += 1
                first_wins += int(w == x)
        if w1 is None or w2 is None:
            continue  # a pair seen in one order only is not used: that is exactly the position-biased case
        if w1 == w2:
            consistent += 1
            results.append((a, b, 1.0 if w1 == a else 0.0))
        else:
            results.append((a, b, 0.5))
    ratings = bradley_terry(n, results)
    order = sorted(range(n), key=lambda i: -ratings[i])
    for rank, i in enumerate(order, 1):
        wins = sum(sa if a == i else 1 - sa for a, b, sa in results if i in (a, b))
        n_played = sum(1 for a, b, _ in results if i in (a, b))
        record.x("INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?,?)",
                 (cid, reports[i]["id"], round(ratings[i], 4), round(wins, 1), n_played, rank))
    bias = {"games": total, "first_shown_won": first_wins,
            "first_shown_share": round(first_wins / total, 3) if total else None,
            "pairs_both_orders": len(results), "pairs_consistent": consistent}
    try:
        record.kb_add(cid, None, "method", "Tournament position bias",
                      json.dumps(bias), ["tournament", "position_bias"])
    except Exception:  # noqa: BLE001
        pass
    return [reports[i] for i in order]
