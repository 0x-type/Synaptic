"""Novelty gate with a real NASA ADS check.

Before this module, the gate promoted a confirmed hypothesis to "supported_new" whenever an ADS token existed and the
agent had written 60+ characters of novelty basis: no search ran. Now, for every hypothesis that passed its
held-out confirmation:
  1. a model writes 3-5 precise ADS queries from the CONFIRMED claim (and code adds two from its key terms);
  2. code runs them on NASA ADS (refereed and preprint Earth-science literature) and keeps the abstracts;
  3. an independent reader (the auditor model, which did not do the work) decides, paper by paper, whether any
     abstract already reports the same finding (same quantity, same region, same period or event, same direction);
  4. only "not_found", with at least 3 searches that ran, makes the result "supported_new". Anything else keeps it
     "supported" and records the closest papers. Failed searches never count as "nothing found".
The full record (queries, hit counts, closest papers, the reader's reasons) is written to the hypothesis's novelty
field so a human can re-check it.
"""
from __future__ import annotations

import json
import re

from .protocol import _loads

QUERY_PROMPT = """You check whether a scientific result is already published. Write 5 NASA ADS search queries that
would find any paper reporting this result or something close to it.

Build each query from 2-3 CONCEPT GROUPS joined by AND; each group is an OR-list of synonyms inside abs:( ).
Example: abs:(GRACE OR "terrestrial water storage" OR groundwater) AND abs:(Sahel OR "Lake Chad" OR Niger) AND
abs:(flood OR "extreme rainfall" OR recharge)
Rules: plain scientific words and place names only. NO statistics or method words (p-value, held-out,
pre-registered, Mann-Kendall), NO dataset ids or product codes (IMERG_PRECIP, RL06), NO months alone.
Query 1: place + quantity only (broad). Query 2: place + the event or period. Query 3: quantity + the mechanism
(e.g. recharge, persistence, decay, lakes). Query 4: a wider region (e.g. "West Africa", "North Africa", drylands)
+ quantity + mechanism. Query 5: your best guess at the title words of a paper that would report exactly this.

Result:
{claim}

Answer with JSON only: {{"queries": ["...", "...", "...", "...", "..."]}}"""

JUDGE_PROMPT = """You are a strict novelty referee for a top journal. A research group claims this result is NEW:

RESULT (confirmed on held-out data):
{claim}

Below are the papers NASA ADS returned for {nq} searches.
Step 1, ON TOPIC: list the numbers of the papers about the same region (or one containing it) AND the same kind of
quantity. If fewer than 3 papers are on topic, the search did not reach the literature: say so (verdict
"search_inadequate"); absence of a match then means nothing.
Step 2, TRIVIAL: if the result is a textbook relationship or an expected consequence of well-known physics (e.g.
"water storage rises when it rains more", "soil is wetter after rain", "it got warmer"), the verdict is
"documented" even if no paper states it for this exact region, unless the result adds a specific new quantity or
behaviour (say which).
Step 2b, WHAT WAS TESTED: judge only what the held-out test established, restated in plain words (e.g. "in this box,
storage is higher in years with more runoff"). Anything in the authors' interpretation that the test did not
measure (a comparison it did not make, a cause, an upstream source, a specific year) is NOT part of the result.
Step 3, MATCH: for each on-topic paper decide whether its abstract ALREADY REPORTS this result: the same quantity
(a rainfall study does not report a water-storage result, and vice versa), the same region, the same event or
period, the same direction. Partial overlap is "partly".

Papers:
{papers}

Answer with JSON only:
{{"on_topic": [paper numbers],
  "tested_result": "what the test established, in one plain sentence",
  "trivial": true | false,
  "verdict": "documented" | "partly" | "not_found" | "search_inadequate",
  "closest": [{{"n": paper number, "overlap": "what it already reports", "missing": "what of the result it does not"}}],
  "reason": "two sentences"}}"""

STOP_WORDS = {"The", "This", "That", "With", "From", "Held", "Confirmed", "Result", "Pre", "Sen", "Mann", "Kendall",
              "Title", "Statement", "Storage", "Rain", "Rainfall", "Satellite", "Heavy", "Water", "Direct", "Dry",
              "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December", "Pearson", "Spearman", "Sep", "Aug", "Mar", "Jul", "Oct", "Box", "Boxes",
              "Central", "Northern", "Southern", "Eastern", "Western", "North", "South", "East", "West", "Test",
              "Basin", "Region", "Grace", "Imerg", "Gldas", "Merra", "Ceres", "Nasa", "Jpl", "Mascon", "Season", "Trend", "Both"}
FILTER = " collection:earthscience -doctype:software"


def _key_terms(text: str) -> list[str]:
    caps = re.findall(r"\b([A-Z][a-z]{3,}(?:\s[A-Z][a-z]{3,})?)\b", text)
    out = []
    for c in caps:
        if not any(w in STOP_WORDS for w in c.split()) and c not in out:
            out.append(c)
    return out[:5]


def claim_text(h: dict) -> str:
    """What is judged is what the held-out test ESTABLISHED, not the hypothesis's story. Found live: a hypothesis
    claimed "Lake Chad's 2024 storage follows upstream floodwater more than local rain"; its pre-registered test only
    showed that the box's storage correlates with the box's own runoff over 2002-2024 (wet years are wet); the
    reader judged the untested story, found no paper reporting it, and called it new."""
    return (f"WHAT THE HELD-OUT TEST ESTABLISHED (judge ONLY this):\n{h.get('confirmation', '')}\n\n"
            f"Hypothesis title (context): {h.get('title', '')}\n"
            f"The authors' interpretation (NOT established by the test unless the test above shows it; do not credit "
            f"it as new): {h.get('statement', '')}")[:3500]


def check(h: dict, llm, search=None, max_hits: int = 45) -> dict:
    """Return {"verdict": documented|partly|not_found|unverified, "queries": [...], "n_ok": int, "papers": [...],
    "closest": [...], "reason": str}."""
    if search is None:
        search = _ads_patient
    claim = claim_text(h)
    queries: list[str] = []
    try:
        r = llm.complete([{"role": "user", "content": QUERY_PROMPT.format(claim=claim)}], "auditor", max_tokens=6000)
        queries = [str(q) for q in (_loads(_json_block(r.text)) or {}).get("queries", []) if str(q).strip()][:5]
    except Exception:  # noqa: BLE001
        queries = []
    terms = _key_terms(f"{h.get('title', '')} {h.get('statement', '')} {h.get('result', '')}")
    if terms:  # code-written queries too, so the model cannot steer the search away from the obvious papers
        places = " OR ".join(f'"{t}"' for t in terms)
        queries.append(f'abs:({places}) AND abs:(GRACE OR "water storage" OR groundwater OR "soil moisture" OR '
                       f'recharge OR albedo OR dust)')
        queries.append(f'abs:({places}) AND abs:("2024" OR "2025" OR "2026") AND abs:(flood OR rainfall OR '
                       f'precipitation OR storage)')
    queries = [q if "collection:" in q else q + FILTER for q in queries]
    seen, papers, n_ok, errors = set(), [], 0, []
    for q in queries:
        try:
            hits = search(q, 15)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{q[:80]}: {type(e).__name__}")
            continue
        for p in hits:
            key = (p.get("title") or "").lower()[:120]
            if key and key not in seen:
                seen.add(key)
                papers.append(p)
    papers = papers[:max_hits]
    out = {"queries": queries, "n_ok": n_ok, "errors": errors, "papers": [
        {"n": i + 1, "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"), "doi": p.get("doi")}
        for i, p in enumerate(papers)]}
    if n_ok < 3:
        return {**out, "verdict": "unverified", "closest": [], "reason": f"only {n_ok} ADS searches ran"}
    if not papers:  # no papers at all means the searches failed to reach the literature, not that it is new
        return {**out, "verdict": "unverified", "closest": [], "reason": "ADS returned no papers for any query"}
    listing = "\n".join(f"[{i + 1}] {p.get('title')} ({p.get('year')}, {p.get('venue')}): "
                        f"{(p.get('abstract') or '(no abstract)')[:700]}" for i, p in enumerate(papers))
    try:
        r = llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(claim=claim, nq=n_ok, papers=listing)}],
                         "auditor", max_tokens=12000)
        j = _loads(_json_block(r.text)) or {}
    except Exception as e:  # noqa: BLE001
        return {**out, "verdict": "unverified", "closest": [], "reason": f"reader failed: {type(e).__name__}"}
    v = str(j.get("verdict", "")).strip().lower()
    on_topic = [n for n in (j.get("on_topic") or []) if isinstance(n, int) and 1 <= n <= len(papers)]
    if v == "search_inadequate":
        v = "unverified"
    if v not in ("documented", "partly", "not_found"):
        v = "unverified"
    if j.get("trivial") is True and v == "not_found":
        v = "documented"
    if v == "not_found" and len(on_topic) < 3:
        # found live: a reader shown only off-topic papers called a textbook result new. "Nothing found" counts only
        # when the searches demonstrably reached the literature on this region and quantity.
        v = "unverified"
    out["on_topic"] = on_topic
    closest = []
    for c in (j.get("closest") or [])[:5]:
        try:
            p = papers[int(c.get("n")) - 1]
            closest.append({"title": p.get("title"), "year": p.get("year"), "doi": p.get("doi"),
                            "overlap": str(c.get("overlap", ""))[:300], "missing": str(c.get("missing", ""))[:300]})
        except Exception:  # noqa: BLE001
            continue
    return {**out, "verdict": v, "closest": closest, "reason": str(j.get("reason", ""))[:600],
            "tested_result": str(j.get("tested_result", ""))[:400]}


def _ads_patient(query: str, n: int) -> list[dict]:
    """ADS with a long timeout and retries: broad OR-queries take ADS 30-90 s at busy times (found live: 5 of 7
    searches timed out at 40 s, and failed searches must never be read as "nothing published")."""
    import time

    import requests

    from .config import SERVICES

    last = None
    for attempt in range(3):
        try:
            r = requests.get("https://api.adsabs.harvard.edu/v1/search/query",
                             params={"q": query, "rows": n, "fl": "title,year,pub,doi,citation_count,abstract",
                                     "sort": "score desc"},
                             headers={"Authorization": f"Bearer {SERVICES.ads_token}"}, timeout=150)
            r.raise_for_status()
            return [{"title": (d.get("title") or [""])[0], "year": d.get("year"), "venue": d.get("pub"),
                     "doi": (d.get("doi") or [None])[0], "cited": d.get("citation_count"),
                     "abstract": d.get("abstract") or ""} for d in r.json().get("response", {}).get("docs", [])]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def summary(res: dict) -> str:
    lines = [f"[ADS novelty check: {res['verdict'].upper()}] judged result: {res.get('tested_result') or '-'}. "
             f"{res.get('n_ok', 0)} ADS searches, "
             f"{len(res.get('papers', []))} distinct papers read. {res.get('reason', '')}"]
    for c in res.get("closest", []):
        lines.append(f"- closest: {c['title']} ({c['year']}{', doi ' + c['doi'] if c.get('doi') else ''}): "
                     f"already reports {c['overlap']}; not reported: {c['missing']}")
    lines.append("Queries: " + " | ".join(res.get("queries", [])))
    return "\n".join(lines)


def _json_block(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.S)
    return m.group(0) if m else "{}"


def dumps(res: dict) -> str:
    return json.dumps(res)[:20000]
