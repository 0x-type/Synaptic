"""Code execution, the `look` tool (direct observation of raw data), literature search."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import HARNESS, SERVICES, SYNLIB_DIR
from .base import ToolContext, ToolResult

_NUMERIC_LINE = re.compile(r"^[\s\d\.\-:+eE,|NA]+$")


def _numeric_chars(text: str) -> tuple[int, int]:
    """Characters of raw numeric data in a block of text, and the longest contiguous run."""
    total, run, best = 0, 0, 0
    for line in text.splitlines():
        digits = sum(ch.isdigit() for ch in line)
        if line.strip() and (digits / max(len(line), 1) >= 0.45 or _NUMERIC_LINE.match(line)):
            total += len(line)
            run += len(line)
            best = max(best, run)
        else:
            run = 0
    return total, best


def python(ctx: ToolContext, code: str = "", timeout: int | None = None) -> ToolResult:
    """Run Python in this task's folder. synlib, numpy, pandas, scipy, matplotlib are importable.
    Files persist between calls; variables do not."""
    if not str(code).strip():
        return ToolResult("No code received. Either put the script in args.code, or write it in a ```python block in "
                          "the SAME reply as {\"action\": \"python\", \"args\": {}}. Your last reply had neither.",
                          is_error=True)
    scripts = ctx.workdir / "scripts"
    scripts.mkdir(exist_ok=True)
    n = len(list(scripts.glob("step_*.py"))) + 1
    script = scripts / f"step_{n:03d}.py"
    script.write_text(code)
    before = {p: p.stat().st_mtime for p in ctx.workdir.rglob("*") if p.is_file()}
    from .. import sandbox

    t0 = time.time()
    try:
        p = sandbox.run_python(script, ctx.workdir, ctx.data_dir, Path(SYNLIB_DIR), timeout or HARNESS.exec_timeout,
                               float(os.environ.get("SYNAPTIC_EXEC_MEM_GB", "2.5")),
                               {"SYNAPTIC_DATA": str(ctx.data_dir), "SYNAPTIC_CAMPAIGN": str(ctx.campaign_dir)})
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, err, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
            f"TIMEOUT after {timeout or HARNESS.exec_timeout}s", -9
    new = sorted(str(p.relative_to(ctx.workdir)) for p in ctx.workdir.rglob("*")
                 if p.is_file() and "scripts" not in p.parts and before.get(p) != p.stat().st_mtime)
    out_t = out if len(out) < 7000 else out[:3000] + f"\n... [{len(out) - 6000} chars truncated] ...\n" + out[-3000:]
    err_t = err[-3000:]
    chars, run = _numeric_chars(out_t)
    if "MemoryError" in err:
        err_t += ("\n[harness] Your script hit the memory ceiling. Load only the columns/rows you need "
                  "(usecols=, filter by variable/years first), avoid wide pivots of the whole region, process in chunks.")
    text = (f"exit code {rc} in {time.time() - t0:.1f}s\n--- stdout ---\n{out_t or '(empty)'}\n"
            + (f"--- stderr ---\n{err_t}\n" if err_t.strip() else "")
            + (f"--- new/updated files ---\n" + "\n".join(new) if new else ""))
    if chars:
        ctx.state["data_chars"] = ctx.state.get("data_chars", 0) + chars
        ctx.state["longest_run"] = max(ctx.state.get("longest_run", 0), run)
    return ToolResult(text, data_chars=chars, files=new, is_error=rc != 0, meta={"longest_run": run})


def look(ctx: ToolContext, path: str, variable: str | None = None, lat: float | None = None,
         lon: float | None = None, months: list[int] | None = None) -> ToolResult:
    """Direct observation. PNG -> the image is put in front of you. CSV -> the raw numbers of one
    location are printed (annual values plus the full monthly table)."""
    p = ctx.resolve(path)
    if not p.exists():
        return ToolResult(f"No such file: {path}", is_error=True)
    ctx.state["looked"] = True
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        ctx.state.setdefault("images_seen", []).append(str(p))
        return ToolResult(f"Image {p.name} is attached below. Describe exactly what you see before interpreting.",
                          images=[str(p)], meta={"image": str(p)})
    if p.suffix.lower() in (".csv",):
        import synlib

        df = synlib.load(p)
        if not {"lat", "lon", "variable", "year", "month", "value"} <= set(df.columns):
            txt = df.head(80).to_string()
            n, run = _numeric_chars(txt)
            ctx.state["data_chars"] = ctx.state.get("data_chars", 0) + n
            ctx.state["longest_run"] = max(ctx.state.get("longest_run", 0), run)
            return ToolResult(f"{p.name}: {len(df)} rows; first 80:\n{txt}", data_chars=n)
        variable = variable or str(df.variable.iloc[0])
        d = df[df.variable == variable]
        cells = d[["lat", "lon"]].drop_duplicates()
        if lat is None or lon is None:
            lat, lon = float(cells.lat.median()), float(cells.lon.median())
        dd = ((cells.lat - lat) ** 2 + (cells.lon - lon) ** 2)
        la, lo = cells.loc[dd.idxmin()]
        s = d[(d.lat == la) & (d.lon == lo)].pivot(index="year", columns="month", values="value").sort_index()
        idx = [m for m in (months or range(1, 13)) if m in s.columns]
        ann = s[idx].mean(axis=1)
        table = s.round(2).to_string()
        body = (f"{variable} at lat {la}, lon {lo} ({len(cells)} cells in file)\n"
                f"Annual{' (months ' + str(months) + ')' if months else ''} means, year:value\n"
                f"{synlib.series_text(ann.index.values, ann.values)}\n\nMonthly table:\n{table}")
        n, run = _numeric_chars(body)
        ctx.state["data_chars"] = ctx.state.get("data_chars", 0) + n
        ctx.state["longest_run"] = max(ctx.state.get("longest_run", 0), run)
        return ToolResult(body, data_chars=n, meta={"longest_run": run})
    txt = p.read_text(errors="replace")[:6000]
    return ToolResult(f"{p.name}:\n{txt}")


# ---------------------------------------------------------------------------- literature

def _openalex(query: str, n: int) -> list[dict]:
    params = {"search": query, "per-page": n, "select": "title,publication_year,doi,cited_by_count,"
              "primary_location,abstract_inverted_index"}
    if SERVICES.openalex_mailto:
        params["mailto"] = SERVICES.openalex_mailto
    import os as _os

    if _os.getenv("OPENALEX_API_KEY"):  # free key from openalex.org: the keyless daily quota is ~100 searches
        params["api_key"] = _os.getenv("OPENALEX_API_KEY")
    r = requests.get("https://api.openalex.org/works", params=params, timeout=40)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((pos, word) for word, ps in inv.items() for pos in ps))
        abstract = " ".join(wd for _, wd in words)
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        out.append({"title": w.get("title"), "year": w.get("publication_year"), "venue": src,
                    "doi": w.get("doi"), "cited": w.get("cited_by_count"), "abstract": abstract})
    return out


def _s2(query: str, n: int) -> list[dict]:
    h = {"x-api-key": SERVICES.s2_api_key} if SERVICES.s2_api_key else {}
    r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                     params={"query": query, "limit": n, "fields": "title,year,venue,externalIds,citationCount,abstract"},
                     headers=h, timeout=40)
    r.raise_for_status()
    return [{"title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
             "doi": (p.get("externalIds") or {}).get("DOI"), "cited": p.get("citationCount"),
             "abstract": p.get("abstract") or ""} for p in r.json().get("data", [])]


def _ads(query: str, n: int) -> list[dict]:
    """NASA Astrophysics Data System: also indexes Earth-science journals (free token)."""
    if not SERVICES.ads_token:
        raise RuntimeError("no ADS_API_TOKEN")
    r = requests.get("https://api.adsabs.harvard.edu/v1/search/query",
                     params={"q": query, "rows": n, "fl": "title,year,pub,doi,citation_count,abstract",
                             "sort": "score desc"},
                     headers={"Authorization": f"Bearer {SERVICES.ads_token}"}, timeout=40)
    r.raise_for_status()
    return [{"title": (d.get("title") or [""])[0], "year": d.get("year"), "venue": d.get("pub"),
             "doi": (d.get("doi") or [None])[0], "cited": d.get("citation_count"),
             "abstract": d.get("abstract") or ""} for d in r.json().get("response", {}).get("docs", [])]


def _crossref(query: str, n: int) -> list[dict]:
    params = {"query": query, "rows": n, "select": "title,issued,container-title,DOI,is-referenced-by-count,abstract"}
    if SERVICES.openalex_mailto:
        params["mailto"] = SERVICES.openalex_mailto
    import os as _os

    if _os.getenv("OPENALEX_API_KEY"):  # free key from openalex.org: the keyless daily quota is ~100 searches
        params["api_key"] = _os.getenv("OPENALEX_API_KEY")
    r = requests.get("https://api.crossref.org/works", params=params, timeout=40)
    r.raise_for_status()
    out = []
    for w in r.json().get("message", {}).get("items", []):
        ab = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
        out.append({"title": (w.get("title") or [""])[0],
                    "year": ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                    "venue": (w.get("container-title") or [""])[0], "doi": w.get("DOI"),
                    "cited": w.get("is-referenced-by-count"), "abstract": " ".join(ab.split())})
    return out


def _europepmc(query: str, n: int) -> list[dict]:
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": query, "format": "json", "pageSize": n, "resultType": "core"}, timeout=40)
    r.raise_for_status()
    return [{"title": d.get("title"), "year": d.get("pubYear"), "venue": d.get("journalTitle"),
             "doi": d.get("doi"), "cited": d.get("citedByCount"), "abstract": d.get("abstractText") or ""}
            for d in r.json().get("resultList", {}).get("result", [])]


def _doaj(query: str, n: int) -> list[dict]:
    """Directory of Open Access Journals: no key needed; covers the open Earth-science journals (EGU/Copernicus,
    ERL, Scientific Reports, ...) with abstracts."""
    import urllib.parse

    q = " ".join(w for w in re.findall(r"[A-Za-z0-9\-]+", query) if len(w) > 2)[:250]
    r = requests.get("https://doaj.org/api/search/articles/" + urllib.parse.quote(q), params={"pageSize": n},
                     timeout=40)
    r.raise_for_status()
    out = []
    for d in r.json().get("results", []):
        b = d.get("bibjson", {})
        doi = next((i.get("id") for i in b.get("identifier", []) if i.get("type") == "doi"), None)
        out.append({"title": b.get("title"), "year": b.get("year"), "venue": (b.get("journal") or {}).get("title"),
                    "doi": doi, "cited": None, "abstract": b.get("abstract") or ""})
    return out


_LIT_CACHE: dict[tuple, list] = {}
_STOP = set("the and for with from that this into over under between during after before about their there these those "
            "using based study analysis change changes trend trends data observed".split())


def _relevance(query: str, hit: dict) -> float:
    """Term overlap between the query and a hit; proper nouns (places, datasets) weigh triple, and a hit that
    misses every proper noun of the query is penalised (a Morocco question is not answered by an Australia paper)."""
    text = f"{hit.get('title') or ''} {hit.get('abstract') or ''}".lower()
    words = [w.strip(",.;:()") for w in query.split()]
    proper = [w.lower() for w in words if w[:1].isupper() and len(w) > 2]
    common = [w.lower() for w in words if not (w[:1].isupper()) and len(w) > 3 and w.lower() not in _STOP]
    score = sum(3 for w in proper if w in text) + sum(1 for w in common if w in text)
    if proper and not any(w in text for w in proper):
        score -= 5
    if common and not any(w in text for w in common):
        return 0  # right place, wrong subject (a Sahel politics paper does not answer a Sahel temperature query)
    if common and sum(1 for w in common if w in text) < min(2, len(common)):
        score -= 3
    return score


def literature_search(ctx: ToolContext, query: str, n: int = 8) -> ToolResult:
    """Search published literature: NASA ADS (if token), OpenAlex, Semantic Scholar, Crossref, Europe PMC.
    Results from all reachable sources are merged and re-ranked by relevance to the query's specific terms.
    Use precise descriptors: variable, region name, period, type of change."""
    n = max(1, min(int(n), 15))
    key = (query.lower().strip(), n)
    errs, used, hits = [], [], []
    import hashlib as _h

    from ..config import RUNS_DIR

    disk = RUNS_DIR / "cache" / "lit" / (_h.md5(repr(key).encode()).hexdigest() + ".json")
    if key not in _LIT_CACHE and disk.exists():  # searches survive restarts and are shared by campaigns
        try:
            c = json.loads(disk.read_text())
            if c.get("hits") and c.get("used") and set(c["used"]) - {"crossref", "europepmc"}:
                _LIT_CACHE[key] = c["hits"]
        except Exception:  # noqa: BLE001
            pass
    if key in _LIT_CACHE:
        hits, used = _LIT_CACHE[key], ["cache"]
    else:
        for fn in (_ads, _openalex, _s2, _doaj, _crossref, _europepmc):
            for attempt in range(2):
                try:
                    got = fn(query, max(n, 10))
                    if got:
                        hits += got
                        used.append(fn.__name__.strip("_"))
                    break
                except Exception as e:  # noqa: BLE001
                    if "429" in str(e) and attempt == 0:
                        time.sleep(2)
                        continue
                    errs.append(f"{fn.__name__.strip('_')}: {str(e)[:60]}")
                    break
            if len(used) >= 3:
                break
        seen, merged = set(), []
        for h in hits:
            t = (h.get("title") or "").lower().strip()
            if t and t not in seen:
                seen.add(t)
                merged.append({**h, "relevance": _relevance(query, h)})
        merged.sort(key=lambda h: -h["relevance"])
        hits = [h for h in merged if h["relevance"] > 0][:n]
        _LIT_CACHE[key] = hits
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_text(json.dumps({"hits": hits, "used": used}))
        except Exception:  # noqa: BLE001
            pass
    if not hits:
        ctx.state.setdefault("lit_queries", []).append(query)
        return ToolResult(f"No relevant literature found for '{query}' (sources tried: {used or 'none'}; "
                          f"{'; '.join(errs)}). Absence of evidence from this search is weak evidence: say so.",
                          is_error=not used)
    ctx.state.setdefault("lit_sources", []).extend(u for u in used if u != "cache")
    weak = set(used) <= {"crossref", "europepmc", "doaj", "cache"}
    used = ",".join(used)
    lines = [f"Literature for: {query}   (source: {used})"]
    if weak:
        lines.append("COVERAGE WEAK: only general indexes answered (the main scholarly indexes were unreachable). "
                     "An off-topic or empty result here is NOT evidence of novelty; say novelty is provisional.")
    for i, h in enumerate(hits, 1):
        ab = (h["abstract"] or "")[:420]
        lines.append(f"[{i}] {h['title']} ({h['year']}) {h['venue'] or ''} | cited {h['cited']} | {h['doi'] or ''} "
                     f"| relevance {h.get('relevance', '?')}\n"
                     f"    {ab}{'...' if len(h['abstract'] or '') > 420 else ''}")
    ctx.state.setdefault("lit_queries", []).append(query)
    return ToolResult("\n".join(lines))
