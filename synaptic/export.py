"""Judge-facing outputs, generated automatically at the end of a campaign (and with `synaptic export <campaign>`):

  results/index.html        one self-contained page: headline results, dossiers, research programme, map, data
                            issues for NASA, how the agents worked (figures embedded; no external requests)
  results/NASA_data_issues.md   the data-issues report for the data providers (GES DISC / POWER): where and when the
                            record disagrees with independent observations, with the evidence
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_LABEL = {"supported_new": ("New result", "new"), "supported": ("Supported", "ok"), "known": ("Already known", "known"),
                "refuted": ("Refuted (contradicted)", "bad"), "not_shown": ("Not shown (not significant)", "muted"), "inconclusive": ("Inconclusive (could not test)", "muted"), "testing": ("Testing", "muted"),
                "open": ("Open", "muted"), "dropped": ("Dropped by PI", "muted"), "reserve": ("Reserve (not tested)", "muted"),
                "awaiting_confirmation": ("Awaiting held-out confirmation", "muted"), "new_signal": ("New signal", "new"), "known_phenomenon": ("Known phenomenon", "known"),
                "artifact": ("Data artifact", "bad"), "not_robust": ("Not robust", "muted")}


def _img(path: str | Path, max_w: int = 1100) -> str | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size > 6_000_000:
        return None
    try:
        from PIL import Image

        im = Image.open(p)
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _md(text: str) -> str:
    try:
        import markdown

        return markdown.markdown(text or "", extensions=["tables", "fenced_code"])
    except Exception:  # noqa: BLE001
        return "<pre>" + html.escape(text or "") + "</pre>"


def _lede(text: str, n: int = 420) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text[:n]
    out = cut[: cut.rfind(". ") + 1] if ". " in cut else cut.rsplit(" ", 1)[0] + "..."
    out = re.sub(r"\s*\(?\d+[.)]$", "", out).rstrip()  # a cut that ends on a list number ("... them: 1.")
    return out[:-1] + "." if out.endswith(":") else out


def _box(text: str):
    """Centre of the first 'lat1 to lat2[N|S], lon1-lon2[E|W]' box in a statement, e.g. '(-35 to -15S, 15-40E)'."""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\s*([NS])?\s*,\s*"
                  r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\s*([EW])", text)
    if not m:
        return None
    la = [float(m.group(1)), float(m.group(2))]
    lo = [float(m.group(4)), float(m.group(5))]
    if m.group(3) == "S":
        la = [-abs(v) for v in la]
    if m.group(6) == "W":
        lo = [-abs(v) for v in lo]
    return (sum(la) / 2, sum(lo) / 2) if all(-90 <= v <= 90 for v in la) else None


def _bias_html(record, cid: str) -> str:
    row = record.one("SELECT body FROM kb WHERE campaign_id=? AND title='Tournament position bias' ORDER BY id DESC",
                     (cid,))
    if not row:
        return ""
    try:
        b = json.loads(row["body"])
    except Exception:  # noqa: BLE001
        return ""
    share = b.get("first_shown_share")
    return (f'<p class="muted">Ranking: {b.get("pairs_both_orders")} pairs judged in both orders; the judge agreed with '
            f'itself on {b.get("pairs_consistent")} of them; the report shown first won {share if share is not None else "-"} '
            f'of games (0.5 = no position bias). Pairs where the order decided the winner count as ties.</p>')


def _badge(status: str) -> str:
    label, cls = STATUS_LABEL.get(status or "", (status or "-", "muted"))
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def _find_fig(cdir: Path, name: str) -> Path | None:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    hits = list(cdir.glob(f"**/{p.name}"))
    return hits[0] if hits else None


def _map_png(cdir: Path, data_index: dict, points: list[dict]) -> str | None:
    """Land cells in grey (the campaign grid is the basemap) and the located results on top."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = None
        for v, f in data_index.items():
            if Path(f).exists():
                base = pd.read_csv(f, usecols=["lat", "lon"]).drop_duplicates()
                break
        fig, ax = plt.subplots(figsize=(11, 5.2))
        if base is not None:
            ax.scatter(base.lon, base.lat, s=3, c="#c9c9c9", marker="s", linewidths=0)
        col = {"new": "#d1495b", "ok": "#2e86ab", "known": "#7a7a7a", "bad": "#edae49", "muted": "#9aa"}
        for p in points:
            cls = STATUS_LABEL.get(p["status"], ("", "muted"))[1]
            ax.scatter([p["lon"]], [p["lat"]], s=60, c=col[cls], edgecolors="k", linewidths=0.5, zorder=3)
            ax.annotate(p["label"], (p["lon"], p["lat"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 82)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        out = cdir / "results" / "map.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return _img(out)
    except Exception:  # noqa: BLE001
        return None


def data_issues(record, cid: str, cdir: Path) -> tuple[str, list[dict]]:
    """Markdown report for the data providers + rows for the page."""
    rows = []
    for c in record.candidates(cid):
        f = c["filters"] or {}
        note = c["note"] or ""
        if c["outcome"] != "artifact":
            continue
        if "[independent check]" in note:
            src = "evidence engine (independent observations disagree)"
        elif "[screened automatically]" in note:
            continue  # summarised by the screen table below
        else:
            src = "agents (deep dive / follow-up)"
        rows.append({"variable": c["variable"], "kind": c["cls"], "where": f.get("place") or f"{c['lat']:.1f}, {c['lon']:.1f}",
                     "cells": c["n_cells"], "onset": c["onset"], "source": src, "evidence": note[:600]})
    trust = cdir / "trust.csv"
    screen_md = ""
    if trust.exists():
        t = pd.read_csv(trust)
        t = t[t.kind != "known_caveat"]
        if len(t):
            brk = (t[t.kind == "break"].groupby(["variable", "year_from"]).size().sort_values(ascending=False)
                   .head(10).rename("cells").reset_index())
            screen_md = ("\n## Breaks found by the automatic screen (MERRA-2 / POWER vs GPCP rain and Berkeley Earth "
                         "temperature)\n\n" + brk.to_markdown(index=False) + "\n")
    md = ["# Data issues found by Synaptic in NASA MERRA-2 / POWER records",
          f"\nGenerated {time.strftime('%Y-%m-%d')}. Each issue is a place and period where the NASA record disagrees "
          "with independent observations (GPCP and GPM IMERG rain, Berkeley Earth temperature, CHIRPS, GHCN stations) "
          "or where the agents traced a signal to data production rather than climate.\n",
          f"\n## Issues ({len(rows)})\n"]
    for r in rows[:60]:
        md.append(f"- **{r['variable']} {r['kind']}**, {r['where']}, {r['cells']} cells"
                  + (f", onset {r['onset']}" if r['onset'] else "") + f" ({r['source']}): {r['evidence'][:400]}")
    md.append(screen_md)
    md.append("\n## Suggested actions\n\n- Check the observation-based precipitation correction (CPC/CMAP) for the "
              "listed regions and years; GPCP and IMERG are the comparison products.\n- Propagation: rain-correction "
              "breaks also move soil moisture, humidity and evaporation in MERRA-2 land fields.\n")
    return "\n".join(md), rows


def export(record, cid: str, out_dir: Path | None = None) -> Path:
    camp = record.one("SELECT * FROM campaigns WHERE id=?", (cid,))
    brief = json.loads(camp["brief"])
    cdir = Path(camp["workdir"])
    out_dir = out_dir or cdir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    di = json.loads((cdir / "data_index.json").read_text()) if (cdir / "data_index.json").exists() else {}
    stats = record.stats(cid)
    tasks = record.tasks(cid)
    hyps = record.hypotheses(cid)
    cands = record.candidates(cid)
    reports = record.q("SELECT r.*, COALESCE(g.rating, 0) rating, g.rank FROM reports r LEFT JOIN ratings g ON "
                       "g.report_id = r.id AND g.campaign_id = r.campaign_id WHERE r.campaign_id=? ORDER BY "
                       "COALESCE(g.rank, 999), r.created", (cid,))
    issues_md, issues = data_issues(record, cid, cdir)
    (out_dir / "NASA_data_issues.md").write_text(issues_md)

    n_screen = sum(1 for c in cands if "[screened automatically]" in (c["note"] or ""))
    n_contra = sum(1 for c in cands if "[independent check]" in (c["note"] or ""))
    n_robust = sum(1 for c in cands if "[robustness suite]" in (c["note"] or ""))
    refs = [t for t in tasks if t["kind"] == "referee" and t["status"] == "done"]
    gpu_jobs = len(list((cdir / "gpu_jobs").glob("j*"))) if (cdir / "gpu_jobs").exists() else 0
    focused = brief.get("mode") == "focused"
    nums = [("land cells", next((pd.read_csv(f, usecols=["lat", "lon"]).drop_duplicates().shape[0]
                                 for f in di.values() if Path(f).exists()), 0)),
            ("years", f"{brief.get('start_year')}-{brief.get('end_year')}"),
            ("anomaly families", len([c for c in cands if not str(c["family"]).startswith("COUPLING")])),
            ("closed automatically", n_screen + n_contra + n_robust),
            ("hypotheses tested", len([h for h in hyps if h["status"] not in ("open", "testing", "reserve")])),
            ("referee reports", len(refs)), ("agent tasks", len([t for t in tasks if t["status"] == "done"])),
            ("agent-hours", stats["agent_hours"]), ("GPU jobs", gpu_jobs)]
    if focused:  # one open problem: no survey of anomaly families
        n_conf = len([h for h in hyps if (h.get("confirmation") or "").strip()])
        nums = [("grid cells", nums[0][1]), ("years", nums[1][1]), ("datasets", len(brief.get("datasets", []))),
                ("held out for confirmation", len(brief.get("holdout", []))),
                ("hypotheses tested", nums[4][1]), ("held-out confirmations run", n_conf),
                ("refuted", len([h for h in hyps if h["status"] == "refuted"])),
                ("not shown", len([h for h in hyps if h["status"] == "not_shown"])),
                ("referee reports", len(refs)), ("agent tasks", nums[6][1]), ("agent-hours", stats["agent_hours"])]

    # headline results
    head = []
    _order = ["supported_new", "supported", "known", "refuted", "not_shown", "inconclusive"]
    for h in sorted(hyps, key=lambda h: (_order.index(h["status"]) if h["status"] in _order
                                         else 9, -h["priority"])):
        if h["status"] in ("open", "testing", "reserve"):
            continue
        rv = {}
        try:
            rv = json.loads(h["review"]) if h["review"] else {}
        except Exception:  # noqa: BLE001
            pass
        dossier = next((r["title"] for r in reports if r["family"] == h["id"] and r["title"]), None)
        head.append({"id": h["id"], "status": h["status"], "title": dossier or h["title"], "claim": h["statement"],
                     "confirmation": h.get("confirmation") or "",
                     "result": h["result"], "novelty": h["novelty"], "referee": rv.get("decision"),
                     "objections": rv.get("objections", "")})
    fam_head = [c for c in cands if c.get("promoted") and c["outcome"] in ("new_signal", "known_phenomenon", "artifact")]
    points = [{"lat": c["lat"], "lon": c["lon"], "status": c["outcome"], "label": c["family"].split(":")[0]}
              for c in fam_head if c["lat"] is not None]
    for h in hyps:  # supported hypotheses at the centre of the box their statement names
        if h["status"] in ("supported_new", "supported"):
            b = _box(h["statement"] or "")
            if b:
                points.append({"lat": b[0], "lon": b[1], "status": "new_signal" if h["status"] == "supported_new"
                               else "known_phenomenon", "label": h["id"]})
    map_uri = _map_png(cdir, di, points)

    def card(h):
        ref = (f'<p class="ref">Referee: <b>{html.escape(h["referee"])}</b>. {html.escape((h["objections"] or "")[:400])}</p>'
               if h.get("referee") else '<p class="ref muted">Not refereed (only supported claims go to a referee).</p>')
        nov = f'<p><b>What is new:</b> {html.escape(h["novelty"][:700])}</p>' if h.get("novelty") else ""
        conf = (f'<p><b>Held-out confirmation:</b> {html.escape(h["confirmation"][:600])}</p>'
                if h.get("confirmation") else "")
        return (f'<article class="card"><div class="row">{_badge(h["status"])}<span class="hid">{h["id"]}</span></div>'
                f'<h3>{html.escape(h["title"] or "")}</h3><p class="claim"><b>Hypothesis tested:</b> {html.escape(h["claim"] or "")}</p>'
                f'<p><b>Result:</b> {html.escape((h["result"] or "")[:900])}</p>{conf}{nov}{ref}</article>')

    def fam_card(c):
        f = c["filters"] or {}
        return (f'<article class="card"><div class="row">{_badge(c["outcome"])}<span class="hid">{html.escape(c["family"])}</span>'
                f'</div><h3>{html.escape(c["cls"])} in {html.escape(c["variable"])}, {c["n_cells"]} cells</h3>'
                f'<p class="claim">{html.escape(f.get("place") or "")}</p>'
                f'<p><b>Independent data:</b> {html.escape(str(f.get("independent") or "-").replace("_", " "))}; robustness '
                f'{html.escape(str(f.get("robustness") or "-"))}</p><p>{html.escape((c["note"] or "")[-700:])}</p></article>')

    dossiers = []
    for r in reports:
        figs = "".join(f'<img alt="figure" src="{u}">' for u in
                       (_img(p) for p in (_find_fig(cdir, x) for x in json.loads(r["figures"] or "[]")) if p) if u)
        rank = f"#{r['rank']} " if r["rank"] else ""
        dossiers.append(f'<details><summary>{rank}{html.escape(r["title"] or "")}</summary><div class="md">'
                        f'{_md(r["body"])}{figs}</div></details>')
    qtext = {m.group(1): m.group(2).strip() for m in re.finditer(
        r"(?:^|\s)(\d+)[.)]\s+(.+?)(?=\s\d+[.)]\s|$)", " ".join((brief.get("question") or "").split()))}

    def _q(q):
        q = str(q or "").strip()
        return f"Q{q}: {qtext[q]}" if q in qtext else q

    board = "".join(f'<tr><td>{h["id"]}</td><td>{html.escape(_q(h["question"]))}</td><td>{html.escape(h["title"] or "")}'
                    f'</td><td>{_badge(h["status"])}</td><td>{html.escape((h["result"] or "")[:260])}</td></tr>'
                    for h in hyps)
    iss = "".join(f'<tr><td>{html.escape(i["variable"])}</td><td>{html.escape(str(i["where"]))}</td><td>'
                  f'{i["onset"] or ""}</td><td>{html.escape(i["source"])}</td><td>{html.escape(i["evidence"][:300])}</td></tr>'
                  for i in issues[:40])
    trust_uri = _img(cdir / "figures" / "trust_map.png")
    cpl_uri = _img(cdir / "learn" / "coupling_change.png")
    title = f"Synaptic: {brief.get('title', 'campaign')}"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synaptic Results</title>
<style>
:root{{--bg:#f7f7f4;--fg:#1d2327;--muted:#66707a;--card:#ffffff;--line:#e2e2dc;--accent:#2e86ab;--new:#d1495b;--ok:#2e86ab;--known:#7a7a7a;--bad:#c98a12}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#14171a;--fg:#e8eaed;--muted:#9aa3ab;--card:#1d2227;--line:#2c333a;--accent:#5fb3d9;--known:#a0a0a0;--bad:#e3b04b;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#14171a;--fg:#e8eaed;--muted:#9aa3ab;--card:#1d2227;--line:#2c333a;--accent:#5fb3d9;--known:#a0a0a0;--bad:#e3b04b;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}h1{{font-size:1.7rem;margin:.2em 0}}h2{{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.3em}}
.lede{{color:var(--muted);max-width:780px}}.nums{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:18px 0}}
.num{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}.num b{{display:block;font-size:1.35rem}}.num span{{color:var(--muted);font-size:.85rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.card h3{{margin:.4em 0 .2em;font-size:1.05rem}}.claim{{color:var(--muted)}}.row{{display:flex;gap:8px;align-items:center}}.hid{{color:var(--muted);font-size:.8rem}}
.badge{{font-size:.75rem;padding:2px 8px;border-radius:99px;color:#fff;background:var(--known)}}.badge.new{{background:var(--new)}}.badge.ok{{background:var(--ok)}}.badge.bad{{background:var(--bad)}}.badge.muted{{background:#8a949c}}
.ref{{font-size:.9rem;border-top:1px dashed var(--line);padding-top:6px}}.muted{{color:var(--muted)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:8px 14px}}summary{{cursor:pointer;font-weight:600}}
.md img,figure img,.fig{{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);margin:8px 0}}.md table,table{{border-collapse:collapse;width:100%;font-size:.88rem}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}.scroll{{overflow-x:auto}}
ol.steps li{{margin:.3em 0}}footer{{margin-top:3em;color:var(--muted);font-size:.85rem}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lede">{html.escape(_lede(brief.get('question', '')))}</p>
<div class="nums">{''.join(f'<div class="num"><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>' for k, v in nums)}</div>
<h2>Headline results</h2>
<p class="muted">Research-programme hypotheses (pre-registered tests, independent data, referee) and the strongest anomaly families. "Already known" and "Refuted" are honest results too.</p>
<div class="cards">{''.join(card(h) for h in head[:12]) or '<p class="muted">No hypothesis closed yet.</p>'}{''.join(fam_card(c) for c in fam_head[:8])}</div>
<h2>Where</h2>{f'<img class="fig" alt="map of results" src="{map_uri}">' if map_uri else '<p class="muted">(no located results)</p>'}
<h2>Dossiers</h2>{''.join(dossiers) or '<p class="muted">No dossier filed yet.</p>'}
<h2>Research programme</h2><div class="scroll"><table><tr><th>id</th><th>question</th><th>hypothesis</th><th>status</th><th>result</th></tr>{board}</table></div>
<h2>Data issues for NASA</h2><p class="muted">Where the NASA record disagrees with independent observations. Full report: NASA_data_issues.md.</p>
{f'<img class="fig" alt="trust map" src="{trust_uri}">' if trust_uri else ''}
<div class="scroll"><table><tr><th>variable</th><th>where</th><th>onset</th><th>found by</th><th>evidence</th></tr>{iss}</table></div>
{_bias_html(record, cid)}
<h2>How the agents worked</h2>
<ol class="steps"><li><b>Data screen</b>: every cell compared with GPCP rain and Berkeley Earth temperature before any agent ran.</li>
<li><b>GPU model</b>: a cross-fitted network learned every cell's month-to-month climate; changes in driver influence with bootstrap CIs and an event-strength control.</li>
<li><b>Research programme</b>: a PI agent wrote falsifiable hypotheses; tests ran beside the survey; lab meetings re-planned.</li>
<li><b>Survey</b>: sweep of every cell, anomaly families, pre-registered filters.</li>
<li><b>Evidence engine</b>: the same statistic recomputed in GPCP, IMERG, Berkeley Earth and GLDAS; robustness suite; field significance; place names; merging. {n_screen + n_contra + n_robust} families closed automatically.</li>
<li><b>Deep dives and science</b>: mechanism, drivers, predictability, impacts.</li>
<li><b>Held-out confirmation</b> (v4): every hypothesis carries a test pre-registered before any result, run by code on a dataset the exploring agents never saw; only a pass makes it "supported". Verdicts can only go down afterwards.</li>
<li><b>Referees and audit</b>: every supported claim attacked by an independent referee agent; a sceptical auditor sets final verdicts.</li>
<li><b>Dossiers and tournament</b>: results written up and ranked, every pair judged in both orders.</li></ol>
{f'<img class="fig" alt="GPU coupling changes" src="{cpl_uri}">' if cpl_uri else ''}
<footer>Data: NASA MERRA-2 (GES DISC), NASA POWER, GPM IMERG, GLDAS (Earthdata), GPCP, Berkeley Earth, CHIRPS, GHCN-Daily, NOAA PSL indices. Agents: NVIDIA Nemotron. Generated {time.strftime('%Y-%m-%d %H:%M')} by Synaptic.</footer>
</main></body></html>"""
    out = out_dir / "index.html"
    out.write_text(page)
    # the same page as a fragment (title + style + content, no document wrapper) for publishing as an artifact
    head = page.split("<head>", 1)[1].split("</head>", 1)[0]
    head = head.replace('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', "")
    body = page.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    (out_dir / "artifact.html").write_text('<meta name="claude:viewport-fit-sides" content="cover">\n' + head + "\n" + body)
    return out
