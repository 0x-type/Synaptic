"""Command line: python -m synaptic <command>

  check                      verify keys, model availability, vision, NASA endpoints, Kaggle
  run BRIEF [--mock]         run a full discovery campaign
  dashboard [--port 8050]    mission-control web UI
  bench [--mock] [--models a,b] [--levels 1,2,3,4,5] [--attempts N]   planted-anomaly benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def cmd_check(_):
    from .config import MODELS, SERVICES

    ok = True
    print("== NVIDIA API")
    if not MODELS.api_key:
        print("  NVIDIA_API_KEY missing in .env")
        ok = False
    else:
        from openai import OpenAI

        c = OpenAI(base_url=MODELS.base_url, api_key=MODELS.api_key)
        ids = [m.id for m in c.models.list().data]
        for role in ("worker", "supervisor", "judge"):
            m = MODELS.model_for(role)
            print(f"  {role:10s} {m}: {'served' if m in ids else 'NOT SERVED'}")
            ok &= m in ids
        try:
            from .llm import NvidiaLLM

            llm = NvidiaLLM()
            t = time.time()
            r = llm.complete([{"role": "user", "content": 'Reply with exactly: {"action": "ok", "args": {}}'}],
                             "worker", max_tokens=300)
            print(f"  text call OK in {time.time() - t:.1f}s: {r.text[:80]!r} (reasoning {len(r.reasoning)} chars)")
            if MODELS.vision:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import tempfile

                from .llm import image_part

                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                plt.figure(figsize=(3, 2))
                plt.plot([1, 2, 3, 10], "o-")
                plt.title("test")
                plt.savefig(f)
                r = llm.complete([{"role": "user", "content": [
                    {"type": "text", "text": "How many points are plotted and which one is the outlier? One line."},
                    image_part(f)]}], "worker", max_tokens=300)
                print(f"  vision call OK: {r.text[:120]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  call FAILED: {e}")
            ok = False
    print("== NASA endpoints")
    import requests

    for name, url in [("POWER", "https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M&community=AG"
                       "&longitude=-7.6&latitude=33.6&start=2020&end=2020&format=JSON"),
                      ("GIBS", "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetCapabilities"),
                      ("CMR", "https://cmr.earthdata.nasa.gov/search/collections.json?keyword=GPM&page_size=1"),
                      ("OpenAlex", "https://api.openalex.org/works?search=drought&per-page=1")]:
        try:
            s = requests.get(url, timeout=30).status_code
        except Exception as e:  # noqa: BLE001
            s = f"error {e.__class__.__name__}"
        print(f"  {name:9s} {s}")
    print("== NASA Earthdata (raw MERRA-2)")
    if SERVICES.earthdata_token:
        try:
            r = requests.head("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXFLX.5.12.4/1998/"
                              "MERRA2_200.tavgM_2d_flx_Nx.199801.nc4", allow_redirects=True, timeout=60,
                              headers={"Authorization": f"Bearer {SERVICES.earthdata_token}"})
            print(f"  MERRA-2 file access: {r.status_code}" + ("" if r.status_code == 200 else
                  " (token expired, or approve 'NASA GESDISC DATA ARCHIVE' in your Earthdata profile)"))
        except Exception as e:  # noqa: BLE001
            print(f"  MERRA-2 check failed: {e.__class__.__name__}")
    else:
        print("  EARTHDATA_TOKEN not set: agents cannot read raw MERRA-2 (optional)")
    print("== Kaggle GPU")
    if os.getenv("KAGGLE_API_TOKEN") or SERVICES.kaggle_key:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            q = api.quota_view()
            g = q.gpu_quota
            print(f"  authenticated as {SERVICES.kaggle_username or api.get_config_value('username')}; GPU used "
                  f"{g.time_used.total_seconds() / 3600:.1f} h this week" if g else "  authenticated")
        except Exception as e:  # noqa: BLE001
            print(f"  Kaggle auth FAILED: {e}")
    else:
        print("  KAGGLE_API_TOKEN / KAGGLE_USERNAME not set: GPU jobs will run locally on CPU")
    print("\nREADY" if ok else "\nFix the items above, or use --mock for an offline demo.")


def cmd_run(a):
    from .harness import Campaign

    c = Campaign(a.brief, mock=a.mock, db=a.db)
    print(f"campaign {c.cid} -> {c.cdir}\nOpen the dashboard with: python -m synaptic dashboard")
    c.run()


def cmd_resume(a):
    from .harness import Campaign

    c = Campaign.attach(a.campaign, db=a.db)
    print(f"resuming {c.cid} ({'mock' if c.mock else 'live'}) -> {c.cdir}")
    c.run()


def cmd_preload(a):
    """Compute and save the Trend Detective's example cases, so a demo answers at once (and offline)."""
    from .dashboard import detective as d

    for demo in d.DEMOS:
        if a.only and demo["id"] not in a.only:
            continue
        if not a.fresh and d.load_saved(demo["spec"]) is not None:
            print(f"  {demo['id']}: already saved")
            continue
        t = time.time()
        try:
            out = d.run_analysis(demo["spec"], lambda m: None)
            d.save_result(demo["spec"], out, shipped=a.ship)
            print(f"  {demo['id']}: saved in {time.time() - t:.0f}s: {str(out.get('verdict'))[:110]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {demo['id']}: FAILED ({type(e).__name__}: {str(e)[:160]})")


def cmd_dashboard(a):
    from .dashboard.app import create_app

    app = create_app(a.db)
    print(f"Mission control on http://127.0.0.1:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


def cmd_bench(a):
    from .bench import run_bench

    run_bench(models=[m for m in (a.models or "").split(",") if m] or None,
              levels=[int(x) for x in a.levels.split(",")], attempts=a.attempts, mock=a.mock, db=a.db)


def cmd_setup(_):
    """Write .env interactively. Keys are typed hidden and never echoed or logged."""
    import getpass
    from pathlib import Path

    from .config import ROOT

    env = ROOT / ".env"
    current = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                current[k.strip()] = v.strip()
    print("Synaptic setup. Press Enter to keep the current value (shown as [set] when one exists).\n")

    def ask(key, prompt, secret=True, default=""):
        has = current.get(key) or default
        label = f"{prompt} [{'set' if secret and has else has}]: "
        v = getpass.getpass(label) if secret else input(label)
        if v.strip():
            current[key] = v.strip()
        elif default and key not in current:
            current[key] = default

    ask("NVIDIA_API_KEY", "NVIDIA API key (nvapi-...)")
    ask("SYNAPTIC_MODEL", "Model", secret=False, default="nvidia/nemotron-3.5-super-vl-120b-a12b")
    ask("KAGGLE_API_TOKEN", "Kaggle API token (KGAT_..., optional)")
    ask("KAGGLE_USERNAME", "Kaggle username (optional)", secret=False)
    ask("EARTHDATA_TOKEN", "NASA Earthdata Login token (optional, urs.earthdata.nasa.gov -> Generate Token; "
                           "approve 'NASA GESDISC DATA ARCHIVE') for raw MERRA-2")
    ask("ADS_API_TOKEN", "NASA ADS token (optional, recommended for the novelty test)")
    ask("OPENALEX_MAILTO", "Your e-mail for OpenAlex/Crossref polite pool (optional)", secret=False)
    lines = ["# Written by `python -m synaptic setup`. Never commit or share this file."]
    lines += [f"{k}={v}" for k, v in current.items() if v]
    env.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(env, 0o600)
    except OSError:
        pass
    print(f"\nSaved {env}. Running the check...\n")
    import subprocess

    subprocess.run([sys.executable, "-m", "synaptic", "check"], cwd=ROOT)


SMOKE_BRIEF = {
    "title": "Smoke test: central Morocco water cycle",
    "question": "Did root-zone soil moisture and daytime heat over central Morocco change between 1981 and 2025 "
                "in ways not yet documented (abrupt shifts, seasonal changes, drying that outpaces rainfall)?",
    "region": {"name": "central Morocco", "lat_min": 30.0, "lat_max": 34.0, "lon_min": -9.0, "lon_max": -5.0},
    "start_year": 1981, "end_year": 2025, "variables": ["GWETROOT", "T2M_MAX"], "max_variables": 3,
    "notes": "Small validation campaign. Be rigorous; it is fine to conclude that nothing is new.",
}


def cmd_smoke(a):
    """A small end-to-end campaign that validates the live model inside the harness, then diagnoses it."""
    from .config import HARNESS
    from .harness import Campaign

    HARNESS.max_promoted, HARNESS.max_tasks, HARNESS.concurrency = 2, 14, 2
    HARNESS.max_followups_per_task = 2
    t0 = time.time()
    c = Campaign(dict(SMOKE_BRIEF), mock=a.mock, db=a.db)
    c.record.x("UPDATE campaigns SET name=? WHERE id=?", (SMOKE_BRIEF["title"], c.cid))
    print(f"Smoke campaign {c.cid} ({'mock' if a.mock else 'LIVE: ' + c.runner.llm.cfg.model_for('worker')})")
    print("Watch it in mission control: python -m synaptic dashboard\n")
    try:
        c.run()
    except Exception as e:  # noqa: BLE001
        print(f"\nCampaign crashed: {e}")
    r, cid = c.record, c.cid
    n = lambda sql: (r.one(sql, (cid,)) or {}).get("n", 0)  # noqa: E731
    tasks = r.q("SELECT kind, status, COUNT(*) n FROM tasks WHERE campaign_id=? GROUP BY kind, status", (cid,))
    parse_err = n("SELECT COUNT(*) n FROM events WHERE campaign_id=? AND type='error' AND content LIKE 'No valid action%'")
    gates = n("SELECT COUNT(*) n FROM events WHERE campaign_id=? AND type='gate'")
    images = n("SELECT COUNT(*) n FROM events WHERE campaign_id=? AND type='image'")
    revs = n("SELECT COALESCE(SUM(revisions),0) n FROM tasks WHERE campaign_id=?")
    fus = n("SELECT COUNT(*) n FROM tasks WHERE campaign_id=? AND kind='followup'")
    crashes = r.q("SELECT task_id, substr(content, -300) c FROM events WHERE campaign_id=? AND role='harness' "
                  "AND type='error'", (cid,))
    st = r.stats(cid)
    print("\n================ SMOKE TEST DIAGNOSIS ================")
    print(f"time {time.time() - t0:.0f}s | sessions {st['sessions']} | tokens {st['tokens_in'] + st['tokens_out']:,}")
    for t in tasks:
        print(f"  {t['kind']:10s} {t['status']:9s} x{t['n']}")
    print(f"action-format errors: {parse_err}   finish refusals (look gate etc.): {gates}   "
          f"images looked at: {images}   supervisor revisions: {revs}   follow-ups opened: {fus}")
    print("\nVerdicts:")
    for k in r.candidates(cid, promoted=1):
        print(f"  {k['family']:30s} {k['outcome'] or '(none)':17s} {(k['note'] or '')[:160]}")
    for rep in r.q("SELECT title FROM reports WHERE campaign_id=?", (cid,)):
        print(f"  report: {rep['title']}")
    for cr in crashes[:3]:
        print(f"\nCRASH in {cr['task_id']}: ...{cr['c']}")
    done = {t["kind"] for t in tasks if t["status"] == "done"}
    ok = {"sweep", "classify", "census"} <= done and not crashes
    print("\nRESULT:", ("PASS - the pipeline runs end to end (scripted mock, no model)." if a.mock else
                        "PASS - the live model drives the full pipeline.") if ok else
          "NEEDS ATTENTION - send this diagnosis (no keys in it) so the harness can be tuned.")


def cmd_export(a):
    from pathlib import Path

    from .export import export
    from .record import Record

    from .config import RUNS_DIR

    r = Record(a.db or RUNS_DIR / "synaptic.db")
    out = export(r, a.campaign, Path(a.out) if a.out else None)
    print(f"results page: {out}\nNASA data-issues report: {out.parent / 'NASA_data_issues.md'}")


def cmd_reproduce(a):
    from .reproduce import run_all

    raise SystemExit(run_all(a.paths or None))


def cmd_sandbox(a):
    from . import sandbox
    from .reproduce import ROOT

    if a.action == "build":
        raise SystemExit(sandbox.build_image(ROOT))
    print(f"sandbox mode: {sandbox.mode()} (network isolated: {sandbox.network_isolated()}); "
          f"secret files protected: {sandbox.secret_files() or 'none found'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="synaptic")
    p.add_argument("--db", default=None, help="SQLite record (default runs/synaptic.db)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sm = sub.add_parser("smoke")
    sm.add_argument("--mock", action="store_true")
    sm.set_defaults(fn=cmd_smoke)
    r = sub.add_parser("run")
    r.add_argument("brief")
    r.add_argument("--mock", action="store_true")
    r.set_defaults(fn=cmd_run)
    rs = sub.add_parser("resume", help="continue an interrupted campaign")
    rs.add_argument("campaign", help="campaign id, e.g. c_1a2b3c4d (shown in the dashboard and at launch)")
    rs.set_defaults(fn=cmd_resume)
    pl = sub.add_parser("preload", help="compute and save the Trend Detective example cases for an instant demo")
    pl.add_argument("--fresh", action="store_true", help="recompute even if a saved result exists")
    pl.add_argument("--ship", action="store_true", help="save into the repository (demo_cache/) instead of runs/")
    pl.add_argument("only", nargs="*", help="example ids (default: all)")
    pl.set_defaults(fn=cmd_preload)
    d = sub.add_parser("dashboard")
    d.add_argument("--port", type=int, default=8050)
    d.add_argument("--host", default="127.0.0.1")
    d.set_defaults(fn=cmd_dashboard)
    ex = sub.add_parser("export", help="write the results page and the NASA data-issues report for a campaign")
    ex.add_argument("campaign")
    ex.add_argument("--out", default=None)
    ex.set_defaults(fn=cmd_export)
    rp = sub.add_parser("reproduce", help="re-check every evidence bundle (evidence/ and campaign results)")
    rp.add_argument("paths", nargs="*")
    rp.set_defaults(fn=cmd_reproduce)
    sb = sub.add_parser("sandbox", help="agent-code sandbox: 'status' or 'build' the Docker image")
    sb.add_argument("action", choices=["status", "build"])
    sb.set_defaults(fn=cmd_sandbox)
    b = sub.add_parser("bench")
    b.add_argument("--mock", action="store_true")
    b.add_argument("--models", default="")
    b.add_argument("--levels", default="1,2,3,4,5")
    b.add_argument("--attempts", type=int, default=10)
    b.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
