# The Synaptic research harness, in detail

The [README](../README.md) gives the overview. This page is the full description of the multi-agent harness: how a
campaign runs, how it maps onto the method it adapts, what changed after an expert review, how it was tested with
Nemotron, and how to run and configure it.

## How a campaign runs (version 3: a senior research group)

Version 2 moved quality control into minutes of code. Version 3 changes how the agents think: a principal
investigator turns the brief into falsifiable hypotheses that are tested from the first hour, beside the survey,
and every claim has to survive automatic independent checks and an adversarial referee before it is reported.

| Stage | Who | What |
|---|---|---|
| Plan | planner | variables, thresholds locked before any data |
| Data screen | code, ~1 min | MERRA-2/POWER vs GPCP rain and Berkeley Earth temperature, gap fills, suppressed variability, known issues: the trust map |
| GPU model | harness, Kaggle 2x T4 | cross-fitted network (every year scored out of sample) learns each trusted cell's month-to-month climate from all variables and ENSO, IOD, Atlantic, PDO, NAO. Change in each driver's influence with a year-bootstrap CI and an **event-strength control** (a "stronger teleconnection" that is only "more big El Ninos in the late period" is rejected) |
| **Research programme** | PI agent | 6-10 falsifiable hypotheses with pre-registered tests and kill criteria, from the brief, the screen and the GPU results. Tests run on reserved worker slots **in parallel with the survey**; **lab meetings** every few tasks kill dead ends, order deeper tests, add hypotheses from surprising survey signals |
| Sweep, classify | agents | change detection on every cell, anomaly families |
| **Census + evidence engine** | code, then agent | for every family: the same statistic recomputed in **GPCP, GPM IMERG, Berkeley Earth, GLDAS 2.0/2.1** (difference-series break test for steps); **robustness suite** (leave-one-year-out, endpoints, without strong ENSO years, block bootstrap, one-extreme-year leverage); **field significance** with spatial correlation kept; place names (GeoNames); families on the same cells merged. Contradicted or fragile families are closed automatically; the rest are ranked by independent support and robustness |
| Deep dives, science | agents | real or not, mechanism, drivers, predictability, impacts; briefs carry the evidence and an automatic literature pre-check |
| **Referees and audit** | Reviewer-2 agents, auditor | every supported claim is recomputed and attacked by an independent referee (upheld / downgraded / refuted, 3+ real checks required); a sceptical auditor sets final verdicts |
| Dossiers, tournament, outputs | agents, editor, judge, code | result dossiers, ranking, **results page** (`results/index.html`) and a **NASA data-issues report** (`results/NASA_data_issues.md`) |

GPU templates any agent can call with one line (inputs are prepared from the campaign's trust-masked tables):
`teleconnection` (per-cell change in a driver's grip with bootstrap CIs, without strong ENSO years and with
amplitude-matched periods, field test, 15-year windows), `causal` (conditional Granger causality of every source on a
target, early vs late, FDR over cells), `predictability` (out-of-sample forecast skill maps, per-cell ridge and a
shared network), `surprise`.

Budget rules: the programme may use half the task budget and half the workers; survey follow-ups 15%;
verification is capped (15%, with a smaller allowance in the survey stages); housekeeping is refused.

## How the paper's method maps onto Earth data

| In the paper (genome mining) | In Synaptic (Earth data) |
|---|---|
| Research brief turned into staged tasks | `briefs/*.yaml` turned into plan, sweep, classify, census, deep dives, reports |
| 1.9 B protein clusters searched with profile HMMs | Every grid cell x variable of NASA POWER (1981-2025) swept with change tests (`synlib`) |
| RT classes | Signal classes: trend, regime shift, variance change, seasonal-cycle shift, recent extreme |
| Neighbour census with promotion filters fixed *before* the census | Anomaly families scored against thresholds the planner **locks before any data is seen** |
| One worker + one supervisor per task, shared record | Same: plan review, result review, revisions, all in a shared SQLite record |
| Agents open their own follow-ups (98 of 119 tasks) | Workers and supervisors propose follow-ups; triage releases them within a budget |
| Curator knowledge base read by later agents | Same (`kb` table, injected into each worker's context) |
| The discovery came from an agent *reading raw DNA* | Mandatory `look` tool: raw numbers and figures in front of the (vision) model; finishing is refused without it |
| Novelty kill-test against literature | Literature search (NASA ADS, OpenAlex, Semantic Scholar, Crossref, Europe PMC) + artifact checklist |
| GPU queue for structure prediction | Kaggle GPU queue (T4 x2 / P100): a sequence model learns each cell's normal behaviour and maps "surprise" (the Fig. 5 analogue) |
| Independent checks against other databases | `independent_series`: CHIRPS rainfall and CHIRTS max temperature (box mean or per POWER cell); `station_series`: NOAA GHCN-Daily weather stations (Tmax, Tmin, rain, to the present) mapped to POWER cells |
| Raw reanalysis check | `merra2_series` (Earthdata token): native MERRA-2 on POWER's grid, to tell POWER processing problems from reanalysis problems |
| Climate drivers | `climate_index`: ENSO (Nino-3.4, SOI), Indian Ocean Dipole, AMO, PDO, NAO, TNA, QBO from NOAA PSL |
| Agents over-claim; humans re-check | **Novelty gate** on `new_signal` (2+ literature searches, a stated novelty basis, warming "documented by default") and a skeptical **verdict auditor** that re-reads every claimed discovery before reports are written |
| Tournament + Bradley-Terry ranking of reports | Same |
| Fixed-input benchmark across models (Fig. 4) | Planted-anomaly benchmark on real NASA data, 5 information levels, raw-data-read analysis |

## What changed in version 4 (after an expert review)

| Expert finding | v4 |
|---|---|
| Shift and recent-extreme tests ignore carry-over (17 false "shift" cells per run of 225 on pure noise) | every test accounts for AR(1) carry-over, bias-corrected, with a regional floor; per-series false rate back to ~5% (tests/test_stats.py) |
| Carry-over + neighbours moving together made a 200-cell false family | families must beat the largest cluster in surrogate fields that keep carry-over AND spatial correlation (phase surrogates); a planted regional step or trend is still found |
| Field significance fails open | fails closed: a family the test cannot check cannot pass |
| No held-out confirmation; the split was chosen after seeing the data | every hypothesis carries a confirmation test pre-registered before any result, run by code on a dataset the explorers never see; a split chosen after looking pays for the search (`step_change_test`, `correlation_change_test`) |
| "44 of 52 cells, p = 2e-7" counted neighbouring cells as independent | `sign_test_spatial` uses the effective number of independent cells; the v3 headline recomputed: NOT established (evidence/southern_africa_enso_winter) |
| The auditor could upgrade a verdict | verdicts can only go down, enforced in code (synaptic/verdicts.py); only the confirmation and novelty gates move a claim up |
| Novelty from weak literature search | without a NASA ADS token novelty is shown as "unverified", never "new" |
| Tournament saw most pairs in one order | every judged pair is played in both orders; disagreements are ties; position bias is reported |
| Most of the budget went to proving the data broken | focused campaigns on one open problem; the data screen and evidence engine are code |
| Hypotheses not grounded in published limitations | the PI reads a literature scan of the open problem first; each hypothesis states its novelty basis |
| No reserve pool; short meeting memory | 10-16 hypotheses, the rest in a reserve pool promoted at meetings; meetings see every result in full |
| Mechanism claims were correlations | `mediation_test` (ablation: does the effect disappear when the mediator is controlled?); otherwise labelled "association only" |
| Format errors, stalled reports, server errors | constrained JSON actions; the editor fixes wording after two revision requests; a fallback model after repeated server errors; PI, judge and auditor on Nemotron 3 Ultra 550B |
| No git, no tests, not reproducible | git history, 39 tests, `python -m synaptic reproduce` re-checks the evidence bundles |
| Agent code unsandboxed | Docker (no network, no credentials, read-only data), or an unprivileged user without network, or a macOS sandbox profile; every credential-like variable is removed |

## The rules every agent follows

1. **Look at the data yourself.** Raw tables and figures go into the model's context; the dashboard shows every image an agent looked at.
2. **Thresholds are pre-registered** by the planner and locked before the sweep. Nobody relaxes them after seeing results.
3. **Try to kill the finding**: data-production artifacts (e.g. POWER's solar fields switch from SRB to CERES SYN1deg in 2001), known phenomena, then the literature.
4. **Never claim a check that was not run.** "Artifact" and "not robust" are good, honest outcomes.
5. **Open follow-ups** for anything unexpected.

The scripted offline demo never claims a discovery: only the live model may call something new.

## Tested live with Nemotron 3.5 Super VL

The harness was deep-tested with `nvidia/nemotron-3.5-super-vl-120b-a12b` on real NASA data (see
[docs/research/harness_test_report.md](research/harness_test_report.md)). Nemotron's habits (dropped closing braces, empty `finish`, long reasoning that
exhausts the output budget, quoting instruction examples) are handled by the parser and the per-role reasoning
settings; the live smoke test passes with about one format error per 230 actions. Workers use low-effort
reasoning (about 4x faster); planner, supervisor, editor and judge use full reasoning.

Highlights from the test:
- In a blind West Africa campaign, Nemotron agents rediscovered on their own the 1999 inhomogeneity in POWER rainfall
  over northern Nigeria. They tested it against CHIRPS and classed it as a data artifact.
- A deep dive refused to call a 2008 western-Sahel rainfall step "climate" because CHIRPS does not show it.
  Following that lead uncovered a **2025 error in NASA POWER rainfall**: the rainy season is missing over central
  Mali (0.3 mm/day vs a normal 3.6-6.4 in CHIRPS and GPCP), with spurious +5 °C Tmax and −26-point humidity anomalies.
  The error was traced to the CMAP input that MERRA-2 uses to correct African rainfall.
  See [docs/research/harness_test_report.md](research/harness_test_report.md) and `docs/research/`.
- The test also caught the harness over-claiming novelty for ordinary regional warming. The novelty gate and the
  verdict auditor were added because of it.
- **Whole-Earth campaign with version 3** (all land 60S-80N, 1981-2025, about 18 h, 695 agent sessions). The PI wrote
  19 falsifiable hypotheses; the group refuted 15 itself. One survived two independent referees: **ENSO's grip on
  winter temperature over western southern Africa (Namibia, Botswana, western South Africa) strengthened around
  2007-2009**, with 44 of 52 cells agreeing, replication in GHCN stations, a soil-moisture mechanism, and a doubling
  of hot winter months in El Nino years. Novelty is provisional until an expert or NASA ADS check. The evidence engine
  closed 415 artifact families and produced a NASA data-issues report (MERRA-2 rain correction inflating trends in
  central Brazil, southern South America, Mindanao and NW China). See [research/harness_test_report.md](research/harness_test_report.md) section 15.

## Running campaigns

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m synaptic setup    # type your keys privately (hidden input) -> writes .env, then runs the check
python -m synaptic smoke    # ~10 min live mini-campaign with Nemotron + a diagnosis of how the model behaved
```

`check` verifies model access, a vision call, NASA endpoints and Kaggle quota. `smoke` runs the whole
pipeline on a small box over central Morocco with 2 deep dives and prints action-format errors,
finish refusals, images looked at, revisions, follow-ups and verdicts. Run it before every big campaign.

Run a campaign and watch it live:

```bash
python -m synaptic dashboard            # http://127.0.0.1:8050  (terminal 1)
python -m synaptic run briefs/morocco_hydroclimate.yaml   # terminal 2, or use "Launch campaign" in the UI
```

If the laptop sleeps, the connection drops or the process is killed, nothing is lost:

```bash
python -m synaptic resume c_1a2b3c4d      # campaign id from the dashboard or the launch line
```

Offline demo (no API calls, real NASA data, scripted agents, clearly labelled in the UI):

```bash
python -m synaptic run briefs/morocco_hydroclimate.yaml --mock
```

Whole-Earth campaign straight from NASA's archive (needs `EARTHDATA_TOKEN`; ~10 GB of MERRA-2 monthly files are
fetched once and cached), then the judge-facing outputs:

```bash
python -m synaptic run briefs/global_science.yaml
python -m synaptic export c_1a2b3c4d     # results/index.html + results/NASA_data_issues.md (also written automatically)
```

Benchmark models like the paper's Fig. 4:

```bash
python -m synaptic bench --models nvidia/nemotron-3.5-super-vl-120b-a12b,nvidia/nemotron-3-super-120b-a12b --attempts 10
python -m synaptic bench --mock --attempts 10    # rule-based statistical baseline to beat
```

## Configuration notes

- **Model**: `SYNAPTIC_MODEL` (default `nvidia/nemotron-3.5-super-vl-120b-a12b`). A stronger supervisor or
  judge can be set per role, e.g. `SYNAPTIC_MODEL_JUDGE=nvidia/nemotron-3-ultra-550b-a55b`.
- **Region**: a brief's box can be up to 40x40 degrees; POWER tiles are fetched 10x10 degrees, one variable at a time.
- **Rate limits**: keep `SYNAPTIC_RPM` under your NVIDIA quota; `SYNAPTIC_CONCURRENCY=4` is a good start.
- **Kaggle**: jobs are pushed as private script kernels with a GPU. Weekly GPU hours are limited; routine
  analysis stays on CPU.

## Safety

Agent-written Python runs in a sandbox (`python -m synaptic sandbox status`): a Docker container with no network
and no credentials (`python -m synaptic sandbox build` once), or, without Docker, an unprivileged user without
network (Linux) or a macOS sandbox profile that blocks the network and the key files. Every environment variable
that looks like a credential is removed in all modes. Data and literature only
come from public NASA and scholarly APIs.
