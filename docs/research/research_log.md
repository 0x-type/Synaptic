# Discovery log

Goal (from the user, 9 Nov 2026): one finding about the Earth, shown by NASA data, that is genuinely new.
A result counts as **new** only if it (1) passes its pre-registered held-out confirmation, run by code on data the
agents never saw, and (2) survives a NASA ADS literature check. A test that is not significant is reported as
**not shown**, never as "refuted". Keys are never printed, logged or committed. Kaggle GPU jobs stay under 2 hours.

Branch: `claude/admiring-fermat-l4nk42` (never `main`).

---

## 0. What the harness can reach (checked 9 Nov 2026)

| Product | Through | Role |
|---|---|---|
| MERRA-2 reanalysis (rain, temperature, soil, runoff, wind, water vapour, aerosol types, clouds) | Sep 2026 | explorers |
| GLDAS Noah 2.1 (soil moisture, evaporation, snow) | Jul 2026 | explorers |
| GRACE / GRACE-FO total water storage (JPL mascons) | Aug 2026 | held out |
| CERES EBAF (reflected sunlight, net energy, cloud area) | Aug 2026 | held out |
| GPM IMERG Final rainfall | Sep 2025 | held out |
| MODIS Aqua (AOD, clouds, droplet size, SST) | 2025-26 | explorers |

I read only the time axes of GRACE and CERES, never their values: they are the held-out test data.

## 1. Deep thinking: where can something new still be found?

What earlier campaigns taught (TEST_REPORT.md sections 13-17):
- Long-term trends in reanalysis are where most "discoveries" died: they were data artifacts (MERRA-2 rain
  corrections, stream changes) or already published (regional warming).
- The one v3 headline (southern Africa ENSO) did not survive honest statistics: with 23-25 years, slow changes
  in relationships are too weak to confirm.
- The 2023 low-cloud campaign found nothing confirmable: a crowded, global topic.

So the best odds are where four things meet:
1. **A big physical signal**, so a pre-registered test on ~25 yearly values can pass (events, not subtle trends).
2. **An independent satellite** measuring something no reanalysis shares (GRACE measures mass by gravity: it
   cannot inherit MERRA-2's rain errors).
3. **Recent years** (2024-2026): the data exist now, papers take 1-2 years to appear.
4. **Few research groups** working on the place and the quantity.

### Candidate A: the 2024 Sahara-Sahel deluge as a natural recharge experiment (CHOSEN)
- What happened: Aug-Sep 2024 brought record rain to the Chadian Sahara, the Sahel (Niger, Chad, Mali, northern
  Nigeria) and SE Morocco / SW Algeria; dry desert lakes (Iriqui) refilled.
- The open question: how much of that water did the land KEEP, where, and for how long? Rain that soaks into
  aquifers or fills lakes stays as mass that GRACE can weigh from orbit; rain that evaporates is gone in weeks.
  GLDAS and MERRA-2 have no groundwater or lakes, so GRACE minus what the land models hold separates the deep
  water from soil water.
- Also testable: did the wet year green the desert edge and change the surface (darker ground in CERES clear-sky
  reflection, less dust in 2025)?
- Data: explorers get MERRA-2 rain, soil wetness, runoff, temperature, water vapour, dust AOD, and GLDAS soil
  moisture and evaporation; held out: GRACE water storage, IMERG rain, CERES clear-sky reflected sunlight.
- Literature (ADS, 9 Nov 2026): **0 papers** match GRACE + Sahara/Sahel + 2024. The event has weather papers
  ("Synoptic drivers of the August 2024 record-breaking rainfall in the Chadian Sahara", 2026; "Meteorological
  ingredients of heavy precipitation and subsequent lake-filling episodes in the northwestern Sahara", 2025;
  Lake Iriqui revival, 2025), but none weighs the water that stayed.
- Risks: GRACE sees ~300 km blobs, so only large regions can be tested; MERRA-2's African rain is corrected
  with CMAP, which had a 2025 error over Mali (TEST_REPORT sec. 8), so rain claims must be confirmed on IMERG.

### Candidate B: did Morocco's water come back when the drought broke (2025-2026)?
- The 2018-2024 drought ended with the 2025-26 rains (2026 Gharb and Loukkos floods). Did groundwater recover, or
  is storage still below what the climate predicts (pumping)?
- Data: same explorers over Morocco; GRACE and IMERG held out; the "residual" confirmation kind.
- Why not chosen: ADS already has 21 GRACE-Morocco papers and 27 recent drought papers (several from 2026),
  and Morocco is only 2-3 GRACE footprints, so tests have little power. Kept as the fallback.

### Candidate C: did Earth's energy imbalance and low-cloud loss persist after 2023 (to Aug 2026)?
- CERES held out, MODIS and MERRA-2 explore.
- Why not chosen: a crowded global topic; the previous campaign on it confirmed nothing.

**Choice: A.** It has the strongest signal, a truly independent held-out instrument, and no published GRACE study.

---

## 2. Harness fixes before the first campaign

| Problem found | Fix | Test |
|---|---|---|
| **The novelty gate never searched ADS.** It set "new" whenever the ADS token existed and the agent had written 60+ characters of novelty basis (`harness.py` `_confirm_all`). | New `synaptic/novelty.py`: for every result that passes its held-out confirmation, a model writes 4 ADS queries, code adds 2 of its own, code runs them on ADS, and an independent reader (the auditor model) checks each abstract. "New" only if the verdict is `not_found` and at least 3 searches ran. The record (queries, papers read, closest papers) is saved per hypothesis in `novelty/<id>.json`. | `tests/test_novelty_and_labels.py`; live check: the 2024 Lake Iriqui refill was correctly found DOCUMENTED (Land, 2025). |
| **"Refuted" used for "not significant".** The confirmation step called a non-significant test "inconclusive", and agents were told "refuted" was a normal close. | New status `not_shown` (test ran, not significant). `refuted` is now only for a significant result in the opposite direction; `inconclusive` only when the test could not run. Prompts, tools, referee decisions, lab-meeting closes and the results page are updated. | `test_not_significant_is_not_shown` |
| **Agents could freeze a verdict** by writing "[independent check]" in a note (`verdicts.py` read code tags as free text). | `record.update_candidate` neutralises code-check tags in any note not written by code (engine, confirmation, novelty gate); tags already written by code are kept. | `test_agents_cannot_write_code_tags` |

Also: `reverse_geocoder` does not build on Python 3.13 (old setup script); copied the pure-Python package by hand in
this environment. Tests: 27 passed.

---

## 3. Round 1: campaign `c_393bde84` (launched 9 Nov 2026, 08:51 UTC)

Brief: `briefs/sahara_2024_recharge.yaml`. Box 8-33N, 18W-30E, 1 degree, 2002-2026.
Explorers: MERRA-2 rain, root-zone soil wetness, runoff, temperature, water vapour, dust AOD; GLDAS top-soil
moisture and evaporation. Held out: GRACE/GRACE-FO storage, IMERG rain, CERES clear-sky reflection.
Budget: 60 tasks, 5 workers. Before launch, the brief's known-literature list was built from ADS: the closest
study (J. Hydrology 2026) covers Africa's GRACE storage to Dec 2024 but studies long-term trends, not the 2024
event or what stayed afterwards.

Fix made while round 1 downloaded data (before any confirmation ran): GRACE and CERES files also hold values over
the ocean, and held-out confirmations averaged them in when a test box touched a coast. Confirmations of
land-only products (GRACE) now use a land mask (union of the Natural Earth country outlines, `regions.land_mask`).

**Round 1, first problem (09:06):** the PI (Nemotron 3 Ultra) wrote a 16-18 hypothesis programme but the harness could
not parse it as JSON in 3 attempts, so the focused campaign had no programme. Reproduced: the same call made fresh
parsed at once. Cause: on failure the harness pasted the whole 24k-character answer back and asked again, which
made later answers worse. Fix (`agents.py` `oneshot`): retry fresh, one repair request, then fresh again (4 tries),
and keep the unparsed text in the record. The campaign was stopped, its programme stage reset, and resumed (09:11).
The data screen found a 2011 break between MERRA-2 and GLDAS soil moisture: the known MERRA-2 stream change, not
climate, and far from 2024.

**Round 1, second problem (09:21): a false "new".** The first confirmed result (H Hd33026ab: across the whole
domain, Aug-Sep GRACE storage rises and falls with IMERG rain from year to year, r = +0.54, p = 0.018) was
promoted to "new" by my first version of the ADS gate. Reading its record: the model wrote over-specific queries
full of method words ("held-out", "IMERG_PRECIP", "p<0.05") that matched nothing; the code's own queries searched
for "August" and returned software releases; 3 searches timed out; the reader saw no relevant paper and said "not
found". Storage rising with rain is textbook hydrology. Fixes (`synaptic/novelty.py`):
- queries are built from concept groups (place OR place) AND (quantity OR quantity), with no method words or
  product codes, restricted to ADS's Earth-science collection and excluding software;
- ADS calls get a 150 s timeout and 3 retries (broad queries are slow), and failed searches never count;
- the reader must list the papers that are on topic (same region and quantity); "not found" only stands with at
  least 3 on-topic papers, otherwise the result is "unverified" (not new);
- textbook relationships are "documented".
I reset Hd33026ab to "supported" as operator and re-ran the gate: all 5 searches ran, 45 papers, 2 on topic, the
reader called the result a textbook expectation: it stays "supported", not new. Tests: 30 passed.

Results so far (held-out GRACE, code-run, pre-registered):
- Central Sahara box 15-20N 15-22E, Aug-Sep 2024 above the trend of the other years by +6.2 cm, p = 0.085:
  **not shown**.
- Same box, Mar-May 2025 and 2026 above trend (persistence): **not shown**.
- Whole domain, Aug-Sep storage vs IMERG rain, r = +0.54, p = 0.018: supported, not new (textbook).

**Round 1, third problem (09:57): a second false "new", a claim-test mismatch.** H846b9dd6 claimed that Lake
Chad's 2024 storage follows upstream floodwater MORE than local rain minus evaporation. Its pre-registered test
only checked that GRACE storage in the box (11-15N 12-16E, Aug-Dec 2002-2024) correlates with MERRA-2 runoff in
the same box (r = +0.90, p = 4e-7): it never compared the two drivers, never measured upstream inflow, and is
not about 2024. The ADS reader judged the untested story and found no match. Fix: the gate now judges ONLY what
the held-out test established (the reader first restates it in one plain sentence; the hypothesis text is shown
as untested interpretation). Re-run: "GRACE storage correlates with runoff in the Lake Chad basin": documented
(textbook, 12 on-topic papers). H846b9dd6 reset to "supported" (not new) by operator.

More results (all code-run on held-out data, pre-registered):
- Central Sahara 15-20N 15-22E, Mar-May 2025+2026 above trend +6.3 cm, p = 0.058; Mar-May 2026 alone +5.5 cm:
  **not shown**. The 2024 peak (+6.2 cm, p = 0.085) and the dry seasons 1.5 years later have about the same
  size: the hint is that the water stayed, but the year-to-year noise is too large to show it.
- Tibesti-Borkou-Ennedi 18-23N 15-22E, Aug-Sep 2024: +2.6 cm, p = 0.15: **not shown**.
- NW Sahara 27-32N 5W-2E, Sep 2024 (Iriqui, Sebkha el Melah lakes): -0.7 cm, p = 0.57: **not shown** (the lakes
  are far smaller than GRACE's ~300 km footprint).
- Niger floodplain minus sandy interfluve 2024: **not shown**. CERES clear-sky darkening of the Chadian Sahara in
  Mar-May 2025: **not shown**. 2025 dust drop: no valid held-out test, inconclusive.

**Round 1, fourth problem (10:15):** H8167bfa2's held-out test (Lake Chad, Mar-May 2026 storage beyond what
accumulated rain minus evaporation explains) used IMERG rain as a driver, but IMERG stops in Sep 2025, so 2026
dropped out and the test could not run: honestly "inconclusive". Fix for round 2: `confirm.validate` now refuses a
spec that needs data past a product's last month (`catalog.Var.end`, set for IMERG), while the PI can still fix it.
Also: across the domain, Mar-May CERES clear-sky reflection does not follow IMERG rain (r = +0.09): not shown.

**Preparing round 2: a second, independent GRACE solution.** Earthdata only hosts JPL's mascons (the "CRI" and
plain grids are the same solution). NASA Goddard publishes its own mascons (RL06v2.0, Apr 2002 - Jun 2026,
0.5 degree) on earth.gsfc.nasa.gov, processed independently. Added as `GRACE_GSFC_TWS` (held out). Sanity check
away from the study region: central Amazon annual storage, JPL vs GSFC, r = 0.99, same amplitude. The download
code now sends the Earthdata token only to Earthdata hosts. (The CSR mascon site failed TLS verification; not
used, verification not bypassed.)

### Round 1 outcome (ended 11:22 UTC; 60 agent tasks, 7 lab meetings)

49 hypotheses on the board. **Nothing new.** 1 supported (textbook: storage follows rain), 15 not shown,
20 inconclusive, the rest in reserve. Results page: `runs/.../results/index.html` (not committed: runs/ is
ignored). I stopped the run during its dossier-writing stage (verdicts were final) to free the NVIDIA quota.

**What went wrong, and the fixes for round 2:**
1. **25 hypotheses had no pre-registered test at all** (empty spec), mostly ones added at lab meetings: the PI's
   spec repair ran only once, after the first programme. They were still queued and used ~40 agent tasks that
   could never confirm anything. Fix: a hypothesis is queued only with a valid spec; lab-meeting hypotheses go
   through spec repair first; the rest wait in reserve.
2. **Agents labelled untested hypotheses "not shown"** (6 cases: "the test could not be executed because GRACE
   is held out"). Fix: such closes become "inconclusive", and every hypothesis with a valid spec still gets its
   held-out test in code even when an agent closed it.
3. **Multiple testing.** With 15-30 held-out tests at alpha 0.05, one or two pass by chance. Fix: to be called
   NEW, a result's p must survive 0.05 / (number of pre-registered tests in the campaign, at least 10).
4. **One GRACE solution only.** Fix: every GRACE confirmation is repeated in code on the independent NASA GSFC
   mascons; both must pass (the weaker p counts).
5. **I have seen round-1 numbers.** Fix: tests of 2024-2026 in any box overlapping the central Sahara box round 1
   looked at four times (15-20N 15-22E) are run at alpha 0.01, enforced in code (`alpha_rules` in the brief).
Tests: 33 passed.

---

## 4. Round 2 (brief `briefs/sahara_2024_round2.yaml`)

Same region, data and question, sharpened: is there water left in 2025-2026 that the land models' soil does not
hold, and where? The brief gives round 1's results and asks for higher-power designs (storage minus what land
models explain; wetted box minus a dry control box; GRACE-sized boxes). Held out: JPL GRACE, GSFC GRACE, IMERG,
CERES clear-sky.

### Round 2, first results (11:39-11:45): the round-1 hint fades under sharper designs
All five code-run tests not shown, at alpha 0.01 (the code applied the stricter rule to boxes overlapping the one
round 1 examined):
- Central Sahara 18-23N 15-22E minus a western control 8-13E, Aug-Sep 2024: +0.15 cm, p = 0.87. Round 1's +6 cm
  was a wide signal shared with the surroundings, not water held in the central Sahara.
- Storage beyond accumulated rain minus evaporation, 2024: central Sahara -0.4 cm (p = 0.56), Lake Chad
  12-17N 13-18E -2.4 cm (p = 0.42). Lake Chad minus western control, 2024: not shown.
- Central Sahara Mar-May 2025 storage beyond what GLDAS soil holds: +1.4 cm, p = 0.33.
Reading: at the scales GRACE resolves, the 2024 rains left no detectable deep-water signal in the Sahara.
Round 2 continues (Lake Chad persistence tests in reserve).

## 5. Round 3, in parallel: Morocco's 2025-26 drought break (candidate B)

Why now: round 2 shows candidate A is unlikely to yield a confirmed finding. Candidate B has a much larger
expected signal (exceptional 2025-26 winter rain after a record 7-year drought) and, by ADS on 9 Nov 2026, no
GRACE study of the recovery (2026 papers map the Gharb and Loukkos floods by radar; GRACE-Morocco papers stop in
2024-25). The non-trivial question: is 2026 storage above or below what the climate alone predicts (did the
aquifers come back, or only the soil)? Designed from the literature and physics; I have not looked at any
Morocco GRACE values. Brief: `briefs/morocco_2026_drought_break.yaml`. Held out: JPL and GSFC GRACE, IMERG.
Running alongside round 2 (no NVIDIA rate limiting seen).

### Round 2, continued (to 12:19)
14 code-run held-out tests so far, **all not shown** (central Sahara vs western, northern and hyper-arid
controls; storage beyond rain minus evaporation and beyond land-model soil in 2024, Mar-May 2025 and Mar-May
2026, Lake Chad and central Sahara; Niger floodplain vs interfluve; Lake Chad vs central Sahara drainage rate;
CERES darkening). Lab-meeting ideas without a valid test are now held in reserve (working as intended).
Spec repair now shows the PI why a spec is invalid; step tests too close to the record's end suggest an event
test instead.

### Round 3 (`c_5142a905`, Morocco), problems found and fixed while it ran
1. **Claim-test mismatch:** "northern Morocco's 2026 storage is above the pre-drought trend" was registered as a
   2002-2026 trend test; it ran (not shown: Sen slope -6.6 cm/decade, p = 0.45), which answers a different
   question. Fix: a claim naming a recent year cannot be tested with a long-term trend (`Campaign.spec_problem`);
   spec repair now also runs when a campaign resumes. Round 3 restarted; 3 of 5 bad specs repaired before any
   result.
2. **A test that could not run:** a residual test with calendar-year drivers lost 2026 because GLDAS stops in
   Jul 2026 (a year needs 2/3 of its months). Fix: product end dates recorded for MERRA-2 (2026-09), GLDAS
   (2026-07), CERES (2026-08), GRACE JPL (2026-08), GSFC (2026-06), IMERG (2025-09); a spec whose final or event
   year would drop out, including through its driver months, is refused before it runs. That test produced no
   number (I checked only counts and years), so I reopened it as operator for a valid spec. Round 3 restarted.

### Interruption and outcome of rounds 2 and 3
The session's worker restarted at about 12:55 UTC and stopped both campaigns (no work lost from git; the
campaign databases are intact). Their held-out tests were essentially complete:
- Round 2 (Sahara, sharper designs): 20 computed held-out tests, **0 supported**.
- Round 3 (Morocco 2026): 11 computed held-out tests, **0 supported**. Closest: northern Morocco 2026 storage
  beyond the climate prediction +9.1 cm (p = 0.15); southern +3.4 cm (p = 0.10); northern May-Jun 2026 level
  +5.0 cm above trend (p = 0.69).
Across rounds 1-3: **41 computed held-out tests, 1 supported (textbook), 0 new.**

**Why so little reaches significance: storage carries over.** GRACE storage is a running total, so a year's level
inherits most of the previous year's (carry-over 0.90 over northern Morocco in round 3); a test of one year's
level has very few independent years. The seasonal GAIN (after the rains minus before) does not carry over:
checked away from the study regions on JPL GRACE, carry-over of Apr-May level vs Sep-Oct to Apr-May gain:
central Amazon 0.02 vs -0.19, Russian plain 0.33 vs -0.01. New confirm kind `gain` (storage gain between two
seasons, optionally minus what rain minus evaporation over the window explains; event or trend test). Synthetic
check: a planted winter gain on a 0.9-carry-over storage series is found; 40 pure-noise series give at most the
nominal false-positive rate (`tests/test_confirm.py`). The family for "new" now includes earlier rounds' tests
(`prior_tests` in the brief, enforced in code). pytest now only collects `tests/` (agent scripts in runs/ were
being collected). Tests: 34 passed.

## 6. Round 4: seasonal gains (brief `briefs/north_africa_gains_round4.yaml`)
Both shocks, measured as gains: northern Morocco's winter 2025-26 gain; the Sahel, Lake Chad and central Sahara
2024 wet-season gain; and the following dry-season drawdown (did the water stay). "New" needs
p <= 0.05 / (round-4 tests + 41).

### Round 4 (`c_17ba1b53`), problems and results
Problem: the PI wrote level tests again, because its prompt did not list the new `gain` kind. Fix: the PI
prompts document `gain`; a brief can require gains for GRACE (`storage_kind: gain`, enforced by
`Campaign.spec_problem`); a spec the campaign rejects is never run. Round 4 restarted before any held-out result
existed; spec repair converted 12 specs to gain tests. (The PI also set its own alpha to 0.05/55 = 0.00091, the
"new" threshold, so results significant at 5% are labelled "not shown"; this changes no conclusion.)

Held-out results, JPL GRACE (seasonal gain vs the other years):

| Test | Result | p |
|---|---|---|
| Lake Chad basin 10-17N 12-18E, 2024 wet-season gain (Jun-Jul to Sep-Oct) | +11.5 cm | 0.0038 |
| Central Sahara 15-23N 10-25E, 2024 wet-season gain | +2.4 cm | 0.0026 |
| Northern Morocco 31-36N 10-1W, winter 2025-26 gain (Sep-Oct to Mar-Apr) | +7.0 cm | 0.012 |
| Sahel 10-17N 18W-30E, 2024 wet-season gain | +8.3 cm | 0.015 |
| Sahel 2024 gain beyond what rain minus evaporation explains | +5.0 cm | 0.036 |
| Central Sahara 2024 gain beyond rain minus evaporation | +0.5 cm | 0.37 |
| Northern Morocco 2025-26 gain beyond rain minus evaporation | +1.8 cm | 0.46 |
| Sahel / central Sahara 2024-25 dry-season drawdown beyond the climate (did the water stay?) | -1.1 / +0.6 cm | 0.55 / 0.78 |
| Northern Morocco winter-gain trend 2003-2026 | -0.9 cm/decade | 0.40 |

Operator replication on the independent NASA GSFC solution (same pre-registered specs, alpha 0.05):
Lake Chad +11.7 cm (p = 0.011), central Sahara +4.1 cm (p = 0.008), northern Morocco +9.3 cm (p = 0.013):
**replicated**. Sahel gain beyond the climate: +1.0 cm (p = 0.63): **not replicated** (JPL-only).
ADS (9 Nov 2026): no paper analyses GRACE storage for the 2024 Sahel-Sahara event or Morocco's 2025-26 recovery
(control searches return 6 and 9 recent GRACE papers on these regions; none covers these events).

---

## 7. Stopping decision and final tally

Four rounds, 55 computed held-out tests (round 1: 10, round 2: 20, round 3: 11, round 4: 14), about 6 h 40 min.
**No finding passes every check.** What the data show robustly (both GRACE solutions, p about 0.003-0.013): the
land gained water where record rain fell (Lake Chad basin and central Sahara in 2024, northern Morocco in winter
2025-26). These do not reach the bar for "new" (p <= 0.05/55 = 0.0009), and they are the expected effect of the
rain: everything that would make them non-obvious (storage beyond what rain minus evaporation explains, water
staying through the dry season, a change in winter recharge) was tested and not shown.

I stopped here rather than open a fifth round on the same questions: every extra round adds chances for a lucky
pass while the evidence has converged. A genuinely new attempt should start from a different question and new
held-out data, with the 55 tests here counted in its correction.

---

## 8. Track B: a continent-wide search with a time holdout (user's choice)

Design: explorers get GRACE (JPL) only for 2002-2015, together with rain, evaporation and soil moisture; they search
all of Africa for places where seasonal storage gains or drawdowns depart from what rain minus evaporation predicts;
each finding is pre-registered as a test on 2016-2026, years the search never saw, and must also pass on the GSFC
solution. A pattern found in one era that reappears in an unseen era is hard to dismiss as chance.

**Leak found and fixed before launch.** In round 1 a referee (allowed all data) fetched held-out IMERG rain with
`nasa_series`, which saved it in the campaign's SHARED data folder; explorers then read it. Checked all agent
actions: no agent read the cached GRACE or CERES files, so GRACE and CERES confirmations are unaffected; in round 1
IMERG served only as a rain driver, never as a held-out target. Fixes: held-out downloads by referees stay in their
own task folder; the cache of whole-record NASA files is now readable only by the harness (agent code runs as an
unprivileged user).

**New harness features:** `time_holdout` in the brief (the data tool and the campaign tables stop at the cut year
for explorers; confirmation specs must test only later years); confirm test `later` for residual and gain kinds
(fit the climate model on the years up to the cut, test the mean departure of the later years, carry-over
estimated from all years). Synthetic check: a planted post-2015 shift is found; 30 noise series give at most
the nominal false-positive rate. Boxes examined in rounds 1-4 are tested at alpha 0.005 (`alpha_rules`).
Tests: 36 passed. Brief: `briefs/africa_water_balance_trackB.yaml`.

**Track B launch (`c_151000c3`, 16:46 UTC).** Data for all of Africa in 5 minutes; the explorers' GRACE table
stops at 2015 as designed (checked: 2002-2015). Problem: the PI's first call sat 25 minutes with no output. The
continent-wide programme takes longer to generate than the 300 s request timeout, and a timeout on the PI's model
"falls back" to the same model, so the call retried the same timeout (up to 8 x 300 s, times 4 one-shot attempts).
The endpoint itself answered short prompts in 1 s. Restarted at 17:20 with a 900 s request timeout.

### Track B, first attempt ended without a test; redesign
1. **The campaign ended at once (17:50).** The PI wrote hypotheses about famous wetlands straight from the
   literature; 8 of 14 had no valid spec and spec repair fixed none. The 5 queue slots went to invalid ones (held
   back by the guard), and reserve promotion picked the same invalid ones again, so nothing ran. Fix: the first queue
   slots and reserve promotion only take hypotheses with a valid spec.
2. **The search step was missing.** Added a code exploration sweep (`synaptic/explore.py`) that runs on the
   exploration years only, before the PI writes anything:
   - seasonal gains and drawdowns beyond rain minus evaporation, per 4-degree block: **392 block-seasons, 18 with
     p < 0.05 where ~20 are expected by chance, none with p < 0.01: no signal beyond chance** (not pursued);
   - annual storage level beyond accumulated rain minus evaporation: **236 blocks, 51 with p < 0.05 (~12 expected),
     4 with p < 0.001**. Strongest: NW Sahara Aquifer System near Ghadames and Hassi Messaoud (documented depletion),
     Sinnar (Sudan) and Tigray (Ethiopia) rises (likely dams), Yemen (documented), central Nigeria near Tegina
     (+9.4 cm/decade, p = 0.004), Gourma, Mali (+2.0 cm/decade; probably the documented "Sahelian paradox").
3. **Ocean in the explorers' GRACE table:** several "signals" were mostly-ocean blocks. Campaign GRACE tables now
   keep land cells only.
Relaunch as Track B v2 (`briefs/africa_trackB_v2.yaml`): candidates come from the level-trend sweep; each is
confirmed with a `residual`/`later` test on 2016-2025 in the direction of its 2003-2015 drift, replicated on GSFC.

### Track B v2 (`c_9b386caa`): programme and a calibration check
The PI wrote 14 hypotheses, all from the level-trend sweep, all with valid `residual`/`later` specs on 2016-2025;
it ranked the documented Sahelian-paradox sites last and put less-studied blocks first (Katakwi, Uganda; Tegina,
Nigeria; Tigray, Ethiopia; Niono, Mali; Dekar, Botswana). First result: Katakwi +2.2 cm beyond the 2003-2015 model
in the predicted direction, p = 0.77 (carry-over 0.73): **not shown**.

**Calibration of the `later` test on ~20 annual values (synthetic):** the pre-registered version has about 10%
false positives at nominal 5% and about 35% power for a 4 cm/decade drift; versions that estimate carry-over from
detrended residuals or by parametric bootstrap gain power (70-90%) but give 18-31% false positives. With ~20 years
of a persistent quantity, a continuing drift and slow natural variability are hard to tell apart. I kept the
pre-registered test unchanged (no deviation after a result); its excess false-positive rate is covered by the "new"
threshold (p <= 0.05/14) and the GSFC replication, and is stated here for any result.

Track B v2, results so far (all code-run on 2016-2025, which no agent saw): Tegina, Nigeria +3.1 cm (p = 0.42);
Katakwi, Uganda +2.2 (p = 0.77); Niono, Mali +3.6 (p = 0.63); Tigray, Ethiopia -14.8 (reversed, p = 0.32); Dekar,
Botswana; Tses and Luderitz, Namibia; Nouakchott, Mauritania; Mbengwi, Cameroon; Al Qusayr, Egypt; Gourma, Mali;
Sinnar, Sudan: all **not shown**. A worker task failed on NVIDIA error 422 ("media could not be screened" on an
image); images are now dropped and the call retried, as for error 415.

### Track B outcome (ended 20:02 UTC)
All 14 pre-registered tests run on 2016-2025 (years no agent saw): **none passed**.

| Block (direction predicted from 2003-2015) | 2016-2025 departure from the early model | p |
|---|---|---|
| Gourma Rharous, Mali (+) | +7.3 cm | 0.02 (needed 0.005: Sahel box examined earlier; documented "Sahelian paradox") |
| Nouakchott, Mauritania (+) | +0.8 | 0.19 |
| Tses, Namibia (+) | +8.6 | 0.23 |
| Ouallam, Niger (+) | +16.8 | 0.23 (documented Sahelian paradox) |
| Luderitz, Namibia (+) | +7.7 | 0.24 |
| Tigray, Ethiopia (+) | -14.8 (reversed) | 0.32 |
| Ad Dabbah, Sudan (+) | -3.1 (reversed) | 0.36 |
| Al Qusayr, Egypt (-) | -4.9 | 0.39 |
| Tegina, Nigeria (+) | +3.1 | 0.42 |
| Mbengwi, Cameroon (+) | +0.9 | 0.45 |
| Sinnar, Sudan (+) | -1.6 (reversed) | 0.56 |
| Niono, Mali (+) | +3.6 | 0.63 |
| Katakwi, Uganda (+) | +2.2 | 0.77 |
| Dekar, Botswana (+) | -0.2 (reversed) | 0.97 |

10 of 14 continued in the predicted direction (sign test p = 0.09): a hint that the pre-2015 drifts persist, not
evidence. The `later` test on ~20 annual values has low power (~35% for a 4 cm/decade drift) and about 10%
false positives at nominal 5% (calibration above), so a continent-wide search with this record length can only
find large, steady drifts, and the ones it found in 2003-2015 that are large are already documented (NW Sahara
Aquifer System, Yemen and Arabia, Sahelian paradox, dams).

## 9. Overall tally (9 Nov 2026, 08:50-20:02 UTC)
Six campaigns (rounds 1-4, Track B v1 and v2), 69 computed held-out tests, 0 findings that pass every check.

---

## 10. 10 Nov 2026: from search to proof (user's decision: "the best solution, with proofs")

**Assessment given to the user:** a publishable new discovery in 5 days is unlikely (5-10%). Power and novelty pull
against each other: with ~20 yearly values only large, steady changes can be confirmed, and those are published.
The door that reliably opens is recency (2024-2026 events), which yields new measurements, not new mechanisms.

**Side bet dropped after its literature check.** Idea: pool all large wet events worldwide and test how much water
is still stored a year later, by aridity. ADS: "Annual memory in the terrestrial water cycle" (HESS 2025) already
quantifies year-to-year storage memory globally, and Rodell & Li (J. Climate 2023) catalogue GRACE wet and dry
extremes. A pooled retention result would be judged known. Not run.

**Independent confirmation, pre-registered.** `PREREGISTRATION.md` (commit 0156c60, pushed 07:45 UTC, before any
SMAP value was downloaded) registers one test per lead on NASA SMAP soil moisture, an instrument never used here:
season-mean surface wetness, event year against the trend of the other years, direction +, "confirmed" only at
p < 0.05/3. Script: `scripts/smap_confirm.py` (daily SPL3SMP v009 subsets through Earthdata OPeNDAP).

**Measurements as volumes** (`scripts/measure_gains.py`, bundle `evidence/water_gains`, reproduces with
`python -m synaptic reproduce`: 3/3 bundles pass). Extra seasonal gain, JPL / GSFC:

| Lead | vs mean of other years | vs trend of other years (p) | rank | still there Apr-May 2025 |
|---|---|---|---|---|
| Lake Chad basin 2024 | 75 / 76 km3 (90% range 43-108) | 60 / 60 km3 (0.002 / 0.006) | 1 of 21 / 1 of 20 | 78% / 74% |
| Central Sahara 2024 | 46 / 73 km3 (29-99) | 38 / 64 km3 (0.002 / 0.006) | 1 / 1 | 69% / 74% |
| Northern Morocco winter 2025-26 | 17 / 21 km3 (2-39) | 23 / 31 km3 (0.012 / 0.013) | 2 of 22 / 3 of 21 | n/a |

Caveats recorded: Lake Chad gains have been larger since 2019, so the excess against the trend (60 km3) is the
conservative number; the two solutions disagree on the central Sahara's size (46 vs 73 km3) and on Morocco's normal
seasonal gain (1.1 vs 5.5 cm), and 2026 is not Morocco's largest winter gain. "Still there" is descriptive: the
dry-season drawdown was normal (round 4: not shown as unusual), so most of the extra water simply had not drained
or evaporated seven months later.

**App:** new page `/findings` (measurements, figures, SMAP verdicts, and the list of what was tested and not shown).

**SMAP confirmation, run after registration (10 Nov, ~08:10 UTC).** 4,379 daily files, no download failures.
Registered verdicts: northern Morocco winter 2025-26 **confirmed** (+0.051 m3/m3, p = 0.0106 < 0.0167); central
Sahara Aug-Oct 2024 **supported, not confirmed** (p = 0.049); Lake Chad basin Aug-Oct 2024 **not shown** (p = 0.10).
In all three the event season is the wettest of SMAP's 11 (descriptive, not registered). The 2022 season is missing
for the Sahel window (SMAP safe mode), as the registered completeness rule requires. No deviation from the
registration. Bundles: `evidence/smap_confirmation` (4/4 bundles reproduce). Summary rewritten: DISCOVERY_SUMMARY.md.

---

## 11. 14 Nov 2026 (hackathon day): the app against the published challenge

Challenge text (Details and Resources tabs) read. The app covers the full scope: choose a variable, a region, a
period; draw the time series; give the rate of change and test its significance (Sen slope, Mann-Kendall with
carry-over, 95% interval), plus a trend map showing where the sign differs.

- **Cross-check against NASA Giovanni** (the first resource the challenge lists). Giovanni's time-series API accepts
  the Earthdata token (`Authorization: Bearer`). 12 monthly grid-cell series (IMERG rain, GLDAS soil moisture and
  evaporation 2001-2024; MERRA-2 temperature 2020 from Giovanni's hourly product; Marrakech, Wisconsin, Lake Chad):
  correlation 1.000000, largest difference 0.0007% of the mean. Bundle `evidence/giovanni_crosscheck`. The API does
  not serve MODIS or monthly MERRA-2, so those are not covered.
- **Ocean chlorophyll added** (`MODIS_CHL`, MODIS Aqua L3 mapped monthly 9 km, R2022, OB.DAAC): the challenge's only
  research paper is a chlorophyll time-series study. OB.DAAC offers no remote subsetting (cloud OPeNDAP 404, Harmony
  no services, the old server rate-limits), so a new `monthfile` backend downloads each 13 MB month once and keeps a
  0.25-degree average (full record: about 4 GB of transfer, a few minutes, then cached).
- **Bug found by preloading:** the challenge video's own example (MODIS aerosol over Wisconsin) failed on a fresh
  environment: `pyhdf` (HDF4 reader for MODIS atmosphere files) was missing from requirements.txt. Added.
- **Saved results.** Every finished analysis is saved and reused; five example cases ship with the repository
  (`synaptic/dashboard/demo_cache/`), so the demo answers in ~0.2 s without the network. `python -m synaptic preload`.
  The page states when a saved result was computed and offers "Fetch again".
- Not done, noted for the team: the challenge names OMI and AIRS; the catalogue has neither.
Tests: 39 passed; `python -m synaptic reproduce`: 5/5 bundles.

---

## 12. 14 Nov 2026: global record scan (user's goal: more results like Morocco). Outcome: none confirmed.

Design and registration: `PREREGISTRATION_2.md` (commit a410385, before any confirming data). New files only.

**Scan (GRACE, both solutions, 4-degree land blocks 45S-50N, seasons ending Nov 2024 - Jun 2026):** 1,534
block-seasons; 97 records on both solutions against ~148 expected by chance; 5 candidates (record and >= 3 sd on
both) against ~13 expected by chance on one solution. No excess of records worldwide. All five candidates are wet
seasons: eastern Bolivia, Loess Plateau (China), SW Colombia, and two blocks in eastern Turkey.

**Power fix:** the confirming test uses ~130 monthly SMAP values with AR(1) carry-over instead of 11 season means.
Synthetic calibration: false alarms 1.8-5.1% at the 5% level; p < 0.01 reachable for strong events.

**Registered SMAP verdicts (Holm over the tests that ran):** Bolivia supported, not confirmed (p = 0.014, Holm
0.057); China, SE Turkey and Colombia not shown (p = 0.40, 0.37, 0.91); NE Turkey could not be tested (snow).
Pooled test p = 0.13: not shown. Placebo windows fire in 4 of 40 cases.

**Reading.** The Morocco result does not generalise into a method: a GRACE record plus this SMAP test did not
confirm any of five new candidates. For China and SE Turkey, rainfall is record or near-record (MERRA-2, and for
China the published literature), so the events are probably real, and surface soil moisture is a poor witness for
them (top 5 cm, memory of days, saturation in a long rainy season); but that is an explanation after the fact, and
the registration does not let rain decide. Bolivia is the opposite pattern (GRACE record and wet soil without
unusual local rain), which would fit water arriving by river from upstream; not tested.

GRACE volumes of the candidates (JPL / GSFC, excess over a normal season): Bolivia 61 / 33 km3; SE Turkey 30 / 30;
China 17 / 25; NE Turkey 17 / 19; Colombia 9 / 14. These are exploratory numbers from a scan whose candidates are
within chance; they are not presented as findings.
Bundle: `evidence/record_scan` (6/6 bundles reproduce).

---

## 13. 14 Nov 2026, afternoon: OMI and AIRS in the app, submission draft (current branch)

- **AIRS (Aqua), monthly v7:** air temperature at 850, 500, 200 and 50 hPa (about 1.5, 5.5, 12 and 20 km),
  surface air temperature, column water vapour. **OMI (Aura):** tropospheric NO2 (OMNO2d v4) and total ozone
  (OMTO3e v4). NASA publishes OMI L3 daily only, so a new `dailymean` backend averages each month's daily boxes
  (about 30 small requests per month, cached per box). 41 variables in all; the challenge's six named sources
  (MODIS, OMI, AIRS, GPM, MERRA-2, GLDAS) are all present.
- Small additive edits in `catalog.py` (two optional fields: `grid_spec`, `level`) and `cloud.py` (regular grids
  without coordinate variables, a vertical level index, nested HDF groups, the `dailymean` backend).
- Bug caught while testing: the shared missing-value filter treats |value| > 1e14 as missing; NO2 columns are
  ~1e15-1e16 molecules/cm2, so every real value was being dropped. The new backend does its own fill handling.
- **DOIs checked against NASA CMR:** five did not match (MODIS SST, MODIS chlorophyll, JPL GRACE, OMI NO2, OMI O3)
  and one (GSFC mascons) could not be verified and had been typed from memory; corrected or removed.
- Two more saved examples: the stratosphere at 50 hPa worldwide (AIRS) and NO2 over eastern China (OMI).
- `SUBMISSION.md`: first draft (summary, description, NASA data with verified DOIs, references, AI-use statement,
  7-slide outline). Items only the team can fill are marked TODO.

---

## 14. Branch `claude/new-search`: the weekend-shift search (started 11:54 UTC, hard stop 23:54 UTC)

Idea: start from dated facts on the ground (countries moving their weekend) and test whether NASA's OMI saw the
weekly NO2 cycle move, with daily data. Literature: weekly NO2 cycles are well studied; no published measurement
of a weekend change found (the closest study, read in full, calls the Saudi cycle stable).
**Where I am (17:15 UTC):** method self-tested on synthetic data (1% false alarms at p < 0.0125; 94-100% power);
`PREREGISTRATION_3.md` written. A session restart cost about 5 hours; the stop time is unchanged.
**Next step:** push the registration, then fetch OMI daily files, run the blind noise check, then the four tests.

**Where I am (17:40 UTC): weekend-shift tests run as registered.** UAE 2022 confirmed (D = +0.38, p = 0.00005,
both cities, placebos null): over Dubai and Abu Dhabi the quietest day in OMI's NO2 moved from Friday to Sunday in
2022. Oman 2013 and Saudi 2013 also passed, but the Kuwait City placebo fired on both 2013 dates (2 of 12 placebos),
so those two are not relied on. Algeria 2009 not shown. Details, post-hoc checks and the literature re-check are in
`PREREGISTRATION_3.md`; figures and the offline check in `evidence/weekend_shift/`.
**Next step:** plain-language summary; nothing goes to the app or the Findings page unless the team agrees.

**14 Nov, evening: the UAE result brought into the submission branch (user's decision), as a secondary result.**
`claude/new-search` fast-forwarded into `claude/admiring-fermat-l4nk42`. The Findings page has a new section after
the water results ("Also detected: a human-made change in the air"), with the year-by-year chart, the registered
p-value, the control cities, and the statement that Oman and Saudi Arabia passed but are not relied on. The "tested
and not found" table gains the record scan and the 2013/2009 weekend tests. `SUBMISSION.md` mentions it under
findings and in slide 6, not in the headline. Test count updated to 81 pre-registered tests. Also fixed: the
Findings page overflowed sideways on phone-width screens.

---

## 15. 14 Nov, evening: the teammate's version and the river-chlorophyll study

- The teammate's version (zip, files dated 10-14 Nov) was compared with this branch and the common base, without
  running it. It is a large change to the harness (see `MERGE_NOTES.md`): 26 conflicts in a trial merge. Decision:
  no harness code merged before the submission.
- Its river campaign (SeaWiFS-era chlorophyll trends off ten river mouths, re-tested on MODIS with VIIRS held out)
  was **re-checked independently here** with this branch's own reader and trend code (`scripts/river_recheck.py`,
  bundle `evidence/river_chlorophyll`): Mississippi West +3.7%/decade (corrected p = 0.43), Mississippi East +5.8%
  (0.40), Eel summer -6.7% (0.37), Po, Pearl, Ganges, Amazon all p > 0.14; Guadalquivir -15.4%/decade (p = 0.00007).
  Mississippi East 2002-2007: p = 0.004 uncorrected, 0.056 corrected. Same conclusions as the campaign.
- Wording corrected for our use: "not detected" instead of "refuted" or "vanished"; no "first"; no claim that the
  2009 paper was wrong (different period and sensor).
- Findings page: the river re-test is now the first result; saved example "Chlorophyll off the Mississippi".
  `SUBMISSION.md` leads its findings with it. 8/8 evidence bundles reproduce.
