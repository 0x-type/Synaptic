# Pre-registration 2: the global record scan (written 14 Nov 2026, before any SMAP or rainfall value for the candidates was fetched)

This file, `scripts/record_scan.py`, `scripts/record_confirm.py` and
`evidence/record_scan/registered_candidates.json` are committed and pushed BEFORE `record_confirm.py fetch` or
`test` is run. The git commit time is the proof of order. Deviations, if any, are listed at the end.

## Step 1 (done, exploratory): the GRACE scan
`scripts/record_scan.py` compared, for every 4-degree land block between 45S and 50N, the latest wet-season storage
gain and dry-season drawdown (seasons ending Nov 2024 - Jun 2026) with every earlier season since 2003, on two
GRACE solutions (JPL RL06.3 mascons to Sep 2026, NASA GSFC RL06v2.0 mascons to Jun 2026). Regions examined in earlier
rounds (Sahel-Sahara, Morocco) were excluded.

**What the scan says, before any confirmation:**
- 1,534 block-seasons were examined. 97 are records on both solutions. If years were interchangeable, about 148
  would be records by chance. **The scan shows no excess of records worldwide.**
- 5 block-seasons are a record on both solutions AND at least 3 baseline standard deviations out on both. About 13
  would do so by chance on one solution. **Five candidates are no more than chance predicts.**
- So the GRACE scan alone proves nothing. Each candidate below is a lead that needs the independent test. All five
  are unusually large wet-season gains; no drought passed the filter.

| Id | Region (box S, N, W, E) | Season | GRACE gain vs normal, JPL / GSFC (cm) | sd out, JPL / GSFC |
|---|---|---|---|---|
| T1 | NE Turkey, SW Georgia [39, 43, 40, 44] | Oct 2025 to Apr-May 2026 | 31.6 vs 17.9 / 34.3 vs 19.1 | 3.5 / 3.3 |
| T2 | SE Turkey, N Syria [35, 39, 36, 40] | Oct 2025 to Apr-May 2026 | 35.8 vs 16.9 / 34.2 vs 15.1 | 3.1 / 4.1 |
| BO | Eastern Bolivia [-17, -13, -64, -60] | Oct 2024 to Apr-May 2025 | 64.7 vs 32.9 / 49.9 vs 32.8 | 4.9 / 3.2 |
| CO | SW Colombia, N Ecuador [-1, 3, -80, -76] | Oct 2024 to Jun-Jul 2025 | 9.9 vs 4.2 / 17.3 vs 8.5 | 3.3 / 3.3 |
| CN | Loess Plateau, N China [35, 39, 108, 112] | Jun to Oct-Nov 2025 | 14.8 vs 4.1 / 19.5 vs 3.5 | 3.1 / 3.9 |

## Step 2 (done): literature (NASA ADS, 14 Nov 2026)
- CN: the 2025 North China rainy season is published as a rainfall record ("The 2025 longest rainy season and
  greatest precipitation in North China", 2026, and others). We found no published measurement of the water storage
  gain.
- T1, T2: GRACE studies of Türkiye and of the Tigris-Euphrates basins exist (2026), on long-term decline. We found
  no published measurement of the winter 2025-26 gain. This is the most crowded literature of the five.
- BO, CO: we found no paper on a 2025 water-storage record there.
Whatever the outcome, the wording is "we found no published measurement", never "first".

## Step 3 (registered now): the independent confirmation
**Instrument:** NASA SMAP L3 radiometer daily 36 km soil moisture (SPL3SMP v009), 6 am pass, valid retrievals
(0 < value <= 0.6 m3/m3), April 2015 - October 2026. Daily box mean = mean over cells centred in the box, counted
when at least 30% of the box's cells are valid; a month needs 8 valid days.

**Statistic and test (fixed; `monthly_event_test` in `scripts/record_confirm.py`):** the mean SMAP anomaly over the
event months (the months of the GRACE gain window after its first month: listed per candidate in
`registered_candidates.json`), against all other months of the record. Anomalies are relative to each calendar
month's mean and to a linear trend, both estimated without the event months. The month-to-month carry-over (AR(1))
sets the variance of the event mean and the effective number of independent months; t-test.

**Why this test can pass (the lesson of the first registration, where 11 season means could not go below p = 0.09):**
it uses about 130 monthly values. Checked on synthetic data only (`calibration.json`, 1,500 runs per case): false
alarms at the 5% level are 1.8-5.1%; for an event of 2.5 monthly standard deviations lasting 4-8 months, p < 0.01
is reached in 81-100% of runs with carry-over up to 0.3, 55-77% at 0.6, and 34-45% at 0.8. For a 1.5 sd event the
power is lower (9-84%). A real but modest event in a region with strong carry-over may therefore be "not shown".

**Direction:** wetter (+) for all five, as the GRACE gains predict.

**Verdicts:**
- Five tests, Holm correction. **Confirmed:** right direction and Holm-adjusted p < 0.05 (the smallest p must be
  below 0.01). **Supported, not confirmed:** right direction and unadjusted p < 0.05. **Not shown:** otherwise
  (never "refuted"). **Could not be tested:** fewer than 60 non-event months, fewer than 60% of the event months,
  or a calendar month with fewer than 5 baseline values (expected risk: snow and frozen ground for T1; dense forest
  for CO).
- **Pooled test (does the scan find real events at all?):** one z per independent event (T1 and T2 are one event:
  their z are averaged), Stouffer combination, one-sided. Significant if p < 0.05.

**Reported with it, not part of any verdict:**
- Placebo: the same window shifted to every other year in SMAP's record, for each candidate (how often the test
  fires on a year with no GRACE record).
- Rain: season-mean precipitation over the same months from MERRA-2 (gauge-corrected over land), 1981-2026, event
  season against the trend of the others (`synlib.event_departure`) and its rank. Rain is the expected cause of a
  storage gain; agreement is a consistency check, not a second confirmation.
- The volume of water for each candidate from GRACE (both solutions), with the year-to-year range.

## What will be claimed
For a confirmed candidate: "GRACE (two solutions) shows a record seasonal gain of water storage, confirmed by an
independent NASA satellite with a test registered in advance; we found no published measurement of it". Nothing
about mechanism. For the others: exactly their verdict. The scan-level facts above (no excess of records, five
candidates within chance) are reported in every summary.

## Deviations
(none at registration)
(none.)

## Results (run 14 Nov 2026, after the registration commit a410385; 4,146 daily SMAP files, no download failures)

| Id | Region | SMAP event departure | p (Holm) | Registered verdict | Rain, MERRA-2 (not part of the verdict) |
|---|---|---|---|---|---|
| BO | Eastern Bolivia | +0.074 m3/m3 | 0.014 (0.057) | **supported, not confirmed** | rank 33 of 45, p = 0.48 |
| CN | Loess Plateau, N China | +0.015 | 0.40 (1) | **not shown** | wettest of 45, p = 0.0006 |
| T2 | SE Turkey, N Syria | -0.024 | 0.37 (1) | **not shown** | 2nd wettest of 45, p = 0.010 |
| CO | SW Colombia, N Ecuador | -0.001 | 0.91 (1) | **not shown** | 2nd wettest of 45, p = 0.073 |
| T1 | NE Turkey, SW Georgia | - | - | **could not be tested** (1 of 6 event months valid: snow and frozen ground, as anticipated) | rank 10 of 45, p = 0.63 |

**Pooled test:** 4 independent events, z = 1.15, one-sided p = 0.13: **not shown**. The scan is not shown to find
real events.
**Placebo** (same window in other years): 4 of 40 placebo tests have p < 0.05 (10%; nominal 5%): the test is
slightly liberal on real data, which makes the verdicts above, if anything, generous.

**Nothing is confirmed.** One candidate (eastern Bolivia) is supported by SMAP but not by local rainfall. Two
(northern China, south-east Turkey) have record or near-record rainfall in MERRA-2 and a GRACE record, but SMAP's
surface soil moisture does not show them; under the registration, rain cannot rescue them. No verdict was changed
after seeing the data.
