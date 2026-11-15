# Pre-registration: independent confirmation with SMAP (written 10 Nov 2026, before any SMAP data were downloaded)

This file is committed and pushed BEFORE the confirmation script fetches any SMAP soil-moisture value.
The git commit time is the proof of order. Nothing below may be changed after the data are seen; any deviation
must be listed at the end under "Deviations", with the reason.

## What is being confirmed
Rounds 1-4 (see DISCOVERY_LOG.md) found, on two GRACE solutions (JPL and NASA GSFC), unusually large seasonal gains
of land water storage in three places. Those results came out of 55 held-out tests, so their p-values
(0.003-0.013) do not survive a correction for multiple testing. They are therefore treated as *leads*, and this
file registers ONE fresh test per lead on an instrument never used in this project.

| Lead | Box [S, N, W, E] | Event |
|---|---|---|
| A. Lake Chad basin | [10, 17, 12, 18] | 2024 rainy season |
| B. Central Sahara | [15, 23, 10, 25] | 2024 rainy season |
| C. Northern Morocco | [31, 36, -10, -1] | winter 2025-26 |

## Instrument and data
NASA SMAP L3 radiometer daily 36 km soil moisture, SPL3SMP version 009 (NSIDC DAAC), April 2015 to present.
SMAP is a microwave radiometer measuring the wetness of the top ~5 cm of soil. It shares no measurement, no
processing and no science team with GRACE (a gravity mission), and none of its data entered rounds 1-4 or Track B.
- Variable: `Soil_Moisture_Retrieval_Data_AM/soil_moisture` (6 am descending pass, baseline algorithm).
- Valid retrieval: 0 < value <= 0.6 m3/m3 (fill values removed). PRIMARY analysis uses all valid retrievals.
- Daily box mean: mean over the 36 km cells whose centre is inside the box; a day counts if at least 30% of the
  box's cells have a valid retrieval. Monthly mean: mean of valid days, needs at least 8 days.

## Statistic (fixed)
Surface soil moisture has a memory of days to weeks, so the SMAP analogue of "the land gained water this season"
is the season-mean wetness:
- Leads A and B: mean of August, September, October, for each year 2015-2026 (12 seasons). Event year: 2024.
- Lead C: mean of December (previous year), January, February, March, April, for season years 2016-2026
  (11 seasons). Event year: 2026.
A season needs all its months.

## Test (fixed)
`synlib.event_departure(season_means, years, [event_year])`: the event year's departure from the linear trend
of the other years, with carry-over accounted for (the same function the harness used for GRACE).
- Direction: positive (wetter), pre-registered. Pass = departure > 0 and two-sided p < 0.05.
- Three tests are run. Each is reported with its own p. Family-wise: a lead is called **confirmed** only if
  p < 0.05 / 3 = 0.0167; with 0.0167 <= p < 0.05 it is called **supported, not confirmed**; otherwise **not shown**
  (never "refuted": a non-significant result is absence of evidence).
- The test needs at least 10 non-event years; if a region has fewer valid seasons the result is "could not be
  tested".

## Sensitivity analysis (reported, cannot rescue a failed primary test)
Same test using only retrievals whose `retrieval_qual_flag` has bit 0 = 0 (recommended quality).

## Descriptive estimates (not tests; reported with uncertainty, on data already seen)
From the cached GRACE files (JPL RL06.3 mascons, GSFC RL06v2.0 mascons), land cells only:
- the volume of water (km3) corresponding to each lead's seasonal gain above the mean gain of the other years,
  for both solutions, with a 90% range from the year-to-year spread of the other years' gains;
- for leads A and B, the share of that excess still present in April-May 2025.
These are measurements, labelled as such, not claims of statistical discovery.

## What will be claimed
- If a lead is confirmed: "first satellite measurement, independently confirmed by SMAP".
- If supported or not shown: reported exactly so, next to the GRACE numbers.
- In no case is a mechanism beyond "the land stored the rain" claimed: rounds 2-4 tested for storage beyond what
  rain minus evaporation explains and for unusual persistence; all were not shown.

## Deviations
(none at registration)
(none. One note: SMAP was in safe mode in August-September 2022, so the 2022 season is missing for leads A and B
under the registered "a season needs all its months" rule; 11 seasons remained, 10 of them non-event years, the
registered minimum.)

## Results (run 10 Nov 2026, after the registration commit 0156c60)

| Lead | Event season vs trend of the other 10 | p | Rank of event season | Registered verdict |
|---|---|---|---|---|
| A. Lake Chad basin, Aug-Oct 2024 | +0.046 m3/m3 | 0.101 | 1 of 11 | **not shown** |
| B. Central Sahara, Aug-Oct 2024 | +0.021 m3/m3 | 0.049 | 1 of 11 | **supported, not confirmed** |
| C. Northern Morocco, Dec-Apr 2025-26 | +0.051 m3/m3 | 0.0106 | 1 of 11 | **confirmed** (p < 0.0167) |

Sensitivity (recommended-quality retrievals only): A p = 0.095; B p = 0.022; C could not be tested (10 valid
seasons, fewer than the 10 non-event years required).
Descriptive, not registered: in all three regions the event season is the wettest in SMAP's record.
Data and code: `evidence/smap_confirmation/`, `scripts/smap_confirm.py`.
