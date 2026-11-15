# Pre-registration 3: did the air notice when a country moved its weekend?

Written 14 Nov 2026 on branch `claude/new-search`, and pushed BEFORE `scripts/weekend_shift.py fetch` downloads any
OMI value for the cities below. The git commit time is the proof of order. Deviations are listed at the end.

## Why this question
Earlier searches looked for something unusual in the data and then needed a second dataset to confirm it, with
about 20 yearly values. This one runs the other way: it starts from dated facts on the ground and asks whether a
NASA instrument saw them, using daily data (thousands of values). The hypotheses come from the calendar, not from
the data, so nothing was selected by looking.

City nitrogen dioxide (NO2) comes mostly from traffic, power and industry, and drops on rest days. Four countries
moved their weekend on a known date within the record of NASA's OMI instrument (2004 to now).

## Data
OMI/Aura tropospheric NO2 column, cloud-screened, daily Level-3 0.25 degree (OMNO2d v4, NASA GES DISC,
doi 10.5067/AURA/OMI/DATA3407), October 2004 to October 2026. For each city: the mean of the valid cells in a
3 x 3 cell box (about 80 km) centred on the city, on days with at least 3 valid cells. Negative retrievals are kept.

## The four tests (fixed)
| Test | Change (date) | Cities (box centres) | Day predicted to RISE | Day predicted to FALL | Reference days | Period |
|---|---|---|---|---|---|---|
| UAE_2022 | Friday-Saturday weekend became Saturday-Sunday, Friday a half day (1 Jan 2022) | Dubai (25.20N, 55.30E), Abu Dhabi (24.45N, 54.40E) | Friday | Sunday | Mon-Thu | 2017-01-01 to 2025-12-31 |
| SAUDI_2013 | Thursday-Friday became Friday-Saturday (29 Jun 2013) | Riyadh (24.70N, 46.70E), Jeddah (21.50N, 39.20E), Dammam (26.40N, 50.10E) | Thursday | Saturday | Sun-Wed | 2005-01-01 to 2021-12-31 |
| OMAN_2013 | Thursday-Friday became Friday-Saturday (1 May 2013) | Muscat (23.60N, 58.40E) | Thursday | Saturday | Sun-Wed | 2005-01-01 to 2021-12-31 |
| ALGERIA_2009 | Thursday-Friday became Friday-Saturday (14 Aug 2009) | Algiers (36.75N, 3.05E) | Thursday | Saturday | Sun-Wed | 2005-01-01 to 2014-12-31 |

**Statistic.** For a city and a period, R_d = mean NO2 on weekday d divided by mean NO2 on the reference days (days
that were working days both before and after). D = (R_rise - R_fall) after the change minus (R_rise - R_fall)
before it, averaged over the test's cities. **Prediction: D > 0** for all four.

**Test.** Weeks are the unit (7-day blocks counted from the change date). Under "the weekly cycle did not change",
the before/after labels of the weeks are exchangeable: they are permuted 20,000 times (the same permutation for all
cities of a test). Two-sided p (`shift_test` in `scripts/weekend_shift.py`). Using ratios within each period removes
seasons and long-term trends in the NO2 level.

**Verdicts.** Four tests, Holm correction. **Confirmed:** D > 0 and Holm-adjusted p < 0.05 (so the smallest p must
be below 0.0125). **Supported, not confirmed:** D > 0 and unadjusted p < 0.05. **Not shown:** otherwise (never
"refuted"). If D is significantly negative, that is reported as a result against the prediction.

**Placebos (reported, not a verdict).** The three Gulf tests are repeated, same dates and days, on four cities whose
weekend did not change then: Kuwait City, Doha, Cairo, Tehran (12 placebo tests). If the method is sound, about
5% of them, not more, should have p < 0.05.

## Can this test pass? (the lesson of the first two registrations)
- **Expected size.** A published study (Sci. Rep. 2020) reports Friday NO2 about 20% below the weekly mean in Saudi
  and Emirati cities. Moving a rest day should therefore change D by roughly +0.2 to +0.4.
- **Synthetic self-test (`selftest.json`, no real data):** two cities, 45% of days valid, 45% day-to-day noise:
  with no change, p < 0.0125 in 1.0% of runs; with a moved weekend and a 20% rest-day drop, in 100%; with a 10%
  drop, in 94%.
- **Known weaknesses:** OMI lost about half its daily coverage after 2007 (row anomaly); the UAE test has only
  four years after the change, and its Friday half day may raise Friday NO2 only a little (OMI passes at about
  13:30 local time); Algiers is small and often cloudy. A "not shown" for Algeria or Oman would not be surprising.
- **Blind noise check, registered here:** `weekend_shift.py noise` reports valid days and day-to-day spread using
  ONLY reference weekdays BEFORE each change date. It cannot show whether the cycle moved, and it cannot change
  the tests; it is run after fetching and reported with the results.

## Literature (NASA ADS and web search, 14 Nov 2026)
The weekly NO2 cycle is well studied ("Satellite evidence for changes in the NO2 weekly cycle over large cities",
Sci. Rep. 2020, OMI 2005-2017 and one year of TROPOMI; "Weekly cycle of NO2 revisited"; and others). The 2020
study, read in full, does not discuss any country changing its rest days, and describes the Saudi weekly cycle as
stable. We found no published measurement of a weekend change in satellite NO2 for any of the four cases. That
study's supplementary table and the wider literature were not read in full, so the wording stays "we found no
published measurement", never "first".

## What will be claimed
For a confirmed test: "OMI shows the weekly NO2 cycle moving with the weekend on the date of the change; we found
no published measurement of it". No claim about emissions in tonnes, health, or causes beyond the calendar change.
Everything stays on this branch and out of the app unless confirmed and the team agrees.

## Deviations
(none at registration)
(none.)

## Results (run 14 Nov 2026, after the registration commit 8daea44; 8,017 daily OMI files, no download failures)

Blind noise check (pre-change reference days only): 180-380 valid days per weekday per city; each weekday ratio
known to about 0.02-0.045. Adequate for the expected shift.

| Test | Day predicted to rise: before -> after | Day predicted to fall: before -> after | D | p (Holm) | Registered verdict |
|---|---|---|---|---|---|
| UAE_2022 | Friday 0.76 -> 0.95 | Sunday 0.95 -> 0.76 | +0.384 | 0.00005 (0.0002) | **confirmed** |
| OMAN_2013 | Thursday 0.96 -> 1.02 | Saturday 1.03 -> 0.93 | +0.152 | 0.0001 (0.0003) | **confirmed** |
| SAUDI_2013 | Thursday 0.96 -> 0.99 | Saturday 0.98 -> 0.93 | +0.075 | 0.010 (0.021) | **confirmed** |
| ALGERIA_2009 | Thursday 0.94 -> 1.03 | Saturday 0.94 -> 0.92 | +0.113 | 0.12 (0.12) | **not shown** |

(Values: NO2 on that weekday relative to the reference working days, mean over the test's cities. The UAE p is the
smallest the 20,000 permutations can give: none reached the observed D.)

**Placebos: 2 of 12 have p < 0.05, more than the 5% a sound test allows.** Both are Kuwait City on the two 2013
tests (D = +0.19 and +0.18, p = 0.001 and < 0.001). The four UAE-date placebos are all null (Doha p = 0.16, Tehran
0.33, Cairo 0.50, Kuwait City 0.98), as are the other six.

## What we rely on, and what we do not (stricter than the registered verdicts)
- **UAE 2022: relied on.** Largest effect, both cities separately (Dubai Friday 0.75 -> 0.97, Sunday 0.97 -> 0.75;
  Abu Dhabi 0.76 -> 0.93 and 0.93 -> 0.77), and all four placebos at its date are null.
- **Saudi Arabia 2013 and Oman 2013: passed the registered test, NOT relied on.** A control city with no weekend
  change in 2013 shows a shift of the same size, so on these dates the test can fire without a dated change.
- **Algeria 2009: not shown.**

## Post-hoc checks (NOT registered; run after the results; descriptive only)
- Kuwait moved its own weekend on 1 Sep 2007, inside the placebo's "before" window: a design error in the placebo.
  With a clean window (2008 on) Kuwait still shows D = +0.17 (p = 0.003, 0.005) around 2013: a slow drift of its
  weekly cycle, not a dated change. Kuwait's own 2007 change tested directly: D = +0.11, p = 0.16 (not shown).
- Year by year, Saudi Saturdays were already falling before 2013 (1.03 in 2005-06, 0.93-0.95 in 2010-12): also a
  drift. Oman's yearly values are too noisy to date a step.
- UAE year by year (`UAE_2022_yearly.csv`): Friday 0.68-0.83 in every year 2005-2021, then 0.90, 0.97, 0.97, 0.96,
  0.96 in 2022-2026; Sunday 0.90-1.12 in 2005-2021, then 0.66, 0.85, 0.78, 0.75, 0.75. Without 2020 (COVID):
  D = +0.39, p = 0.0002. With a false change date inside one regime (mid-2019; 1 Jan 2024): D = +0.07 and +0.02,
  p = 0.31 and 0.81.
- Literature re-checked after the result (ADS and web): UAE NO2 studies of 2022-2026 cover trends, periodicity,
  forecasting and COVID; one ground-station study of Abu Dhabi (2019-2022) notes the weekly pattern "is expected to
  change" after 2022 but does not measure it. We found no published measurement of the change.
