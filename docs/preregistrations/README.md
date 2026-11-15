# Pre-registrations

Each confirming test was written down and pushed to git **before** any of the data it uses were downloaded.
Each file shows when the test was written, followed by its results exactly as they came out.

| Registration | What it fixes in advance | Registered (commit, UTC) | Results added (commit, UTC) | Outcome |
|---|---|---|---|---|
| [1. Water gains, SMAP check](1_water_gains_smap.md) | The SMAP test of the three GRACE water gains, the threshold (p < 0.0167 for 3 tests), the season windows | `0156c60`, 10 Nov 2026 07:45 | `bc8c145`, 10 Nov 2026 07:54 | Morocco confirmed (p = 0.011); Sahara supported, not confirmed; Lake Chad not shown |
| [2. Global record scan](2_record_scan.md) | The SMAP test of 5 record-season candidates from a worldwide GRACE scan | `a410385`, 14 Nov 2026 11:33 | `1d2f7ad`, 14 Nov 2026 11:46 | None confirmed (1 supported, 3 not shown, 1 could not be tested) |
| [3. Weekend shift](3_weekend_shift.md) | Four dated weekend changes, the statistic, the cities, the control cities and the threshold | `8daea44`, 14 Nov 2026 17:10 | `97d096b`, 14 Nov 2026 17:34 | UAE 2022 confirmed (p = 0.00005); Oman and Saudi 2013 passed but are not relied on; Algeria 2009 not shown |

Each file holds the registered text, then the results exactly as they came out, then any deviations from the plan.
"Not shown" means the test was not significant. That is not evidence against the idea.

Every number in these results is recomputed offline by `python -m synaptic reproduce` from the bundles in
[`evidence/`](../../evidence).
