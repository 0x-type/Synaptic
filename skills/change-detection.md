# Change detection with synlib

```python
import synlib, numpy as np
df = synlib.load("path/to/power_region_xxx.csv")      # long format
cube = synlib.to_cube(df, "GWETROOT")                   # [lat, lon, year, 12]
i, j = cube.cell(31.5, -7.5)
r = synlib.analyze_series(cube.data[i, j], cube.years)  # every test at one cell
sw = synlib.sweep(df, "GWETROOT")                       # every cell, with FDR q-values
sw.to_csv("sweep_GWETROOT.csv", index=False)
synlib.plot_field("map.png", sw, "sen_per_decade", "GWETROOT trend", q_column="mk_q")
ann = cube.annual()[i, j]
synlib.plot_series("cell.png", cube.years, ann, "GWETROOT", cp_year=r.get("cp_year"), months=cube.data[i, j])
print(synlib.series_text(cube.years, ann))              # raw numbers into your context
```

Signal classes (`synlib.classify_cells`):
- **trend**: Mann-Kendall + Sen slope; effect = total change / interannual SD.
- **shift**: Pettitt break, kept only when a step model beats a linear trend by BIC (a regime shift
  is physically different from gradual forcing, and is also the signature of data artifacts).
- **variance**: Brown-Forsythe test of variability before vs after the break.
- **seasonality**: phase of the first annual harmonic, first vs second half of the record;
  bootstrap p-value. Reported in days (positive = later peak).
- **recent_extreme**: last 3 years vs all earlier years.

Seasonal analysis: `synlib.analyze_series(monthly, years, months=[6,7,8])` or `sweep(df, var, months=[...])`.
Corroboration: `synlib.corroboration(family_dict, {var: sweep_df, ...})` uses `LINKED_VARIABLES`.
