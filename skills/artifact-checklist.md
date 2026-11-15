# Artifact checklist for NASA POWER / reanalysis signals

Run every item that applies and SAY which ones you ran.

1. **Data-production changes.** POWER solar and cloud fields (ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN,
   ALLSKY_SFC_LW_DWN, ALLSKY_KT, CLOUD_AMT) come from GEWEX SRB up to 2000 and from CERES SYN1deg from
   2001 (check the `sources` printed by the download tools), and start in 1984. A step near 2000-2001
   in these fields is an artifact until proven otherwise: test whether the step is identical across
   all months and all cells (instrument-like) or structured (physical). `synlib.source_transition_flag`
   flags it automatically.
   That switch only affects the solar/cloud fields. Never attribute a break in precipitation, temperature,
   humidity, wind or soil wetness to SRB->CERES: those come from MERRA-2.
2. **Reanalysis is a model.** MERRA-2 fields (temperature, humidity, soil wetness, precipitation) are
   model output constrained by observations whose mix changes over time (new satellites, station
   density). Precipitation and soil wetness are the least constrained. A strong claim needs an
   independent dataset: GPM IMERG / GPCP for rain, GRACE/GRACE-FO for water storage, MODIS LST or
   station records for temperature, MODIS/VIIRS NDVI for vegetation (use cmr_search and gibs_layers).
3. **Edge and coast cells.** Grid cells straddling coastlines or the box edge mix land and ocean.
4. **Gap filling and constant runs.** Look for repeated identical values or a suddenly smooth record.
5. **Known global/regional forcing.** Warming trends in T2M are expected (documented globally); a
   trend alone is not a discovery. What can be new: *where*, *when*, *how fast*, *which season*, or a
   decoupling between variables that normally move together.
6. **Multiple testing.** Hundreds of cells x several tests: trust FDR q-values (field significance),
   not single-cell p-values; spatial coherence (min_cells) is the second guard.
7. **Autocorrelation.** synlib inflates Mann-Kendall variance for lag-1 autocorrelation; annual
   means further reduce it. Be cautious with monthly-resolution claims.

8. **Mechanism for a MERRA-2 break.** If a POWER meteorological field breaks while an independent product
   (CHIRPS, CHIRTS) does not, the usual suspects are changes in what MERRA-2 assimilates or uses to correct its
   precipitation (gauge networks reporting or dropping out, new satellites), not POWER's solar source change.
   Rain-forcing errors propagate: drier forcing -> drier soil -> lower humidity -> hotter daytime maxima.
