# Independent corroboration beyond POWER

POWER is convenient but reanalysis-based. Strong claims need a second, independent line of evidence.
- **Independent observations (no login):** `independent_series(product='chirps'|'chirts', box, years, months)`
  returns box-mean monthly CHIRPS rainfall (gauges + satellite, 1981-2025) or CHIRTS max temperature
  (stations + satellite, 1983-2016). Build the difference (temperature) or ratio (rain) POWER/independent over
  the same box and months and run `synlib.pettitt` on it. Both products changing together = real climate;
  a break only in POWER = reanalysis inhomogeneity (often from changes in the observations it assimilates or
  in its precipitation correction), which then propagates to soil moisture, humidity and temperature.
- **NASA archives with the Earthdata token:** `nasa_series(product='imerg', variable='PRECIP', box, years)` gives
  GPM IMERG V07 rain (satellites + GPCC gauges, 1998-2025) on the campaign's own cells: the strongest check of a
  MERRA-2 rain signal, because MERRA-2 corrects its rain with CPC/CMAP, not IMERG. For land signals,
  `nasa_series(product='gldas21', variable='ROOTMOIST'|'EVAP'|...)` (2000-2025) or `'gldas20'` (1948-2014) gives
  the GLDAS Noah land model: a different model with different forcing, so agreement with MERRA-2 GWETROOT/EVLAND
  is real support (it is a model, not an observation; say so). Never join gldas20 and gldas21 into one series.
  Use correlations of anomalies and of trends/steps; units differ (GLDAS soil water is kg/m2, MERRA-2 is 0-1).
- **Imagery (no login):** `gibs_layers('NDVI')`, `gibs_layers('snow')`, `gibs_layers('TrueColor')`,
  then `gibs_snapshot(layer, date, box)` for dates before and after the change; `look` at both.
  Useful for vegetation loss/gain, reservoirs, irrigation expansion, snow cover, burn scars.
- **Dataset discovery:** `cmr_search('GPM IMERG monthly')`, `cmr_search('GRACE-FO mascon')`,
  `cmr_search('MODIS land surface temperature monthly')`, `cmr_search('SMAP soil moisture')`.
  Report which dataset would confirm the signal even if you cannot download it (most need a free
  Earthdata login), so NASA scientists know the exact next step.
- **Learned expectation (GPU):** `gpu_submit(template='surprise', inputs=[data.csv],
  params={'train_end_year': 2005})` trains a sequence model on each cell's early record and maps
  where later behaviour is most surprising. Agreement between the statistical family and the
  surprise map is independent evidence that something changed; disagreement is informative too.

## Weather stations (ground truth, to the present)
`station_series(element, box, years)` returns NOAA GHCN-Daily monthly means (TMAX, TMIN, PRCP) for the stations
in a box, each tagged with its POWER grid cell. Merge with POWER on lat, lon, year, month, then per station:
difference (POWER minus station) for temperature, log ratio for rain. Test the difference for a step at the
candidate year. Station records have gaps: compare matched months only, and say how many stations and months
the conclusion rests on. Use it especially after 2016, where CHIRTS ends.

**Elevation.** In highlands, a station can sit hundreds of metres above or below its grid cell's mean elevation.
Correct temperatures with a lapse rate of about 0.65-0.7 °C per 100 m, or compare only stations within ~100 m of
the cell elevation. Mean offsets of several °C (Dire Dawa, Combolcha in the 2026 East Africa test) are usually
elevation, not data error. Breaks in the POWER-minus-station difference are still meaningful after correction.
