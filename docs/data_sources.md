# NASA data sources

All accessed with one NASA Earthdata login. DOIs as listed in NASA's Common Metadata Repository on 14 Nov 2026.

| Data | Product | Used for | DOI or address |
|---|---|---|---|
| MERRA-2 reanalysis (GMAO) | Monthly means: temperature, rain, soil wetness, runoff, snow, wind, pressure, water vapour, clouds, sunlight, aerosols | App; agents' exploration | 10.5067/AP1B0BA5PD2K and related M2TMNX collections |
| GPM IMERG Final V07 | Monthly rainfall, 0.1 degree | App; held-out tests | 10.5067/GPM/IMERG/3B-MONTH/07 |
| GLDAS Noah 2.1 | Monthly soil moisture, evaporation, snow depth | App; agents' exploration | 10.5067/SXAVCZFAQLNO |
| MODIS Aqua atmosphere (MYD08_M3 C6.1) | Aerosol optical depth, cloud fraction, droplet size | App | 10.5067/MODIS/MYD08_M3.061 |
| MODIS Aqua ocean | Sea surface temperature; chlorophyll-a | App; river re-test | 10.5067/MODSA-MO9D9; 10.5067/AQUA/MODIS/L3M/CHL/2022.0 |
| OMI on Aura | Tropospheric nitrogen dioxide (OMNO2d v4); total ozone (OMTO3e v4) | App; UAE weekend result (daily data) | 10.5067/AURA/OMI/DATA3407; 10.5067/Aura/OMI/DATA3402 |
| AIRS on Aqua (AIRS3STM v7) | Temperature at 850, 500, 200, 50 hPa; surface air temperature; water vapour | App | 10.5067/UBENJB9D3T2H |
| CERES EBAF-TOA Ed4.2.1 | Reflected sunlight, net energy, cloud area | App; held-out tests | 10.5067/TERRA-AQUA-NOAA20/CERES/EBAF-TOA_L3B004.2.1 |
| GRACE / GRACE-FO, JPL mascons RL06.3 v4 | Total water storage | App; water findings | 10.5067/TEMSC-3JC634 |
| GRACE / GRACE-FO, NASA GSFC mascons RL06v2.0 | Total water storage (independent processing) | Water findings: replication | earth.gsfc.nasa.gov/geo/data/grace-mascons |
| SMAP L3 radiometer soil moisture (SPL3SMP v009) | Daily surface soil moisture | Water findings: independent confirmation | 10.5067/4XXOGX0OOW1S |
| NASA Giovanni | Time-series service | Cross-check of the app's data path | giovanni.gsfc.nasa.gov |
| NASA POWER, GIBS, CMR | Point data, imagery, catalogue search | Agents' tools | power.larc.nasa.gov |
| NASA ADS | Literature search | Novelty check | ui.adsabs.harvard.edu |

Other open data used by the agents' tools: CHIRPS rainfall (UCSB), GHCN-Daily stations and climate indices (NOAA),
Natural Earth country outlines, OpenAlex, Crossref.

## References

1. Yoon et al. (2026), *Autonomous AI agents discover reverse transcriptases with tandem repeat arrays* (the
   multi-agent method the harness adapts).
2. *Time-Series Analysis of Remotely-Sensed SeaWiFS Chlorophyll in River-Influenced Coastal Regions*, NASA technical
   report NTRS 20090042747 (in the challenge resources).
3. Rodell et al. (2018), *Emerging trends in global freshwater availability*, Nature.
4. *How Have Hydrological Extremes Changed over the Past 20 Years?* (2023), Journal of Climate.
5. *Annual Memory in the Terrestrial Water Cycle* (2025), Hydrology and Earth System Sciences.
6. *Long-term and interannual variability of African terrestrial water storage revealed by GRACE/GRACE-FO and
   models* (2026), Journal of Hydrology.
7. *Synoptic drivers of the August 2024 record-breaking rainfall in the Chadian Sahara* (2026), Weather and Climate
   Dynamics.
8. *A multi-source geospatial and meteorological analysis of the 2026 Gharb Plain floods, Morocco* (2026).
9. *When climate variability partly compensates for groundwater depletion: an analysis of the GRACE signal in
   Morocco* (2022), Journal of Hydrology: Regional Studies.
10. *Satellite evidence for changes in the NO2 weekly cycle over large cities* (2020), Scientific Reports.
11. Statistical methods: Mann-Kendall trend test; Sen's slope estimator; Holm's sequential correction; permutation tests.

Software: Python, NumPy, pandas, SciPy, statsmodels, xarray, netCDF4, pyhdf, Flask, Chart.js, matplotlib.
