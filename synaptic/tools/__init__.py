"""Tool registry. Each task kind gets the common toolbox plus its stage-specific tools."""
from __future__ import annotations

from .base import Tool, ToolContext, ToolResult
from . import catalogdata, climate, compute, earthdata, gpu, independent, nasa, nasa_archive, science

COMMON = [
    Tool("power_region", "Download NASA POWER monthly data for a lat/lon box (1981-2025, MERRA-2/CERES grid "
         "~0.5x0.625 deg). One CSV in long format; tiled automatically.",
         {"lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "parameters": "list of POWER parameter names, e.g. ['T2M','PRECTOTCORR','GWETROOT']",
          "start_year": "int (>=1981)", "end_year": "int (<=2025)"}, nasa.power_region),
    Tool("power_point", "Download NASA POWER monthly data for one site (several parameters at once).",
         {"lat": "float", "lon": "float", "parameters": "list of names", "start_year": "int", "end_year": "int"},
         nasa.power_point),
    Tool("gibs_layers", "Find NASA GIBS satellite imagery layer names by keyword(s), e.g. 'NDVI', 'snow', "
         "'TrueColor MODIS Terra'.", {"keyword": "str"}, nasa.gibs_layers),
    Tool("gibs_snapshot", "Fetch a satellite image (GIBS) of a box on a date; then look() at it.",
         {"layer": "str", "date": "YYYY-MM-DD", "lat_min": "float", "lat_max": "float", "lon_min": "float",
          "lon_max": "float", "width": "int pixels (default 900)"}, nasa.gibs_snapshot),
    Tool("cmr_search", "Search NASA's Common Metadata Repository for datasets (independent corroboration).",
         {"keyword": "str", "limit": "int"}, nasa.cmr_search),
    Tool("python", "Run a Python script in your task folder. synlib, numpy, pandas, scipy, statsmodels, "
         "matplotlib are available. print() what you need; save figures as PNG and then look() at them.",
         {"code": "str (full script)", "timeout": "int seconds (optional)"}, compute.python),
    Tool("look", "DIRECT OBSERVATION. A .png is shown to you as an image. A data .csv prints the raw "
         "numbers of one location (annual series + full monthly table). Use it before concluding anything.",
         {"path": "str", "variable": "str (csv only)", "lat": "float (csv only)", "lon": "float (csv only)",
          "months": "list of month numbers for a season (csv only)"}, compute.look),
    Tool("independent_series", "Independent observations to corroborate or refute a POWER signal: 'chirps' (rain gauges + "
         "satellite rainfall, 1981-2025) or 'chirts' (station + satellite max temperature, 1983-2016), box-mean monthly "
         "series. Compare POWER minus this product over the same box: a break in the difference = data inhomogeneity.",
         {"product": "chirps | chirts", "lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "start_year": "int", "end_year": "int", "months": "list of months (optional, e.g. the rainy season)",
          "per_cell": "bool: true = one series per NASA POWER grid cell (to MAP where a break is), false = box mean"},
         independent.independent_series),
    Tool("station_series", "Ground truth: NOAA GHCN-Daily weather-station monthly means (TMAX, TMIN, PRCP; global, to the "
         "present), each station mapped to its POWER grid cell. Independent of MERRA-2: use it to test whether a POWER "
         "step or trend also appears at real stations (difference series per station).",
         {"element": "TMAX | TMIN | PRCP", "lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "start_year": "int", "end_year": "int", "min_days": "int valid days per month (default 20)",
          "max_stations": "int (default 40)"}, independent.station_series),
    Tool("merra2_series", "Native NASA MERRA-2 monthly fields (Earthdata token required), on the same grid as POWER: "
         "PRECTOTCORR, PRECTOT (uncorrected model rain), T2M, T2MDEW, QV2M, TS, SLP, U850/V850 winds, TQV, GWETROOT, "
         "GWETTOP, EVLAND, SWGDN, CLDTOT "
         "and more. Tells whether a problem is in POWER processing or in the reanalysis; PRECTOT vs PRECTOTCORR shows "
         "what the observation-based rain correction changed.",
         {"variable": "str", "lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "start_year": "int", "end_year": "int", "months": "list of months (optional)"}, earthdata.merra2_series),
    Tool("nasa_series", "More NASA Earthdata archives on the campaign's own grid (Earthdata token): 'imerg' = GPM "
         "IMERG V07 satellite + gauge rain, 1998-2025 (independent of MERRA-2's rain correction: the best check of a "
         "MERRA-2 rain signal); 'gldas21' (2000-2025) / 'gldas20' (1948-2014) = GLDAS Noah land model, an independent "
         "model estimate of root-zone soil moisture (ROOTMOIST), layers SOILM_0_10/SOILM_10_40/SOILM_40_100, EVAP, "
         "TRANSP, POTEVAP, RUNOFF, LATENT, SENSIBLE, TAIR, QAIR, RAIN, SWE. Never splice gldas20 with gldas21.",
         {"product": "imerg | gldas21 | gldas20", "variable": "PRECIP for imerg; a GLDAS name above",
          "lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "start_year": "int", "end_year": "int", "months": "list of months (optional)"}, nasa_archive.nasa_series),
    Tool("nasa_data", "NASA catalogue datasets from Earthdata Cloud on a regular grid (long table lat, lon, variable, "
         "year, month, value). Ids include MERRA2_T2M, MERRA2_PRECIP, IMERG_PRECIP, MERRA2_CLDLOW, MERRA2_CLDTOT, "
         "MODIS_CLOUD, MODIS_CLOUD_REFF (droplet radius), MODIS_AOD, MERRA2_AOD, MERRA2_AOD_SULFATE, MERRA2_AOD_DUST, "
         "CERES_SW_UP, CERES_SW_CLR, CERES_NET, CERES_CLOUD, MODIS_SST, MERRA2_SST, MERRA2_WIND, MERRA2_SLP, MERRA2_TQV, "
         "GLDAS_SOILM, GRACE_TWS, MERRA2_SNOW. Datasets held out for the campaign's confirmation are refused.",
         {"dataset": "catalogue id", "lat_min": "float", "lat_max": "float", "lon_min": "float", "lon_max": "float",
          "start_year": "int", "end_year": "int", "resolution": "grid degrees (default 1.0)"}, catalogdata.nasa_data),
    Tool("climate_index", "Monthly climate-driver indices from NOAA PSL: NINO34 (ENSO), SOI, DMI (Indian Ocean Dipole), "
         "AMO, PDO, NAO, TNA, QBO. For drivers, lags, predictability and changing teleconnections.",
         {"names": "list of index names", "start_year": "int", "end_year": "int"}, climate.climate_index),
    Tool("literature_search", "Search published literature (ADS, OpenAlex, Semantic Scholar, Crossref, Europe PMC). Use precise "
         "descriptors (variable, region, period, kind of change) for novelty tests.",
         {"query": "str", "n": "int <= 15"}, compute.literature_search),
    Tool("where", "Real place names (GeoNames, offline) for a lat/lon box: centre town, countries and provinces. Use it "
         "before naming a region in any result.", {"lat_min": "float", "lat_max": "float", "lon_min": "float",
                                                    "lon_max": "float"}, science.where),
    Tool("kb_search", "Search the campaign's shared knowledge base (findings, rejections, methods).",
         {"query": "str"}, science.kb_search),
    Tool("read_task", "Read another task's brief and summary from the shared record.", {"task_id": "str"},
         science.read_task),
    Tool("family_info", "Details of an anomaly family: cells, bbox, onset, effect, filter results.",
         {"family": "str"}, science.family_info),
    Tool("gpu_submit", "Submit a GPU job to Kaggle (2x T4, PyTorch). Ready templates that read the campaign's own "
         "trust-masked tables and all climate indices automatically (no inputs needed, just params):\n"
         "        'teleconnection' params {target, driver, season, lag, split_year}: per-cell |r| early vs late with "
         "bootstrap CIs, the change WITHOUT strong ENSO years and with amplitude-matched periods (the sample-effect "
         "control), field test, 15-year windows, 10-degree blocks;\n"
         "        'causal' params {target, sources, max_lag, split_year}: conditional Granger causality of each source on "
         "the target in every cell, early vs late, FDR over cells (new or lost links);\n"
         "        'predictability' params {target, season, lead, predictors, drivers, split_year}: out-of-sample "
         "forecast skill maps (per-cell ridge + shared network), early vs late;\n"
         "        'surprise': where reality departs from a learned model. Or pass your own `code` (reads ./inputs/<file>, "
         "writes ./out/...). Large inputs are attached as a private dataset automatically.",
         {"code": "str (optional)", "template": "str (optional)", "inputs": "list of file paths",
          "accelerator": "NvidiaTeslaT4 | NvidiaTeslaP100", "params": "dict (optional)"}, gpu.gpu_submit),
    Tool("gpu_status", "Status of a GPU job (waits up to wait_seconds).", {"job": "str", "wait_seconds": "int"},
         gpu.gpu_status),
    Tool("gpu_fetch", "Download a finished GPU job's outputs into your folder.", {"job": "str"}, gpu.gpu_fetch),
]

STAGE = {
    "sweep": [Tool("register_sweep", "Register a sweep table (synlib.sweep output CSV) for one variable.",
                   {"path": "str", "variable": "str"}, science.register_sweep)],
    "classify": [Tool("build_families", "Classify significant cells and group them into anomaly families "
                      "(uses every registered sweep).", {}, science.build_families)],
    "census": [Tool("apply_filters", "Score all families against the locked pre-registered thresholds.", {},
                    science.apply_filters),
               Tool("promote", "Promote filter-passing families to deep dives.",
                    {"families": "list of family ids", "rationale": "str"}, science.promote)],
    "deep_dive": [Tool("set_verdict", "Record the verdict on a family. new_signal needs 2+ literature searches and a "
                        "novelty_basis; known_phenomenon (a blind rediscovery) is a valid, useful outcome.",
                       {"family": "str", "outcome": "new_signal | known_phenomenon | artifact | not_robust",
                        "note": "str", "novelty_basis": "str (required for new_signal)"}, science.set_verdict)],
}
STAGE["followup"] = STAGE["deep_dive"]
STAGE["science"] = [Tool("record_finding", "Record one scientific result about a real signal: a clear statement, "
                         "the evidence (numbers, tests, data), confidence 0-1, and what is new versus the literature.",
                         {"family": "str", "statement": "str", "evidence": "str", "confidence": "float 0-1",
                          "novelty": "str (optional)"}, climate.record_finding)]


_BOARD = Tool("hypothesis_board", "The research programme: all hypotheses, their status and results.", {},
              science.hypothesis_board)
STAGE["hypothesis"] = [
    Tool("record_hypothesis_result", "Close your hypothesis test: supported | supported_new (supported and not in the "
         "literature: needs 2+ literature searches and a novelty_basis) | not_shown (test ran, not significant) | "
         "refuted (ONLY when a significant result contradicts the claim) | known | inconclusive (could not test). A claimed "
         "success is checked by the harness on held-out data before it counts. Mechanism words need mediation_test "
         "(synlib) in your code, or the result is labelled an association. Not significant = not_shown, never refuted.",
         {"hypothesis_id": "str", "status": "str", "result": "str (one or two precise sentences)",
          "evidence": "str (numbers, tests, data, periods)", "novelty_basis": "str (for supported_new)",
          "confidence": "float 0-1"}, science.record_hypothesis_result),
    STAGE["science"][0], _BOARD]
STAGE["referee"] = [
    Tool("referee_report", "File your referee decision on a claimed result (hypothesis id or family id): upheld | "
         "downgraded (true but weaker or not new) | not_shown (your recomputation is not significant) | refuted "
         "(your checks significantly contradict it), with your objections and the checks you ran.",
         {"target": "str", "decision": "upheld | downgraded | not_shown | refuted", "objections": "str", "checks_run": "str"},
         science.referee_report), _BOARD]
STAGE["science"] = STAGE["science"] + [_BOARD]


def toolbox(kind: str) -> dict[str, Tool]:
    tools = {t.name: t for t in COMMON}
    for t in STAGE.get(kind, []):
        tools[t.name] = t
    return tools


__all__ = ["Tool", "ToolContext", "ToolResult", "toolbox"]
