"""The NASA dataset catalogue used by the Trend Detective app and by campaigns.

Every entry is a MONTHLY Level-3 or Level-4 product (the challenge video: "it is easier to determine trends over
time for data points that are weekly or monthly") reachable with one Earthdata token through NASA's cloud services:
  opendap   OPeNDAP in Earthdata Cloud: only the requested variable and box are transferred
  harmony   NASA Harmony variable subsetting (MODIS Atmosphere monthly files are 374 MB each; the subset is 0.3 MB)
  file      one file holds the whole record (CERES EBAF, GRACE/GRACE-FO mascons): downloaded once, cached
  download  one file per month, downloaded and cropped (slow the first time)
"""
from __future__ import annotations

from dataclasses import dataclass, field

K2C = (1.0, -273.15)
PER_S_TO_DAY = (86400.0, 0.0)


@dataclass(frozen=True)
class Var:
    id: str                    # catalogue id used by the app and the harness, e.g. "MERRA2_T2M"
    label: str
    units: str
    collection: str            # CMR concept id
    variable: str              # variable name (path) inside the files
    backend: str               # opendap | harmony | file | download
    source: str                # human-readable product name
    start: int                 # first year with data
    domain: str = "global"     # global | land | ocean
    dims: str = "tyx"          # dimension order of the variable: t=time, y=lat, x=lon
    lat: str = "lat"
    lon: str = "lon"
    convert: tuple = (1.0, 0.0)  # value * a + b gives `units`
    group: str = ""            # theme in the app menu
    doi: str = ""
    note: str = ""
    file_url: str = ""         # for backend 'file'
    tags: tuple = field(default_factory=tuple)
    end: str = ""              # last available month "YYYY-MM" when the record stops before the present
    granule: str = ""          # CMR granule-name pattern when a collection mixes periods and resolutions
    grid_spec: tuple = ()      # regular grid without 1-D coordinates in the file: (lat0, dlat, nlat, lon0, dlon, nlon)
    level: int = -1            # index along the vertical dimension ("z" in dims), e.g. a pressure level


M2 = "MERRA-2 reanalysis (GMAO), 0.5 x 0.625 deg"
M2_DOI = "10.5067/AP1B0BA5PD2K"

CATALOG: list[Var] = [
    # --- temperature and air
    Var("MERRA2_T2M", "Air temperature at 2 m", "degC", "C1276812859-GES_DISC", "T2M", "opendap", M2, 1980,
        convert=K2C, group="Temperature", doi="10.5067/AP1B0BA5PD2K"),
    Var("MERRA2_TS", "Surface skin temperature", "degC", "C1276812859-GES_DISC", "TS", "opendap", M2, 1980,
        convert=K2C, group="Temperature"),
    Var("MODIS_SST", "Sea surface temperature (MODIS Aqua, day)", "degC", "C2036877944-POCLOUD", "sst", "opendap",
        "MODIS Aqua L3 SST thermal IR monthly 9 km (OBPG / PO.DAAC)", 2002, domain="ocean", dims="yx",
        group="Ocean", doi="10.5067/MODSA-MO9D9"),
    Var("MODIS_CHL", "Ocean chlorophyll-a (MODIS Aqua)", "mg/m3", "C3380709133-OB_CLOUD", "chlor_a", "monthfile",
        "MODIS Aqua L3 mapped chlorophyll-a monthly 9 km, R2022 (NASA OB.DAAC), averaged here to 0.25 deg", 2002,
        domain="ocean", dims="yx", group="Ocean", doi="10.5067/AQUA/MODIS/L3M/CHL/2022.0",
        granule="*.L3m.MO.CHL.chlor_a.9km*",
        note="The amount of microscopic plant life (phytoplankton) at the sea surface. Each month is one 13 MB file, "
             "downloaded once and kept; the first run of a long period takes a few minutes. Aqua's orbit has "
             "drifted since 2022: recent trends need care"),
    Var("MERRA2_SST", "Sea surface (skin) temperature, reanalysis", "degC", "C1276812857-GES_DISC", "TSKINWTR",
        "opendap", M2, 1980, domain="ocean", convert=K2C, group="Ocean"),
    # --- water
    Var("MERRA2_PRECIP", "Rainfall (bias-corrected)", "mm/day", "C1276812868-GES_DISC", "PRECTOTCORR", "opendap", M2,
        1980, convert=PER_S_TO_DAY, group="Water",
        note="Over land the model rain is replaced by gauge/satellite products (CPCU, CMAP over Africa)"),
    Var("IMERG_PRECIP", "Rainfall (GPM IMERG satellite)", "mm/day", "C2723754851-GES_DISC", "/Grid/precipitation",
        "opendap", "GPM IMERG Final V07 monthly 0.1 deg", 2000, dims="txy", lat="/Grid/lat", lon="/Grid/lon",
        convert=(24.0, 0.0), group="Water", doi="10.5067/GPM/IMERG/3B-MONTH/07", end="2025-09"),
    Var("MERRA2_TQV", "Water vapour in the air column", "kg/m2", "C1276812859-GES_DISC", "TQV", "opendap", M2, 1980,
        group="Water"),
    Var("MERRA2_SOILM", "Root-zone soil wetness", "0-1", "C1276812856-GES_DISC", "GWETROOT", "opendap", M2, 1980,
        domain="land", group="Water"),
    Var("GLDAS_SOILM", "Soil moisture 0-10 cm (GLDAS Noah)", "kg/m2", "C1342986036-GES_DISC", "SoilMoi0_10cm_inst",
        "opendap", "GLDAS Noah 2.1 monthly 0.25 deg", 2000, domain="land", group="Water",
        doi="10.5067/SXAVCZFAQLNO"),
    Var("GLDAS_EVAP", "Evapotranspiration (GLDAS Noah)", "mm/day", "C1342986036-GES_DISC", "Evap_tavg", "opendap",
        "GLDAS Noah 2.1 monthly 0.25 deg", 2000, domain="land", convert=PER_S_TO_DAY, group="Water"),
    Var("MERRA2_RUNOFF", "Surface runoff", "mm/day", "C1276812856-GES_DISC", "RUNOFF", "opendap", M2, 1980,
        domain="land", convert=PER_S_TO_DAY, group="Water"),
    Var("GRACE_TWS", "Total water storage incl. groundwater (GRACE/GRACE-FO)", "cm", "C3195527175-POCLOUD",
        "lwe_thickness", "file", "JPL GRACE/GRACE-FO mascon RL06.3 v04", 2002, domain="land", group="Water",
        doi="10.5067/TEMSC-3JC634", note="Monthly with gaps (2017-2018 mission gap)", end="2026-08"),
    Var("GRACE_GSFC_TWS", "Total water storage, second GRACE solution (NASA GSFC mascons)", "cm", "GSFC-mascons",
        "lwe_thickness", "file", "NASA GSFC GRACE/GRACE-FO mascons RL06v2.0, 0.5 deg", 2002, domain="land",
        group="Water",
        note="Processed independently of the JPL mascons (Loomis et al. 2019): a pass on both is a replication",
        file_url="https://earth.gsfc.nasa.gov/sites/default/files/geo/"
                 "GSFC.glb_.200204_202606_RL06v2.0_OBP-ICE6GD_HALFDEGREE.nc", end="2026-06"),
    # --- snow
    Var("MERRA2_SNOW", "Snow water (snow mass)", "kg/m2", "C1276812856-GES_DISC", "SNOMAS", "opendap", M2, 1980,
        domain="land", group="Snow"),
    Var("MERRA2_SNOWDEPTH", "Snow depth", "m", "C1276812856-GES_DISC", "SNODP", "opendap", M2, 1980, domain="land",
        group="Snow"),
    Var("GLDAS_SNOWDEPTH", "Snow depth (GLDAS Noah)", "m", "C1342986036-GES_DISC", "SnowDepth_inst", "opendap",
        "GLDAS Noah 2.1 monthly 0.25 deg", 2000, domain="land", group="Snow"),
    # --- weather
    Var("MERRA2_SLP", "Sea-level pressure", "hPa", "C1276812859-GES_DISC", "SLP", "opendap", M2, 1980,
        convert=(0.01, 0.0), group="Weather"),
    Var("MERRA2_WIND", "Surface wind speed", "m/s", "C1276812868-GES_DISC", "SPEED", "opendap", M2, 1980,
        group="Weather"),
    # --- clouds, sunlight, radiation
    Var("MERRA2_CLDTOT", "Cloud cover, total (reanalysis)", "%", "C1276812858-GES_DISC", "CLDTOT", "opendap", M2,
        1980, convert=(100.0, 0.0), group="Clouds & radiation"),
    Var("MERRA2_CLDLOW", "Low cloud cover (reanalysis)", "%", "C1276812858-GES_DISC", "CLDLOW", "opendap", M2, 1980,
        convert=(100.0, 0.0), group="Clouds & radiation"),
    Var("MERRA2_SWGDN", "Sunlight reaching the surface", "W/m2", "C1276812858-GES_DISC", "SWGDN", "opendap", M2,
        1980, group="Clouds & radiation"),
    Var("MERRA2_ALBEDO", "Surface albedo", "0-1", "C1276812858-GES_DISC", "ALBEDO", "opendap", M2, 1980,
        group="Clouds & radiation"),
    Var("MODIS_CLOUD", "Cloud fraction (MODIS Aqua)", "%", "C1443775657-LAADS", "Cloud_Fraction_Mean_Mean",
        "harmony", "MODIS Aqua MYD08_M3 C6.1, 1 deg", 2002, dims="yx", convert=(100.0, 0.0),
        group="Clouds & radiation", doi="10.5067/MODIS/MYD08_M3.061",
        note="Aqua's orbit has drifted since 2022 (later overpass time): cloud trends after 2022 need care"),
    Var("MODIS_CLOUD_REFF", "Liquid cloud droplet radius (MODIS Aqua)", "um", "C1443775657-LAADS",
        "Cloud_Effective_Radius_Liquid_Mean_Mean", "harmony", "MODIS Aqua MYD08_M3 C6.1, 1 deg", 2002, dims="yx",
        group="Clouds & radiation", note="Larger droplets = fewer aerosol particles to condense on"),
    Var("CERES_SW_UP", "Sunlight reflected to space, all-sky (CERES)", "W/m2", "C3880497643-LARC_CLOUD",
        "toa_sw_all_mon", "file", "CERES EBAF-TOA Ed4.2.1, 1 deg", 2000, group="Clouds & radiation",
        doi="10.5067/TERRA-AQUA-NOAA20/CERES/EBAF-TOA_L3B004.2.1",
        file_url="https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/"
                 "CERES_EBAF-TOA_Edition4.2.1_200003-202608.nc"),
    Var("CERES_SW_CLR", "Sunlight reflected to space, clear-sky (CERES)", "W/m2", "C3880497643-LARC_CLOUD",
        "toa_sw_clr_c_mon", "file", "CERES EBAF-TOA Ed4.2.1, 1 deg", 2000, group="Clouds & radiation",
        file_url="https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/"
                 "CERES_EBAF-TOA_Edition4.2.1_200003-202608.nc"),
    Var("CERES_NET", "Net energy into the Earth at the top of the atmosphere (CERES)", "W/m2",
        "C3880497643-LARC_CLOUD", "toa_net_all_mon", "file", "CERES EBAF-TOA Ed4.2.1, 1 deg", 2000,
        group="Clouds & radiation",
        file_url="https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/"
                 "CERES_EBAF-TOA_Edition4.2.1_200003-202608.nc"),
    Var("CERES_CLOUD", "Cloud area fraction (CERES)", "%", "C3880497643-LARC_CLOUD", "cldarea_total_daynight_mon",
        "file", "CERES EBAF-TOA Ed4.2.1, 1 deg", 2000, group="Clouds & radiation",
        file_url="https://data.asdc.earthdata.nasa.gov/asdc-prod-protected/CERES/CERES_EBAF-TOA_Edition4.2.1/"
                 "CERES_EBAF-TOA_Edition4.2.1_200003-202608.nc"),
    # --- atmosphere by level (AIRS on Aqua) and air composition (OMI on Aura): instruments the challenge names
    *[Var(f"AIRS_{vid}", label, "degC", "C1701805662-GES_DISC", "/ascending/Data_Fields/Temperature_A", "opendap",
          "AIRS/Aqua L3 monthly standard retrieval v7, 1 deg (daytime pass)", 2002, dims="zyx", convert=K2C,
          group="Atmosphere by level (AIRS)", doi="10.5067/UBENJB9D3T2H", grid_spec=(89.5, -1.0, 180, -179.5, 1.0, 360),
          level=lev, note=note)
      for vid, label, lev, note in (
          ("T850", "Air temperature at 850 hPa, about 1.5 km up (AIRS)", 2, "Lower troposphere. Missing over high ground."),
          ("T500", "Air temperature at 500 hPa, about 5.5 km up (AIRS)", 5, "Middle troposphere"),
          ("T200", "Air temperature at 200 hPa, about 12 km up (AIRS)", 9, "Near the top of the troposphere"),
          ("T50", "Air temperature at 50 hPa, about 20 km up (AIRS)", 13,
           "Stratosphere: greenhouse gases warm the lower atmosphere and COOL this layer"))],
    Var("AIRS_TSURF_AIR", "Surface air temperature (AIRS satellite)", "degC", "C1701805662-GES_DISC",
        "/ascending/Data_Fields/SurfAirTemp_A", "opendap", "AIRS/Aqua L3 monthly standard retrieval v7, 1 deg (daytime pass)",
        2002, dims="yx", convert=K2C, group="Atmosphere by level (AIRS)", doi="10.5067/UBENJB9D3T2H",
        grid_spec=(89.5, -1.0, 180, -179.5, 1.0, 360)),
    Var("AIRS_TQV", "Water vapour in the air column (AIRS satellite)", "kg/m2", "C1701805662-GES_DISC",
        "/ascending/Data_Fields/TotH2OVap_A", "opendap", "AIRS/Aqua L3 monthly standard retrieval v7, 1 deg (daytime pass)",
        2002, dims="yx", group="Atmosphere by level (AIRS)", doi="10.5067/UBENJB9D3T2H",
        grid_spec=(89.5, -1.0, 180, -179.5, 1.0, 360)),
    Var("OMI_NO2", "Nitrogen dioxide in the lower atmosphere (OMI)", "10^15 molecules/cm2", "C3333493715-GES_DISC",
        "/HDFEOS/GRIDS/ColumnAmountNO2/Data_Fields/ColumnAmountNO2TropCloudScreened", "dailymean",
        "OMI/Aura NO2 tropospheric column, cloud-screened, daily L3 0.25 deg v4 (monthly means computed here)", 2004,
        dims="yx", convert=(1e-15, 0.0), group="Aerosols & air quality", doi="10.5067/AURA/OMI/DATA3407",
        grid_spec=(-89.875, 0.25, 720, -179.875, 0.25, 1440),
        note="A gas from traffic, power plants and industry. NASA publishes it daily; the app averages each month "
             "(about 30 small requests per month, so a long period takes several minutes the first time). "
             "Part of the instrument's view has been blocked since 2007 (row anomaly): fewer measurements per day"),
    Var("OMI_O3", "Ozone layer thickness, total column (OMI)", "Dobson units", "C3377057279-GES_DISC",
        "/HDFEOS/GRIDS/OMI_Column_Amount_O3/Data_Fields/ColumnAmountO3", "dailymean",
        "OMI/Aura TOMS-like total ozone, daily L3 0.25 deg v4 (monthly means computed here)", 2004, dims="yx",
        group="Aerosols & air quality", doi="10.5067/Aura/OMI/DATA3402",
        grid_spec=(-89.875, 0.25, 720, -179.875, 0.25, 1440),
        note="The ozone layer that shields the surface from ultraviolet light. Daily files averaged per month here"),
    # --- aerosols
    Var("MODIS_AOD", "Aerosol optical depth at 550 nm (MODIS Aqua)", "", "C1443775657-LAADS",
        "AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean", "harmony", "MODIS Aqua MYD08_M3 C6.1, 1 deg", 2002,
        dims="yx", group="Aerosols & air quality", doi="10.5067/MODIS/MYD08_M3.061",
        note="The challenge video's example: smoke over Wisconsin"),
    Var("MERRA2_AOD", "Aerosol optical depth, total (reanalysis)", "", "C1276812866-GES_DISC", "TOTEXTTAU",
        "opendap", M2, 1980, group="Aerosols & air quality"),
    Var("MERRA2_AOD_DUST", "Dust aerosol optical depth", "", "C1276812866-GES_DISC", "DUEXTTAU", "opendap", M2, 1980,
        group="Aerosols & air quality"),
    Var("MERRA2_AOD_SULFATE", "Sulfate aerosol optical depth", "", "C1276812866-GES_DISC", "SUEXTTAU", "opendap", M2,
        1980, group="Aerosols & air quality",
        note="Shipping fuel rules cut sulfur emissions at sea in 2020 (IMO 2020)"),
]

# last available month of records that stop before the present (checked 9 Nov 2026); confirmation specs that need
# later months are refused before they run (confirm.validate)
_ENDS = {"MERRA2_": "2026-09", "GLDAS_": "2026-07", "CERES_": "2026-08"}
from dataclasses import replace as _replace  # noqa: E402

CATALOG = [(_replace(v, end=next(e for k, e in _ENDS.items() if v.id.startswith(k)))
            if not v.end and any(v.id.startswith(k) for k in _ENDS) else v) for v in CATALOG]
BY_ID = {v.id: v for v in CATALOG}


def get(var_id: str) -> Var:
    if var_id not in BY_ID:
        raise KeyError(f"unknown dataset {var_id}; choose one of {sorted(BY_ID)}")
    return BY_ID[var_id]


def menu() -> list[dict]:
    return [{"id": v.id, "label": v.label, "units": v.units, "group": v.group, "source": v.source, "start": v.start,
             "domain": v.domain, "note": v.note, "doi": v.doi} for v in CATALOG]
