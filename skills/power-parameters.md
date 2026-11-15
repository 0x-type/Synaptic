# NASA POWER monthly parameters (verified names and units)

Meteorology (MERRA-2, 1981-present): T2M, T2M_MAX, T2M_MIN, T2M_RANGE, T2MDEW, T2MWET, TS (skin
temperature) [C]; RH2M [%]; QV2M [g/kg]; PRECTOTCORR [mm/day]; PS [kPa]; WS2M, WS10M, WS50M [m/s];
WD2M, WD10M [degrees]; GWETTOP, GWETROOT, GWETPROF (surface / root-zone / profile soil wetness, 0-1);
SNODP [cm]; EVPTRNS (evapotranspiration energy flux) [MJ/m2/day]; FROST_DAYS [days].

Solar & cloud (SRB to 2000, CERES SYN1deg from 2001; start 1984): ALLSKY_SFC_SW_DWN,
CLRSKY_SFC_SW_DWN, ALLSKY_SFC_LW_DWN [MJ/m2/day]; ALLSKY_KT (clearness index); CLOUD_AMT [%].

Grid: about 0.5 deg latitude x 0.625 deg longitude. Regional requests: one parameter and at most
10x10 degrees per call (the power_region tool tiles and loops for you).

Physically linked pairs for corroboration: rain <-> soil wetness <-> humidity; T2M <-> T2M_MAX/MIN,
soil wetness (drying amplifies daytime heat); shortwave <-> cloud amount.
