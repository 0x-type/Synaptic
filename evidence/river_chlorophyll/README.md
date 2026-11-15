# River-mouth chlorophyll: an independent re-check

**What this is.** A teammate's Synaptic campaign re-tested the trends of the NASA study "Time-Series Analysis of
Remotely-Sensed SeaWiFS Chlorophyll in River-Influenced Coastal Regions" (NTRS 20090042747, in the challenge's
resource list): MODIS Aqua 2002-2025 for the agents, VIIRS 2012-2025 held out. This folder repeats the MODIS part
with this branch's own code (`scripts/river_recheck.py`), for the same boxes. It is a check, not a new campaign.

**What both versions agree on (MODIS Aqua, 2002-2025).**
- No significant trend off Mississippi West, Mississippi East or the Eel in summer, the three rivers where the
  paper reported one. Not detected is not the same as absent: the 95% ranges still allow changes of several
  percent per decade.
- Off Mississippi East, the 2002-2007 overlap is significant without a correction for carry-over and not
  significant with it.
- Off the Guadalquivir, a significant decrease.

**What only the teammate's campaign tested.** VIIRS as a held-out second sensor (most re-tests too short to be
conclusive; Guadalquivir decrease p = 0.19 on VIIRS alone), links to river discharge, and method controls (the two
sensors agree off the Mississippi and the Pearl).

**Wording.** A non-significant test is "not detected" or "not shown", never "refuted" or "vanished". The paper's
period (1998-2007) and sensor (SeaWiFS) differ from ours, so this does not show the paper was wrong. We found no
published re-test; we do not say "first".
