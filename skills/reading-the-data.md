# Reading the data directly (the single most important habit)

In the reference study (Yoon et al., Anthropic 2026), the new reverse-transcriptase family was found
because an agent read raw DNA into its context and *saw* a tandem repeat that no tool had been asked
to find. In their benchmark, models that only ran tools without reading the sequence missed the
feature most of the time; recognition rose steeply with the amount of raw sequence actually read.

The Earth-data equivalent:
1. `look(path='<data>.csv', lat=..., lon=...)` prints the annual series AND the full year x month
   table for one cell. Read it. Scan columns (months) and rows (years) for:
   - a run of identical values (gap filling), sudden NA blocks, values at physical limits;
   - one month whose behaviour changes while others do not (seasonal shift, e.g. rains arriving later);
   - an abrupt jump that is the same in every month (often a data-production change, not physics);
   - a year that is extreme in several months (a real event: check it against known events).
2. Make the figure with `synlib.plot_series(..., months=monthly_array)`; the monthly-anomaly strip
   under the series shows seasonal structure at a glance. Then `look` at the PNG.
3. When you look, write down *what you see* first ("from 1998 the July-September cells turn
   persistently red while winter does not change"), then interpret.
4. Compare the extreme cell with an ordinary neighbour. A signal confined to one cell on the edge of
   the box is suspicious; a signal that fades smoothly with distance is more physical.
5. For spatial signals, fetch a GIBS image (e.g. MODIS NDVI or true colour) for a date before and after
   the change and look at both. Land-cover change, reservoirs, irrigation and fires are visible.
