# Figure size and declutter spec (round 11, legibility at 900 and 600 px wide)

All figures use a 2056 px canvas at 200 dpi (pt = px × 72 / 200). Each text role must stay readable at 900 px wide (×0.438) and stay near 8 px at 600 px wide.

| Role | Canvas px | pt @200dpi |
|---|---|---|
| Title | 40 | 14.5 |
| Panel title / group header | 34 | 12 |
| Subtitle (ONE line max) | 30 | 11 |
| Axis label | 30 | 11 |
| Tick label | 28 | 10 |
| Legend | 28 | 10 |
| Value label (bold) | 28 | 10 |
| Diagram box title / sublabel | 30 / 26 | 11 / 9.5 |
| Annotation, callout, edge label (floor) | 26 | 9.5 |

| Element | Canvas px |
|---|---|
| Data line | 5 |
| Whisker / CI | 3.5, caps ≥ 24 wide |
| Reference / threshold dashed line | 3, dark grey #4A5568 |
| Bar / box outline | 3 (FT boxes in diagrams 4.5 = the non-color cue) |
| Diagram arrows | 3, arrowhead ≥ 18; dashed arrows mid-grey, not light |
| Gridlines | 1.5, light |
| Leader lines | 2 |
| Primary markers | 32 diameter |
| Per-run dots | 14 with a 2 px white edge, offset so they don't sit on bar edges |
| Shaded bands | ≥ 12% darker than the background |

## Declutter rule
Everything a chart needs to be read stays on it: title, axes, legend, values and n. Definitions, caveats, method notes, job IDs, CLAIMS refs and code identifiers move to the doc caption.

## Per figure
- **fig3 placement:**
  - e16–e23 tags at 26 px, or drop them (the caption names them);
  - rank ticks only on the bottom panel;
  - the "2 copies" dot at 14 px.
- **fig6 forest:**
  - "worse with FT →" once, by the x axis;
  - subtitle one line;
  - whiskers 3.5 px;
  - markers 32 px.
- **Recovery sequence:**
  - horizontal edge labels at 26 px;
  - dashed ZeroMQ arrows 3 px mid-grey;
  - more height if needed.
- **figA1:**
  - drop the "upper CI ≥ 1%" tags (the caption covers it);
  - "1% threshold" at 26 px;
  - each-pair dots 14 px;
  - threshold line 3 px;
  - subtitle one line.
- **figA4:**
  - subtitle one line (no "CLAIMS 211"; definitions to the caption);
  - FT dot 32 px;
  - diamonds 24 px;
  - range line 3.5 px.
- **fig7b:**
  - callouts 26 px;
  - leader lines 2 px;
  - "parts may not sum" to the caption.
- **fig7:**
  - drop the per-row "n = x" (keep n in the group headers);
  - drop "hang_confirm_s";
  - "excl. 30 s confirmation" at 26 px;
  - whiskers 3.5 px with caps;
  - markers 32 px;
  - open/filled meaning to the caption.
- **fig2 ghost slots:**
  - drop the "lost" tag (the red border + legend carry it);
  - "from rank 2" and "scratch" at 26 px;
  - zoom lines 3 px.
- **fig10:**
  - each-run dots 14 px with a white edge, offset off the bar top;
  - subtitle one line.
- **figA3:**
  - glosses to the caption, or 26 px;
  - subtitle one line;
  - if a whisker is narrower than the bar end, drop it and state the max range in the caption;
  - reference lines 3 px.
- **fig4 late writes:**
  - drop the "FT" tags and keep the thicker green border (legend explains);
  - small labels at 26 px;
  - dashed outlines 3 px.
- **fig9:**
  - "kill" at 26 px;
  - subtitle one line (drop the duplicate "10 s bins");
  - darker pre-kill band (≥ 12% darker), no dotted extensions;
  - offered-load line 3 px dark grey;
  - data line 5 px.
- **figA5:**
  - callouts 26 px;
  - subtitle one line (no job ID);
  - drop invisible whiskers and state the range in the caption.
- **fig1 architecture:**
  - drop the "FT" tags and keep the thicker green border;
  - edge labels 26 px;
  - arrows 3 px.
- **fig8:**
  - drop the in-bar mode markers (row labels carry the mode; the outline greens stay);
  - n folded into the row label, e.g. "No EPLB (n = 4)";
  - per-run dots 14 px, jittered vertically;
  - subtitle one line;
  - labels 26 px+.
- **fig5:**
  - per-run dots 14 px with a white edge;
  - pair lines 2 px.
- **All charts:** ticks, legend and values at 28 px; subtitles at 30 px, one line.
