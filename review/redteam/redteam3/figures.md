# Red-team lane: figures (visual design / data-viz)

Scope: 15 PNGs in `.../claude/mvp/figures/draft/`. DESIGN_DOC.md uses only `fig_throughput.png` (Fig 5) and
`fig_recovery.png` (Fig 6); the other 13 are older drafts. Style references checked: tech_blog10 pareto
(`tech_blog10_tps_ttft_pareto_curve.png`: TPS/GPU vs TPS/User, matplotlib tab10, a distinct marker per series,
boxed legend, bold titles) and tech_blog4 heatmap.

## Ranked fixes (most important first)

1. **Mode labels disagree between draft versions; settle which data belongs to which mode before finals.**
   - `fig5_cost.png` labels the step-cost bars +1.9/+1.1/+1.0/+2.2% as **No EPLB**. `fig5_healthy_cost.png`,
     `_alt` and `fig5_step_cost.png` show the same numbers as **Offline EPLB**.
   - `fig7_recovery*.png` shows 1.00/2.08/6.63 s as **FT on, offline EPLB**. `fig6_recovery.png` shows the same
     numbers as **FT on, no EPLB**, with offline at 1.30/1.80/6.00 s (mock).
   - Either the older generator mapped the modes wrongly or the newer one does. Audit the mode-to-run mapping in
     the generator, then delete or archive the stale drafts so nobody copies a wrong one.
2. **Fig 5 (`fig_throughput.png`): the caption and footer say "bars: min-max", but no error bars are drawn.** The
   points are n=1-2 and the whiskers are missing or hidden under the markers. Either draw them (capsize, a darker
   tone) or drop the claim. With n=1 say "single run" and don't imply a spread.
3. **Fig 5 text does not match the figure.** The doc says "overlap at every load in both measured modes ... within
   ±2%". The figure has three modes. `table_healthy` shows offline −8.3% at 8 and −6.9% at 512, and No EPLB TPOT
   +5.5% at 128. Also, offline 512 shows −6.9% throughput but −7.1% TPOT (FT faster per step but lower throughput);
   flag that or explain it, because a reader will spot it. Either fix the text or label the offline points as
   outliers.
4. **Fig 6(a) (`fig_recovery.png`): log-scale bars are misleading.** On a log axis, a bar's length depends on where
   the axis happens to start (~0.3 s here), so the bar ratios carry no meaning. Use a dot or lollipop at the value,
   or a broken linear axis, and annotate "~1000x faster" directly.
   - Provenance gap: the FT-on bars are labelled "final code, n=1", while the grey 477/539 s bars appear unchanged
     in every draft (probably an older run). Give the FT-off source and n in the caption.
   - Drop the 10^4 decade (the `fig7_recovery_time` x-limit) and show tick labels as 1 s / 10 s / 100 s / 1000 s,
     not 10^0.
5. **Colour semantics are inconsistent across figures.**
   - Current figures use tab10 colours per mode (blue = no EPLB, orange = offline, green = online), with FT off in
     grey.
   - The 15:3x drafts use NVIDIA green for FT on and a dark/light green split for online/offline.
   - Pick one scheme: FT off = grey, and each mode keeps one hue in every figure, FT on and off alike. Fig 6(b) uses
     a green ramp for phases, which collides with "green = online EPLB". Use a neutral sequential ramp (blues or
     greys) for phases, or tie the ramp to the online hue on purpose and say so.
6. **Colour-blind safety.**
   - Blue/orange/grey is fine. Green against orange (tab10 C1/C2) is weak under deuteranopia.
   - The dark-green/light-green online/offline pair in the `_alt`/`step_cost`/`fig7` drafts is hard to tell apart
     under some vision types and in greyscale print.
   - Add redundant markers as blog10 does (circle = no EPLB, square = offline, triangle = online; hollow for FT off
     or a dashed FT-off line).
   - Fig 6(b)'s five phases sit on a light-green ramp with no in-bar labels. "install" and "first step" are close in
     lightness, so label the segments directly.
7. **`table_healthy.png` is broken as a figure; render it as a Markdown table instead.**
   - The last header ("n off/on" + "range tok/s/GPU off / on") overlaps and spills past the table border, and the
     range column is misaligned.
   - The n=1 rows show degenerate ranges ("1,051-1,051").
   - The "Throughput" and "TPOT" delta columns don't say their sign convention: + is good for throughput and bad
     for TPOT. Use "Δ tok/s/GPU (FT on vs off)" and "Δ TPOT".
   - Remove the ~150 px of empty space at the top.
   - In Markdown the table stays accessible, searchable and diffable.
8. **Fig 5 layout nits.**
   - The concurrency labels overlap in the No EPLB panel near x≈50 and at "8" (a half-hidden glyph shows under the
     marker). Offset the labels or label only one series.
   - Three identical legends: use one shared legend, or direct line-end labels.
   - "No EPLB" vs the doc's "w/o EPLB": settle on one spelling across doc, figures and table (the doc and figures
     currently use "no EPLB"/"No EPLB"; the prompt's convention is "w/o EPLB").
   - Use tick separators (8,000), or "k".
   - Units: use "tok/s/GPU" and "tok/s/user" to match blog10 ("TPS/GPU vs TPS/User") and the table.
9. **PRELIMINARY stamps.**
   - The stamps are good and consistent (red, top right) but sit in a corner and are easy to crop off.
   - Fig 5's stamp text ("rc1, n=1-2; No EPLB FT-off used different GEMM tactics") is a confound note. It belongs in
     the caption, and the No EPLB panel should carry a visible "not comparable" marker (a hatched or hollow FT-off
     series).
   - The memory rule "design doc: final data only" means these drafts must not reach the doc unstamped. Today they
     are linked from the doc body, so make sure the final swap replaces the files rather than editing stamps out.
10. **Older drafts (archive; listed only in case any is revived).**
   - `fig5_cost`: a bar chart of % deltas from 0. Fine, but its "Offline EPLB (mock)" bars carry error bars, i.e.
     mock data drawn with fake uncertainty. Don't draw whiskers on mock data.
   - `fig5_healthy_cost`: absolute bars hide 0.3-2% deltas (an honest scale but useless for the claim). The panels
     have different y-ranges (80 vs 50; 8000 vs 6000), which invites wrong cross-panel comparisons.
   - `fig5_healthy_cost_alt` / `fig6_throughput(a)`: the online 8 point's whisker runs off the axis (±15%) with no
     clipping indicator. The y-range is asymmetric (−3..4 vs −12..16) between neighbouring panels. "FT cost, above
     0 = FT lower throughput" inverts the usual sign; readers misread it.
   - `fig6_throughput(b)` / `fig7_recovery(c)`: mock "after a loss" bars use a y-axis truncated at 60% (or 0.6),
     which exaggerates a 2-3 pt gap. Start at 0, or plot the gap. Percent in one draft and fraction in the other.
   - `fig6_throughput_load`: a linear x-axis for 8/32/128/512 (equal spacing, so effectively log) with no "log"
     note. Its y-axis doesn't start at 0, which makes the FT-on/off 512 gap look larger.
   - `fig5_placement_v2`: a clear schematic. Fixes:
     - Legend: "lost GPU" (red) shows on slots, so call it "slots on the lost GPU".
     - The hatched-ghost vs light-blue contrast is low.
     - The top-row "e16..e23" labels are tiny (about 7 pt at doc width).
     - The bottom panel has no expert labels.
     - Add a title or a one-line caption saying what the green slots mean per row ("restored expert, evenly
       spread").
   - `fig7_recovery_time`: in-bar labels ("agree" clipped against the edge, "graphs") are cramped. The 1 rank row's
     "agree" overlaps the detect segment.

## Cross-figure checklist for the final generator

- One palette and marker map for modes, plus a fixed FT-off style (grey, hollow or dashed). Matplotlib tab10 order
  matches blog10.
- Use one font size of at least 9 pt at the doc's rendered width: the 2000-2500 px PNGs get downscaled about 2-3x
  in GitHub/Sphinx, and the point labels and footers (~11 px) become unreadable. Render at the target width with
  dpi 200, or ship the SVG.
- Every quantitative caption states n, the statistic (median/mean; the drafts disagree, e.g. "median" in Fig 5
  vs "mean" in `_alt`), the spread definition, the build/run IDs, and ISL/OSL (1k/1k), batch and MTP.
- Units: tok/s/GPU, tok/s/user, ms for TPOT, s for recovery. "8-9 minutes" in the text matches 477/539 s, but don't
  mix minutes and seconds across panels (fig6/7(c) is in minutes while (a) is in seconds).
- Start delta plots at a symmetric range around 0 with the ±1% target band. Start absolute plots at 0 or state the
  truncation.
- Mock panels: a hatched fill plus a "MOCK" label inside the panel, not only in the corner banner, and no error bars.
