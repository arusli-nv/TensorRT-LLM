# TensorRT-LLM tech-blog chart style (reference for the data figures)

Charts viewed (docs/source/blogs/media/): tech_blog26_dsv4_performance_evolution_8k1k, tech_blog21_e2e_tept8_osl1k_bar,
tech_blog21_real_data_bars, tech_blog25_scaling_lines, tech_blog28_latency_step_down, tech_blog28_bf16_time_breakdown,
tech_blog27_coder_bc_frontier, tech_blog18_bandwidth, tech_blog16_blackwell_decode, tech_blog19_e2e_pareto_frontier,
tech_blog20_aggr_all_models, tech_blog14_perf, tech_blog8_perf-8k-1k-dep.

The older blogs (8, 14, 19, 20) are Excel / Plotly / default-matplotlib exports with no shared style. The recent
matplotlib blogs (21, 25, 26, 28) converge on one look; that is the target.

## Colour
- NVIDIA green `#76B900` marks "ours" / the latest or best result (blog 26 newest line, blog 25 main line, blog 28 best bar).
- Green families for variants of our own method: dark `#1B4F12`, mid `#458F19`, light `#76B900` (blog 28 bars and pie).
- Baseline / "other" in neutral grey or slate: `#B3B8C0` (blog 28 pie "Others"), dashed grey `#999999` for ideal lines (blog 25).
- Text and axes in dark slate rather than pure black: `#1F2A44` labels, `#8C96A3` spines (blog 21, 28).
- Highlight / callout colour: dark red `#8B0000` only for a derived metric (speedup line, blog 21); otherwise none.

## Fonts and sizes
- matplotlib default DejaVu Sans throughout (blogs 18, 21, 25, 26, 28).
- Title 16-18 pt bold, black; subtitle 11-12 pt regular grey (`#6B7280`) directly under it (blog 26, 25).
- Axis labels 12-13 pt, often bold in the bar charts (blog 21, 25, 28); tick labels 11-12 pt slate.
- Value labels on bars 11-13 pt bold dark slate (blog 21), or regular "value · speedup" (blog 28).

## Background, grid, axes
- White background, no panel fill.
- Top and right spines removed (blog 21, 26, 28); left/bottom spines thin slate.
- Light gridlines on the value axis only, dashed or very light solid (`#E5E7EB`); no grid on the category axis of bar charts.

## Legend
- Horizontal row above the plot area, or upper-right inside the axes (blog 21, 28: above; blog 25, 26: upper right).
- Square patches for bars, line+marker for lines; light or no frame.

## Titles, labels, annotation
- One short title stating what is shown; one grey subtitle with the setup (hardware, model, headline number).
- Values written on or just above the bars (bold); speedups as "1.19x" next to the value.
- Small grey italic annotation for reference lines ("avg 1.88x", "NVLink BW (900 GB/s)") placed at the line's right end.
- Footer optional: blog 26 and 27 use one small grey line at the bottom (setup or caveat); most have none.

## Aspect
- Wide: about 1.6-2.2 : 1 (2146x1200, 2160x1296, 2143x973, 1714x1064); multi-panel figures share y and sit side by side.

## Comparisons
- Baseline vs ours: baseline grey or muted, ours NVIDIA green (blog 21, 25, 28).
- Step-by-step improvements: same-family greens from dark to bright, with "value · speedup" labels (blog 28).
- Ideal / reference as a dashed grey line with a short label (blog 25, 18).
