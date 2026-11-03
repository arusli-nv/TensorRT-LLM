# Figure quality rounds (user, 10-26 ~12:00 PT: at least 10 rounds, deeper each time)

Process for each round:
1. Build a blinded set with tools/blind_round.sh <N> [refs|norefs] [design].
2. A fresh judge agent reads only that set, through the round's lens.
3. Verify the findings and send fixes to the two figure owners (make_figs.py and make_figs_extra.py).
4. View the result.
5. Log one line in the table below.

Rules: data and CSVs stay unchanged unless a round finds a data bug, which is then disclosed. Short titles, with takeaways only in the doc captions. Design figures change only for accuracy or consistency.

| Round | Lens | Refs |
|---|---|---|
| 1-5 | general quality vs TRT-LLM blog charts (done; ours ranked above blog-30 by round 5) | blog |
| 6 | general quality again after the round-5 fixes | blog-30 |
| 7 | cross-figure consistency, data and design figures together: color meaning, fonts, sizes, label case, number formats, legend placement, spacing | design |
| 8 | data honesty: misleading encodings, axis starts, n shown, uncertainty, rounding consistency, label vs value arithmetic | none |
| 9 | accessibility: color-blind simulation (deuteranopia, protanopia), grayscale print, contrast ratios | none |
| 10 | legibility at doc width: images downscaled to 900 px and 600 px | none |
| 11 | comprehension test: the judge answers 3 questions per figure from the image alone; answers checked against the CSVs | none |
| 12 | numeric audit: every printed number vs its CSV and CLAIMS row | none |
| 13 | expert skeptic: an NVIDIA perf engineer reviewing for a design review | blog-30 |
| 14 | captions + figures in the doc: caption/figure agreement, takeaway accuracy, order | doc |
| 15 | final blind comparison vs blog-30 | blog-30 |

## Log
- 12:0x round 6 fixes applied (fig5 deltas from plotted means: 32/Online 0.0%, 128/No −1.2%, 128/Off −0.4%; fig8 2-dec + gaps; shared phase colors; A3/A4/A5/fig9 fixes). Round 6 judge running.
- 11:58 round 6 judged: ours 7-8 (fig5 8, fig6 8, fig7 8, A4 8, A5 8, fig9 8; A1/A3/fig7b/fig8 7; fig10 7.5), blog-30 refs 5-6. Round-7 fixes sent.
- 12:03 round 7 fixes applied (agent caught my wrong trim definition -> '≤1.5x median step'; A1 label '1% threshold'). Round 7 consistency judge (16 data+design figs) running.
- 12:06 round 7 (consistency incl design): set 6/10 (charts alone ~8.5). Fixes sent to 3 agents: shade+shape for EPLB modes, grey reference lines, fixed title margin, unified wording (Lose 1 rank / FT on-off / DEP32 · 32 GB200), 2056 px widths; design figs minimal (bg #F5F5F5, text #2D3142, no blue for actors, hatch = ghost only, term 'All-to-all workspace').
- 12:13 round 8 fixes landed (consistency: shapes+shades, grey refs, wording, 2056 px, header geometry identical across scripts; design figs minimal fixes). Round 8 data-honesty judge running.
- 12:17 round 8 (data honesty) judged: 5-7 per chart. Main issues: derived labels not reproducible from printed values (rule adopted: they must), no spread for small n (whiskers/dots added), undefined stats/windows/CI methods (subtitles), A3 summed abort row (split), restart confound note, A5 selection stated. Cross-chart metric conflicts -> doc captions. Fixes sent.
- 12:23 round 9 (honesty fixes) landed both scripts: derived labels reproduce (check scripts: main 16/16, extra 35/35), whiskers/n everywhere, stats defined in subtitles, A3 abort split + FT-off round trip 28.1/38.4 us, A4 runs + clock definitions, A5 survivor whiskers. fig10 '~0%' rule corrected (opposite-sign pairs only). Title x aligned 61-63 px. Next: round 10 accessibility.
- 12:25 fig10 '~0%' rule fixed by monitor (main-figure agent no longer reachable; future make_figs.py fixes via a new agent). Round 10 accessibility judge (deutan/protan/gray sims of 16 figs) running.
- 12:31 round 10 accessibility judged: charts 7-9; design figs weak in grayscale (Fig1 4, Fig3 4, Fig4 5, Fig2/FigR 6): FT-vs-existing and restored/lost color-only. Fixes sent: design (thicker FT outline + 'FT' badge, darker restored fill, lost hatch, dark-green labels, failed-rank dashed), charts (dark strokes on lime markers, darker FT-on outlines, fig7b detect fill, fig9 line).
- 12:35 round 10 fixes landed: design (FT tabs + thicker outlines, Fig3 restored/lost/2-copies cues, dark-green labels, failed-rank dashed), charts (rims on lime markers, #458F19 FT-on outlines, fig7b detect fill, fig9/A4 green, A3 sublabel contrast). Side effect fig8 Offline/Online look alike -> shape markers fix running. Next: round 11 legibility.
- 12:41 round 11 legibility judged: 900px 6-8, 600px 4-6 (annotations ~8 px, tags ~6 px, whiskers/dots vanish). SIZE_SPEC.md written (sizes, one-line subtitles, declutter to captions, FT tags dropped for thicker borders). Fixes dispatched to 3 agents; removed text will go to doc captions.
- 12:48 round 11 fixes landed (all 3 scripts); header geometry re-aligned by monitor (extra -> main values; subtitle row at y=100 in both); CAPTION_NOTES.md complete. Next: round 12 comprehension test.
- 12:51 round 12 comprehension test: ~40 answers, all values checked correct (e.g. 0.85 s, 10.1 min, 7.31 s, -1.1%, 85.6%, 2.13 us, 992 ms); judge also caught my wrong premise (import+map does NOT dominate No-EPLB kill 1: wait 475 > 348). Hard spots: Fig4 mask meaning, Fig9 offered-load value, Fig6/A1 open-vs-filled, A4 bar-vs-diamond, Fig2 scratch. Fixes dispatched.
- 13:01 round 12 fixes landed; monitor moved fig9 'offered load' labels below the trace (agent's boxed label hid ~2 min of data). Round 13 numeric audit next.
- 13:16 round 13 numeric audit: all plotted values match CSV + CLAIMS; design facts match code. 10 mismatches in labels/doc: Fig7b mean vs 'median' caption, A3 median not mean, Fig7 frozen labels subtract 30 s not measured fence, Fig8 gap range 0.6-2.2, Fig5 change stat differs from table, doc bullets with numbers not printed. Fixes: figure agent (fig7 frozen, A3 median) + round-14 caption agent (doc).
- 13:26 round 14 (doc captions vs figures) applied: all audit mismatches fixed in text, CAPTION_NOTES folded in, takeaways quote only printed numbers, figure refs verified. Round 15 expert-skeptic next.
- 13:32 round 15 expert skeptic: 10 must-address (stress 'after/before' is served-vs-offered not capacity; user impact after loss buried; Fig6 caption vs TTFT effect; step cost vs flat client metrics; Fig10 selection; Fig8 proportional not upper bound; restart baseline is lower bound; no-EPLB remap regression since host-copy fix; coverage limits; what numbers don't cover). Doc agent + fig7 marker fix running.
- 13:43 round 15 doc fixes applied (10 items; Fig 10 now states no selection rule was set; capacity-after-kill column; TTFT +24% after a kill; coverage limits). Fig 7 hollow frozen markers removed, caption updated. Round 16 final blind judge vs blog-30 running.
- 13:58 round 16 (blind vs blog-30): ours 6.8 vs blog 5.9; 7 chart fixes landed (A4 legend, A1 clip, fig10 dots, fig6 direction in headers, fig7 labels+whiskers, A3 axis, fig7b whiskers; checks 20/20 + ALL PASS). Design-encoding audit (user's Fig 4 catch): fig4 same geometry both panels, 'Without remap', dead rows labelled, legend 'late write lands'; fig1 legend 'added or changed', both restore arrows to one slot; fig3 2-copy dots from the after state; captions Fig 2/4/G1. Skipped: Fig1 Experts grey (GEMM tactics changed by FT: Fig 4 green is right), 'lost' encodings unified (style only).
