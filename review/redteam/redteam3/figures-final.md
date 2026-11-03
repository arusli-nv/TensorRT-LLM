# Red-team lane 9 (figures), final pass

Scope: the six figures DESIGN_DOC.md references. Fig 1 `figures/fig1_architecture.png`, Fig 2 `fig2_ghost_slots.png`,
Fig 3 `fig5_placement.png`, Fig 4 `fig3_late_writes.png`, Fig 5 `figures/draft/fig_throughput.png` (generator
`analysis/std_figs.py` on `analysis/finals_client_all.json`), Fig 6 `figures/draft/fig_recovery.png` (generator
`analysis/rec_figs.py` on `analysis/kills_final_8115497.json` plus a hard-coded `COLD` dict).

Spot-check: every Fig 5 point matches the client medians in `finals_client_all.json`, and every Fig 6 FT-on dot and band
matches `kills_final_8115497.json`. Two cells are exceptions (items 2 and 3). Most damaging first.

1. **Fig 5: the No EPLB 32/GPU cost is invisible, and the figure shows FT-on gaps the doc's table says do not exist.**
   - Evidence, the hidden cost: the plotted client tok/s/GPU at No EPLB 32 is 2999 (off) vs 2982 (on), -0.6%, and the
     two points sit on top of each other. The doc's headline (bold +2.68%, UB95 +3.86%, readout section 2) is a
     rank-0 step metric that the figure does not show.
   - Evidence, the visible gaps: the figure's largest gaps are at 8/GPU. No EPLB is 1115 vs 1070 (-4.1%) and offline is
     1153 vs 1096 (-4.9%), readout section 3. The table puts the step cost there at -0.08% and -0.26%. Offline 32 is
     -1.7% and online 128 is -2.3% on the client. A reader of the figure sees the opposite of the table, and no text
     explains it (client throughput includes ramp, prefill and stall time; the step metric is decode-only at
     ≥0.9 x load).
   - Fix: add a cost panel next to or below Fig 5. Plot the step cost % (y) against load 8/32/128 (x, categorical), one
     marker per mode, whiskers to the one-sided UB95, a dashed +1% line, and No EPLB 32 annotated "+2.7% (≤ +3.9%)".
     Plot 512 hollow as "beyond recipe". Add one caption sentence: "Client throughput includes ramp-up, prefill and
     stalls, so its FT on/off gaps (up to -5% at 8/GPU) differ from the decode-step cost in the table." Alternatively,
     drop Fig 5 as the cost evidence and call it "absolute operating points".
2. **Fig 6(a): the grey FT-off values are not what the caption says, and they come from another build and setup.**
   - Evidence, the values: `rec_figs.py` hard-codes `COLD = {"rank5": (477, 452, 517), "node1": (539, 488, 642)}`.
     These are CLAIMS #117's **"down"** times. The caption says "FT off: time from the failure to the first token after a
     relaunch", which in #117 is 467 [441-507] s and 528 [477-630] s.
   - Evidence, the axis: the x-axis label says "Failure to the survivors' first completed step", which applies to the
     FT-off row as well.
   - Evidence, the provenance: #117 is wheel build-final-b18f0b64ed, job 8082563, online EPLB, 128/GPU. The doc says
     "Kill and accuracy runs use the same build" (Oct 21 `v2`).
   - Fix: plot the first-token values (467, 528) or reword the caption to the "down" definition. Relabel the axis
     "Time from the failure to serving again (s, log scale)". Add to the caption: "FT off: CLAIMS #117, earlier build
     b18f0b64ed, online EPLB, 128/GPU, n=3". Or rerun the cold restarts on `v2`.
3. **Fig 5 vs readout_final: the excluded 512 row is excluded in the figure but not in the readout, and the caption
   does not say so.**
   - Evidence: `finals_client_all.json` offline/512 FT off has n=2 (STD-8116111, STD-8118573), so STD-8118134
     std-offline-1 is dropped and no whisker reaches 7329. `readout_final.txt` section 3 still lists offline/512
     std-offline at n=3 with range [7328.66-7970.59] and TPOT [46.05-50.08]. Its "rows with >5% failed/short dropped"
     rule did not catch the late-arrival row.
   - Fix: in `std_collect.py` / readout section 3, drop the row by the pre-registered late-arrival rule and print
     "excluded: STD-8118134 std-offline-1 (512)". Add to the Fig 5 caption: "Offline EPLB 512 FT off: n=2; one run
     excluded (late arrivals, rule fixed before the runs)". Also change "median over 3-5 runs" to "2-5 runs".
4. **Fig 5 includes an unpaired FT-on run, so the online series compares n=5 to n=4.**
   - Evidence: online-ft at every load has 5 runs, including `STD-G1-8118134/online-ft-1`. Readout section 7 says
     "Unpaired FT cases (reported, not in any cell): STD-G1-8118134:std-online-ft". Its group's FT-off arm was the
     static-LB control.
   - Fix: pass only paired runs to `std_figs.py` (drop that run), or draw it as a hollow marker and footnote it. Then
     state n per mode in the caption: "No EPLB 4, offline 3, online 4 node groups".
5. **Fig 4 (`fig3_late_writes.png`): the caption contradicts the layout.**
   - Evidence: the caption says "packed buffer (top) vs per-sender mailboxes (bottom)". The figure puts the MoE pipeline
     on top, packed at bottom-left and mailboxes at bottom-right.
   - Fix: change the caption to "A late write in a packed buffer (left) vs per-sender mailboxes (right), in the receive
     workspace of one MoE layer (top)."
6. **Fig 6: the FT-on n is not stated, and it differs per mode.**
   - Evidence: the kills JSON has online rank5 n=2, online node1 n=2, and the other modes n=3. The caption's "median and
     range of 3 runs" grammatically attaches to FT off only. Panel (b) is online only, so each bar is the sum of
     per-phase medians of 2 kills. The text says "2-3 kills per case".
   - Fix: append to the caption "FT on: 3 kills per case (online EPLB: 2); (b) online EPLB, per-phase medians of 2
     kills; job 8115497."
7. **Fig 6(a): the lollipop stems on a log axis start at an arbitrary 0.3 s, and the decade labels are hard to read.**
   - Evidence: `x0 = np.log10(0.3)`, and each stem is drawn from x0. Stem length therefore encodes log(v/0.3), which
     means nothing. The online rank band minimum (0.33 s) sits against the axis edge. Any kill under 0.3 s would be
     clipped silently. The ticks render as 10^0 ... 10^3, and the empty decade above 10^3 runs to 6000.
   - Fix: drop the stems and keep dot plus band. Set xlim to roughly (0.2, 1000). Label the ticks "0.3 s, 1 s, 10 s,
     1 min, 10 min". Optionally annotate "~1000x" between the groups.
8. **Fig 5: the FT-off points are hidden under the FT-on points, and the load labels collide.**
   - Evidence: at No EPLB 512/128 and offline 128/512 the grey marker is fully covered (same marker, ms=5, FT on drawn
     on top). At "8" the grey label overlaps the whisker cap. The labels are 8.5 pt on a 2501-px PNG, so they render at
     about 5 pt at doc width.
   - Fix: hollow grey markers (mfc white, ms 6) for FT off, drawn after FT on. Use a different marker shape for FT on.
     Make the labels at least 10 pt with a white halo. Use the shared legend "FT off (hollow grey) / FT on (filled,
     mode colour)".
9. **Fig 5 caption and footer: the error-bar meaning is incomplete.**
   - Evidence: the bars are min-max over node groups in both x (tok/s/user) and y. The footer says "median over runs".
     The x quantity is benchmark `user_throughput`, not 1000/TPOT; the doc's table and text talk about TPOT and step
     time.
   - Fix: change the caption to "Points: median over node groups (n per mode as above); bars: min-max over groups in
     both axes. x: benchmark output tok/s per user (includes TTFT); 1k/1k, MTP 1, 32 GB200."
10. **Fig 1 and Fig 2 show only the KV-cache source for restore memory, but the text says free GPU memory comes
    first.**
    - Evidence: doc line 104 says "memory comes from free GPU memory first, and from the KV cache only if ...". Fig 1
      has only "KV cache: lends memory". Fig 2 maps e16 into the KV-cache block with no "fallback" note.
    - Fix: in Fig 2, add a small "free GPU memory" block left of the KV cache with the arrow to it, and mark the KV
      cache "(if needed)". In Fig 1, relabel to "free memory / KV cache: lends memory".
11. **Fig 2 labels the restored expert as "e16 lost" in a green (restored) box.**
    - Evidence: green means "restored" in the Fig 3 legend and "added by FT" in Fig 1. The label "lost" inside the green
      box reads as a contradiction.
    - Fix: change the label to "e16 (restored)", or "ghost slot ← e16".
12. **Fig 3 (`fig5_placement.png`): the fill order and the legend are slightly inconsistent.**
    - Evidence: in "Rank 2 lost" the restored experts fill the *second* ghost row (the top row stays hatched). In
      "Node 1 lost" the second row fills first, then the top row only on GPUs 0-3. The legend entry "lost GPU" (red) is
      drawn on slots. The counts check out: 8 lost to 8 GPUs, and 32 lost to 28x1 + 4 extra on GPUs 0-3.
    - Fix: name the legend entry "slot on a lost GPU". Add a caption clause: "GPUs 0-3 take a second expert because
      32 experts spread over 28 survivors, fewest first, lowest rank first".

Checked and OK:
- Fig 5 values for every plotted median.
- Fig 6 medians: 0.39/0.35/0.51 and 1.19/0.95/1.45 s match the kills JSON. The no-EPLB 2.85 s outlier and the offline
  0.95 s outlier are visible as bands.
- Fig 6(b) phase colours are a neutral blue ramp, distinct from the mode hues.
- Mode names "No/Offline/Online EPLB" are consistent between the figures and the table.
- Fig 3's placement counts match the "fewest first, lowest rank first" rule.
