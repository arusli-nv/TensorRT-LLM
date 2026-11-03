# DESIGN_DOC.md final red team, lane 7 "doc" (technical writer + skeptical reader)

Read 2026-10-21. Doc: mvp/DESIGN_DOC.md (391 lines, "Lnn"). Sources: analysis/readout_final.txt/.json (R), ctx_final.txt (CTX),
analysis/kills_final_8115497.json + runs/KILL-FINAL-*-8115497/kill_final.txt (K), kills_prelim_8111350.json (KP),
runs/ACCURACY-KILL-*-8115497, runs/INTEGRATION-build-v2-cef4670789-8118061/test.log, CLAIMS.md (last row #139).
Verified OK: every cell of the L219-223 table (mean, UB95, n) = R §2; L227 +0.37 ms (R: +369 us); L228 13.82/13.80/14.34 = R §7;
L349-352 = R §7; L354 per-group = R §2; L241 58 x 24.8 MB = 1.44 GB = 1.34 GiB; L252 medians = K "total" (rank 0.35/0.39/0.51,
node 0.95/1.19/1.45), 2.8 s outlier = K 2.85; L256 "5 reruns" = KILL-FINAL-OFF-B rank5-2 reruns=5; L370-372 accuracy
(0.9469 vs 0.9454, 2SE 0.0177, 19/21 flips); L378 12/12 passed (test.log).

## Ranked issues (top 5 first)

1. **Overview and Related Work quote the preliminary build's recovery times, not the final build's (contradiction).**
   L15 "resumes in 0.4-0.8 s", L16 "1.0-1.6 s", L283 "0.4-1.6 s (preliminary)" are KP (rc1, n=1: rank 0.37-0.81, node
   1.02-1.61). Final build K (n=2-3 per case): rank medians 0.35-0.51 s (range 0.33-2.85), node 0.95-1.45 s (range 0.90-1.50);
   L252 says "0.3-0.5" / "0.9-1.5". Client-visible resume (K resume_s medians) is longer: rank 0.99-1.07 s, node 1.40-1.93 s;
   the doc never gives it (prior lane #4 still open).
   Fix L15: "~125 requests end with a retryable error; the survivors complete a step again in 0.3-0.5 s (median; clients see
   tokens again after ~1.0 s)". Fix L16: "... in 0.9-1.5 s (clients: 1.4-1.9 s)". Fix L283: "0.3-1.5 s to the first step,
   1.0-1.9 s to the client (final build, n=2-3 per case)". Add to L252: "Clients see tokens again 0.5-0.7 s later (median
   resume 0.99-1.07 s rank, 1.40-1.93 s node; harness poll)."

2. **Figure 5 visibly contradicts the "<1%" text, and nothing reconciles them.** Figure 5 plots client output tok/s/GPU
   (R §3), where FT on is lower by 4.9% (offline 8: 1096 vs 1153), 4.1% (no EPLB 8: 1070 vs 1115), 2.3% (online 32:
   2934 vs 3001; online 128: 5432 vs 5563), 1.7% (offline 32). A reader sees gaps of 2-5% next to a table that says <1%.
   Fix (after L214): "Client throughput is descriptive: it includes request arrival, stalls and the slow-step tail, and its
   FT-on/off medians differ by up to 5% at 8 requests/GPU with overlapping min-max ranges; the step-time table below is the
   cost measure fixed before the runs." If the gaps are not overlapping noise, say so instead of this sentence.

3. **L225 "FT costs under 1% at every load" is stronger than the statistics.** R §2: the every-cell test fails; Holm
   establishes <1% only for no-EPLB 8/128, offline 8/128, online 8/128. Offline 32 UB95 +1.27% (n=3); online 32 Holm p=0.082
   (not established, two-sided CI [-0.97, +1.04]). Device-step offline 32 (R §5): +0.90%, UB +1.04%.
   Fix: "With EPLB, FT's measured cost is under 1% in every cell (-0.28 to +0.56%), and the 95% upper bound is under 1% in
   five of six cells; offline EPLB at 32 requests/GPU (+0.47%, bound +1.27%, three groups) is not established below 1%."

4. **The no-EPLB +2.7% cost is missing wherever a reader looks for the cost.** L178 "target: within 1%", L187 "When to
   enable", L283 "see Evaluation" and the Overview give no result; the one failing cell appears only inside Evaluation.
   Source R §2: +2.68%, UB95 +3.86%, n=4, CI [+1.08, +4.28]; device step +2.29% (UB +2.43%, R §5), so it is not noise.
   Fix L178: "Healthy throughput: under 1% with online or offline EPLB at 8-128 requests/GPU (offline at 32 not established);
   without EPLB, +2.7% per decode step at 32 requests/GPU (bound +3.9%, 4 node groups), under investigation." Put the same
   phrase in the L283 "Healthy cost" cell.

5. **No CLAIMS row backs any final number.** CLAIMS ends at #139 (preliminary rc1 rows #137/#138); no row cites 8116111,
   8118573, 8118134, 8115497, 8118061 or 8114247/8115508/8117713. Every Evaluation number is therefore uncited under the
   project's own rule. Fix: add CLAIMS rows #140 (healthy cost, R §2, n=3-4 groups, host loop-step median, decode-only
   num_active >= 0.9 load), #141 (static-LB control, R §7, n=1 group), #142 (kills, K, n=2-3), #143 (accuracy), #144 (tests),
   #145 (prefill, CTX), and mark #137/#138 superseded.

6. **Slow-step paragraph (L237-240) misstates the data.** R §4: no EPLB at 8/GPU is n=4 groups, not "one group of two"; the
   FT-minus-off differences are +4.4, +1.2, +1.7, +1.9 points (HL +1.86, fail). "After that both stay under 3%" is false:
   steady stall share reaches 4.6% (offline 8, FT on) and 3.1% (offline 32, FT off). The pre-set stall rule fails in 8 of 12
   cells, including online 8 (max pair +4.9), online 32 (+3.95), offline 8 (+4.08). The "first minute: 8% vs 2%" figure has
   no source in R. Fix: "Slow steps (steps > 2x the median): the pre-set rule (FT adds <= 0.5 points, no pair > 2 points)
   passes in 4 of 12 cells. At 8 requests/GPU FT adds 1.2-4.4 points without EPLB (4 groups) and single groups add up to
   4-5 points in other modes; the median-step cost above excludes these episodes. Cause not established." Cite the
   first-minute 8%/2% source or delete it.

7. **Prefill (L232-233) mixes two metrics.** "2-25% slower per prompt token" is wrong: CTX per-token cost for online EPLB
   ranges -8.8 to +12.5%; the +25-29% values are the median prefill step (+25.1, +27.3, +29.0 at 32/GPU). The quoted
   "+7.5, -4.2, +12.5, +6.1%" is per token (correct). Offline also shows +13.3% per token in one group (STD-8118573, 32/GPU).
   Fix: "Prefill steps: per prompt token, FT on was -8.8 to +12.5% versus FT off across online-EPLB groups and loads (32
   requests/GPU: +7.5, -4.2, +12.5, +6.1%; one offline group +13.3%), with too few prefill steps (~60-270 per row) to
   establish a cost either way; prefill is not covered by the healthy-cost claim."

8. **512 requests/GPU (L234-236) omits the one real cost there and the load caveat.** R §2: online 512 is +0.94%, two-sided
   CI [+0.23, +1.66] (excludes zero), UB +1.47%. Load attainment at 512: only 12-52% of decode steps reach 0.9 x load, and one
   FT arm (STD-8118573 online) has median active 4. The "12% of requests ~63 s late" exclusion is not in readout_final
   (source only MORNING_SUMMARY L26), and R §3 still lists offline 512 FT-off with n=3 (min 7329), i.e. not excluded there.
   Fix: "512 requests/GPU is beyond the reference recipe and often did not reach full load (12-52% of steps at >= 90% of the
   target); step cost there was -0.55% (no EPLB), -0.42% (offline) and +0.94% (online, 95% CI +0.23 to +1.66%). One offline
   FT-off run (STD-8118134) was dropped from the client comparison by the pre-set rule (requests arriving ~63 s late at the
   network level)." Make the figure/readout agree on which n they use.

9. **Recovery-history numbers are miscounted or unsourced (L258, L331).** L331 "1 of 10 kills": 8115497 has 16 kills
   (K: 3+3+3+3+2+2; MORNING_SUMMARY says 15/15). L258 "0.43-0.81 s (job 8111350)": KP rank totals are 0.37/0.43/0.81. "1.21-1.50
   s (job 8102532)" has no CLAIMS row (#135 lists client resume 1.53/1.57 s), and 8102532 (547a6aa7ed) vs 8111350 (rc1 +
   overlay) differ by more than the spreading change. Fix L258: "Spreading restored experts evenly across GPUs is the main
   change between build 547a6aa7ed (rank loss 1.2-1.5 s, job 8102532) and rc1 (0.37-0.81 s, job 8111350); other changes
   also landed between them." Fix L331: "(a 2.8 s outlier in 1 of 16 final kills ...)".

10. **Figure 6's FT-off bars are a different metric, build and mode (prior lane #5 still open).** Plotted 477/539 s are
    CLAIMS #117 "down" (failure to serving), not "first token after relaunch" (467/528 s), on build b18f0b64ed, online EPLB
    only, relaunch in the same allocation with warm caches; the x-axis label says "survivors' first completed step" for all
    bars. Fix caption: "FT off: time from the failure until the relaunched server serves again (CLAIMS #117, online EPLB,
    build b18f0b64ed, same allocation with warm caches, so optimistic for FT off; median and range of 3)."

11. **n per cell is inconsistent between caption, table and readout.** L214 "median over 3-5 runs"; the table uses 3-4
    node groups; online FT has 5 client points because the unpaired STD-G1-8118134 FT case (R: "Unpaired FT cases") is
    plotted but not in any cost cell. Fix L214: "Points: median over node groups (3 offline, 4 no-EPLB/online; online FT on
    includes one unpaired group); bars: min-max."

12. **Post-failure throughput is promised (L35-37) but never given (prior lane #10 still open).** K est tok/s after/before
    (medians): rank loss 0.94-0.95, node loss 0.89 (no EPLB 314/329 and 293/331 k tok/s; offline 315/334, 296/333; online
    344/366, 316/354). Fix (add to Recovery): "After a loss the deployment served 94-95% (rank) and ~89% (node) of its
    pre-failure token rate (harness estimate, n=2-3 per case, 8115497)."

13. **Accuracy run did not exercise lost requests.** ACCURACY-KILL-ON: lost_held=0, reruns=0, recovery_ms=794,
    eval_after_kill_s=-40.3. "No request failed or was retried" is then expected, not evidence. Fix L370-372: append "Rank 5
    held no requests at the kill, so this run checks the survivors' outputs after a restore, not reruns."

14. **Metric definition is unclear to an outside reader.** L216 "median step time at full load on rank 0" is R's "rank-0
    host loop-step median over decode-only steps with num_active >= 0.9 x load"; "node group", "one-sided 95% bound over
    groups" (t over n=3-4 groups) and "paired" are undefined; L203 and L205-207 repeat the protocol with different warm-up
    wording. Fix L216: "FT's cost = (FT on - FT off) / FT off of rank 0's median decode-loop step time, over steps with at least
    90% of the target requests active; each node group (8 nodes) runs both arms, and the bound is the one-sided 95% t bound
    over the 3-4 groups." Merge L203 into L205-207.

15. **Remaining unsourced statements in Evaluation/Limitations.** L229 "slot layout explains ~0.07 ms" and "CUDA-graph
    padding ... ruled out" (not in R; cite the run/analysis); L298-299 ULFM TCP "within noise at 128 requests/GPU" (no
    final-build run); L238 8%/2% (see 6). Jargon still undefined at first use (prior #19): ULFM, ob1, CFT, ADP, BAR1, "tier".
    Fix: cite a run ID + CLAIMS row after each number, or delete the number.

## Prior lane (redteam3/doc.md) status
Fixed: #1-3 (final data, pinned tactics, headline table), #7 (5 reruns cited), #8 (12/12 on v2), #11 (pre-set rule stated),
#13 (median step named), #14. Still open: #4 (recovery stated several ways, no client metric) -> item 1; #5 -> item 10;
#10 -> item 12; #12 (Related Work cell) -> items 1 and 4; #17 (unsourced ~64 B, "roughly doubled", +39 us, 13 GiB BAR1);
#19 jargon -> item 15.
