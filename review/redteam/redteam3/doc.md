# DESIGN_DOC.md red team 3: "doc" lane (technical writer + skeptical reader)

Read on 2026-10-21. Sources: DESIGN_DOC.md (304 lines, "Lnn"), CLAIMS.md (#n, last row #136), runs/STD-8110144 and
STD-G1-8110144 (ab_stats, tactic_check), runs/KILL-FINAL-*-8111350 (kill_final.txt), figures/draft/fig_{throughput,
recovery}.png, branch v2 cef4670789 (code + guide), redteam2/doc.md (R2-n). Most R2 items have been applied. Still open
from R2: #14 (replica-restore simulation unsourced), #15 (~64 B underived), #33 (post-loss throughput), #34 (tests).

## A. Ranked issues (most damaging first)

1. **The healthy-cost text contradicts its own data.** L186-187: "FT on and FT off overlap at every load in both
   measured modes ... within ±2% in both directions." Figure 5 has three modes, not two, and the n=1 data (STD-8110144,
   STD-G1-8110144, rc1 78836d1723) exceeds ±2%. Offline EPLB output throughput: FT worse by 8.3% at 8/GPU and 6.9% at
   512 (window tok/s: 15.2% at 8). No EPLB steady step at 32/GPU: +3.1%. In the figure, offline at 512 is the visibly
   largest gap. No row in CLAIMS covers these runs.
   Rewrite: "At n = 1 per point (rc1, runs STD-8110144/-G1), FT on vs off ranges from −2.9% to +8.3% in output
   throughput (offline EPLB at 8 and 512/GPU is the largest), which is not resolvable at n = 1. Final runs (n ≥ 5)
   decide." Add a CLAIMS row for these runs.
2. **A known confound is shown on the figure but not in the text.** fig_throughput's banner says "No EPLB FT-off used
   different GEMM tactics". tactic_check.txt confirms: std-noeplb-1 PIN_CHECK FAIL (autotune=1216; the other arms are
   pinned). So the no-EPLB comparison is not like for like. Rewrite: add to Setup "No-EPLB FT off autotuned its own
   tactics in this round (the other arms used a pinned table); that panel is not a valid comparison yet", or drop the
   panel.
3. **The doc has no cost headline, and drops earlier decisive evidence.** L191 says "not used for the cost
   headline", but the doc never states one. A reader who knows the history will ask where these went: #123 (static, n=9:
   +1.3-1.5% step, CIs exclude 0, aae00e6c65); #122 (online +5-10%, root-caused #126 and fixed); #136 (online 128/GPU
   +1.15-1.37%, n=2, compact-copy revert); #135 (w/o EPLB +0.6-1.6%, FT worse in every pair). Calling the current
   ±2% "noise" while earlier n=9 data showed a real ~1.4% cost reads as spin. Rewrite: state the goal (<1% healthy cost),
   the best current estimate with n/CI/build, and one sentence: "earlier builds measured +1.3-1.5% (n = 9, #123); the
   fixes since then (R2 dispatch, ...) are expected to remove X µs; the final runs check this."
4. **Recovery time is stated three ways.** L8: "regroup in about 1-3 s". L203: "0.4-0.8 s ... rank and 1.0-1.6 s ...
   node". Figure 6: 0.37-1.61 s. CLAIMS #119/#121 (n=3, aae00e6c65): rank 0.92-1.79 s, node 0.94-2.64 s. The client-side
   resume in KILL-FINAL-ON/OFF-A-8111350 (n=1) is 0.93/1.65 s (online) and 1.47/2.43 s (no EPLB), about 0.5-0.8 s
   more than the figure. The figure times the survivors' first step, not the client. Rewrite L8: "regroup in about
   1-2.5 s from the kill to the first client token (0.4-1.6 s to the first completed step)". Give both metrics and n
   in Figure 6, and add a CLAIMS row for 8110820/8111350. (Per the numbers-cite-CLAIMS rule, every number needs one.)
5. **Figure 6's metrics are not comparable.** FT on is "first committed step" (server). The cold restart is "first
   token after relaunch" (client): #117, online EPLB only, older build b18f0b64ed, same allocation with warm caches.
   State this in the caption. The warm caches favour FT off, so this is conservative; say so, since it helps the
   claim. #117's SIGSTOP cold path (855 s vs FT ~6 s) is the strongest contrast and is missing.
6. **"Most of the recovery time is mapping memory" (L207) is only half supported.** Panel (b) is online EPLB. For a
   rank loss, "plan" is 0.19 of 0.43 s (44%), not "most". The claim "spreading ... cut a rank loss from 1.2-1.5 s to
   0.4-0.8 s" has no CLAIMS row or run ID. Rewrite: "Mapping memory behind ghost slots ('plan') is the largest phase:
   ~45% of a rank loss and ~70% of a node loss (n = 1, run ...)". Cite the before/after runs for the spreading claim
   or drop it.
7. **The kill runs never exercised the rerun path.** L205-206: "requests that had not streamed yet run again on the
   survivors". Every final kill shows reruns=0 (lost_held = client_short; #119/#121, 8111350). At DEP32, all of a lost
   rank's requests had already streamed. Rewrite: "... run again (covered by the 4-GPU integration tests; at 32 GPUs
   every lost request had already streamed, so all ended with the retryable error: 128 per rank, 512 per node)."
8. **Test claims are not backed for the final build.** L291: "12 integration cases ... run in CI once a ULFM image is
   available". The last recorded passes are 10/10 (#114/#116, b18f0b64ed). INTEGRATION-build-rc1-78836d1723-8108846
   holds only install logs, no test.log. Rewrite: "N integration cases (DeepSeek-V3-Lite, 4 GPUs, one node); last full
   pass 10/10 on b18f0b64ed (#116). Not yet run on the measured build. CI skips them (no ULFM image)."
9. **The GEMM-tuning claim is not what the benchmark measured.** L104-105: "Kernel tuning keys on the loaded slots,
   so the expert GEMMs tune as with FT off" (true in code: ff55962c59, in rc1). But every FT-on arm ran a pinned or
   transplanted tactic table (autotune=0), so the healthy-cost data never tests FT's own autotune. In #110, tactic
   choice alone was +5% on an older build. Rewrite: add "(the benchmark pins one tactic table for both arms; FT's own
   autotune result is checked by <run>)" or run one unpinned FT-on arm.
10. **Post-failure throughput is promised but never given.** Goals L24-25 promise "reasonable" throughput after
   failover. The doc has no number. #120 (aae00e6c65, n=3) gives after/before deployment throughput 0.63-0.92 vs an
   expected 0.875-0.97. That is a gap a reader should see. Add it with its build caveat, or mark it TBD in Limitations.
11. **Excluding 512/GPU after seeing the data looks like cherry-picking.** L190-191 excludes 512/GPU "for the cost
   headline". That is where offline EPLB shows its biggest gap. Either say the exclusion rule was fixed in advance
   (cite the plan), or report 512 with its queueing caveat.
12. **The Related Work row contradicts Evaluation.** L232 shows "This design ... Recovery TBD | Healthy cost TBD".
   Fill it: "0.4-1.6 s to first step, ~1-2.5 s client (n = 1) | <TBD final>". Also note that the SGLang/ReviveMoE
   times may be client-visible, so compare like with like.
13. **The headline metric changes between sections.** Figure 5 and L186-189 use throughput and TPOT. Appendix L283
   says "The headline uses the median step". L273 says "per-step device time as the precise cost". Pick one primary
   metric (the median decode step is the right one, given the online-EPLB tail) and put it in Setup.
14. **Status words read badly in a design doc.** L176 "tonight's runs" and L270 "Filled from the final runs". Use dates
   and build IDs: "rc1 78836d1723, 2026-10-20, n = 1-2".
15. **Slot terminology drifts.** The doc uses "ghost slots", "spare slots", "restore slots" (L171, L192), "extra
   slots", "spare slots backed by memory", "empty". Use the config names: extra slots = `ghost` (no memory until a
   restore) or `empty` (memory held). Define them once in Ghost Slots and use only those words afterwards.
16. **Online EPLB's slot kind depends on the platform, which the doc does not say.** L106-108: "If GPU memory can't be
   made CPU-writable, online EPLB falls back ...". The code (config.py extra_slots, 42978bac5e) defaults online EPLB
   to `empty` wherever GDRCopy can't map ghost memory or GPU memory is a NUMA node. Then FT holds ~1.3 GiB/GPU. That
   contradicts L192 "ghost slots take 0 GiB" for those platforms. The measured runs used ghost (rank logs: "Mapped ...
   behind ghost expert slots"). Say which kind the GB200 runs used, and that the memory bullet is per kind.
17. **Unsourced numbers (R2 #14/#15 still open).** L213 "roughly doubled peak expert load" (cite the simulation file
   and setup). L131 "~64 bytes" (show the derivation: top-8 × (4 B slot id + 4 B fp32 scale) = 64 B). L106 "+39 µs"
   (has no row of its own; #133 mentions it as "known"; cite the nsys run). L107 "13 GiB of BAR1" (KILL-FINAL-ON-A-8102532,
   build 547a6aa7ed, not rc1). L257 "MTP ... leak rejected-draft KV pages" (cite the analysis).
18. **Scope is too thin in Evaluation.** Add to Setup or Limitations: R1 NVFP4 only, GB200 only, DEP32 only, random 1k/1k
   tokens, MTP acceptance forced (A2). Process kills only: the host stays up, and no real node loss was tested.
19. **Jargon is undefined for a TRT-LLM reader:** ULFM, ob1, CFT, GDRCopy, BAR1, ADP, "fence", "generation" (L144),
   "tier" (L106), "main" (L172, meaning upstream main). Expand each at first use or add a short glossary.
20. **Figure file names are out of order:** Figure 3 = fig5_placement.png, Figure 4 = fig3_late_writes.png, Figures 5/6
   come from figures/draft/. Renumber the files to fig1..fig6 before review. Also, Figure 5's caption says "bars:
   min-max", but no bars are visible at n = 1.
21. **Requirements and features are mixed** in L31-42 "This design currently supports". KV Cache Manager V2 and
   ULFM are requirements, not features. Split the list into "Requires" (ULFM Open MPI + launch flags, KV cache V2, an
   EPLB load_balancer config, `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0` where CFT exists, 368 GB of /dev/shm per node
   with replicas) and "Supported/tested".
22. **Smaller fixes.**
   - L56 "the executor's hang timeout": give the value (300 s).
   - L53 "~6-7 s": the measured range is 5.0-6.8 s (#119/#121/#130).
   - L289: the accuracy result is n = 1-2 evals (#124); state the n.
   - L9-10 / L17: two internal docs are cited by title only; link them.

## B. Proposed outline (reorder; cuts ~15% of the length)

1. Summary (5 lines): the problem, what FT does, a headline recovery number (client-visible, n, build), and a
   headline healthy cost (or "pending final runs, goal <1%").
2. Motivation (current L13-18, unchanged).
3. Scope: Requires / Supported and tested / Refused at startup / Failures covered / Not covered (merge L31-56 and
   Limitations' fail-closed paths).
4. Design: one terms box (extra slot, ghost/empty, mailbox, authority, generation, fence), then Ghost slots, Placement,
   Late writes, Recovery flow, Health/serving behaviour, Configuration.
5. Evaluation: Setup (including the confounds: transport, pinned tactics, forced MTP acceptance, random tokens, build
   IDs), Healthy cost (one primary metric, table with n/CI per mode × load, earlier-build history in one line),
   Recovery (client and server timing, n, cold-restart baseline caveats), Post-recovery throughput, Accuracy.
6. Alternatives considered.
7. Related work (table with this design's numbers filled in).
8. Limitations and open questions (keep; move fail-closed paths to Scope).
9. Appendix: run index (run ID → CLAIMS row → figure), method notes, the online-EPLB tail table.
