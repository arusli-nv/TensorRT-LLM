# WideEP FT: presenter notes (2026-10-26, updated 2026-10-27)

Source of truth: review/DESIGN_DOC_final.md (all figures embedded). Every number below is in the doc with
its CLAIMS row. Code: v5-final 27442bd91d; DeepSeek-R1 NVFP4, 32 GB200, DEP32.

## Five headline messages

1. **A rank loss no longer ends the job.**
   - Serving resumes 0.8-1.6 s after a rank kill (1.7-2.7 s if killed in prefill) and 1.4-2.8 s after a node kill
     (kill to rank 0's first step; range over all v5-final runs).
   - A restart takes 6-10 min on average (301-693 s, n=20). It is a lower bound: FT off, page and autotuner caches
     kept, same allocation. (CLAIMS 193, 202, 203, 224, 227, 228; Figures 7, 7b, A4)
   - Not covered: losing rank 0, rank 0's node, or a node plus its ring-copy node ends the job, as without FT.
2. **Survivors keep serving with no recapture.**
   - Throughput after a loss, measured at the client (medians): 0.945x with 31/32 live (proportional 0.969), 0.860x with 28/32
     (proportional 0.875) (CLAIMS 209, 220, 239; n=13 each incl. THIN-CELLS 8302191 and RECOV4 8363511; Figure 8).
   - Proportional is a reference, not an upper bound: each survivor then serves more requests (~132 / ~146 vs 128).
   - Survivors' requests all complete. The lost rank's in-flight requests (128 per rank, 512 per node) end with a
     retryable error; a restart drops all 4096. (CLAIMS 193)
3. **Accuracy is unchanged:** GSM8K on the recovered survivors 0.9477 vs FT off 0.9469. (CLAIMS 193)
4. **Healthy cost is small but not zero.**
   - Output tok/s vs FT off at 32/GPU, pooled n=8: no EPLB +0.25%, offline -0.69%, online +0.05% (all CIs span 0).
   - The original offline 32/GPU -2.0% did not replicate: pooled n=8 -0.7% [-2.8, +1.4]; all modes within noise
     (CLAIMS 206).
   - The decode step (trimmed mean) is +0.35% to +0.87% slower at 32/GPU (51-120 us, pooled n=8) and +0.39% to +0.56% at 128/GPU
     (99-150 us, pooled n=8); no EPLB pays the most, partly the static load balancer. "< 1% per step" is not claimed.
   - Client numbers are flat while the step is slower: FT also runs fewer, larger prefill steps (-4.89%, CLAIMS 208),
     which can offset it (likely, not measured). Matching client numbers do not mean zero cost.
   - (CLAIMS 194, 206; Figures 5, 6, A1)
5. **The compaction stalls found during the work are reduced to the FT-off level, as far as measured.**
   - Startup stall on a used node: 2.7 / 5.7 s vs 3.2 s with FT off (21-46 s before, on another metric).
   - Mid-run bursts: 3/12 windows before, 0/8 FT and 0/9 FT off now (not proof of removal: 0/8 happens ~10% of the
     time at the old rate). (CLAIMS 184, 187, 194, 196, 197; Appendix E)

## Say these plainly (open, explained, fixes recorded for later)

- One healthy FT run hung (1 of 36; 0 of 31 FT-off): every rank stuck in a host allgather over FT's TCP transport,
  which FT's hang detection does not cover. Cause not isolated; the soak found 0 hangs in 7.1 / 7.2 h (CLAIMS 222); detector fixes are a follow-up.

- Offline EPLB at 32/GPU, -2.0%:
  - FT on ran as on earlier builds; this batch's FT-off runs were fast.
  - Over 12 pairs and 3 builds it averages ~-1% (post hoc): ~0.45% steady cost plus ~0.6% occasional stalls.
  - Those stalls are other ranks prefilling out of phase (CLAIMS 201). FT off had them too, but across 16 pairs FT
    arms had slightly more (12 of 16, p=0.035; ~0.1% of the window in the replication; CLAIMS 207), carried by the
    pre-fix and fix-1 builds; on v5-final there is no clear sign. Small, cause not attributed.
- Steady per-step cost:
  - A kernel-only run at 32 GPUs: +0.88 us per a2a round trip without EPLB stats (~50-59 us/step; CLAIMS 210) and +3.0 us with them
    (~178 us/step if back to back; CLAIMS 200, diagnostic). In serving, online costs no more than offline, so overlap
    likely hides most of it (inference).
  - We tested our leading explanation (the mask spilling to local memory) and it was refuted: removing the spill did
    not reduce the stats-path cost and trimmed the rest by only ~10 us/step (CLAIMS 205).
  - An item-by-item ablation then explained it (32 GPUs, n=3 per item, CLAIMS 210, diagnostic): with stats, 2.13 of
    3.00 us is the per-peer mask check in the serial stats loop; without stats, the released-round load before the
    fence (0.45 us) and the lane-parallel mask checks (0.49 us). All removed leaves +0.09 / +0.02 us. Measured fixes
    for later (not in the presented code): a ballot stats loop cuts 1.62 us; an 8-byte flag load 0.18 / 0.14 us.
- Median TTFT is +7.3% [+1.3, +13.3] pooled at 32/GPU (24 pairs over modes); p99 unchanged; not seen at 128/GPU
  (CLAIMS 208). A pre-registered soak test says it is an FT effect in this synchronized-start benchmark: FT runs start
  in the slower "locked" state more often (14/18 vs 6/18, p = 0.018), and locked FT windows are +161 ms slower; steady
  windows show no difference (CLAIMS 223). Under realistic open-loop traffic median TTFT is +0.9% [-0.2, +1.9] (6 pairs;
  CLAIMS 217, 234, 240, Figure 9). Cause not known.
- Healthy hang soak: 0 hangs in 7.1 / 7.2 loaded h (FT on / off); rate < 0.42 per hour each (CLAIMS 222); rarer hangs
  are not ruled out.
- What the healthy numbers do not cover: one model and shape (R1 NVFP4, 1k/1k), MTP acceptance forced to 1, a
  closed-loop benchmark at 32 and 128/GPU, and one open-loop load (~80%, Figure 9).
- Slow online restarts are main's online-EPLB host-copy build (~250 s extra model load on every launch, CPU-bound),
  not page cache (inferred from job 8206870; CLAIMS 211 corrects 198).
- Rank and node kills have 4 runs each on the final code (online rank: 8, incl. four in an EPLB update; CLAIMS 193,
  203, 227, 235; means: no EPLB 1.50 / 2.57 s, online 0.88 / 1.55 s). Offline rank and node kills have 5 (means 1.28 /
  1.62 s), frozen rank 3 and two losses 2 per mode (CLAIMS 213, 227, 235, 238). Prefill kills have 2-3 per mode (8 in total). The pre-fix build's
  node kill (3.2 s script, 2.94 s server) was ~0.1-0.5 s slower server-side than v5-final; cause not isolated.
- No-EPLB remap is slow (~0.8-1.0 s, about twice online): since the host-copy fixes 0/12 no-EPLB rank-type kills were
  fast (fix-1 0/5, v5-final 0/7 through THIN-CELLS; later rows not checked), vs 15/36 before (Late Writes; Appendix I.1, no CLAIMS row). Whether the fixes cause
  this is not established.
- Authority loss: serving continues, /health reports 'unprotected', and a further loss ends the job (EP4 only,
  CLAIMS 88, 101).
- Hang threshold 30 s: at 5 s a healthy rank holding the GIL was falsely fenced. At 30 s, a 45 s GIL hog and all-rank
  and node host stalls were not fenced (older builds, n=1 each; CLAIMS 77, 97, 103).
- Not tested on the final code: two frozen ranks at once, CFT (needs driver >= 615; all runs used driver 580's fence
  path), host death, NVLink faults. (Offline-EPLB kills now pass on the final code, n=5 each: rank 1.23-1.37 s, node
  1.51-1.72 s, CLAIMS 202, 213, 238.)
- Benchmark settings: forced MTP acceptance (as in TRT-LLM's perf recipes), all requests released at once, exact
  1k/1k. Absolute tok/s are not comparable to published numbers.

## Hard questions

The 17 hardest reviewer questions, with cited answers: review/diag/redteam_methods.md, section (b).

Likely questions on the stress tests and figures:

- "Served ~1.0x after a kill, so no capacity was lost?" No: the offered load stayed below the post-loss capacity.
  Capacity (closed-loop calibration row, before vs on the survivors at the end) is 0.919-0.93x (CLAIMS 216, 219).
- "What do users see after a kill?" At ~80% load, median TTFT 826 -> 1021 ms (+24%) and 842 -> 1049 ms (+25%),
  median TPOT 11.8 -> 13.2 ms (one case), same offered load (CLAIMS 216).
- "Why only 3 losses in Figure 10?" No selection rule was set; all 8 losses have the same client records. The other 5,
  including the node kill (CLAIMS 219), are not plotted; their numbers are in the tables.

## Code

- PR stack `v5-pr` (a23efa7228, local): the measured code plus behaviour-equivalent cleanups; tests pass per commit (review/V5PR_EQUIVALENCE.md,
  review/diag/v5pr_per_commit_table.md).
- Follow-up `v5-pr-fixes` (v5-pr-fixes; 3 small fixes + 1 test commit off the measured path; unit tests + GPU kill smoke PASS, job
  8282693), to merge with the stack or right after it.
- Separate PRs first: the attention-DP burst-release rule and the mpi4py bump (which needs a container image bump).
- Future fixes: review/POST_FINAL_ANALYSIS_PLAN.md, "Future fixes".
- Early-user stress (realistic open-loop traffic, R1 32 GPUs): at ~80% load (79-81% of calibrated), FT on vs off differs
  only slightly (median TTFT +0.9% [-0.2, +1.9], median TPOT +0.1 ms in 5 of 6 pairs, p99s noisy with one outlier run
  each way, served throughput within 0.2%; S1 8310100 + S1B 8359063 + S1C 8363138, CLAIMS 217, 234, 240, n=6 pairs;
  Figure 9). Random kills under traffic (S2 8299473, CLAIMS 216; 8318006, CLAIMS 219): 3 rank
  kills resumed in 1.03-1.05 s and a node kill in 1.96 s, with only the lost ranks' requests failing; /health missed
  the recovering state in 2 of these 4 (3 of all 8 stress-test losses; 1 s cache). Figure 10 shows 3 of the 8 losses; over all 7 rank kills the first 10 s dips -10% on average (-3 to -16%), the node kill -30% (CLAIMS 236).
- A second random loss 15 min later also recovers (S3 8314070, CLAIMS 218): 4 losses, resume 0.87-1.03 s, only the lost
  ranks' requests failed, served ~1.0x of the offered load (not capacity); 30 ranks serve at the end.
