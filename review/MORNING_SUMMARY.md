# WideEP FT: morning summary (2026-10-25, updated 2026-10-27 overnight: see FINAL/OVERNIGHT_LOG.md)

Data frozen (your decision 07:20 PT). Doc and figures: review/DESIGN_DOC_final.md (all graphs embedded).
Final for presenting (11:15 PT; clarity rewrite and CLAIMS 201-203 added 14:35 PT).
Presenter notes: review/PRESENTER_NOTES.md. Future fixes: review/POST_FINAL_ANALYSIS_PLAN.md.

## Final results (v5-final 27442bd91d; DeepSeek-R1 NVFP4, 32 GB200, 1k/1k)

- Code: the finals build plus two compaction fixes: no peer prefault, and the owner unmaps its copy after writing.
- Recovery (CLAIMS 193, 203, 227, 235; 128/GPU; all rows pass; kill -> rank 0's first step after the commit; mean of 4 runs, online rank
  kill 8 incl. four in an EPLB update):
  - rank kill 1.5 s without EPLB, 0.88 s online; node kill 2.6 s / 1.55 s;
  - range over all v5-final runs: rank kill 0.8-1.6 s (1.7-2.7 s if killed in prefill, n=2-3 per mode), node kill 1.4-2.8 s;
  - frozen rank fenced 29.5-34.1 s after the freeze (64/GPU; n=3 per mode: no EPLB 34.1 / 34.1 / 30.3 s, offline
    30.1 / 29.5 / 30.0 s, online 30.0 / 30.5 / 33.4 s; first step 32.9-37.4 s from the stop; THIN-CELLS 8302191,
    RECOV2 8353506, RECOV4 8363511; CLAIMS 213, 227, 238);
  - restart instead 5.7 / 6.2 / 10.4 min mean (no / offline / online EPLB, n=6 / 7 / 7, pooled warm restarts; CLAIMS 224, 228;
    weights, kernel compile + tuning, graph capture); a lower bound: page and autotuner caches kept, same allocation;
  - client-measured throughput after a loss, median 0.945x (31/32 live) and 0.860x (28/32), 1.5-2.4 percentage points below proportional (pooled; per-mode medians 0.6-4.4)
    (0.969x / 0.875x; a reference, not an upper bound) (CLAIMS 209, 220, 239; n=13 each with THIN-CELLS 8302191 and RECOV4 8363511, CLAIMS 213, 238).
  - Requests: survivors' all complete; the lost rank's 128 (node: 512) in-flight requests end with a retryable
    error (3.1% / 12.5% of 4096). A restart drops all 4096.
- Accuracy: GSM8K served by the survivors after a kill: 0.9477 vs FT off 0.9469 (CLAIMS 193). All four runs
  (three builds) are within 2 SE.
- Healthy cost (benchmark_serving, same-node pairs; original n=4 per cell, Appendix F; CLAIMS 194). Originally only offline
  32/GPU was significantly below FT off, and it did not replicate (below). One-sided 95% bounds on FT's throughput loss:
  32/GPU pooled n=8 0.58% no EPLB, 2.38% offline, 0.63% online; 128/GPU pooled n=8 1.37% / 1.27% / 1.10% (CLAIMS 206, 232).
  Pooled n=8 means at both loads:

| Mode | Load | Output tok/s/GPU | Median TPOT | p99 TPOT | Median TTFT |
|---|---|---|---|---|---|
| No EPLB | 32 / 128 | +0.25% / -0.33% | -1.8% / +0.8% | -1.2% / +0.4% | +6.1% / -0.7% |
| Offline EPLB | 32 / 128 | -0.69% / +0.08% | +0.1% / -0.3% | +1.0% / -0.1% | +3.6% / +1.3% |
| Online EPLB | 32 / 128 | +0.05% / -0.38% | -1.4% / +0.0% | -3.9% / +0.8% | +12.2% / +0.9% |

- Every one of these 24 CIs includes 0.

- Median TTFT pooled over 24 pairs at 32/GPU: +7.3% [+1.3, +13.3] (CLAIMS 208, 223); not seen at 128/GPU; under
  open-loop traffic at ~80% load median TTFT is +0.9% [-0.2, +1.9] (6 pairs; CLAIMS 217, 234, 240).
- Not covered by these numbers: other models or shapes, unforced MTP acceptance, other loads; the soak bounds hangs
  at < 0.42 per loaded hour (CLAIMS 222), not rarer ones.
- Stall bursts: 3 of 12 online FT windows before the fixes; 0/8 FT and 0/9 FT off on the final code (CLAIMS 194,
  196, 197). Startup stall on a used node: 2.7 / 5.7 s vs 3.2 s FT off (CLAIMS 187).

## Open, explained as far as the data allows

- One healthy FT run hung in this afternoon's replication (1 of 36 FT-on runs, 0 of 31 FT-off): all ranks stalled in
  main's attention-DP allgather over FT's TCP transport; FT's hang detection did not catch it (job ended at 600 s).
  Cause not isolated; disclosed in the doc (Known costs, Known Limits; CLAIMS 204). Soak done (0 hangs in 7.1 / 7.2 h, CLAIMS 222); follow-up: detector fixes.

- Healthy-cost replication (CLAIMS 206, 4 new pairs per 32/GPU cell): the offline -2.0% did not replicate (pooled
  n=8 -0.69% [-2.80, +1.42]); online +0.05%, no EPLB +0.25% pooled. All within noise.
- Offline 32/GPU -2.0% (original): FT on ran as on earlier builds; this batch's FT-off arms were fast. Over 12 pairs and 3
  builds, ~-1.0% (post hoc) = ~0.45% steady cost + ~0.6% occasional stalls. A follow-up with all-rank logs found
  those stalls are other ranks prefilling out of phase (CLAIMS 201). FT off has them too, but across 16 pairs FT arms
  had slightly more (12 of 16, p=0.035; ~0.1% of the window in the replication; CLAIMS 207), carried by the pre-fix and
  fix-1 builds; on v5-final no clear sign. Host memory does not
  explain the build-to-build spread.
  (review/diag/offline32_analysis.md)
- Steady cost (trimmed mean): +0.35% to +0.87% per step at 32/GPU (51-120 us, pooled n=8), +0.39% to +0.56% at 128/GPU (99-150 us,
  pooled n=8), mostly in GPU time (medium confidence). The a2a kernels: ~40 us on one node (CLAIMS 191); a kernel-only
  sweep to 32 GPUs gives ~59 us/step without EPLB stats, up to ~178 us back to back with them (CLAIMS 200,
  diagnostic); serving does not show the stats-path growth. The mask-in-local-memory explanation was tested and
  refuted for the stats path (shared-memory variant: +3.21 vs +2.96 us; no-stats +0.69 vs +0.85 us; CLAIMS 205).
  "< 1% per step" is not claimed under either rule. (review/diag/steady_cost_analysis.md)
- Offline-EPLB kills on v5-final: rank 1.28 s mean (1.23-1.37), node 1.62 s (1.51-1.72), n=5 each, all pass
  (CLAIMS 202, THIN-CELLS 8302191, CLAIMS 213). Victims rank 5 / 22, node 1 / 5: position moved
  nothing beyond the run-to-run spread.
- Two losses (5 -> 9 no EPLB, 13 -> 17 online, 5 -> 9 and 22 -> 26 offline), n=2 per mode, all pass (THIN-CELLS 8302191,
  CLAIMS 213; offline RECOV3 8361282, CLAIMS 235): server first / second no EPLB 1.73 / 2.04 s and 1.65 / 1.69 s,
  offline 1.32 / 1.55 s and 1.36 / 2.18 s, online 0.82 / 1.77 s and 1.14 / 0.85 s (second from its detect).
- Replication (CLAIMS 203): 2 more rank and node kills each for no EPLB and online on v5-final, all 8 pass. The pre-fix build's 3.2 s
  no-EPLB node kill was 2.94 s by server timestamps vs 2.42-2.81 s on v5-final: partly polling, partly server-side.
- No-EPLB remap is slow (~0.8-1.0 s): 0/12 fast since the host-copy fixes vs 15/36 before (Appendix I.1, no CLAIMS
  row); whether the fixes cause it is not established.
- Not tested on v5-final: CFT (needs driver >= 615; runs used driver 580), two frozen ranks at once, host death,
  NVLink faults. Not recoverable: rank 0 or its node, a node plus its ring-copy node.
- Benchmark settings disclosed in the doc (forced MTP acceptance, all requests at once, exact 1k/1k, pinned tactics).
- Reviewer Q&A prep: review/diag/redteam_methods.md section (b).
- Follow-up diagnostics (explain, not fix): review/POST_FINAL_ANALYSIS_PLAN.md.

## Decisions for you

1. Option B (driver-allocated host memory): recommend not adopting; documented as the next step. No gain at full
   scale, +128-170 s startup, not shown immune to compaction (CLAIMS 195, 196).
2. PR stack: your audit before any push (see PR status below).
3. Deletions: review/WORKSPACE_CLEANUP_PLAN_2026-10-25.md section 7 (41.4 GiB; +24.8 GiB if no more C++ builds of
   this stack). 72 stale items were already moved to archive/TRASH; nothing deleted.

## PR status

- Local only, not pushed; backup branches kept.
- Product code = v5-final plus PR cleanups (moved/inlined helpers, dead code, license headers), reviewed hunk by
  hunk as behaviour-equivalent (review/V5PR_EQUIVALENCE.md). The restacks did not change the final tree.
- GPU smoke on its code: PASS (FT-off serve; online rank kill 0.85 s to first step, script-observed 1.21 s; CLAIMS 199).
- Per-commit testing (review/diag/v5pr_per_commit_table.md) found and fixed:
  - 2 commits that did not work alone. Restacked: two commits squashed, ring pieces moved.
  - 2 misplaced tests, moved.
  - 11 of main's unit tests broken by our new gather argument / live_ranks field. Fixed in the tests only.
- Final: v5-pr = a23efa7228 (local; same code as 85f781942d, two commit messages corrected). Product code vs the measured stack: 0 bytes changed; 4 test files changed.
  Every commit's tests pass, except:
  - one env-only failure (long pytest tmp path; passes with a short one);
  - main's MoE kernel matrix (test_moe_module): one CUDA illegal memory access in a single-GPU dense NVFP4 GEMM
    test (8293032), unreproduced in 16 reruns including main's code (8300459, 8301335); FT involvement not shown.
    test_moe_comm passed 41/41.
- A sweep vs main found no other regressions. The remaining failures also fail on main.
- Open: red-team PR findings (review/diag/redteam_pr.md). Separate PRs first: the burst-release rule and the mpi4py
  bump (needs an image bump). Follow-up v5-pr-fixes (59eee8140a, local; same tree as
  the tested 2673a630c4): 3 fixes + 1 test commit off the measured path; unit tests + GPU kill smoke PASS (job 8282693); merged with or right after the stack (your decision, ~10:20 PT).
