# Post-final analysis plan (explain, don't fix)

User decision (2026-10-25 ~07:20 PT): the presented data is final (v5-final 27442bd91d; CLAIMS 193-197). No more
fixes. After the package is delivered and GPU slots are free, run diagnostics that explain the open items. Results
go to CLAIMS as DIAGNOSTIC rows and to a doc appendix. They do not replace the presented numbers.

Order: item 2 first (cheapest decisive), then item 1. Smallest decisive test first. Each needs a decision rule written before data.

1. Offline EPLB at 32/GPU, -2.0% (-0.1% and -1.0% on earlier builds).
   - Question: is it offline FT's ~390 GB/node of host copies, or run-to-run noise?
   - Existing data (review/diag/offline32_analysis.md): FT-on arms match earlier builds; this batch's FT-off arms
     were fast; 12-pair mean ~-1.0% = ~0.45% steady + ~0.6% stalls far from rank 0's prefill (mostly FT on).
     Add iteration logs from all ranks to name the stalling rank; >= 8 ABBA pairs on one node set.
   - Ballast A/B (FT off + 390 GB memfd ballast) drafted (review/drafts/ballast_ab_*) and PARKED: existing logs
     already rule out occupancy as the build-to-build cause, and same-node noise (+-2-3%) makes it likely
     inconclusive at 2 x 16 nodes x 1.75 h.
   - Test instead: one 16-node job, offline 32, FT on/off ABBA, per-step logs from ALL ranks.
   - Rule: attributed if one non-rank-0 rank's context step overlaps >= 80% of rank 0's far stalls.
2. Steady per-step cost, +0.3% to +1.0% (trimmed mean); ~0.3% is explained by the all-to-all kernels.
   - Question: where is the rest at 32 GPUs?
   - Existing data (review/diag/steady_cost_analysis.md): fixed ~84 us/step with EPLB, in GPU time; a2a kernels
     explain ~34-48 us at EP4; ~36-50 us unexplained.
   - Test: no-server 8-node a2a microbench sweep (tools/a2a_microbench_ep.sh), EP 4/8/16/32, 32/64/256 tokens/rank,
     off/ft alternating, graph + PDL, on v5-final.
   - Rule: EP32 at 64 tokens >= 1.3 us per round trip -> the a2a kernels explain it all; ~0.7 us -> the rest is
     outside the a2a (then a same-node serving pair with a plain-a2a FT arm).
3. Startup stall residual: FT 2.7 / 5.7 s vs FT off 2.2 / 3.2 s (CLAIMS 187, n=1 each).
   - Only if 1 and 2 finish early: more pairs to see whether the gap is real.
4. Main's own compaction exposure (online EPLB host copies), for the upstream note.
   - Only after 1-3. FT-off used-node A/B with vm.compaction_proactiveness 20 vs 0.

Results so far (2026-10-25):
- Item 2 (A2A-EP2 8282103, CLAIMS 200): verdict C, partial. EP32 T=64: +1.0 us per round trip without EPLB stats, +3.0 us
  with stats (grows with EP). Likely mechanism: the FT rank-mask register copy spills to local memory
  (review/diag/a2a_ep_scaling_coderead.md).
- Follow-up fix candidate (product change, user decision): keep the mask in registers or __shared__. Verify with
  cuobjdump -res-usage (24 B stack) and one EP32 stats bench (predicted +3.0 -> ~+1.0 us), then a same-node serving A/B.

## Follow-up fix candidate (from A2A-EP2 + code read, CLAIMS 200/200b; not in v5-pr or v5-pr-fixes unless the user says so)
- FT rank mask without the local-memory spill: walk the mask's set bits with a compile-time word index, or keep it in
  `__shared__` (moeAlltoAllKernels.cu:238-248 and the serial stats loop :679-688).
- Verify: (1) CPU: `cuobjdump -res-usage` shows STACK 24 for the MASK=1 instantiations (64 today); (2) one EP32 bench of
  the stats variant (tools/a2aep_bench.py --eplb, EP 8/32, T=64): predicted delta +3.0 -> ~+1.0 us (if it stays +3.0 with
  the spill gone, the mechanism is wrong); (3) then a same-node serving A/B (online and no-EPLB, FT on vs off).

## Future fixes (user, 2026-10-25 ~11:15 PT: finalize now, explain open items, note fixes for later)
- Offline far-from-prefill stalls: STALL-ATTR 8282497 found them spread over other ranks' out-of-phase prefill, with FT
  off as often as on, so this is not an FT fix. A general attention-DP prefill-phasing improvement is out of FT scope.
- FT rank-mask local-memory spill in the a2a kernels (CLAIMS 200b): see the candidate above.

## Healthy-serving host-collective hang (CLAIMS 204, review/diag/repl_hang_8285727.md)
- Soak: FT arm on its ob1/TCP launcher vs the same FT arm on the default transport, healthy serving, several hours
  each, same nodes; count stalls (no progress > 60 s) per hour per arm. Decides whether the transport is the cause.
- Fix candidates for code owners (not done):
  1. treat "every live rank waiting in a host collective with no progress for confirm_s" as a stall in HangMonitor;
  2. check for a rank-0 loss before comm.Shrink() in peer-loss recovery (end the job at once instead of hanging in Shrink);
  3. end the job through the runtime (authority / mpirun kill) instead of per-rank MPI_Abort when FT is on.

Update 16:20 PT: MASKSPILL bench 8289096 (CLAIMS 205): REFUTED for the stats path (new +3.21 vs ctrl +2.96 us at EP32
T=64, stack 24 vs 64 B verified); no-stats PARTIAL (+0.69 vs +0.85 us). The mask spill is not the cause of the EP
growth; the candidate fix is dropped. Next idea, if pursued: the serial per-peer stats loop itself (parallelize over
lanes/warps), measured the same way.

## Update 2026-10-25 ~16:30: mask-spill candidate refuted (CLAIMS 205)
- Removing the spill (STACK 64 -> 24) left the EP32 stats delta at +3.21 us (CTRL +2.96): the spill is not the cause
  of the EP-scaling stats-path cost; it accounts for at most ~0.16 us of the no-stats delta. Drop it as a fix
  candidate; the stats-path growth is still unexplained (next: per-peer cost of the serial EPLB stats loop itself,
  e.g. its remote stores, with ncu or a variant that skips the stats writes).

## Update 2026-10-25 ~21:55: a2a FT cost explained by ablation (CLAIMS 210)
EP32 T=64: with EPLB stats (+3.00 us) the per-peer rank-mask check in the stats loop is 2.13 us; without stats (+0.88 us)
the release-flag read before the fence (0.45) and the lane-parallel mask checks (0.49). Future fix candidates, measured
on the experiment branch a2a-ablate d47a3bcf88 (not on v5-pr): S1 ballot stats loop (cuts 1.62 of 2.13 us, keeps FT),
F8 8-byte flag load (0.14-0.18 us). Need a FT-correct design + kill tests before any use; the lane-check cost (L) has no
candidate yet.
- 2026-10-26 ~00:00: /health status cache (serve/fault_tolerance_status.py max_age_s=1.0) hides or delays "recovering" and
  returns stale 503 after the commit (S2 8299473 s2-rank13 FAIL k1_health; review/diag/health_recovering_gap.md). Fix
  candidate: max_age_s ~0.05 s or 0, cache only failed answers; needs a kill test reading /health at 0.1 s.
