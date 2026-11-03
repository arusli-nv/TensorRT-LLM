# WideEP fault tolerance: owner review summary (2026-10-19)

Code: branch `wideep-ft-dev` 6e8657d0be (final design = final-design3 2945b0f5bb + test fixes 2e19c3b6d2, 6e8657d0be).
Review fixes on `fix-review` (12 commits on top; the first 4 built, b18f0b64ed). PR stack: review/PR_PLAN.md. Mechanisms
in detail: mvp/DESIGN_DOC.md. Every number below cites its mvp/CLAIMS.md row (#n); runs, scale and n are in that row.
Evidence vs code: all GPU evidence is from 2945b0f5bb or earlier. fix-review (M1, M2, L1, F1, L2/L4/L5, staging
quarantine, count trap; request-lifecycle fixes in progress) has unit tests and gtests only so far.

## 1. Final design
A wide-EP MoE job (DeepSeek-R1 NVFP4, DEP32 on GB200) is meant to keep serving when a non-zero rank or a non-0 node
dies or hangs, on the survivors' existing CUDA graphs. The FT layout has 320 EPLB slots in total, 10 per rank: online
EPLB runs its standard 288 (9 per rank) plus 32 ghost slots (1 per rank); the static layout is 256 experts + 64 extra
slots (8 + 2 per rank). Ghost slots take no memory while healthy (CudaVirtualMemoryManager chunks, scratch mapped
behind them); the failure domain is the node. Lost experts go ghost-first, spread evenly (minimize the max extra
experts per GPU); when ghosts run out, a slot of the most-replicated expert is taken. Load-aware placement (hottest
first) is designed, not coded. Without online EPLB each node keeps host copies of the next node's experts (ring),
staged through the idle all-to-all workspace; with online EPLB, EPLB's full host copies are the source. Healthy and
recovered serving use the same graphs: the all-to-all kernels read a rank mask and an abort record at run time, each
sender writes fixed receive slots, and CuteDSL/Cutlass MoE read them in place. A coordinator (the recovery authority
beside rank 0, ZeroMQ + HMAC) runs Propose / Ready / Install / Commit; ULFM detects deaths and shrinks the host
communicators. Late writes: the lost rank is fenced (SIGKILL; PMIx kill if it cannot run) before survivors resume, and
a dead sender's receive slots are not reused for serving; but on the built wheels ring staging reuses them during a
restore. fix-review (unbuilt) keeps staging off every byte a lost rank may write, including a rank lost mid-staging,
and splits weights into pieces so small workspaces fit. Hangs: fenced after Open MPI's detector (~5 s, frozen process)
or the 30 s hang rule (alive but stuck; section 3). Requests: survivors pause during recovery; uncommitted steps are
discarded and re-run; a request that delivered no tokens re-runs once on a survivor and fails if hit again; streamed
requests on the lost rank (almost all its in-flight requests) fail with a retryable error. /health: 200 for `healthy`,
`degraded`, `unprotected` (authority not answering); 503 for `starting`, `recovering`, `terminal`, `unknown`; the body
carries the authority's status.

## 2. Not in it (this phase)
- Phase-B stream continuation (a lost rank's streamed requests continue on a survivor): parked.
- Intended fail-closed (the job ends): host / PRRTE-daemon death (#105), rack / NVLink-domain loss, rank 0 / node 0 /
  coordinator loss (an authority lost while serving leaves the job "unprotected", no restart). Known gaps: with a
  PRRTE backport the job survives a daemon loss, then MPI_Finalize hangs (#111); a rank-0 freeze left 27 ranks hung in
  Shrink (#69, hang detection off); exit after a recovery is source reading only (SHUTDOWN-R1 not run); no evidence
  rows for rank-0 or install-time loss. A red-team of the fail-closed claims is running.
- In-place receive slots for DeepGEMM, TRTLLM-Gen, CuteDSL Rubin/BF16/FC12, Marlin (compact copy under FT): next phase.
- FT as one launch switch: designed, not coded (FT_ONE_SWITCH.md); config stays a file + env until design acceptance.
- MegaMoE / fused-comm backends and CFT counted writes: FT refuses them. FT forces the fence dispatch path where main
  would use CFT (driver >= 615, <= 128 tokens/rank); that cost is unmeasured (#87: our driver has no CFT).

## 3. Evidence so far (R1 NVFP4 DEP32 unless noted)
Final cost, kill-matrix and correctness runs on the final wheel (build-final-2945b0f5bb) are in progress; the
verdict rule is fixed in advance (COST if the CI excludes 0; noise if within the FT-off A/A spread; else unresolved).
- Ghost layout cost decomposed (#110, FT off, 128/GPU, n=4): 320 vs 256 slots +6.16% steady step (CI 4.55..7.76) =
  autotuner tactic choice +5.04% (CI 3.08..6.99) + load-balancer route path +1.08% (CI 0.13..2.03); empty ghost
  slots -0.01% (CI -0.57..0.56). Fixes A (tune with loaded slots) and B (skip the route kernel) follow; fix A's
  effect is not yet measured.
- FT on vs off: the final config's cost is being measured now (packs on 2945b0f5bb). Earlier wheels, 128/GPU steady
  step: +2.09% (CI 1.72..2.45, #104) and +1.14% (CI 0.37..1.91, #106), both CIs excluding 0; #112 (288 slots, no
  ghosts, pre-final wheel, n=4) +1.93% (CI -1.68..5.53), unresolved; 32/GPU -0.27% (CI -1.37..0.82). Launch pieces
  alone (ULFM ob1/TCP + detector, FT off): +0.32% (CI -0.76..1.40, n=2, #98).
- Profile (#113, nsys, 2 groups x 4 ranks): one fixed FT cost in all 8 pairs, the MTP layer's receive-slot copy,
  +122 us/step (~0.45% of a 25.9 ms step); the final wheel removes it (MTP in-place slots). The 32-rank follow-up
  estimates another ~+90 us/step (~0.35%: combine floor, in-graph idle, between-replay gap), partly
  profiler-sensitive (DECISIONS 18:20, 18:55); the unprofiled final packs are the measure.
- Integration (V3-Lite NVFP4 EP4): 10/10 on final-design2 (#114: rank lost idle / in flight, sync / overlap, two
  ranks, loss during recovery, restore into ghost slots static and online, hung rank fenced, authority lost) and on
  the final wheel (#115, with gtests 34/34 and all unit groups; test_moe_comm's rank-0-masked failure was a test bug,
  fixed in 6e8657d0be, GPU rerun pending).
- Hangs: a SIGSTOPped rank is fenced and survivors resume ~5 s later (#102, #109). 30 s threshold: 10 s falsely fenced
  a 45 s GIL hog, 30 s passes the negative controls (#103, #109); its R1 positive case (alive native threads, no
  progress) is pending (kill matrix "stop" row). Detector false-positive evidence is thin (none in ~6 runs, #63).
- Recovery time on the final ghost design: unmeasured (kill matrix queued). Proxies with borrowed memory: survivors
  resumed 0.78 s (rank, #79) and 1.25 s (node, #80) after the interruption, n=1 each. Ring host-copy restore has no R1
  restore-time number; the gate (<= 2 s per node, DECISIONS 16:00) is unmet so far.
- Node loss with online EPLB (1 ghost per GPU): by the code, lost experts beyond the 28 surviving ghosts take
  replica slots; if the survivors cannot place every expert or map ghost memory, the round aborts and the job ends
  (fail closed; on built wheels survivors could disagree on that, fixed by M1 on fix-review). Untested on GPU at R1:
  kill matrix row node1 on online EPLB (KILL-FINAL-ON, dep32-final.yaml; queued).
- Exact tokens: survivors matched the healthy run after a rank kill (#95: n=2, static 288, one kill timing, a few
  requests, routing oracle from startup; also #100); EXACT-FINAL pending. After a real loss batch sizes change, so
  exactness holds only for batch-invariant kernels.
- Late writes (#100): abort delayed 10 s after the victim's memory freed: no Xid; teardown window unmeasured (Q1).
- Earlier recovery evidence (older designs): #1-#8, #12, #16, #18, #53, #72, #79, #80, #94-#96, #99.

## 4. Changes to main that owners must review
Unless a row says FT only, these run with FT off. FT-off identity to main is unmeasured (noft-main n<=2, #112); an
identity pack is planned.
| Change | What it does to main | Owner area | PR |
|---|---|---|---|
| Ghost memory on CudaVirtualMemoryManager | one manager tag per slot piece, unmaterialized while healthy, scratch-swap configurator, graph-safe mapping, free() syncs before unmap | MoE + runtime (VMM) | 12 (wired in 25) |
| Planner unusable-slot mask | one input to doReplication/doPlacement; all-usable mask = planner unchanged; keeps EPLB off unbacked ghost slots | MoE / EPLB | 8 |
| loadAndPublishPlacement refactor | main's EPLB update split into plan + one shared slot-load/routing-publish tail that FT's install also calls; same calls, same order | MoE / EPLB (C++) | 7 |
| Active-slot autotune profile (fix A) | GroupedGemmInputsHelper profiles only slots holding an expert; keys/shapes unchanged; zero empty slots = same profile | runtime (autotuner) + MoE | 9 |
| Route-kernel skip (fix B) | one-slot placement: NVLinkOneSided dispatch maps expert -> slot from a device table, route() skipped; gated on FT on fix-review (F1) | MoE + runtime | 15 |
| MTP / Cutlass in-place slot gather | Cutlass expandInputRows reads receive slots (MoeInputSourceSlots); rejected combos: LoRA, scales, W4AFP8, min-latency | MoE + runtime | 10 |
| Launcher | trtllm-llmapi-launch ends by its worker's signal (PRRTE aborts at the second non-zero exit); FT mode only | runtime + infra | 5 |
| ULFM transport | ULFM Open MPI (stock image has it, off until `--with-ft ulfm`), mpi4py 4.1.2, host collectives on ob1/TCP, Open MPI's in-band detector | runtime + infra | 4, 16 |
Also main-side and small: fake moe_a2a op args + shared EPGroupHealth (2), EPLB repacked-layout slot loading (3), KV
Manager V2 step rollback (6), host-sharer expert subset (11), chat stream error event (1); ADP routing answers an
out-of-range attention_dp_rank with an error response (was IndexError, or a silent wrap for negatives).
Not gated, in FT-off hot paths, unmeasured in isolation: a never-taken expert_to_slot branch and a per-token
`__shared__` routed-id store in the alltoall dispatch kernels, larger kernel parameter structs, one per-row select
in Cutlass expandInputRows, an `all_of` over experts per non-graph routing call, and a per-iteration mutex/notify
in the EPLB worker.

## 5. Open owner questions
1. Late writes: FT's guarantee is fence-before-resume plus never reusing a dead sender's receive slots (#100; ring
   staging breaks it on built wheels, fix unbuilt). Unmeasured: NVLink writes in flight during CUDA context teardown
   after SIGKILL. Does process exit idle its NVLink channels before teardown returns; is there a bound on in-flight
   posted writes; can an exporter revoke one importer's mapping? Accept fence + quarantine as Phase 1's guarantee?
2. ULFM dependency + one-switch launch: ULFM Open MPI + mpi4py >= 4.1 in CI/release images, ob1/TCP host transport,
   an experimental detector. Accept ULFM as the first runtime adapter, and the one-switch launcher
   (FT_ONE_SWITCH.md: `mpirun --with-ft ulfm -x TLLM_FAULT_TOLERANCE_MODE=1 ...`, authority started by rank 0's launcher)?
3. The section-4 changes to main: accept each in its owner's PR (ghost memory as RFC with its wiring or alone)?
4. Healthy-cost target: FT on vs off on the identical deployment within noise, < 1% (aim ~0) on throughput and
   TTFT/TPOT p50/p99 at 8/32/128 per GPU. Accept this gate (final numbers pending)?
5. Both heartbeats: native C++ UDP (only liveness signal of a GIL-starved rank, #97, #103) and Python (carries
   `waiting` for the hang rules and the host/pid that admits native datagrams). Keep both, accept UDP, and take a
   graph-safe device-row reader into alltoall_watchdog.py?
6. FT-off behaviour changes from the review: F2 (EPLB host-copy madvise thread on every online-EPLB run) and F3
   (chat streaming ends with an SSE error event instead of a broken connection; harmony/responses error events and
   completions' double [DONE]: fixes in progress). Gate on FT or accept on main? (F1 is gated on FT on fix-review.)
7. Scope: agree that phase-B continuation, host/daemon death, rack loss and rank 0/coordinator loss stay out of
   Phase 1 (fail-closed, with the section-2 gaps), and other backends' in-place slots come next phase.

## 6. PR plan and review status
- review/PR_PLAN.md: 27 PRs over wideep-ft-dev (89 commits on main fc52f0adee). Part A, 12 main-side PRs with no FT
  dependency (send first: 1, 2, 3); Part B, 15 FT-core PRs. Restack splits/squashes listed there; nothing restacked yet.
- Correctness review (review/CODE_REVIEW_2026-10-19.md): 0 high, 2 medium, 6 low, 3 FT-off changes. On fix-review:
  M1 (online-EPLB recoverability from rank 0's placement), M2 (aborted source gets a masked source's rows), L1 (fence
  before Propose), F1 (route skip only with peer-loss recovery): built (8079634, gtests/units pass), integration
  pending; L2, L4, L5 (Python, overlay-tested); L3, L6 left with reasons.
- Audit of today's records (review/AUDIT_2026-10-19.md): corrections adopted into CLAIMS.
