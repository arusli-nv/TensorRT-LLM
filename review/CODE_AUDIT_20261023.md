# Code-verified audit of v5-final 7e8dfb37df (monitor, 2026-10-23 ~09:30 PT)

Read-only audit (3 forks; monitor re-checked A15, A18, the ADP burst commit, the detection-only graph refusal, the key
length and the 2x30 s deadline). Paths relative to /tmp/claude-161238/wt-split. PLR = peer_loss_recovery.py,
LB = moe_load_balancer.py, FD = docs/source/features/wide-ep-fault-tolerance.md.

## Affects benchmark numbers (and how the finals cover it)
| Item | Effect | Finals coverage |
|---|---|---|
| FT forces ob1/TCP (communicator.py:959-962) | ~8 host collectives/step slower than UCX, in FT-on vs off | FT-off ulfm arm (2 groups) separates transport from FT |
| ADP burst-release rule 17a047ea76 (py_executor.py:7002-7042), not FT-gated | changes scheduling/TTFT vs main for ADP-balance users | main parity arm measures our FT off vs main; disclose as "gate-fix effect" (Fig 4); own PR |
| mpi4py 3.1.5 -> 4.1.2 (docker/common/install_mpi4py.sh) | unmeasured vs stock main | every arm (main incl.) installs mpi4py 4 via install_wheel.sh: no delta inside the finals; disclose |
| Healthy path: rank mask + abort polling + expert_to_slot | CLAIMS 173 ~0.86% @32 no-EPLB; online ~0 (n=1) | headline itself |
| KV lent to remap / restore into ghost slots | lowers post-kill KV capacity | inside post-loss throughput; disclose |

## Feature doc fixes (doc-only; no rebuild)
- FD:78 ghost slots: "hold no device memory while healthy" -> only chunks wholly inside empty slots are unbacked;
  boundary chunks stay backed, plus a per-process scratch (1+2+4+8 chunks). Drop stale "accepted without a warning"
  (LB:1260-1267 warns, A14).
- FD:72 key: any hex key >= 16 bytes (client.py:38,69), not exactly 64 hex chars.
- FD:79/188: up to ~2x recovery_timeout_s after agreement (fresh deadline at Stopped; PLR:964-965, 1013-1014).
- FD:70 + matrix:48: detection-only mode (TLLM_FAULT_TOLERANCE_MODE=1 without authority) refuses CUDA graph capture
  (nvlink_one_sided.py:1036-1042).
- FD:142/153/244-245: a peer failure during remap exchanges re-plans (RecoveryInterrupted); the job ends only if the
  mapping/release breaks.
- FD:246-248: the same recovery's commit already tries to return remap KV; it stays lent only under overlap + steady load.
- Requirements: add "a generation sampler" (PLR:448).
- Fill the TODO numbers from the finals (FD:65, 78, 186, 188, 215, 276-277, 293, 330, 351-352, 364).

## Design doc ($R/mvp/DESIGN_DOC.md): rewrite needed
Mailbox/per-sender regions/drained-writes sections (:92-96, 156-179, 243-246, 334, 358-359, 371-372, 388-389) are the
old v2 design; recovery flow lacks the remap (:185-196; code PLR:1003-1089); "mpi_ft_detector ~5 s" is wrong (frozen
rank ~35 s); tuning keyed by loaded slots (cute_dsl_custom_ops.py:211-229); MPI_Finalize bound 30 s; fail-closed list
lacks the remap window/KV loan; numbers are v2-build data (replace with finals only).

## FREEZE_REVIEW_FIXES A-items
Fixed: A1-A14, A16, A17 (thread_local cache), A19. Open: A15 (shm mappings leak per FT-on weight reload, LB:313-315,
follow-up); A18 (FLAG_VAL 8 B with FT off, moeAlltoAllOp.cpp:238, layout delta vs main, no perf effect: disclose).

## Open limits to disclose (code-verified)
Frozen rank fenced after ~35 s (heartbeat 5 + confirm 30); two ranks hung at once not confirmed; a loss after
Install/commit or a remap mapping failure ends the job; losing a node with its ring copy node, or rank 0, is
unrecoverable; startup refusals (PLR:416-460, communicator.py:941-967); per-request rejections (PLR:507-520); spec
decoding other than one-model MTP untested (warned); NUMA-memory GPUs fall back to `empty` ghost slots; streaming error
now ends with an SSE error + [DONE] (openai_server.py, FT-off behaviour change).

## Follow-ups (PR follow-up list)
A15; split_unservable_requests on every FT-on step with arrivals (py_executor.py:6793); loaded-slot tuning key only
for Cutlass/CuteDSL; FT-off kernel instantiation not SASS-verified (moeAlltoAllKernels.cu SWITCH_BOOL 1393); split
PLR (1796 lines); worker os._exit may drop a queued shutdown ack (unverified).
- PR follow-up (design-draft agent, monitor-checked): moe_scheduler.py:266-278 docstring says the slot-table path is 'only with peer-loss recovery'; the gate is in C++ (moeLoadBalancer.cpp:660-664 routesEachExpertToOneSlot). Comment-only fix at PR time (keeps the frozen wheel unchanged).
