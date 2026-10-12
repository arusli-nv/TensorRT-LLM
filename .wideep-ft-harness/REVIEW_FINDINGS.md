# Pre-PR review findings — tracking

Legend: [ ] open, [x] done, [~] documented/deferred with reason.
Round 1 (2026-10-11): all items done except those carried into round 2 below.
Round 2 reports: review-r2/{design,kernels,python}.md.

## Blockers
- [x] B1 recovery waits unbounded (Shrink before Stopped; rearm syncs) -> recovery runs under the executor HangDetector
- [x] B2 PeerFailureWatch.stop() not idempotent (masks RecoveryError)
- [x] B3 ULFM test must skip without real ULFM (distro Open MPI 4.1)
- [x] B4 authority/mpi_survivors/a2a tests missing from CI lists (l0_cpu.yml, GPU list)

## Kernels
- [x] K-S1 op requires expert-id sanitization with a descriptor
- [x] K-S2 gather zeroes invalid non-expert-id rows
- [x] K-S3 aborted combine withholds readiness
- [x] K-S4 gather reads the abort request once per block, no status write
- [x] K-S5 abort copy vs CUDA_DEVICE_MAX_CONNECTIONS=1 -> reject in create()
- [x] FT-off SASS parity: 150 kernels opcode-identical to main (sass-diff/opcode-parity.txt)
- [ ] K-S6 healthy-path gather copy (~1 ms/step EP64 decode): measure; payload-major alias alternative
- [x] ep_group_health extra_attrs fallback restored
- [x] K-N watchdog comment, MOE_DEVELOPER_GUIDE section
- [x] K-N alias annotation comment, combine capacity check dropped, status documented diagnostic-only
- [~] K-N snake_case locals match the file; reset_local_workspace mirrors init without its barrier

## MPI / session
- [x] S1 ULFM capability gate always true with mpi4py 4 -> real check
- [x] S4 sync tasks to a degraded session loop forever -> fail them
- [x] S5 Abort(0) path: flush TRT-LLM logger, exit code, doc, test
- [~] S6 free revoked communicators (two comms leaked once per recovery; freeing revoked comms is implementation-defined)
- [x] FT-off: _shared_barrier tolerance gated on recovery

## Executor
- [~] H1 interrupted post-commit response gather skips rest of iteration (no-op tail for supported profile; documented; recheck with V2)
- [x] S3 create() rejects: lm_head_tp_in_adp, hybrid/Mamba managers, PEFT, CUDA_DEVICE_MAX_CONNECTIONS=1
- [x] S7 nvtx_range moved onto _publish_responses
- [~] S8 gather_all_responses: only KV-cache estimation sets it; documented terminal
- [~] S2 n>1: is_final is all-sequences; one error ends the client request — comment only
- [ ] stats/perf events of a discarded step; penalties/seed/stop-word state (document per-request limits)
- [x] remove unused installed_generation; registration logs its wait
- [~] client env helpers in leaf module (zmq is a core dependency; imports are lazy)
- [ ] device timeout vs ULFM detection documented
- [ ] create() validation before model load (UX) — consider
- [ ] tests: stop idempotent + RecoveryAborted surfaces; stuck shrink bounded; loop-level hook placement; real ULFM revoke unblocks; KV revert GPU round trip; serve() errors after departure

## Series / PRs
- [ ] drop 44eded2a5c (admission) from the series
- [ ] split: ADP-absent-ranks refactor, ConfigurableMoE memo fix, unrelated fixes
- [ ] KV manager V2 rollback (DeepSeek/Kimi default to V2)
- [ ] user doc + integration test + experimental label
- [ ] verify the documented trtllm-serve launch (docs example) on EP32
- [x] single-node EP4 integration test (idle + in-flight) passes; added to l0_gb200_multi_gpus post_merge
- [ ] mpi4py.futures threads print MPI_ERR_PROC_FAILED tracebacks at the loss and at shutdown (cosmetic; part of the futures/ULFM gap)
- [ ] test_rpc.py::test_incremental_task_async hung once (7 s before) — rerun to classify
- [ ] gpu_moe: failures + PMIx step kill from the trapping child (child env now isolated) — rerun verbose
