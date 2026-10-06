# WideEP failure behavior

**Scope:** current error reporting, teardown and explicit restart—not N−1 recovery.
Tested stack: DeepSeek-R1-0528 NVFP4, EP32/expert TP1/attention DP/PP1,
288 static slots, GB200, R580, Ray 2.55.1, NVLinkOneSided fence,
decode graphs `[1,2,4,8]`; CFT, rank-mask FT, overlap and autotuning disabled.

| Case | Detection → client → teardown | Explicit restart on the same GPUs |
|---|---|---|
| Healthy | Weights, communication, warmup and capture initialize → serve | Fresh workers repeat initialization; no live engine state reused. |
| Middle worker SIGKILL | Ray actor DEAD → sticky executor fatal cause → tested waiting/new requests received `EngineDeadError` → shutdown workers/RPC. Peer Gloo calls can block or fail; the crash watchdog can kill peers. | Observed working; workers, contexts, communicators, KV and graph executables are recreated. Weights reload; warmup and graph capture rerun. |
| `host_collective_abort` | New lean reproducer awaiting physical qualification. | Pending; no result claimed. |
| `fence_round_mismatch` | New lean reproducer awaiting physical qualification; requires native dispatch/combine timeout evidence. | Pending; no result claimed. |

**No automatic return to serving was observed.** Ray-service survival is not model
survival. SIGKILL victims cannot acknowledge. No committed survivor agreement,
communicator shrink or unchanged-graph recovery was demonstrated. CUDA probes
prove only the operation that replied; missing replies remain unknown.

The branch adds failure-only propagation fixes: forward worker errors, preserve
the first cause, fail pending results on confirmed Ray death, bound RPC close,
and preserve `EngineDeadError` at the LLM entry point. Before the Ray fix, four
of five streaming attempts timed out; afterward ten consecutive attempts passed.
MPI/PMIx canceled the model step despite `--kill-on-bad-exit=0`; Ray is the sole
runtime maintained here. This does not rule out a future qualified MPI adapter.

**Restart evidence:** job 7736326 used the same 32 GPU UUIDs with new workers.
Client error: 0.267 s. Historical Ray-service launch → readiness: 324.1/264.2 s; independent
resource cleanup: 48.8/47.4 s. Rank-0 loading: 106.1/54.3 s;
initial warmup: 84.4/84.8 s; final warmup: 9.7/7.8 s;
initial capture: 1.75/1.66 s; final capture: 1.72/1.71 s.
These intervals overlap; caches were uncontrolled, autotuning disabled.
The job’s final cross-host timing guard **failed**; same-host post-hoc analysis
supports the facts above, not a formal qualification pass.

Historical evidence: `.wideep-ft-runs/process-restart/` and checkpoint
`6dd6a1fdbd`; raw logs remain uncommitted. The rebased lean runner is not yet
physically qualified. CFT safety and survivor recovery remain later work.
