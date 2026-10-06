<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WideEP failure behavior

Runs used `764bde9d6d`, based on `cb03225c13`, with branch propagation fixes.
DeepSeek-R1-0528 NVFP4, EP32/expert TP1/attention DP/PP1, 288 static slots,
GB200/R580/CUDA 13.4, Ray 2.55.1, LLM API, NVLinkOneSided fence, decode graphs `[1,2,4,8]`.
CFT, rank-mask FT, overlap and autotuning were disabled. Target rank was 16.
These runs measured detection, teardown and explicit restart, without N-1 recovery.

| Fault | What happened | Client error / fresh restart readiness |
|---|---|---|
| SIGKILL between requests | Ray confirms actor death → saves the fatal cause → next request gets `EngineDeadError` → workers/RPC shut down. | 0.164 s / 216.6 s; PASS |
| SIGKILL during a stream | First nonfinal output → verified kill → waiting stream gets `EngineDeadError` → teardown. | 0.156 s / 209.2 s; PASS |
| `process_group_destroy` | Local WORLD teardown → request broadcast loses its group → rank-crash watchdog self-SIGKILL after 10 s → peer Gloo failure → RPC error → `EngineDeadError`. | 10.3 s / 205.1 s; PASS |
| `fence_round_mismatch` | Verified round +2 → dispatch completion-flag timeouts → CUDA launch failure/KV-manager error → RPC error → `EngineDeadError`. | 291.5 s / 280.4 s; PASS |

WORLD teardown includes registered CPU/CUDA groups. Communication faults are synthetic, not
physical link/device loss. Gloo `abort()` left inference working, so that attempt failed.

After verified process/GPU release, new workers on the same GPU UUIDs reload weights and
recreate contexts, communicators, KV and graph executables. Warmup and both capture passes rerun.
Filesystem/compiler caches and Ray services may persist. No automatic serving restart,
communicator shrink or committed survivor agreement occurred. A SIGKILL victim cannot acknowledge.

CUDA probes succeeded on 31/31 peers after process kills. After the fence fault,
31 failed and one was missing. Survivors were not preserved. Probe success does not
prove transfer drain or safe recovery.

Healthy startup follows workers → weights/communication → warmup/capture → inference checks.
Startup/restart readiness took 280.5/175.5 s. Rank-0 seconds, initial start / fresh restart:

- Weight loading 116.5/64.0
- Initial engine warmup 83.5/23.1
- Final engine warmup 11.6/6.5
- Initial graph capture 1.62/1.62
- Final graph capture 1.69/1.63

Capture is included in warmup. Intervals overlap. Communicator init has no separate timer.
Caches were uncontrolled. Greedy, fixed-seed Paris/Rome checks are not an exact-token reference.
Readiness uses parent monotonic receipt time. Client delay uses its local injection-RPC clock.
Allocation/Ray startup are excluded. The nominal 300-second fence timeout depends on the GPU clock.

Branch fixes preserve fatal causes, fail pending Ray results and bound RPC close.
Streaming timed out in 4/5 earlier attempts, then passed 10 consecutively.
The current runner has one new pass per case, without repeated qualification.
Startup interruption released all GPUs. All 32 pre-constructor identities matched live workers.
Ray was chosen because the tested MPI/PMIx setup canceled its step despite `--kill-on-bad-exit=0`.
Evidence is in `.wideep-ft-runs/lean-characterization/physical-{7738662,7739144,7740233}/`.
Failed attempts are retained. Raw logs are uncommitted. Identity, evidence and restart checks
can carry over to CFT. Transfer injection and safety need fresh qualification.

RPC review: submission confused request-local serialization errors with fatal transport
errors and missed `zmq.ZMQError`. Four targeted cases fail before the correction.
Restricting the fatal catch to RPC/ZeroMQ errors fixes classification without changing healthy sends.
