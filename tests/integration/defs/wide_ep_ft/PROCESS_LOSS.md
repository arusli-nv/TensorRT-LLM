<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WideEP process-loss findings

This milestone qualifies failure reporting and cleanup, not recovery.
[README.md](README.md) defines the deployment, scenario contracts and commands.
Raw logs, frozen sources, JUnit and every failed/interrupted attempt are retained
under the local ignored `.wideep-ft-runs/` tree. No retries replace failed runs.
The complete chronological report is archived as described below.

## Configuration and response paths

| Item | Configuration |
|---|---|
| Model/hardware | DeepSeek-R1-0528 NVFP4; eight four-GPU ARM64 GB200 nodes |
| Parallelism | EP32, expert TP1, attention DP with local TP1, PP1 |
| Placement | Static cyclic 288 slots for 256 routed experts, nine slots/rank, 58 MoE layers |
| Execution | CUTEDSL, NVLinkOneSided fence path; decode graphs `[1, 2, 4, 8]` |
| Capacity | Batch 8, 256 tokens, sequence 512, KV fraction 0.1 |
| Disabled | CFT, rank-mask FT, overlap scheduler and autotuning |
| Runtime | Ray 2.55.1; host driver 580.126.20; CUDA 13.4 development runtime |
| Native source | Base image `f388b7c6a5`, with explicitly hashed Python-only fixes |
| Workload | Fixed France/Italy prompts, greedy decoding, seed 2026 |

Injection requires all 32 identities, 1,856 rank/layer placement/communication
records and final graph capture. Default native crash policy remains enabled.
The 288-slot layout retains routed-expert coverage for nonzero ranks
`[1, 2, 29, 30, 31]`; this is not full-model recovery admission. Rank 16 tests
terminal reporting without relying on redundancy. The optional 512-slot layout
is a comparison, not a recovery claim.

MPI/PMIx cancels the model step even with `--kill-on-bad-exit=0`; the plain-process
control survives. Job 7647040 used Open MPI 5.0.10rc2/HPC-X and reported PMIx −61.
Ray exposes death independently of model communication, so this suite maintains
one runtime. This does not exclude a future qualified MPI adapter or prove ULFM
integration. Historical launcher source is retained in `c057cd602a`.

Native Gloo failures occur in request-count broadcast, attention-DP allgather
and response `tp_gather`. The loop wrapper can arm a 10-second crash watchdog;
survivors may self-SIGKILL. Ray service survival is not inference survival.
The failed rank never acknowledges its SIGKILL; diagnostic replies are not
membership agreement. A successful operation on an actor's existing CUDA
context proves only that operation; missing replies remain unknown.

## Failure-propagation diagnoses

These diagnoses were recorded before their fixes; original notes and paired
controls remain in the archived report and artifact roots.

**Worker RPC error forwarding and bounded close.** An empty response poll could
hide `_event_loop_error` (7647930), while root loss wedged `zmq.Context.term`
(7648681). `rpc_worker_mixin.py` forwards the terminal cause;
`rpc_proxy_mixin.py` preserves the first cause, fails waiting results and rejects
new work. `rpc/rpc_client.py` closes sockets with zero linger. Jobs
7648583/7648602 delivered errors; 7648908 closed boundedly but still timed out
before actor monitoring. Result registration precedes dispatch to handle an
immediate response. Risks are submission/teardown races, covered by focused tests.

**Streaming timeout.** Ray had no ongoing actor-death path to `_set_fatal_error`.
Root RPC could remain blocked in `tp_gather` → `gather_object` → `all_gather`,
with `_event_loop_error=None`, leaving already-waiting requests unresolved.
Job 7651980 recorded rank-16 `ActorDiedError` at 0.138 s after injection RPC
issuance, but request 4 timed out at 30 s with no fatal cause and prefix ` Paris`.
Producer clocks establish local ordering only. Four of five investigation runs
timed out; the fifth delivered an error.

Fix: `executor/ray/executor.py` subscribes to GCS actor events before reconciling
a snapshot, filters exact owned actor IDs and invokes the existing sticky fatal
handler on confirmed DEAD. `test_rpc_executor_mixin.py` holds root RPC silent
with two waiting requests and tests deaths before/after subscription, unrelated
events and shutdown during subscription. Paired job 7652444 failed both death
cases before and passed after. A review correction uses typed `ray.JobID` in
`state.actors`, checked explicitly by the test.

Risk: private Ray API compatibility and a new control-plane task. No per-request
polling, GPU work or model collective is added. Reported control-channel errors
fail closed without inventing rank death; silent subscription/snapshot/poll
stalls have no established bound. Healthy performance equivalence is unmeasured.
SIGSTOP is not a DEAD event.

**New-request cause loss.** TERM job 7652491 had the correct rank-16 fatal cause
but the next request raised `RuntimeError("LLM is shutting down")` at 0.172 s;
root-loss job 7652553 behaved similarly. `LLM.generate_async()` checked
`is_shutdown()` before submission, replacing the known cause. The user approved
a failure-only fix in `llmapi/llm.py`: raise `EngineDeadError` when a fatal cause
exists; retain the intentional-shutdown exception otherwise. No API/configuration
change. Paired job 7652706 failed the fatal case before/passed after, with
intentional shutdown passing both. Corrected rank-zero tests now require PASS;
historical strict timeout XFAILs remain recorded.

## Qualification and retained attempts

| Scenario | Passes / attempts | Maximum client interval | Frozen source |
|---|---|---|---|
| Middle worker kill between requests | 5 / 5 | 0.207, `EngineDeadError` | `9578ca36c7` |
| Middle worker kill during stream | 10 / 10 | 0.268, `EngineDeadError` | `d3823cc910` |
| Middle worker TERM | 5 / 5 | 0.210, `EngineDeadError` | `9578ca36c7` |
| Middle worker STOP | 5 / 5 | 30.0002, observation timeout | `d3823cc910` |
| Rank-zero kill | 5 / 5 | 0.190, `EngineDeadError` | `9578ca36c7` |
| Frontend kill | 5 / 5 | 0.317, connection error | `d3823cc910` |
| Healthy cold startup | 5 / 5 | Not applicable | `d3823cc910` |

These 40 accepted attempts passed consecutively within each named batch.
`regression-foundation/{stream-native-port-v3,worker-accounting-v4,
startup-stall-frontend-native-port-v3}` retain `batch.json`, `batch_result.json`,
`qualification.xml` and linked `process-loss-JOB_ID` directories.
`qualification-native-port-summary.json` reconciles every result and source.
Five native overlay modules are identical across both cohorts; only test
accounting differed. The compiled base image is unchanged.

Worker latency is injection RPC issuance to result receipt on the client clock;
frontend latency is pinned signal issuance to HTTP error on the controller clock.
All accepted runs have eight verified process/GPU checks, no new compute contexts,
zero memory delta, archived Ray logs and pinned storage removal receipts.
CUDA reply counts vary (idle 12–31, stream 27–31, TERM 22–30, root 16–31);
missing replies remain unknown. Ten streaming and five other attempts are
project qualification counts, not a repository-wide rule. Routine runs use one.
Any failure stops its qualification batch and requires diagnosis.

Failed/superseded attempts retain their original dispositions, job IDs, JUnit
and source hashes in the archived chronological report and original batch roots
under `regression-foundation/`. These include preparation errors, historical
strict root XFAILs, four-of-five streaming timeouts, interrupted qualifications,
bootstrap errors and the accounting failure below. None are replacement passes.

### Test-lifecycle fixes

**Bootstrap.** Job 7654628 could not bind GCS port 44628 and hit the 720 s
readiness bound; independent cleanup passed in 37.8 s. Handwritten port ranges
overlapped host ephemeral ports (32768–60999, then 9000–65000 on compute nodes).
A TCP self-connect/TIME_WAIT control disproved plain TCP readiness, without
identifying the actual holder of port 44628. Controls 7655074 and 7655254
validated owned metadata plus bounded GCS `CheckAlive`; the final launcher uses
native `--port=0`. Control 7655254: readiness 12.12 s, false listener rejection
2.10 s, cleanup 5.63 s. Risk is bootstrap endpoint propagation; no Ray-library
or inference changes. Evidence: `regression-foundation/launcher-control/`.

**Accounting.** TERM 7655831 delivered `EngineDeadError` in 0.221 s and exited
`srun` zero, but cleanup issued cancellation before eventual `COMPLETED / 0:0`.
The first accounting row was not retained; publication lag is likely, not proven.
`slurm_lifecycle.py` now gives an exited launcher bounded accounting progress
(up to 10 s and half the remaining cleanup budget) before exact-step cancellation.
Two CPU cases failed before/passed after; permanently active accounting still
fails boundedly. Control 7656046 passed ten natural exits without cancellation
but did not reproduce the lag. Evidence: `regression-foundation/accounting-control/`.

**Interrupted host probe.** In failed control 7684726, controller death during
cleanup left an archive writer active. Rescue overlapped deletion/receipt
publication and failed on three missing receipts (`FAILED / 1:0`); later valid
receipts do not convert that control to PASS. A late preparation probe could
similarly create storage after rescue certified absence.
`process_loss_controller.py` now names probes and waits for same-run batch
launchers and registered probes before validation/rescue, under one deadline.
It preserves active archive writers. Risk is conservative timeout when accounting
cannot establish termination. Two deterministic controls failed before/passed
after. Post-fix frozen `c60b3f6292`, job 7685333: all eight cleanup checks passed
in 2.66 s; original allocation remains `FAILED / 137:0`. Healthy 7685334 passed
(readiness 322.86 s, cleanup 46.70 s). Artifacts:
`lean-foundation/{interrupt-cleanup,probe-ownership-cleanup-control,
probe-ownership-healthy}`, `probe-race-{before,after}.xml` and
`interruption-independent-audit.json`. Late cleanup validation cannot pass.

### Native validation and compatibility

Paired native controls 7652444/7652706 validate the diagnoses above; 7652091
retains two initial waiting-request failures. RPC suite 7652490 exceeded 420 s
after 28 cases, not PASS. The incremental case passed before/after (7652630).
An immediate-reply/future-registration reproducer observes `RPCTimeout` on both
base and candidate with byte-identical relevant methods: a preexisting healthy
RPC race, not a proven cause of that timeout. That path is unchanged.
Final RPC validation 7653434: 21 focused PASS, broader suite 65 PASS/five skips;
7654132: 19 proxy/Ray/LLM PASS. Final executor 7684738: 24 PASS, including
reported subscribe/snapshot/poll errors with already-waiting requests.

Private actor API controls passed on isolated Ray 2.55.1/2.55.0/2.49.2
(7683270, two cases each); final 2.55.1 isolation passed (7684222).
These had no NVML compute contexts. Other Ray versions are not qualified
WideEP runtimes. Artifacts: `lean-foundation/ray-*`.

The lean harness passed 151 CPU cases (seven physical skips), then all seven
EP32 routine scenarios (7683477–7684453), separate from formal qualification.
All historical cleanup proofs passed rechecking against raw GPU CSV/memory deltas.
Independent interruption controls passed on eight nodes: cooperative startup
7683478 (90.49 s), controller kill during startup 7683537 (46.06 s), and controller
kill after verified STOP 7684727 (86.55 s). Original scenarios remain errors.

Additional four-node EP16 compatibility smokes, target rank 8:

| Profile | Healthy startup | Idle kill | Streaming kill |
|---|---|---|---|
| DeepSeek-R1-0528 NVFP4 EP16, 288 slots | 7683536, 302.87 s | 7683753, native error 0.161 s | 7683869, native error 0.223 s |
| Qwen3-235B-A22B NVFP4 EP16, 144 slots | 7683535, 239.98 s | 7683683, native error 0.130 s | 7683870, native error 0.253 s |

All six passed the same source, rank/layer, client and cleanup contracts.
These are smokes, not formal qualifications; artifacts: `lean-foundation/`.
No native production code changed during harness simplification.

## Startup measurements and interpretation

Fresh Ray services, API process and actors define cold startup. Readiness is
service launch to receipt of healthy completions for both fixed prompts, on the
controller clock; queue time uses scheduler timestamps separately. Autotuning is
disabled; filesystem/page/compiler cache state is uncontrolled. Native loading,
initial/final warmup and capture intervals overlap/nest and must not be summed.
Unmarked communicator/JIT costs remain unavailable. [README.md](README.md#evidence-and-startup-measurement)
defines the markers. Clean restart-to-readiness remains a follow-up.

Five frozen `d3823cc910` cold-start runs: 7655314/7655409/7655457/7655572/7655679.
Seconds below; native phases are rank zero. Full rank trees: `cold_start.json`.

| Interval | Minimum–maximum | Median |
|---|---|---|
| Service launch to healthy readiness | 280.59–329.42 | 319.24 |
| API construction | 186.90–229.79 | 223.71 |
| Model loading | 63.30–98.83 | 86.36 |
| Initial / final warmup | 84.63–85.37 / 6.97–8.65 | 84.69 / 7.12 |
| Initial / final graph capture | 1.61–1.86 / 1.64–1.73 | 1.68 / 1.67 |
| External cleanup | 44.33–48.99 | 47.99 |
| Allocation queue, excluded from readiness | 3–498 | 74 |

Conservative bounds use observed maxima across 40 accepted attempts plus explicit
margins for this shared cluster; they are not production latency guarantees:

| Check | Observed maximum | Bound | Margin above maximum |
|---|---|---|---|
| Service launch to healthy readiness | 394.75 s | 720 s | 325.25 s |
| Native error / frontend connection error | 0.317 s | 30 s | 29.683 s |
| External cleanup | 52.48 s | 180 s | 127.52 s |
| Allocation queue, separate from readiness | 810 s | 7200 s | 6390 s |
| Memory delta per GPU after cleanup | 0 MiB | 64 MiB | 64 MiB |

STOP asserts its 30 s observation deadline (up to one additional second for
scheduling/recording), not death detection. No new compute PID is permitted,
regardless of the 64 MiB memory allowance.

## Autonomous decisions and MVP boundaries

| Decision | Reason / boundary |
|---|---|
| Ray only; retain MPI evidence | Tested PMIx cancellation prevents survivor/client observation; avoid a second partial runtime. |
| EP32/288 slots; fixed middle rank | Reproducible terminal characterization; coverage-based recovery admission is later. 512 slots remain a comparison. |
| One driver/controller/probe/checker | Preserve ownership, durable evidence and independent cleanup; no generic callback or recovery framework. |
| Persisted nonfinal stream trigger | Observable synchronization; no exact GPU phase, execution deadline or agreement claim. |
| Default crash policy; STOP timeout asserted | SIGSTOP is host silence, not necessarily frozen GPU work or death evidence. |
| Existing startup markers; cold start first | Queue separated, unmarked costs unavailable, nested phases never summed. |
| Config-derived topology and optional site segment | No EP-size allowlist or site-specific topology flag; new configurations and clusters still need physical qualification. |
| Additional DeepSeek/Qwen smokes | Exercise a second topology and model without a separate injector or stronger qualification claim. |
| Independent memory reconciliation and owned-probe drain | Reject contradictory cleanup summaries and prevent rescue racing storage writers; keep original failed controls. |
| Isolated live Ray controls and reported-error tests | Verify the private subscription contract and already-waiting requests; silent GCS loss stays unqualified. |
| Subscribe before snapshot; filter exact actors | Wake waiting requests without blocked root RPC; separately approved LLM guard preserves the cause. |
| Register results before dispatch; check fatal state | Close submission races; healthy API/configuration unchanged. Host-check overhead has no separate performance-equivalence measurement. |
| Freeze sources; no retries | Every failed/interrupted attempt retained; raw logs are artifacts, not committed files. |
| Native GCS port and owned protocol gate | Both handwritten ports and plain TCP readiness failed physical controls. |
| Accounting grace only after local exit | Preserve bounded cancellation; local exit alone does not prove remote cleanup. |
| Signed local commits; no publication | User owns human communication and CFT access requests. |

The graph buckets prove healthy capture, not post-failure graph identity/reuse.
Host stream markers do not target a GPU layer/phase. Current rank-mask FT rejects
graphs; sticky errors must not be cleared to simulate recovery. CFT requires a
qualified R615/platform stack, active counted writes, completion/failed-issuer
teardown proof and safe endpoint/counter/payload reuse. R580 fence-path probes
and clean resource accounting supply none of those transfer-safety proofs.
Survivor authority, host collectives, coherent graph state, request disposition
and correct N−1 inference remain future work.

## Line-reduction validation

The reduced runtime at frozen `717386b8f5` passed EP32 cold startup (7687217,
311.75 s readiness), idle kill (7687321, native error 0.191 s) and streaming kill
(7687524, 0.163 s). All three passed eight-node process/GPU/storage cleanup;
all 32 rank logs matched the previous algorithm byte-for-byte. Native executor
validation 7687197 passed 24 tests; CPU validation retained 151 passes/seven
physical skips. Production modules are byte-identical to the earlier qualification.
These are targeted refactor checks, not a replacement 40-attempt qualification.
Artifacts and frozen source bundle: `line-reduction/`.

## Archived history

The full pre-reduction report is retained locally at
`.wideep-ft-runs/line-reduction/PROCESS_LOSS.md` and in
`.wideep-ft-runs/line-reduction/before.bundle` (tip `6c54e4dbd5`, SHA256
`d03b62566ecacc8a3729e8c926aa6f505721bd3fc7306ba08768c9df1e9a476c`).
The bundle requires base `f388b7c6a5`. Fetch its `refs/heads/wideep-ft` into
`FETCH_HEAD` in a checkout with that base to inspect history without a new branch.
Earlier development history and physical source IDs are retained in
`lean-foundation/final-development.bundle` (tip `e7d31a6578`, SHA256
`736673072a45c5043f2f5b5535689c1de80cd329151a949d14d7aa3061ec880d`).
Bundles/raw logs remain local artifacts; committed documentation is the concise
finding and reproducibility record.
