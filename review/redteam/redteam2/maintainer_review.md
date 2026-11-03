# WideEP FT stack: CODEOWNER review (MoE/comm, runtime/executor, llmapi/serve)
Each PR is reviewed as `git diff <parent>..<branch>`, and line numbers are in that branch. The reviewer re-read the items marked [v] in the code. The rest come from persona reviewers who read the code (nothing was built or run). Stack-wide facts: no `llm_args.py`, golden-manifest, schema or api_stability change; 4 new `TLLM_FAULT_TOLERANCE_*` env vars; 28 call sites of `peer_loss_recovery_enabled()`; every PR says "Not run yet".

## Stack-wide (raise once, on PR 1)
- **No build, no pytest, PERF all `<PERF_TBD>`.** No owner approves C++/kernel PRs (02/03/04/06/08/09) without a green build plus l0_b200/l0_cpu runs and an FT-off TPOT row. Run them before requesting review.
- **The e2e proof cannot run in CI [v].** The `docker/common/install_base.sh:103` image has `apt-get install openmpi-bin` (no `ftmpi`), and nothing in jenkins/docker sets `TLLM_REQUIRE_ULFM`. So every `wide_ep_ft/*` test (l0_gb200_multi_gpus post-merge) and `test_mpi_survivors_ulfm` skips silently and shows green. Say so in every description, or add a ULFM stage that sets `TLLM_REQUIRE_ULFM=1`.
- **Config through env plus a side YAML, not LlmArgs.** `FaultToleranceConfig` (StrictBaseModel, user-facing) loads from `TLLM_FAULT_TOLERANCE_CONFIG`. That bypasses the AGENTS.md manifest, schema and telemetry rules, and the guide (PR 10) documents env vars as the UI. `TLLM_FAULT_TOLERANCE_MODE` already exists upstream; `_AUTHORITY`, `_AUTHORITY_KEY` and `_CONFIG` are new surface.
- **Too much "no in-tree caller".** PR 2's MoeExecutionState and its about 300 Python lines are first used in 3; `expert_to_slot` in 6. PR 4/5/6 APIs are first used in 7a. PR 5 Status/StatusQuery are first used in 10.
- **Hygiene is mostly clean.** All new files carry 2026 SPDX headers, and there are no history-narrating comments or TODOs. Broad `except Exception` with `noqa: BLE001`: communicator.py:131 (04), peer_loss_recovery.py:919 (07a), hang_evidence.py:294,301 (08). `print` in mpi_exit.py:103,118 (04).

## PR 1/10: config and mode switch (+141/-10, tests +142)
**Blocking**
- `executor/recovery/config.py:33-53`: a user-facing config that lives outside `TorchLlmArgs`. Add `fault_tolerance_config: Optional[FaultToleranceConfig] = Field(None, status="prototype")` now and regenerate the manifest and schemas. Keep env vars only for the bash launcher and the authority sidecar. If you defer it, the description must justify the deferral to llmapi owners.
**Important**
- Dead in this PR: `peer_loss_recovery_enabled` (`_utils.py:525`), `fault_tolerance_config()` (config.py:56), `failure_domains` (:71), `FailureDomain`. Move each to its first caller.
- `failure_domains` (config.py:71-84) has no test anywhere in the stack, yet it decides which losses the placement must survive. Add tests for rank vs node, rank 0's host excluded, a 1-host world and de-duplication.
- `trtllm-llmapi-launch:174-189`: SIGKILL on **any** non-zero exit under FT also covers deliberate exits (config error, 143). PR 8 has to carve out 75, which shows the rule is too broad. Define the exempt codes here. Also justify the comment at :177 "a second non-zero exit ends the whole job" with a PRRTE reference.
- Why are the switches split between `_utils.py` and `executor/recovery/config.py`, with the worker side in `_torch/pyexecutor/`? Pick one home.
**Nits**: `failure_domain: str` should be `Literal`. Use `functools.cache` rather than `lru_cache(maxsize=None)`. The Raises section omits FileNotFoundError and YAMLError. `extra_slots: Literal["empty"]` has no reader, and PR 9 retypes it. The launcher test uses a different header style.
**Tests**: l0_cpu, no ULFM needed. Good, but they have to actually run.

## PR 2/10: NVLinkOneSided rank mask, abort, fixed slots (+1077/-148 product, C++)
**Blocking**
- `cpp_custom_ops.py:736-771` [v]: the fake `moe_a2a_dispatch` returns `[ep, runtime_max_tokens_per_rank, ...]`, but with `source_valid_counts` the real op returns `[ep, MAX_NUM_TOKENS, ...]` (`moeAlltoAllOp.cpp:629`). Pass the slot capacity to the fake and add a fake-vs-real shape test to `test_schema_and_fake_signatures`.
- `wide_ep_ft.py:59-65`: the CUDA-graph refusal is dropped although nothing binds a state in this PR. Detection-only FT (mode without authority) stays unbound for the whole stack. Graph replays then bypass `_publish_rank_mask`, the watchdog skips captured collectives (alltoall_watchdog.py:473), and there is no abort record, so a peer loss becomes a 300 s device trap. Keep the refusal unless `_execution_state is not None`.
**Important**
- `moeAlltoAllKernels.cu:311`/`.h:62`: every waiting lane polls pinned host memory (`ld.relaxed.sys`) every 50 us. At DEP32 that is blocks×ep host reads per period on normal stragglers. Poll from lane 0 and `__shfl`, and measure FT-on TPOT.
- `moeAlltoAllOp.cpp:1203`: `abort_state` is written by the kernels but the schema says `Tensor?`. Make it `Tensor(f!)?`.
- Silent contract change: `active_rank_mask` must now be a CUDA tensor; it used to be CPU. The local-bit check moved to Python. Declare this in the API section.
- `moeAlltoAllKernels.cu:1601` `SWITCH_BOOL(expert_to_slot)` doubles every fence-dispatch instantiation, FT-off included, for a feature first used in PR 6. Move it to PR 6, or instantiate only under `ENABLE_RANK_MASK`.
- `MOE_DEVELOPER_GUIDE.md:166`: "graphs follow membership changes" is false for unbound comms.
- PDL: the `receive_source` trigger (k.cu:612) fires before its outputs are stored. Move it after the stores, or justify it in a comment.
**Nits**: inline `_utils.device_visible_host_zeros` (one caller). Byte-wise zeroing on the abort path (k.cu:2071). `thread_local` pointer cache is unneeded on UVA (op.cpp:185). The `mnnvl_memory.py:830` cleanup leak only warns.
**Scope**: move the Python binding (execution_state, bind/_bound_layout, pin/unpin, about 300 lines) to PR 3, leaving PR 2 as kernel + op + tests. Kernel lines are not churned later.

## PR 3/10: Cutlass/CuteDSL read receive slots in place (+484/-73, C++)
**Important**
- `moe_kernels.cu:1646,1770`: the source-slot expand kernel is instantiated for every expand variant of all 24 runner instantiations, though Python enables it only for unquantized, no-LoRA inputs (`fused_moe_cutlass.py:958`). Instantiate only for `BlockScalingType::NONE && !AWQ`, `TLLM_THROW` otherwise, and tighten the `:4531` check. The description's "one extra instantiation" understates the binary growth.
- `cute_dsl_custom_ops.py:4225` [v]: `key=(unique_id, source_tokens, slot_tokens)`, where `source_tokens = runtime_max_tokens_per_rank` (`nvlink_one_sided.py:1042`) changes each iteration. The class-level tuning cache therefore grows up to max_num_tokens entries. Key on `source_tokens>0` instead.
- `moe_scheduler.py:697`: the `do_finalize=False` refusal fires after dispatch, while peers wait in combine, so it hangs instead of erroring. Reject it at bind time in `MoeExecutionState.for_model`.
- Layering: `moe_scheduler.py:139-143,629,645,1118` read an executor env switch in each MoE layer's `__init__`. Gate on "comm is bound" (`dispatch_source_slots() is not None`) so `for_model` is the single source of truth.
- `runner.py:204`: `for_model` raises for any DecoderRunner without NVLinkOneSided layers once recovery is on. Confirm no second runner exists (draft or spec). Non-Decoder runners silently get None (`model_engine.py:801`); make that an error.
**Nits**: `torch_custom_ops.py:310` adds an undisclosed slice on the FT-off path that only the unit test reaches. Remove it. The same invariant is checked 3 times (moe_kernels.cu:1769, :4531, moeOp.cpp). The second 30-arg virtual `runMoe` overload should become a params struct. Cache `supports_dispatch_receive_slots()`.
**Tests**: `test_moe_slot_gather` is in l0_b200; its CPU-only scheduler tests could move to l0_cpu. Missing: `CutlassFusedMoE.run_moe` with `enable_alltoall=True` + slots, CuteDSL `run_moe_nvfp4` with `source_slots`, C++ rejection of quant/LoRA with slots.
**Scope**: the runner/model_engine binding is runtime code and could move to 7a.

## PR 4/10: ULFM survivor comms, bounded Finalize (+898/-19, C++)
**Blocking**
- `nanobind/runtime/peerFailureWatch.cpp:74-85,103-105` [v]: destroying the watch on its own thread (last ref dropped in `on_failure`) makes `stop()` return false, `~std::thread` sees a joinable thread, and the rank dies in `std::terminate`. Detach in that case, or keep the object alive while the thread runs. The fake watch hides this, and RESTACK lists it as a known unapplied finding.
**Important**
- communicator.py:973-986: `watch_peer_failures` does not check that `enable_peer_loss_recovery()` ran, so `MPI_Test` can run on another thread without `MPI_THREAD_MULTIPLE` checked. communicator.py:1018: `barrier_among_survivors` has no `_shrunk is None` guard. Raise as the sibling methods do.
- `_utils.py:641`, `mpi_exit.py:36`: `Get_failed` raises AttributeError on mpi4py<4, and the paths are reachable whenever the env is set. The 4.1.2 pin lives in another PR. Fail fast once at startup.
- `peerFailureWatch.cpp:139` / communicator.py:815: `MPI_Test` every 5 ms under THREAD_MULTIPLE contends with host collectives. The comment "costs the serving loop nothing" is unmeasured. communicator.py:142 adds `MPIX_Comm_get_failed` after every ADP `tp_allgather` (FT-on).
- `mpi_exit.py:109,123`: `os._exit` skips every atexit handler (ZMQ/shm cleanup). Document it, or run them before Finalize. Possible circular import at `mpi_session.py:27` → `executor.recovery` → `executor.py` → `mpi_session`; import lazily and check with a fresh `python -c`.
- `communicator.py:911`: the `OMPI_MCA_mpi_ft_enable` read adds yet another env switch and is Open MPI only.
**Nits**: `except Exception` at communicator.py:131 should list the real exception types. Revert the formatting-only `tp_broadcast` rewrite (:1153). `mgmn_worker_node.py:26` imports at the bottom with `noqa: E402`. FT-off has 2-3 `_host_world` checks, not 1 as described.
**Tests**: the native watch is never exercised (fully faked). `test_mpi_survivors_ulfm` always skips and never checks returncode or the bounded Finalize. `test_..._before_install_` names an API from 7a.
**Scope/churn**: 7a rewrites about 11 of the communicator lines added here (`allgather` signatures, `shrink`/`revoke` docstrings, `_world_comm`). Land the final shapes here. mpi_session/mpi_exit/mgmn could be their own PR.

## PR 5/10: external recovery authority (+1057/-5)
**Important**
- Wire format: pickle + HMAC (`protocol.py:5`, `ipc.py:442`). It follows `ZeroMqQueue` precedent, so this is not blocking. But the messages are frozen dataclasses of scalars, and PR 8 adds its own datagram codec. Use one JSON/msgpack codec with a type allowlist, so a leaked `_AUTHORITY_KEY` (passed with `mpirun -x`, visible in /proc environ) is not RCE on every worker.
- No protocol version. PR 9 already changes `Stopped`. Put `PROTOCOL_VERSION` or `__version__` in `Register` and refuse skew. Today skew surfaces as pickle AttributeErrors (and as /health 500s in PR 10).
- `client.py:74-82`: `unauthenticated()` classifies errors by exception text ("HMAC verification failed", "values to unpack"). Raise a typed `AuthenticationError` from `ZeroMqQueue`.
- `authority_server.py:44,60`: the `key` parameter is unread until PR 8. `:77` reads the private `router._last_identity`.
- Launch: `python -m tensorrt_llm.executor.recovery.authority` imports torch and the full wheel for a "GPU-less sidecar". There is no console script and no `main()` test. Make `authority_server` the target or add `trtllm-ft-authority`, and test `main([...])`.
- `authority.py:139,277`: abort reasons embed full reprs and can exceed `MAXMSGSIZE`=64 KiB (protocol.py:31). The connection then drops and workers time out instead of aborting. Truncate to 1 KiB.
- `config.py:64-67`: setting only `recovery_timeout_s: 15` fails because the default `authority_timeout_s` (20) is not below it. Derive the default or say so in the error.
**Question**: why is a generic coordinator under `executor/`, with no caller until 7a, instead of a torch-free tool package?
**Tests**: test_recovery_authority runs in l0_cpu over ipc:// (good, not ULFM-gated). Missing: CLI/main, over-size message, a real 3-frame unauthenticated peer, version skew.

## PR 6/10: restore into empty extra EPLB slots (+1048/-117, C++)
**Blocking**
- `moeLoadBalancer.cpp:1257-1286` [v]: `restorePlacement` checks `!mIterActive && mIterInfoQueue.empty()`, but the worker pops the queue and then walks layers unlocked (:1341-1364). Without a prior `quiesce`, `installPlacement` races `updateWeightsRoutine`. 7a calls quiesce first, but the public binding does not enforce it. Add `TLLM_CHECK(mWorkerDoneIterId == mIterId)` as `resetUpdateSchedule` (:1320) does, plus `TLLM_CHECK(mFaultToleranceMode)`. With FT off, the updater rejects -1 halfway through installation, after some layers were already installed.
**Important**
- `moeLoadBalancer.cpp:1363`, kernels.cu:657-669: an abandoned iteration still runs `maybeStartUpdateWeights` on partial statistics, so placements diverge. 7a relies on installing rank 0's placement to cover this. Skip the update when the iteration is abandoned.
- `moeLoadBalanceOp.cpp:170-185` `one_slot_table`: a `from_blob` view on `current_device()` (not the balancer's device) that Python caches and that dangles after shutdown. The op sets a flag but its schema has no side effect.
- `moe_load_balancer.py:1082`: static EPLB + FT keeps a full host copy of every expert on every node in /dev/shm. A rough R1 NVFP4 estimate is hundreds of GB per node, plus prefault time before serving. Quantify it in the description; PR 9 rings it only later.
- C++ gates FT on `TLLM_FAULT_TOLERANCE_MODE` and Python on mode+authority, so detection-only mode pays the C++ FT costs. Pass the flag from Python (`finalize_model(fault_tolerance=...)`).
- `_gather_hosts` (:1013) raises unless EP == COMM_WORLD, inside the constructor. Validate this in config instead.
**Tests**: every restore test mocks `load_balancer_impl`. Nothing drives the real `restorePlacement`/`validatePlacement`/updater -1, so cutting `MoeLoadBalancerRestoreTest` x9 removed the only real coverage. Restore 2-3 gtests, and move `moeLoadBalancerTest` to `add_tllm_gtest` with OWNER and LABELS (AGENTS.md).
**Scope/churn**: PR 9 rewrites about 343 lines of `moe_load_balancer.py` and 119 of `moeLoadBalancer.cpp` from this PR, including a pure 32-line rename `appended_slot_count`→`ghost_slot_count`. Use a neutral final name (`extra_slot_count`) here. `impl_blocks.slot_rows` (DEEPGEMM repacked scales) is a separate concern.

## PR 7a/10: executor recovery (+1866/-102, tests +1916; over the 3.5k cap)
FT-off: the claims hold. Each step adds about 10-20 attribute checks or no-op calls, with no allocations or syncs [v spot-checked].
**Important**
- FT-off behaviour change [v]: `request_utils.py:134`, `adp_router.py:248`. The per-rank counts become a dict, so a strict negative `attention_dp_rank` that used to route to the last rank now raises KeyError, and an out-of-range one raises KeyError instead of IndexError. Validate the range in `_validate_request` (400).
- `py_executor.py:8681`: `discard_interrupted_steps` sets `previous_batch=None` after `commit_step` + `_update_requests` (5966) but before `_process_previous_batch` (6214). That can drop the committed step's radix commit, `update_resources` and canceled-request handling. Prove this is safe with a test, or run that bookkeeping in the discard.
- FT-on healthy cost: `py_executor.py:5973,6051` defers `_update_v2_context_resources` by one step, adds a host wait (`_wait_for_model_engine_input_copy`) on the critical path, and runs `abort.any()` as a host torch op every step. Measure these before claiming <1%.
- `peer_loss_recovery.py:542` / `py_executor.py:8735`: context changes are recorded twice in overlap. Non-overlap discards `step_prepared`'s return (4934). `step_released()` guards differ between 4975 and 5881.
- `py_executor.py:1500-1508`: the event-loop crash path calls `os.kill(getpid(), SIGKILL)` from library code. Runtime owners will ask for this to be a launcher/worker decision. At minimum, gate it on the authority having admitted the rank.
- Three more env readers deep in scheduler and inputs (`request_utils`, `adp_router`, `inputs.py:476`). Pass a flag down instead.
**Nits**: `shrink_timeout_s()` (config.py:69) is an alias until PR 8. `mpi_exit.py:97` reaches into `_torch` via `sys.modules` string. Hard-coded `_REGISTRATION_TIMEOUT_S=600`. `_outputs_requested` latches forever. Class-level `RecoveryHook()` default exists only for `object.__new__` tests. The `kv_cache_manager_v2.py` docstring rewrite edits a wave-1 PR; drop it.
**Tests**: 56 unit tests in l0_cpu, all mocks. No GPU test of an overlap discard with KV V2, chunked context or MTP, and no FT-off parity run of `_executor_loop_overlap`.
**Split**: 7a-0 "ADP absent ranks" (about 0.8k: `live_ranks`/`departed=`, adp_router dict, `split_unservable_requests`, request_utils, `test_adp_absent_ranks` and adapted tests), then 7a (about 3.0k). Pre-shape `_retire` and the recovery-gather NamedTuple so 7b only appends.

## PR 7b/10: quarantine, 503, e2e (+140/-20, tests +828)
**Blocking**: `l0_gb200_multi_gpus.yml:101-105` [v]: all 5 e2e tests skip in CI (see stack-wide). Do not list them until a ULFM stage exists, or add the stage.
**Important**
- `peer_loss_recovery.py:1162-1169,1215-1220`: "fails only if it made no progress" is false for requests on a departed rank. Progress reaches rank 0 only through the next recovery's gather. A request from a lost rank that was active at loss 1 fails at loss 2 even after hundreds of steps. Ship `_progressed` with the regular response gather and add the test.
- `:1155-1173`: two close losses (cascading node failure before the first post-resume commit) quarantine a survivor's whole batch. Narrow the rule, or list it as a limitation.
- `openai_server.py:268,1200` [v]: gating on `peer_loss_recovery_enabled()` in the HTTP process depends on that process's env. It raises ValueError inside the error handler if AUTHORITY is set without MODE. Drop the gate; only recovery emits the tag.
- This is a public HTTP contract change (400 becomes 503 + `Retry-After` + `ServiceUnavailableError`) classified by message text. There is precedent (context-length), but any echoed user text containing the tag becomes a 503. Get a serve-owner sign-off and file a follow-up for a typed code across IPC. `Retry-After: "1"` with "takes about a second" is unsourced. Disagg and gRPC pass-through are unverified.
**Scope**: three concerns (runtime policy, serve contract, e2e for 7a). Fold the e2e tests into 7a, or split the 503 into a serve-owned PR.

## PR 10/10: /health and guide (+185/-2, docs +279)
**Blocking**: `wide-ep-fault-tolerance.md:115,178` [v] documents `hang_detection_timeout` as a user setting. It is only a `PyExecutor.__init__` kwarg with no caller and no LlmArgs field (py_executor.py:454). State the real constraint instead (`recovery_timeout_s` vs the fixed 300 s).
**Important**
- `fault_tolerance_status.py:50` + `openai_server.py:1709`: an `unknown` phase (version skew) returns 503 on a serving server, which pulls every replica out of rotation. A Status that fails to unpickle raises a 500 on every probe (only HMAC errors are swallowed, :128-145). Map both to 200 with an alerting body.
- `failed_requests` on /health counts departed-rank requests only, not survivors' quarantined ones (`peer_loss_recovery.py:910,1101-1110` vs 1217). Rename the field or count both.
- `/health` without an authority changes FT-mode detection-only from an empty 200 to a JSON body, and re-reads the env per call (:1712). Read it once at init. The probe's thread pool and socket are never closed; close them in lifespan teardown.
- Doc numbers without a run, scale or n: "1.4 GiB per GPU" (l.72), "within milliseconds" (l.140,187), "GSM8K within two standard errors" (l.200), "about a second" (l.234), "being measured" (l.60). Give numbers or mark them `<PERF_TBD>`. Key length: "64 hex" (l.66), but the code accepts 32 or more.
**Tests**: `test_fault_tolerance_health` covers only the no-probe path. Add a fake probe parametrized over recovering, terminal and unknown. No e2e asserts the `degraded` state after a kill.
**Scope**: split the guide (doc owners) from /health (serve owners). Since 10 sits before 08/09, the guide is written twice (28 lines churned).

## PR 8/10 (wave 2): hang detection and fencing (+1576/-34, C++)
**Blocking**
- `peer_loss_recovery.py:1078-1092` [v] `_shrink` calls `_end_ranks` (PMIx kill) on **every** recovery, and `authority.py:450-457` [v] sends `Fence` for every failed rank before Propose, also unconditionally. A `Fenced` error then aborts the recovery (:463). Neither is gated on `hang_detection`, so this PR makes core crash recovery depend on PRRTE/PMIx job control (srun+PMIx and node loss are untested), and the change is undisclosed. Gate both on `hang_detection`, or treat `PmixUnavailable` and an already-dead process as fenced.
- `pmix_fence.py:79` / guide l.92 claim a kill of an already-exited process is a no-op, but nothing tests it (only fakes). Run `[exit]`/`[replan]` through `kill_rank`.
**Important**: the hang_evidence reporter issues CUDA work concurrently with main. Its only capture guard is an `assert` on the private `graphs._capture_allowed` (`peer_loss_recovery.py:543`), which `-O` strips; make it a real check. Each survivor kills each failed rank, about 112 duplicate HNP requests per node loss at DEP32, so let rank 0 do it. `pmix_fence.py:95` has an unchecked `PMIx_Info_create` NULL. Refuse `hang_detection` at startup without PMIx. The guide's l.168 "Measured under an earlier rule: 10.6/15.9/36.7 s" is history plus preliminary data; drop it.
**Nits**: contradictory comments at authority.py:317-318. Broken rewrap at `trtllm-llmapi-launch:174-176`. Unreaped `Popen` (zombies) at authority_server.py:235. `in_host_collective` belongs on `MPIDist`, not the base. `RecoveryAuthority.__init__` inserts positional parameters; make them keyword-only.
**Tests**: the GPU progress-row test is reachable only through l0_cpu, so it always skips; list it in l0_b200. `test_hung_rank_fenced` covers SIGSTOP only and skips without ULFM. `[fenced]` may kill before the SIGSTOP lands.
**Split**: hang evidence/monitor | PMIx fence | kills-before-agreement (the core behaviour change).

## PR 9/10 (wave 2): ghost slots and ring host copies (+1889/-110, C++)
**Blocking**
- `moe_load_balancer.py:1077,1150`: ring host copies replace full copies for every static FT placement with no switch. Losing a copy node makes its experts unrecoverable. `peer_loss_recovery.py:441` [v] `restore_copy_plan([])` at startup can refuse configs that worked at PR 8 (no free workspace half). Add `host_copies: all|ring` (default `all`), or fall back to full copies.
**Important**
- `config.py:54` [v]: the default flips from `empty` to `None`→ghost, so FT users now get a KV cut at recovery. Pydantic rules say no `Optional` as "auto". Use `Literal["auto","empty","ghost"]="auto"` from PR 1 and say "default changed".
- `moe_load_balancer.py:1189`: explicit ghost + online EPLB on GB200 silently switches **all** EPLB updates to GPU memcpy. Warn or reject. The GDRCopy host-access path has zero CI coverage (the gtest is skipped on GB200, and the C++ was never built); defer it to 9c.
- `ghostSlotMemory.cpp:674-705` `releasePending` unmaps with no device sync, while `free()` (:558) syncs. `:697`/`:647`: a failed release leaves a piece marked unbacked with stale scratch state. Use `dropPiece` and fail closed.
- `py_executor.py:5583` `resize_kv_cache_quota` copies the `_rebalance_kv_pools_now` suspend/resume block without its `OutOfPagesError` handling. Factor out a context manager and test the failure path.
- The description says a `shared_mutex` was added; it already existed (`hostAccessibleDeviceAllocator.h:171`). FT-off: `weight_slots` (:604) pins every slot tensor even without ring copies.
**Nits**: an empty `catch` at ghostSlotMemory.cpp:261. `moeLoadBalancer.h:33` pulls `cuda.h` into a widely included header. Magic `_GHOST_MARGIN_BYTES`. Top-level `import pynvml` for a log line. The `ghostSlotMemory.h:34` 16-line design comment belongs in the guide. `moe_ghost_slot_storage` has no `register_fake`.
**Tests**: `GhostSlotMemoryTest` x7 sits in the legacy `add_gtest` target; it needs `add_tllm_gtest` with OWNER, LABELS and REQUIRES_GPU. The ghost e2e is post-merge and skips (no ULFM).
**Split**: 9a ghost slots + KV cut | 9b ring copies/staging + protocol barrier fields, behind a switch | 9c GDRCopy host-access, once x86 CI exists.

## Cross-PR churn (worst; about 230 rewritten lines in total, low for 20k)
06→09 rename (32 lines, plus 9 in tests); 10→08/09 guide (28); 07a→07b/09 `_forget`→`_retire`, gather tuple and `requests_cancelled` (20); 05→08 authority `_sequences` and filter (16); 04→07a communicator signatures (15); 01→05/09 config docstring and `extra_slots` default (14); 06→09 moeLoadBalancer.cpp empty→ghost paths (13); 01→08 launcher condition and broken wrap. Fix: land final names and shapes early, and move PR 10 to the end of the stack (or write the 08/09 guide sections in those PRs).

## Top 10 merge-stallers (ranked) and the cheapest fix for each
1. **Nothing built or run; PERF all TBD.** Fix: one build-only job over boundaries 02/03/04/06/08/09 plus the l0_cpu/l0_b200 suites, then fill in the FT-off TPOT rows before requesting review.
2. **The e2e proof never runs in CI (no ULFM image).** Fix: say "skips in CI today" in 7b/10/8/9, attach the external GB200 run logs, and open an infra ticket for a ULFM stage with `TLLM_REQUIRE_ULFM=1`.
3. **Env plus side-YAML config, not LlmArgs.** Fix: add a prototype `fault_tolerance_config` field to TorchLlmArgs in PR 1 and regenerate the manifest and schemas. Keep env vars only for the launcher and authority.
4. **PR 8 makes every recovery depend on PMIx kills/Fence.** Fix: gate `_end_ranks` in `_shrink` and the pre-Propose `Fence` on `hang_detection`.
5. **PR 2 fake/real shape mismatch, plus the dropped CUDA-graph refusal for unbound comms.** Fix: a capacity argument in the fake with a shape test, and keep the refusal unless a state is bound.
6. **PR 6 `restorePlacement` race / FT-off -1 install, with no real C++ test.** Fix: two `TLLM_CHECK`s and 2-3 restored gtests.
7. **PR 9 silent behaviour changes (ring copies, ghost default, all-GPU-memcpy).** Fix: `host_copies` and `extra_slots="auto"` switches with old defaults, and a warning on the GPU-memcpy fallback.
8. **PR size and "no in-tree caller"** (7a at 3.8k; PR 2 Python, PR 5, `expert_to_slot`). Fix: split out 7a-0 ADP absent ranks, move PR 2's binding to 3 and `expert_to_slot` to 6.
9. **PR 4 `std::terminate` on self-thread destruction**, plus missing guards and the mpi4py<4 AttributeError. Fix: detach in the destructor, add 2 RuntimeError guards, and fail fast on mpi4py<4.
10. **Serve contract and docs accuracy** (503 by message text, `unknown`→503, unsourced numbers, `hang_detection_timeout` documented as settable). Fix: drop the env gate in openai_server, make `unknown` a 200, close the probe, and replace doc claims with cited numbers or `<PERF_TBD>`.
