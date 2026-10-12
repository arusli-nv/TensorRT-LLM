# Review round 4: per-PR review of the 14-commit fold (`wideep-ft-folded`)

Line numbers refer to each file as of its own commit. Items already reported in rounds 1 to 3 are left out unless they are still open in a way that affects this round. Those carried over are marked **(r3 open)**.

## Summary

- **No correctness blockers.**
  - Commit 14's suspected V2 bug was checked and is mitigated. The V2 first-chunk revert does not rewind the context cursor, but on re-admission `prepare_context` calls `_settle_context_cursor` for any first chunk when block reuse is on (`kv_cache_manager_v2.py:3866-3867`). With reuse off, the cursor is already 0.
  - The remaining gap there is a missing test, not a bug: add a reuse-match case to the V2 revert test.
- **Cross-PR, still open:**
  - The `jenkins/current_image_tags.properties` bump (PR 6). Without it, every ULFM test skips and passes green. **(r3 open)**
  - The single cached env predicate in a leaf module (PRs 7, 10, 13). **(r3 open)**
  - The single `Get_failed` helper (PRs 8, 13). **(r3 open)**
- **Lines that could go:** about 250 to 300 of source and 300 to 350 of tests, with no lost guarantees.
- **Biggest single trims:**
  - PR 12's three hunks that `create()` makes unreachable: about −30.
  - The commit-14 tests that mock the rollback: about −50.
  - The commit-10 duplicate abort or exclusion tests: about −70.
  - PR 7's re-export `__init__`: −32.
  - Commit 10's `commit_failed`: about −57.
- **Rejection risks with a main reviewer:**
  - PR 12 advertises paths it cannot reach, and its FT-off claim holds only for `DefaultADPRouter`.
  - PR descriptions are stale or wrong in 4, 5, 8, 9 and 11.
  - PR 11's GPU tests check a Python re-implementation of the routing, not the kernel.
  - One assertion in the ULFM test cannot fail.

Keep the following (bounded waits, fail-closed checks, quarantine):
- In the authority: the `generation` fields, `Propose.survivors` validation, `MAXMSGSIZE`, `SNDTIMEO` and the recovery deadline.
- `install_survivors(failed)` checking against the shrink result.
- `HostCollectiveInterrupted.__reduce__`.
- `MnnvlMemory.pin`, the fixed-slot bound, and the abort-record check in `install_mask`.
- The `ENABLE_RANK_MASK` template parameter, which keeps FT-off opcode parity.
- `_join_device_work`, recovery under the hang detector, the `create()` requirements, and the revoke in `finally`.

---

## 1. 3f472f3cb5 [None][fix] propagate fatal RPC worker errors
**Verdict:** mergeable once trimmed. It stands alone, and the FT-off behaviour change is intended and described.

1. `rpc_proxy_mixin.py:207-223`: the fetch wrapper checks `_shutdown_event.is_set()` three times. Compute `error` in the try, `except CancelledError` and `except Exception` branches, then run one `if not shutdown: _set_fatal_error(error)`. **−4.**
2. `rpc_proxy_mixin.py:111-121`: drop the `sent` flag and the `finally`.
   - `except (RPCError, zmq.ZMQError)`: pop the result, set fatal, raise `EngineDeadError`.
   - `except BaseException`: pop the result, then `raise`.
   - **−3.**
3. Test bloat in `test_rpc_executor_mixin.py`, which has 13 tests in 254 lines:
   - `:130` repeats `:79` and `:121`. Keep only "a ZMQError on send is fatal and keeps its cause". **−8.**
   - `:148` parametrizes three exception types and an `Unpicklable` class over one code path. One `TypeError` side-effect is enough. **−8.**
   - Merge `:172` and `:204` into one test parametrized over {raise, clean end, cancel}. **−12.**
   - `:49` drives `LLM.generate_async` with a `SimpleNamespace` self, from the executor-mixin file. Move it to the llmapi tests or accept it (low priority).
4. `test_rpc_worker_mixin.py`: the `_event_loop_error_delivered` assertion and its stub check a detail the fix never touches. **−2.**

## 2. 8416218cb9 [None][fix] bound RPC socket teardown with unreachable peers
**Verdict:** good and minimal.
1. `test_rpc.py:151`: the `client.close()` after the `with RPCClient(...)` block is redundant. **−1.**

## 3. b160de1ba6 [None][fix] retain DLPack metadata until the last storage alias
**Verdict:** correct; trim the tests.
1. `test_mnnvl_memory_lifecycle.py:1057` (`test_dlpack_failed_conversion_releases_metadata`) duplicates `:1034`. It needs two monkeypatches to reach a path the fix does not change. **−20.**
2. `_dlpack_utils.py:78-83`: add one comment saying why it uses `ctypes.pythonapi["Py_IncRef"]`, which returns a private prototype that other modules cannot mutate. This is what justifies the test at `:1042`.
3. Optional: `_dlpack_utils.py:223`, `_capsule_wrapper` on the tensor, is no longer needed for ownership; only the tests read it. **−1.**
4. Follow-up outside this PR: `_torch/modules/dwdp/vmm.py:382` uses the same tensor-attribute pattern and may have the same alias bug.

## 4. fe743066db [None][fix] skip request updates without a sampling result
**Verdict:** good; stands alone (the r3 stub of a later symbol is gone).
1. `test_py_executor.py:642` (`test_update_requests_preserves_valid_state_contract`) pins unchanged behaviour with three Mocks. Delete it. **−19.**
2. `:626`: the `AttributeError` side-effect sits on a mock that is asserted never called, and the `_handle_errors` and `_accumulate_spec_dec_stats` asserts are redundant. `update_requests.assert_not_called()` is enough. **−5.**

## 5. 518c4f79f3 [None][fix] keep EPGroupHealth identity across ConfigurableMoE deepcopy
**Verdict:** the fix works, but it puts FT knowledge in the wrong layer, and the PR description is stale.
1. **Better fix:** give `EPGroupHealth` a `__deepcopy__(self, memo): return self`, with a one-line docstring calling it a shared membership object.
   - Revert `configurable_moe.py:344-347` to plain `copy.deepcopy(model_config)` and drop the `wide_ep_ft` import at `:54`.
   - Drop the `_HEALTH_KEY` → `EP_GROUP_HEALTH_KEY` rename (3 hunks in `wide_ep_ft.py`).
   - This also covers the legacy `extra_attrs["ep_group_health"]` key (`wide_ep_ft.py:68`), which the memo misses, and every other deepcopy of a config.
   - **About −8**, and the MoE layer no longer imports FT internals.
   - The test can become `assert copy.deepcopy(health) is health`, plus the existing `ConfigurableMoE` check.
2. `pr-descriptions/05` and the README still say "TODO: add a test", but the test exists at `test_configurable_moe.py:177`.

## 6. 533c0ef182 [None][infra] upgrade mpi4py to 4.1.2
**Verdict:** blocked on the image tag; otherwise lean.
1. **(r3 open)** `jenkins/current_image_tags.properties` is not bumped. CI keeps mpi4py 3.1.5, so every ULFM test later in the series skips and passes green.
2. `test_mpi_session.py:49` (`test_mpi_comm_executor_singleton_bypasses_kv_device_selection`) behaves identically on 3.1.5. It proves neither the upgrade nor the patch, and the image build already fails if the patch does not apply. Delete it (**−12**) or replace it with `hasattr(MPI.Comm, "Shrink")`.
3. I could not verify locally that the patch context matches 4.1.2's `futures/_core.py`. Say in the description that the image build is the proof.

## 7. 72e7694c1e [None][feat] WideEP recovery authority
**Verdict:** mergeable after small fixes.
- The state machine fails closed.
- Every client wait has a deadline, sends are capped at 5 s, and HMAC uses `compare_digest`.
- It stands alone: nothing imports it yet.
- The test list (`l0_cpu`) and the headers are correct.

1. `__init__.py:17-48` re-exports 14 names that only the test uses, and `authority.__all__` (`:50`) is missing `router_socket`. Reduce `__init__` to the header plus the docstring, and point the test at the submodules. **−32.**
2. `Register.pid` (`protocol.py:52,56`, `client.py:150`) is never read by the authority. Delete it, or log it in `_on_register`. **−3 src, about −5 test.**
3. `RecoveryAborted.__init__` and `.reason` (`client.py:98-100`) have no reader. Raise with an f-string at `:197`. **−4.**
4. Drop the `TimeoutError` base on `RecoveryTimeout` (`client.py:103`); nothing catches it.
5. `RecoveryClient.recv` (`client.py:164-178`) is public, but only `wait` calls it. Rename it `_recv` and give it a one-line docstring. **−3.**
6. Optional: the constructor checks at `authority.py:96-99` guard inputs that already fail closed. Drop them and `test_invalid_construction` (`test:206-215`). **−4 src, −10 test.**
7. **(r3 open)** Neither the commit message nor the PR description should say "torch-free"; running the authority needs a TRT-LLM install. `authority.py:25,52,312` uses stdlib `logging`; switch to `tensorrt_llm.logger`.
8. `protocol.py:17-19` says the key "authenticates peers". It authenticates messages, not routing ids: with `ROUTER_HANDOVER` (`authority.py:294`) and guessable `rank-N` ids, a peer without the key can steal a route. That fails closed (timeout, then Abort), so state it accurately.
9. Public API: the `RecoveryAuthority.survivors` property (`:114-116`) lacks a docstring. `serve()` (`:266`) is in `__all__` but does not document `key`, `stop` or `poll_interval`.
10. Tests (`test_recovery_authority.py`):
    - `:84-88` is covered by the `_register_all` and `_registered` asserts. Delete it. **−6.**
    - Fold the three pre-registration violation tests at `:161-180` into one parametrized test. **About −8.**
    - `:361-367` uses `sleep(0.2)` and then asserts nothing changed. Instead, register normally afterwards and assert admission (deterministic).
    - `:377`: delete the duplicate wrong-tag frame. **−1.**
    - The e2e clients at `:326-358` need a closing fixture, so a failed assertion cannot hang `context.term()`.

## 8. 46cb99c8a0 [None][feat] ULFM survivor communicators for host collectives
**Verdict:** mergeable after trims.
- FT-off parity holds: the main path is unchanged, and there is one `None` check per collective.
- The watch loop is non-blocking. `Shrink` is bounded by the hang detector in PR 14.

1. **The ULFM test's last step cannot fail.** `test_mpi_survivors_ulfm.py:110-113` says "a clean stop never fires" but asserts nothing; `failure_seen.clear` as the callback hides that. Use a fresh event and `assert not fired.is_set()` after `stop()`.
2. **The PR description is wrong.** It says "the base `Distributed` and the test fakes get the same interface", but `tests/unittest/_torch/disaggregation/fake_dist.py` has neither `fill_value` nor `live_ranks`. Fix the fake in the PR that first passes `fill_value` unconditionally (PR 12), or drop the claim.
3. **`is_mpi_peer_failure` has no test that runs in CI.** The fakes monkeypatch it, and the ULFM test skips. Add about 10 lines testing `_peer_failure_error_classes()` against a fake `MPI` in which one class aliases `ERR_UNKNOWN`. Keep the alias filter.
4. **The layout check runs too late and is duplicated.** `tp == moe_ep == world` is checked in `shrink()` and again in `PeerLossRecovery.create()` in PR 14. Move it into `enable_peer_loss_recovery()`, which runs at startup, and delete the copy in `create()`. **About −8 across the series.**
5. `_interruptible(operation)` is a decorator factory, yet every call passes the method's own name. Make it a plain decorator that uses `func.__name__`. **−5.**
6. `tp_cp_allgather_int64(fill_value=)` is dead: recovery requires `cp_size == 1`, and only a test uses it. Revert to main's signature and drop the two test lines. **−10.**
7. Three fields carry one fact. `_survivor_comm`, `_live_ranks` and `_compact_ranks` are always set together, but methods test different ones.
   - Drop `_compact_ranks`; `_compact_root` can use `_live_ranks.index(root)`.
   - Use `_live_ranks is None` as the single "installed?" test.
   - Make `shrink()` require `enable_peer_loss_recovery()`, so `allgather` and `barrier` collapse to a single `_host_world is None` branch.
   - **About −9.**
8. The `__init__` field comments (9 lines) can be one short line per field. **−5.**
9. `PeerFailureWatch(poll_interval=)` is never passed. Use a module constant `_WATCH_POLL_S`. **−3.**
10. Code that defends against impossible states:
    - The `MPI.UNDEFINED` filter in `_world_ranks`: every group derives from `COMM_WORLD`. **−1.**
    - **(r3 open)** The `AttributeError` in `_known_failed_ranks`, together with `test_failed_ranks_empty_without_ulfm_query` (`test:288-293`). **−7.**
11. `Distributed` has four stubs nothing can reach: `revoke`, `shrink`, `install_survivors` and `watch_peer_failures` all raise `NotImplementedError` behind `enable_peer_loss_recovery()`'s stub. Drop them (**−20**), or, if they are meant to document the Ray contract, give each a one-line docstring.
12. `_world_ranks(group)` frees its argument; say so in the docstring.
13. Tests (`test_mpi_survivors.py`, 468 lines):
    - `:259-279`: the `error_class` parametrization tests the fixture's own lambda; 14 cases become 7.
    - `:313-317`: redundant, and asserts the private `_survivor_comm`. **−6.**
    - Merge the watch-stop tests at `:431-447` and drop `FakeComm.freed`. **−9.**
    - Parametrize `:413-428`. **−6.**
    - `FakeComm.shrink_members` (`:79,106-108`) is never set. **−3.**
    - Trim the `:250-256` overlap. **−2.**

## 9. 1269f756ec [None][feat] fault-tolerant execution descriptor for the NVLinkOneSided all-to-all
**Verdict:** mergeable after small fixes.
- No kernel correctness issues.
- It stands alone: only the ops change, and they are tested through `torch.ops`.
- FT-off kernels are untouched apart from template plumbing. The trailing optional schema argument is backward compatible.

1. `moeAlltoAllKernels.cu:301-318` has a wrong comment and confusable names.
   - The comment on `load_abort_request` says "recording the cancellation", but `abort_requested` is the function that records it.
   - Rename them `is_abort_requested` (pure load) and `observe_abort` (load plus record), or apply item 2.
2. **`MoeA2AAbortState::status` is redundant** (`.h:69-79`, `.cu:310-318`).
   - It is written only after `requested` is seen, and only the test reads it. PR 5's `abort.any()` needs only `requested`. `kHealthy` is never used.
   - Fix: keep `int32_t requested` plus padding, so the descriptor stays `kRankMaskWords + 1` words, and delete `status`, `kHealthy`, `kCancelled` and the per-thread `atomicExch`.
   - **About −14 C++, −3 Python and tests.**
   - At minimum, delete `kHealthy`.
3. **The PR description names a test that does not exist.** `test_pending_fence_waits` does not exist. The trap is covered by `test_fence_timeout_still_traps` (`:277`) and the abort release by `test_cancel_and_exclusion_on_replay`. Its EP32 figure (2.6 s) also disagrees with DESIGN §5 (1.4 s): name the run or use one number.
4. `l0_b200.yml:160`: the entry sits inside the KDA/Kimi block. Give it its own `# MoE A2A execution descriptor` comment line, or move it next to the MoE comm tests. The stage itself is right.
5. `moeAlltoAllOp.cpp:170-180` `validateFixedSlotWorkspace`:
   - `dim()==2` is already checked at `:643`, and total size at `:649`.
   - The stride and cacheline checks guard the layout of a workspace that is always contiguous.
   - Keep only the runtime ≤ static capacity check, inline. **−6.**
6. The launcher `TLLM_CHECK`s at `.cu:1229,2336` repeat `resolveExecutionDescriptor` (`.cpp:158`); the op is the only caller. **−2** (optional).
7. `.h` `MoeA2AGatherParams::expertIdPayloadIndex`: the comment's "-1 if none" describes an impossible state, because the op requires the index (`.cpp:552`).
8. Nits:
   - `slotTokens` (`.cpp:590`) is a misnomer without a descriptor; call it `tokensPerPayload`.
   - The code mixes `executionDescriptor.has_value()` and `params.execution_descriptor != nullptr`; use one `bool const fixedSlots`.
   - The new snake_case fields (`active_rank_mask`, `source_stride_bytes`) should be camelCase like the other new structs.
9. Tests (`test_moe_a2a_abort.py`):
   - `:249-251` asserts the historical name `"abort_state"`; delete the loop. **−3.**
   - Three of the six descriptor-validation cases at `:217-226` hit the same check, and the `del` hack at `:238` can go. **−4.**
   - `:125-131,159`: derive `source_stride` from one eager dispatch instead of re-deriving the 128-byte alignment. **−5.** Keep the test itself; it is the quarantine evidence.
   - Explain or drop the `fill_(0xFF)` at `:105`.
   - Consider renaming the file `test_moe_a2a_execution_descriptor.py`.

## 10. bb5081bcf0 [None][feat] MoE execution state for WideEP recovery
**Verdict:** approve after trimming.
- FT-off: nothing is bound, and the workspace size, watchdog and graph rejection are unchanged.
- Keep `MnnvlMemory.pin`, the fixed-slot bound, the abort-record check in `install_mask` and the CFT rejection.

1. **`EPGroupHealth.commit_failed` has no caller here, and its check can never fire** (`ep_group_health.py:149-185`, tests `:492-513`). Its only caller (PR 11/14, `model_engine.py:975-979`) reads `health.generation` and passes it straight back on the same thread.
   - Delete `commit_failed` and `_mark_failed_locked` and restore `mark_failed`.
   - The caller becomes `if not health.mark_failed(rank): raise RuntimeError(...)`.
   - **About −35 src, −22 test.** This also makes the PR stand alone.
2. `install_mask(words)` (`execution_state.py:135-155`) compares its argument with its own source. Make it `install_mask(self)` and have it publish `self.health.snapshot().mask`. Keep the sync and the abort-record `RuntimeError`. **−7 src, −6 test.**
3. `dispatch_kwargs()` and `combine_kwargs()` (`:116-122`) are identical one-liners. Pass `execution_descriptor=state.descriptor if state else None` at `nvlink_one_sided.py:970,1127`; the schema default keeps FT-off parity. **−8, −2 test.**
4. `bind_execution_state` (`nvlink_one_sided.py:613-622`): `for_model` already rules out the idempotent re-bind, a health mismatch and `phase != "idle"`. Keep the "already bound" and CFT checks. **−6, −8 test.**
5. `MoeExecutionState.__init__` (`:31-34,52-60`):
   - The world-size check duplicates `get_mask_words()`.
   - `_mask_words` re-implements it; use `torch.tensor([...], dtype=torch.uint64).view(torch.int64)`.
   - The "full initial membership" check guards an impossible state; delete it and `test_construction_requires_full_membership` (`:139-143`).
   - **−9, −6 test.**
6. `retain()` keeps two dicts keyed by the same id; use one `dict[int, tuple[MnnvlMemory, Callable]]`. **−3.**
7. `reset_local_workspace` (`nvlink_one_sided.py:626-635`):
   - Only `retain` calls it, so make it `_reset_local_workspace`.
   - Comment the magic `2*ep*4` as `recv_counters[2][ep] int32` (it mirrors `moeAlltoAllOp.cpp:310`).
   - The tests re-implement it: `test_moe_execution_state.py:175-186` and `test_moe_a2a_late_writes.py:165-170`. Call the real method instead, or state that it is an oracle.
8. Test placement:
   - **(r3 open)** `test_moe_execution_state.py` is single-GPU but listed in `l0_gb200_multi_gpus.yml:21`. Move it next to the abort test in `l0_b200`.
   - Add `TIMEOUT (10)` to the late-writes entry, which spawns 3×2 processes against a 300 s store timeout.
9. Duplicate tests:
   - `test_graph_captured_before_install_runs_after_peer_exclusion` (`:146-163`) is a subset of PR 9's `test_cancel_and_exclusion_on_replay`. **−18.**
   - `:129-136` duplicates `:207-209`. **−8.**
   - The late-writes "cancel" mode (`:187-209,234-239,284-302`) repeats the single-process abort tests. **−45.** Keep the kill test; it is the 1a.8a evidence.
10. **`test_moe_a2a_late_writes.py` belongs in neither PR.** Its only PR 5 dependency is `fixed_source_slots=True`. Either move it to PR 4 with an explicit slack, or make it drive `MoeExecutionState` (`request_abort`, `rearm`, `install_mask`).
11. Nits:
    - The `runner.py:253-254` comment names the scheduler, but the model engine installs the mask. Rewrite it as "Bound to every NVLinkOneSided layer when recovery is enabled; None otherwise."
    - `wide_ep_ft.py:85` `or recovery` is redundant. **−1.**
    - The `recovery_enabled` docstring (`:46-52`) can be one line.
    - The `MOE_DEVELOPER_GUIDE.md` addition restates DESIGN, and one line is over the wrap width. **−5.**
    - `unpin()` lacks a docstring.
    - Bound layers still create a watchdog (`nvlink_one_sided.py:544-545`). Verify it does not start a thread per layer.
    - DESIGN §6 says "needs no extra workspace", but `calculate_required_workspace_size` adds an `ep*MAX_PAYLOADS*128` pad.
    - **(r3 open)** The leaf-module env predicate, so FT-off MoE construction does not import zmq.

## 11. ea428767c2 [None][feat] resident-replica EPLB placement after a rank loss
**Verdict:** the logic is fine; fix the test list, the tests and the description. FT-off is unchanged: the new methods have no caller, and the `_shared_barrier` tolerance is gated.

1. **The GPU tests do not test the device routing** (`test_resident_placement.py:31-58,150-170`).
   - `_FakeLayer.route` re-implements round-robin in Python. CUDA is needed only because `moe_load_balancer.py:950` hard-codes `device="cuda"`.
   - Either use one real static `SingleLayerMoeLoadBalancer` with one accept and one reject case, or take the device from the layer and make both tests `cpu_only`.
2. **The `l0_b200.yml:161` entry is redundant, and the PR description is stale.**
   - The `l0_b300.yml:53` pre_merge stage already collects all of `unittest/_torch/executor`.
   - The description and the README still say `l0_cpu`, plus a TODO.
   - Delete the line and rewrite Test Coverage.
3. **(r3 open)** `prepare_resident_placement` (`model_engine.py:943-950`) is a pass-through; call `masked_placement_digest` directly. **−8.**
4. `_layer_routing_shapes` (`moe_load_balancer.py:799-800,884`) is an index-aligned shadow list. Store `top_k` and the slots per rank on each `SingleLayerMoeLoadBalancer` in `add_layer`.
5. Tests:
   - `:87-95` is covered by the install test. **−9.**
   - The module comment at `:27` restates the markers. **−2.**
6. Docstrings:
   - Document the `ValueError` for out-of-range ranks.
   - Trim `install_resident_placement` (`:952-967`) to the ordering rule. **−3.**
7. Not standalone: the APIs get their first caller in PR 14. Say so in the description.

## 12. 1d0315e0fb [None][refactor] let attention-DP routing skip absent ranks
**Verdict:** request changes. Three hunks are unreachable with FT on, and the FT-off claim is too broad. Confirmed against the `create()` requirements in `peer_loss_recovery.py:153-160`.

1. **Unreachable hunks.** Delete them and drop them from the description (**about −30** including docstrings):
   - **ADP balance fill:** `py_executor.py:144-147` and `:6680-6686`. They need `attention_dp_enable_balance`, which `create()` rejects. **−9.**
   - **Fetch-status gather:** `_allgather_model_parallel_status` `:4019-4026`. Only the benchmark-disagg gate uses it, and that needs a transceiver, which `create()` rejects. **−6.**
   - **KV-pool rebalance:** `:5315-5317`. It runs only with `_enable_remote_tail_adp`, which needs a transceiver. **−3.**
2. **(r3 open)** The `List | Dict` unions (`request_utils.py:109,133-137,151`; `adp_router.py:513-514,538-546`) keep a list branch that only tests use. Make them `Dict[int, int]` and convert the test literals with `dict(enumerate(...))`. **About −10.**
3. Stale hint: `py_executor.py:6403` `all_ranks_num_active_requests: Optional[List[int]]` now receives a dict.
4. **The FT-off claim holds only for `DefaultADPRouter`.** `KVCacheAwareADPRouter` (`adp_router.py:754,782`) and `ConversationAwareADPRouter` (`:991,1012`) still index dense lists. A strict pin to a rank at or above tp_size now passes admission and then raises `IndexError` inside them. Check `unservable_requests` before those routers, or scope the claim.
5. `num_admitted_requests` (`:6486-6501`) counts occupied capacity, so call it `num_occupied_requests`.
6. Optional: `_fail_unservable_requests` (`:6545-6562`) repeats the error-response construction at `:8540-8564`.
7. Tests:
   - No test covers the absent-rank admission accounting or the CUDA-graph vote filter. Add one `_gather_adp_graph_batch_info` case with 3 live ranks (**about +10**).
   - `test_default_router_honours_strict_pin_to_live_logical_rank` (`:81-86`) duplicates existing coverage. **−6.**

## 13. 928596db92 [None][feat] keep the MPI session serving after a peer loss
**Verdict:** OK after trims. FT-off is a no-op apart from an uncached env read on every task.

1. **(r3 open)** `mpi_session.py:37-39` `_recovery_enabled()` duplicates the cached `wide_ep_ft.recovery_enabled` and runs on every serve-loop message and future callback. Use one cached leaf predicate. **About −4 here, −8 in `wide_ep_ft`.** That also moves the `ValueError` for an authority address without FT mode out of a callback thread and into startup.
2. **(r3 open)** `_num_failed_ranks` (`:42-56`) duplicates `MPIDist._known_failed_ranks`.
   - Both catch an impossible `AttributeError`.
   - `_known_failed_ranks` never calls `Free()` on the group.
   - Use one `_utils.mpi_failed_ranks(comm)`. **About −10.**
3. Comments:
   - The `_live_worker_futures` docstring (`:983`) says "None", but it returns `[]`.
   - Trim `:1045-1047` to "mpi4py.futures cannot release a world that lost a rank." **−2.**
4. Optional: the shutdown at `:1044-1054` could reuse `_close_global_comm_executor(grace=0, abort=...)` (**−3**). Keep the direct `Abort(0)` if the concurrent `__exit__` race matters.
5. Tests (`test_mpi_session.py`):
   - Drop the unused `Get_error_class` and `ERR_*` in `_fake_ulfm_mpi`. **−6.**
   - Use a parametrized fixture for the env setup copied into two tests. **−8.**
   - Fold `test_broken_executor_counts_as_departure_only_after_a_failure` into the async-callback test. **−8.**
6. Not standalone: the degraded mode does nothing until PR 14. Say so in the description.

## 14. 7448f93dee [None][feat] recover the executor from a WideEP peer loss
**Verdict:** mergeable after trimming the tests.
- FT-off parity holds: every hook is behind `is not None`, enqueue-then-terminate is preserved, and the dummy-retire extraction is equivalent.
- Keep the bounded waits and fail-closed checks.

1. **The V2 first-chunk revert does not rewind the cursor** (`kv_cache_manager_v2.py:4282-4284`). Verified: this is not a bug. Re-admission runs `_settle_context_cursor` for first chunks when reuse is on (`:3866-3867`), and with reuse off the cursor is 0.
   - Add a reuse-match case to `test_reverted_context_is_not_offered_for_reuse`, which today only uses a fresh cache.
   - Optionally add `rewind_context_after_cache_drop(req, self.tokens_per_block)` for symmetry with V1 and `revert_allocate_context` (+1).
2. The non-first-chunk V2 branch (`:4287-4292`) re-implements `revert_allocate_context` with weaker semantics: no history length, and no history-larger-than-capacity case. Consider `self.revert_allocate_context(req)`, but check its `suspend()` for a request that is re-scheduled on the next step. **About −6.**
3. Fold the three `except HostCollectiveInterrupted: raise` blocks (`py_executor.py:8283-8286,8470-8471,8502-8503`) into `_raise_if_peer_failed`, with `abort_forward()` called first in `_forward_step`. **About −8.**
4. `_held_request_ids` (`py_executor.py:8359-8366`) has one caller in `PeerLossRecovery`; inline it there. **−4.**
5. The `revert_prepare_resources` guards (`kv_cache_manager_v2.py:4271-4282`, `resource_manager.py:1251-1255`) repeat `create()`. Replace them with a one-line precondition in the docstring (**−14**), or keep them if the KV owners want self-checking public methods.
6. Docstrings:
   - The "holds no request state" sentence in the `PeerLossRecovery` class docstring (`peer_loss_recovery.py:90-99`) is a review justification. **−5.**
   - The V2 `revert_prepare_resources` docstring (`:4260-4270`) can be 3 lines. **−6.**
7. `create()` (`peer_loss_recovery.py:141-197`):
   - Hoist `dist`; `:171` re-reads `executor.dist.mapping`.
   - Drop `getattr(..., None)` on attributes that always exist: `moe_load_balancer`, `spec_config`, `moe_execution_state`, `llm_args.sleep_config`. Keep the `getattr` calls on the KV manager.
   - **−3.**
8. Nits:
   - `RECOVERY_TIMEOUT_ENV` (`:42`) is module-only, so make it `_RECOVERY_TIMEOUT_ENV`.
   - `requests_routed(routed: Dict[int, list])` should be `Dict[int, List[RequestQueueItem]]`.
   - `_expect_generation` lacks a type hint for `message`.
9. Tests (`test_peer_loss_recovery.py`, 696 lines):
   - `:445-498` assert exact `remove_sequence` and `resize` calls on managers built with `__new__`. The real round trip in `test_kv_cache_revert.py` covers both. **About −50.**
   - `:142-149` repeats the happy path; `_recover` never reads `error.failed_ranks`. **−9.**
   - `:331-365`: parametrize the commit-gate tests and delete the setter test. **−15.**
   - `:309-314` tests a two-branch `if`. **−7.**
   - `:477,596,642` carry stale `enable_block_reuse` remnants.
   - `:590,661,672,684` repeat local imports; move them to the top. **−5.**
   - `:257-259` asserts `watch.stop.call_count == 2`, an implementation detail.
10. Integration test and docs:
    - `tests/integration/defs/wide_ep_ft/README.md`: the EP32 qualification section (`:37-61`) is not CI, and `:43-44` is garbled. Delete the README and put one sentence in the test docstring (**−61**), or at least delete the section (**−25**).
    - `docs/source/features/wide-ep-fault-tolerance.md:42-44` has a broken line wrap.
    - The doc's requirements list omits "hang-detection timeout above the recovery timeout" and "no conversation manager".
    - `peer_loss_driver.py`: `--victim`'s default and the `mode`/`victim` result keys are unused. **−2.**
11. Test placement: `test_kv_cache_revert.py` has no SM100 dependency. Move it from `l0_b200.yml:162` to `l0_a10.yml`, next to the KV-manager tests.
