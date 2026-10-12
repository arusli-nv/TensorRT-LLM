# Round 3 holistic review: `git diff 55279c109f` plus untracked files

Verdict: **not merge-ready yet.** The code is close: there are no correctness blockers in the kernels, the MPI adapter or the authority. What blocks merging is one failing unit test, a PR split that does not build or test PR by PR, CI coverage that silently skips, and docs that contradict the code.
Line numbers refer to the worktree as of this review.

## Blockers

1. **A unit test fails against the current code.**
   - `tests/unittest/_torch/executor/test_peer_loss_recovery.py:1047` still contains the parametrized case `(… "enable_block_reuse", True), "block reuse")`.
   - The uncommitted change removed the block-reuse requirement from `PeerLossRecovery.create()` (`peer_loss_recovery.py:142-170`). `create()` therefore no longer raises; it reaches `_register()` and fails with something other than the expected `ValueError`.
   - The last CPU run (`build-stage-r2-tests2`) predates this edit.
   - Fix: delete the case, or re-add the requirement (see Docs 1).
2. **The PR split in PR_PLAN does not stand alone.** It misses these edges:
   - **PR3 needs PR5.** `wide_ep_ft.py:46-55` `recovery_enabled()` imports `tensorrt_llm.executor.recovery.client`. PR_PLAN puts PR5 after PR3.
   - **PR3 is half-wired without PR8's files.** PR3 lets rank-mask mode run with CUDA graphs when recovery is on (`wide_ep_ft.py:85`). But the only binder is `MoeExecutionState.for_model`, called from `engine/runners/decoder/runner.py`, and the `model_engine.moe_execution_state` property and `close()` call are in `model_engine.py`. PR_PLAN assigns all of these to PR8, so at the PR3 tip, recovery env plus CUDA graphs fails at capture. Move the runner hook, the property and `close()` into PR3.
   - **PR7 needs PR4 and py_executor hunks.**
     - `cuda_graph_runner.py:301` uses `dist.live_ranks` (PR4).
     - `test_adp_host_exchange.py:218` expects `fill_value=1` (PR4 communicator, plus the py_executor `_can_queue` hunk at about line 3875).
     - `ADPRouter.unservable_requests` (`adp_router.py:369`) is only called from `py_executor.py:~6563` and `_fail_unservable_requests` (`~6590-6607`). Without those hunks it is dead code, and the advertised FT-off behaviour change does not exist.
   - **PR8 needs PR4 and has no tests of its own.** `moe_load_balancer.py:23,1076` uses `is_mpi_peer_failure` from PR4. Its new API (`covers_loss_of`, `masked_placement_digest`, `check_routing_excludes`, `install_resident_placement`) is tested only in `test_peer_loss_recovery.py:208-382`, which belongs to PR9. `check_routing_excludes` has no test anywhere.
   - **PR6 is inert until PR9.** `mpi_session.py`'s degraded mode cannot trigger without PR9. Fold it into PR9, or state this in the PR description.
   - **PR_PLAN is stale.**
     - `executor_request_queue.py` (PR9) has a zero net diff.
     - 44eded2a5c is fully reverted inside ed20a243ad, so the fold must skip both.
     - U4's test (`test_py_executor.py:643-662`) stubs `_raise_if_peer_failed` from PR9.
3. **CI does not actually exercise the feature.**
   - PR1 changes `docker/common/install_mpi4py.sh` to 4.1.2 but does not bump `jenkins/current_image_tags.properties`, so CI keeps mpi4py 3.1.5.
   - `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:51-59` and `test_mpi_survivors_ulfm.py` skip without mpi4py≥4 and an Open MPI that reports `ftmpi`.
   - Until a ULFM-capable image is in CI, the post-merge GB200 entries (`l0_gb200_multi_gpus.yml:97-98`) skip green. Confirm the CI image's Open MPI has ULFM and bump the tag, or the DESIGN §9 claim that CI covers the feature is false.
4. **The late-write reproducer (v7 1a.8a acceptance) is untracked and not in any test list.** `tests/unittest/_torch/moe/multi_gpu/test_moe_a2a_late_writes.py` is the only hardware evidence for 1a.8a, and MVP_ACCEPTANCE still marks 1a.8a open. Add it to the `l0_gb200_multi_gpus` pre_merge block, with a TIMEOUT, in PR2.

## Should-fix

### Layering and conventions

- **The MoE layer imports the executor control plane.**
  - `wide_ep_ft.py:53` imports from `executor.recovery.client`. That module imports zmq at module level, and `nvlink_one_sided.py` `__init__` calls `recovery_enabled()` unconditionally, so FT-off MoE construction loads the recovery package.
  - `mpi_session.py:36-39` re-implements the same predicate without the cache. It is called per serve-loop message and per future callback.
  - `peer_loss_recovery.py:28` reaches through MoE to read an env gate.
  - Fix: one cached env predicate in a leaf module (`wide_ep_ft.py` or `_utils`), which `recovery/client.py` imports. This also removes the PR3→PR5 edge.
- **The layout requirement is checked too late.** `tp == moe_ep == world` is only checked in `MPIDist.shrink()` (`communicator.py:~921-929`), so a mis-configured job passes startup and dies at the first loss. Move the check into `enable_peer_loss_recovery()` or `create()`.
- **`Get_failed` handling is duplicated** in `mpi_session.py:42-57` and `communicator.py` `_known_failed_ranks`. Use one `_utils` helper next to `is_mpi_peer_failure` (about −10 lines). The `AttributeError` in its except tuple cannot happen with mpi4py≥4.
- **The authority is not torch-free**, although DESIGN §6 says so. `python -m tensorrt_llm.executor.recovery.authority` imports `tensorrt_llm/__init__` and therefore torch. Fix the claim and require a TRT-LLM install on the authority host. Use `tensorrt_llm.logger` instead of stdlib `logging` (`authority.py:25,52`).
- **Wrong docstring.** `py_executor.py:~1494-1505` `_run_event_loop` says the hang detector "is stopped for recovery and restarted". `peer_loss_recovery.py:212` actually runs recovery inside it. Replace it with one accurate line.
- **Test placement.**
  - `test_moe_a2a_abort.py` and `test_moe_execution_state.py` are single-process, single-GPU tests sitting in the 4-GPU stage (`l0_gb200_multi_gpus.yml:21-22`). Move them to a single-GPU SM90+/SM100 list.
  - The ADP-router tests (`test_peer_loss_recovery.py:112-205`) belong in PR7's test files.
  - The EPLB tests (`208-382`) belong in a load-balancer test in PR8.

### FT-off parity

Everything else is gated: one `is not None` per hook, the `_interruptible` wrapper re-raises unchanged when `_host_world is None`, there is no extra `Dup`, and the kernels are opcode-identical. The following need attention:

| Change | Verdict |
|---|---|
| Strict ADP pin to an out-of-range rank now returns an error response (one extra `tp_gather`) instead of `IndexError` (`request_utils.py:133`, `adp_router.py:352`) | Acceptable only as its own PR7, with the py_executor hunks and tests in `test_adp_router.py` |
| Op schema gains a trailing `Tensor(b!)? execution_descriptor=None` (`moeAlltoAllOp.cpp:1190,1200`), plus host-side `TLLM_CHECK`s (`.cu:1229,2328`) | Acceptable |
| `cpp_custom_ops.py:728-764` fakes gain three args that main's fakes were missing. This silently fixes a pre-existing fake/schema mismatch on main | Acceptable; state it in the PR2 description or split it as a U-fix |
| Detection-only mode (`TLLM_FAULT_TOLERANCE_MODE=1` without an authority) also gets the new combine `__syncthreads_or` and descriptor checks | Not covered by the SASS parity claim; note it in PR2 |
| `tests/unittest/_torch/disaggregation/fake_dist.py:136` `tp_allgather_int64` lacks `fill_value` | Run the disaggregation unit tests once |
| FT-off performance | Parity is proven by the SASS diff only; the perf A/B is still listed as running in MVP_ACCEPTANCE. Attach it to PR2 and PR9 before claiming "performance unchanged" in the docs |

### Leanness: concrete deletions, about 150 source lines and about 60 test lines; about 270 test lines should move

- `moe_load_balancer.py:937-967` `reconfigure_mask_only_with_timeout`, plus the `deadline` plumbing in `model_engine.install_resident_placement` and 2 tests: **−45**. HangDetector already bounds recovery (B1), and the helper thread can leave a daemon stuck in `cudaEventSynchronize`.
- `execution_state.py`: **−18**.
  - `dispatch_kwargs`/`combine_kwargs` (`:117-123`) are identical; pass `state.descriptor` instead.
  - `_mask_words` (`:30-34`) duplicates `EPGroupHealth.get_mask_words()`.
  - `install_mask(words)` (`:139-157`) re-validates words derived from the same health object. Make it `install_mask()`.
- `authority.py` `--admitted-ranks`/`admitted_ranks` (`:86-88,101-109,192-199,320-324`) has no production user, and v7 admission is by intersection: **−20**, plus 2 tests.
- `model_engine.prepare_resident_placement` is a one-line pass-through: **−8**.
- `wide_ep_ft.recovery_enabled` wrapper, after the layering fix: **−10**.
- `peer_loss_recovery.py`: **−20**.
  - Delete the impossible rank-mismatch check at `:71-73`.
  - Trim the 20-line class docstring at `:77-98` to about 8 lines. Its V1-stores-blocks sentence reads like a contradiction of the rollback.
  - The Optional `_request_owners` None checks at `:117-119,273,291` should become a plain dict.
- `communicator.py` `tp_cp_allgather_int64` `fill_value` and the `np.tile` branch: no caller, and CP must be 1: **−6**.
- `request_utils.py:109` and `adp_router.py:513` take `List | Dict`, but production always passes a dict: **−8**.
- `test_peer_loss_recovery.py`: **−60**.
  - The mock KV-revert tests at `767-819` duplicate the real-manager round trips at `895-940`.
  - `1079-1086` can fold into the parametrize.
  - Drop the `# ----` banners, which neighbouring tests do not use.
- `test_moe_a2a_abort.py:253` asserts that a removed `abort_state` argument is absent. That is history in a test; delete it.
- `tests/integration/defs/wide_ep_ft/README.md` (95 lines) duplicates the user doc: the env table, authority launch and limits. Keep only the test procedure and the EP32 qualification steps: **about −40**.
- `MoeA2AAbortState::kHealthy` (`moeAlltoAllKernels.h:74`) is unused.

## Docs (`docs/source/features/wide-ep-fault-tolerance.md`) versus the code

1. **Block reuse.** The doc says "KV cache manager V1 or V2 without block reuse". The integration README says "no … KV cache block reuse". `create()` no longer rejects reuse, and DESIGN §1 says "block reuse on or off". Pick one answer:
   - If the reuse-safe rollback GPU test (MVP_ACCEPTANCE: "reuse test to run") has not passed, re-add the requirement and keep the test case.
   - Otherwise fix the doc and README and delete the test case.
2. **Missing requirements.**
   - The doc omits expert TP 1 and the requirement that `tp == moe_ep == world`.
   - It omits disabled sleep/wakeup and `CUDA_DEVICE_MAX_CONNECTIONS != 1`, both of which `create()` enforces.
   - It says "NVLinkOneSided" without saying how to select it. The test uses `TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED`.
   - `TRTLLM_MOE_A2A_FORCE_CFT=0` in the README and test is redundant, because CFT is opt-in (`nvlink_one_sided.py:400`). Drop it or explain it.
3. **mpi4py version.** "mpi4py 4 or later" understates the requirement. The session relies on the futures patch in `install_mpi4py.sh` (4.1.2). Say "the TRT-LLM container's mpi4py (4.1.2)" and Open MPI 5 with ULFM.
4. **Unverified launch example.** The documented `trtllm-llmapi-launch trtllm-serve` launch has never been run; REVIEW_FINDINGS lists it as open. Every qualification run used the LLM API driver. Verify it, or show the driver pattern.
5. **Unbacked performance claim.** "With it disabled, behavior and performance are unchanged" needs the FT-off perf A/B. The strict-pin change also alters FT-off behaviour, marginally.
6. **Authority host requirements.** The authority needs a full TRT-LLM install. "Any host the workers can reach" is fine, but the doc should say this.
7. **`MOE_DEVELOPER_GUIDE.md:231`.** It says `TLLM_FAULT_TOLERANCE_MODE=1` enables recovery. Recovery also needs `TLLM_FAULT_TOLERANCE_AUTHORITY` and its key; MODE alone gives detection-only mode, which rejects CUDA graphs.
8. **DESIGN §6.** It says the fixed-slot layout "needs no extra workspace". `nvlink_one_sided.py:290-292` adds `ep_size*MAX_PAYLOADS*128` B.

## v7 MVP gaps

- **1a.8a:** the reproducer exists but is untracked, has not been run in CI, and is still open in MVP_ACCEPTANCE.
- **1a.11:** the captured-collective audit and the all-active perf A/B are still listed as running.
- **1d.5:** degraded goodput and tail latency have not been measured. K-S6 (healthy-path gather cost at EP64) is still open.
- **Release gate "truthful health/metrics"** (v7 detail C; 1d.2 is not MVP-tagged):
  - Nothing in the diff changes health or metrics reporting, so `/health` reports healthy throughout recovery and in the degraded state.
  - Document this as a limit, or add a degraded flag.
- **Admission pause:** v7 says "stop replay/admission". DESIGN §7 justifies dropping the frontend pause because requests wait in rank 0's queue. This is an acceptable deviation, but state it in the PR9 description.
- **1d.1 launcher survival:** on multiple nodes, `mpirun` can outlive every rank after `MPI_Abort`. This is documented as a caveat. It is acceptable for an experimental feature, but it does not fully meet "bounded teardown".

## Nits

- `py_executor.py:396`: class-level `_peer_loss_recovery = None` duplicates the `__init__` assignment.
- `HostCollectiveInterrupted` is also raised for "MoE all-to-all" and "forward" interruptions (`peer_loss_recovery.py:266`, `py_executor.py:~8363`). The name is misleading.
- `communicator.py:221`: the `live_ranks` docstring says "TP ranks", but the ranks are world ranks. They are equal only because of PP=CP=1.
- `nvlink_one_sided.py`: `**({...} if state else {})` per call. Pass `execution_descriptor=self._execution_descriptor` instead. The `"MoeExecutionState | None"` string annotation differs from the file's `Optional`.
- `reset_local_workspace` hard-codes `2*ep_size*4`. Use a named constant or `itemsize`.
- `executor/recovery/__init__.py:17-48` re-exports everything, but production imports the submodules directly. `router_socket` is missing from `authority.__all__`.
- `test_mpi_survivors_ulfm.py:37-45` runs `ompi_info` at collection time.
- `tests/integration/defs/wide_ep_ft/__init__.py` is empty. That is fine, since sibling dirs do the same.
- Copyright headers on all new and modified files say 2026.
- Env-var prefixes are mixed (`TLLM_FAULT_TOLERANCE_*` vs `TRTLLM_ALLTOALL_WATCHDOG_*`). Prefer one prefix for new variables.
