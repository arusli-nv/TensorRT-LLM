# Red team 3: Python code review (lane code-review-py)

Scope: `git diff upstream/main v2 -- tensorrt_llm/` (v2 = cef4670789, 50 files, +7.5k/-0.4k), plus the test diff.
Method: read-only and static. This host has no torch, so nothing was imported or run. Three helper lanes covered
MoE, recovery/serve/launch and tests; I re-checked every finding of theirs that appears below.
Labels: CONFIRMED = the code path was traced and re-read. PLAUSIBLE = the mechanism was traced but its impact or
precondition is unverified.

## Verdict
- No blocker found. I found no import cycle in v2. The FT-off paths I traced match main: communicator `departed=`,
  ADP routing (dict vs list and negative pins), `_flush_pending_transfer_responses`, `_update_requests`,
  executor_request_queue, the MoE kernels and the autotuner `unique_id`.
- The one FT-off behavior change still open is the serve streaming path (M3).
- The most important new correctness gap is an overlap-loop window (M1). Neither a unit test nor the CI ULFM lanes
  exercise it (M4).

## Major
M1. Overlap loop: a step committed but not yet processed loses its KV update when a peer loss interrupts the flush gather.
- **Where:** py_executor.py:5991-6005 (`commit_step(prev)`, then `_update_requests(prev)`); :6032 (the per-step ADP
  gather in `_flush_pending_transfer_responses`, and optionally `_emit_first_token_responses`); :6090 →
  `_process_previous_batch` → `_update_v2_generation_resources(prev)` at :6250. Also `discard_interrupted_steps`
  at :8678-8714, which sets `previous_batch = None`.
- **Scenario** (overlap + MTP + KV V2, the standard recipe):
  1. Rank A commits step N, applies its tokens, then a loss interrupts the flush gather.
  2. The recovery discards only step N+1 and drops step N without running
     `kv_cache_manager.update_resources(N)` or `resource_manager.update_resources(N)`.
  3. Each generation request of step N keeps a stale `_inflight_draft_lens` entry (kv_cache_manager_v2.py:3570,
     popped oldest-first at :6597). Every later update rewinds by the width one step older.
  4. Step N's unaccepted draft slots are never rewound, and `try_commit_generation_blocks(N)` is skipped.
- **Impact:** fixed MTP width: up to `draft_len` extra KV slots per affected request until it ends. Variable
  width (speculation gate, dynamic draft len): a wrong rewind, the bug kvm2:6612 warns about (freeing pages step
  N+1 still writes). Step N's tokens stream one step late.
- **Status:** window CONFIRMED; impact PLAUSIBLE; untested. **Fix:** under recovery, run
  `_update_v2_generation_resources(prev)` right after `_update_requests(prev)`, as the non-overlap loop does
  (:5091-5094), and skip it in `_process_previous_batch`; or have the discard finish a committed `previous_batch`.
  Test: interrupt the flush gather after a commit; assert `_inflight_draft_lens` lengths and KV capacity.

M2. The configuration is env- and file-based, outside LlmArgs (CONFIRMED). Policy issue.
- **Where:** `TLLM_FAULT_TOLERANCE_MODE`, `_AUTHORITY`, `_AUTHORITY_KEY`, `_CONFIG` (a YAML path read by every
  worker and the authority, recovery/config.py:131); _utils.py:516-541. AGENTS.md requires LlmArgs, the golden
  manifest, serve schemas and API-stability tests for user-facing config; the user's code bar is "config not env".
- **Fix:** add a `fault_tolerance: FaultToleranceConfig` field to `TorchLlmArgs` and regenerate the manifest and
  schemas. Keep only the HMAC key in env. Pass the authority its settings by CLI or the same YAML.

M3. Serve streaming changes with FT off (CONFIRMED; open since round 2, #4).
- **Where:** openai_server.py:285 (`_end_stream_on_error`), wrapped unconditionally at :2417, :3209 and :3379.
- **Change from main:** main logged and re-raised (connection aborted). Now every deployment ends the stream with
  `data: {"error": {"message": str(exc)}}` + `[DONE]`: raw exception text to clients, bare `except Exception`
  with no noqa/reason. Anthropic `/v1/messages` (:2473) feeds that body to `reframe_openai_stream`, whose
  `ChatCompletionStreamResponse(**...)` likely rejects the error chunk (PLAUSIBLE). Completions went from two
  `[DONE]` to one (a fix, but unrelated to FT). Tests cover only the helper, not the endpoints.
- **Fix:** gate the wrapper on `fault_tolerance_mode()`, or split it into its own PR with endpoint tests
  (chat, Responses, completions, Anthropic).

M4. The tests do not reach the integration points (CONFIRMED / PLAUSIBLE).
- (a) CONFIRMED: no test runs the real executor loops with the hooks. test_peer_loss_recovery.py:162 mocks
  `discard_interrupted_steps`. A misplaced `step_prepared`/`step_released`/`commit_step` call
  (py_executor.py:4934/4975/5069, 5844/5910/5995) passes every pre-merge test.
- (b) PLAUSIBLE: the ULFM tests skip unless `TLLM_REQUIRE_ULFM=1` is set
  (tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:64-80, test_mpi_survivors_ulfm.py:~25-45).
  Nothing in jenkins/, docker/ or scripts/ sets it or installs ULFM. The 12 new l0_gb200_multi_gpus post-merge
  entries (:102-113) can stay green while testing nothing. They also cost up to about 9 h of post-merge time
  (12 x 45 min with model loads).
- **Fix:** a CPU test that runs a few overlap and sync iterations with a stub engine, interrupts between them,
  and asserts the discard order [N+1, N]. Set `TLLM_REQUIRE_ULFM=1` on the GB200 stage, or do not register
  those tests until the image has ULFM. Fold the parametrizations into fewer server launches.

M5. Import layering: `_torch/moe` now loads the executor and llmapi packages. No cycle exists today; the edge is
  CONFIRMED.
- **Where:** moe_load_balancer.py:30 imports `tensorrt_llm.executor.recovery.config`. That runs
  `executor/__init__` (worker → base_worker → pyexecutor; executor.py:29 → `llmapi/__init__`), and config.py:22
  imports `llmapi.utils`.
- **Problem:** a low-level MoE module joins the existing executor↔llmapi cycle, so import order now matters. This
  is the same shape as tonight's breakage (820e66b4ff, reverted by 8d0ec8cf1c). It also means the "standalone"
  authority needs torch and the whole product tree.
- **Fix:** make recovery/config.py depend on pydantic only (its own `extra="forbid"` base), and import it lazily in
  moe_load_balancer, as `_largest_failure_domain` already does. Better: move config/protocol/client to a leaf
  package outside `tensorrt_llm.executor`.

## Minor
m1. Detection-only FT with CUDA graphs now fails at warmup, not construction (CONFIRMED): main's
  `get_wide_ep_ft_options` raised ValueError on `use_cuda_graph`; v2 (wide_ep_ft.py:66) builds the model and
  raises RuntimeError during capture (nvlink_one_sided.py:1150). Fix: restore the early raise when
  `rank_mask_enabled and not peer_loss_recovery_enabled() and use_cuda_graph`.

m2. Detection-only `/health` changed (CONFIRMED): `TLLM_FAULT_TOLERANCE_MODE` exists on main; with it set and no
  authority, `/health` now returns `{"state":"unprotected"}` instead of an empty 200 (openai_server.py:1710).
  Fix: call it out in the PR, or keep main's bare response.

m3. FT-on cost off CuteDSL NVFP4 (CONFIRMED path): `_compact_receive_slots` (moe_scheduler.py:1098) allocates
  and copies `x`/`x_sf` per MoE layer per step for TRTLLM-Gen, DeepGEMM, quantized Cutlass; interface.py:577 counts
  ghost slots in `num_slots`, so their tuner keys differ from FT-off (PLAUSIBLE tactic drift). Fix: skip when
  `source_tokens == slot_tokens`; require a receive-slot backend for recovery or map tuner keys to the loaded layout.

m4. `mpi_failed_ranks` hides errors (CONFIRMED): _utils.py:647 maps any `MPI.Exception` from `Get_failed` to "no
  failures" (docstring promises NotImplementedError). So the ULFM probe (communicator.py:339-346) can pass on
  ERR_UNSUPPORTED, and `_interruptible`/`_num_failed_ranks` can silently see 0. Fix: map unsupported classes to
  NotImplementedError; re-raise the rest.

m5. `peer_loss_recovery_enabled()` raises a ValueError on a misconfiguration (PLAUSIBLE).
- _utils.py:526, called in `mpi_async_error_callback` → `_is_departed_peer_failure` (mpi_session.py:102);
  concurrent.futures swallows the raise, so a non-rank-0 worker death is not forwarded. Fix: validate once at
  startup; keep the predicate non-raising.

m6. (PLAUSIBLE) authority.py:687 `serve()` catches only RuntimeError/ValueError from `router.get()`: an
  authenticated but unloadable pickle (version skew) ends the authority loop. fault_tolerance_status.py:124-150:
  `Status` has no correlation id (a late answer reads as current); an unpickle error makes `/health` 500.

m7. hang_evidence.py:192/286: with the default `pmix_ranks=()`, `_kill_rank` raises an IndexError that the
  `except (PmixUnavailable, RuntimeError)` does not catch, and the fence thread dies. Production passes
  `pmix_ranks`. CONFIRMED in code.

m8. mpi_session.py:131-172: `_finalize_and_exit` stops only the startup watch. The hang-evidence and `rank_fence`
  threads can still be inside PMIx/MPI during `MPI_Finalize`; a 30 s timer bounds this. It still reaches
  `peer_loss_recovery` through `sys.modules` (open round-2 #32). PLAUSIBLE.

m9. Discard restores `py_draft_tokens` by reference (py_executor.py:8693, peer_loss_recovery.py:643);
  drafter.py:97 / speculative/utils.py:1039 pad it in place, so the "restored" list is padded and
  `py_draft_tokens_effective_len` counts padding. Fix: record a copy. PLAUSIBLE; acceptance stats only for MTP.

m10. Still open from earlier rounds: P1 (a loss after Commit escapes `_recover`, peer_loss_recovery.py:1041/1061,
  ends the job); module sizes (peer_loss_recovery.py 1657, authority.py ~890, moe_load_balancer.py +~760);
  HangMonitor/hang_evidence beside HangDetector with their relationship undocumented.

## Nits
- tests/unittest/_torch/disaggregation/fake_dist.py:136 names the parameter `fill_value=`, but the product
  keyword is `departed=`. Its `tp_allgather` also lacks `departed`. Any fake-dist test reaching
  `_can_queue`, the adp_router gather or `_flush_iter_stats_synced` would raise TypeError.
- Hook gating is inconsistent. `requests_cancelled` and `step_released` (non-overlap, :4975) run unguarded on the
  base hook, while the others check `.enabled`. Pick one style.
- trtllm-llmapi-launch:183 hard-codes 75, duplicating `STALLED_EXIT_CODE`. An `MPI_Finalize` error during a
  clean FT shutdown (status 1) self-SIGKILLs.
- Comments narrate rationale or history rather than explaining the code: mpi_session.py:83, the launcher comment
  at :174 ("PRRTE survives aborted processes ..."), the moe_load_balancer docstring timings (round-2 #19).
  Move them to the PR description.
- `_enqueue_responses` still detects a `ResponseContribution` by duck typing (`tuple`, `len == 4`,
  `list`, py_executor.py:9329). A NamedTuple would make the check explicit.
- moe_load_balancer.py:1573 `restore_copy_plan` divides by `len(live)` without a guard (ZeroDivisionError with an
  external `base`). :1252 sets `copy_nodes` inside `add_layer` (mixed layers would leave `home_ranks` None).
  :1357 calls `nvmlInit` without `nvmlShutdown`.
- Flaky timing margins: test_peer_loss_recovery.py:427-438 (0.2 s deadline), :594/:607;
  test_hang_evidence.py:185-213 (at least 10 heartbeats per second). Use a fake clock.
- test_hang_evidence.py lacks a module `cpu_only` mark; test_kv_cache_revert.py runs twice (l0_b200.yml:199 + dir).

## Checked, no finding
`LogitsStorage`/`PyResult` checkpoint/rollback (output keys pre-created); `revert_prepare_resources` order and
reused-prefix rewind; quarantine failures buffered via `_pending_transfer_responses`; `SchedulingParams` is a
dataclass; `Distributed._tp_ranks` cached and every subclass calls `Distributed.__init__`; no new
`.item()`/`.tolist()` on the FT-on MoE decode path; product broad `except`s carry noqa+reason except
openai_server.py:291 (M3).
