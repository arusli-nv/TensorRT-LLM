# Red-team lane "pr-stack": WideEP FT stack as upstream maintainers will see it
Personas: MoE/comm owner, runtime/executor owner, llmapi/serve owner. Read-only. Each PR reviewed as `git diff <parent>..<branch>`.
Base prmain-all 6837295b9c. Tip ftwave2/09 f1a810619e equals v2 cef4670789 in tensorrt_llm, cpp/tensorrt_llm, docs and examples;
only tests differ (28 files +2581/-4938, the 052627db76 trims). Checked here: diffs, commit metadata, cross-PR blame churn, and that
every test the descriptions cite exists at its boundary. Nothing built or run in this lane; run evidence is quoted from redteam3/tests.md.

## Stack-wide facts
- Commits: 11 single-commit branches, titles `[None][feat] WideEP FT n/10: ...`, exactly one `Signed-off-by: Andrew Rusli`,
  author = committer, no Co-Authored-By or AI lines. PR bodies end with "Generated with Claude Code" (user's call to keep or drop).
- Build/test state. Descriptions say "Not run yet", which is wrong both ways:
  - The older tree ft-final2 8d0ec8cf1c was built and tested (tests.md §2): moeLoadBalancerTest 39/39; GPU MoE group 127 passed;
    CPU group 953 passed, 5 failed (stale fakes, later fixed in tests only); real-ULFM test_mpi_survivors_ulfm passed (custom OMPI).
  - test_nvlink_one_sided: first case F, then the job step was killed; not triaged. test_ring_restore_staging never ran.
  - Never compiled: the v2 C++ fixes (peerFailureWatch detach, restorePlacement checks, 2 new gtests) and the trimmed
    moeLoadBalancerTest.cpp. No stack boundary (02/03/04/06/08/09) has been built on its own. cef4670789's test edit: no rerun.
- CI: the 9 post-merge `wide_ep_ft/*` e2e cases (TIMEOUT 45) and test_mpi_survivors_ulfm skip (no ULFM in the image). Numbering:
  "10/10" merges before 8 and 9, and 11 PRs are numbered "/10".
- New user surface: 3 env vars (`TLLM_FAULT_TOLERANCE_AUTHORITY`, `_AUTHORITY_KEY`, `_CONFIG`) plus a side YAML
  (`FaultToleranceConfig`). No LlmArgs field, so no golden-manifest or schema regeneration was triggered.
- Cross-PR churn (git blame of each PR's deleted lines): about 245 lines. Largest: 09→06 56; 07b→07a 28; 08→05 27; 08→10 17;
  07a→04 15; 09→10 12. Against wave 1: 7a rewrites 14 docstring lines of prmain/04 (revert_prepare_resources), 7b 3 lines of prmain/01.

## Per PR
**01 ft-config** (8 files, +259/-10; product 117)
- One concern. Foundation fails strictly: `fault_tolerance_config()`, `extra_slots`, `failure_domain`, `recovery_timeout_s` and
  `peer_loss_recovery_enabled()` have no product reader here (first readers in 03/06/07a); the description says only "only adds configuration".
- Launcher SIGKILLs on any non-zero exit under recovery. PR 8 then adds `-ne 75` and rewraps the comment, leaving a broken wrap
  ("crashed\n# by a signal"); putting the 75 exemption here removes that churn.
- Description: FT-off line accurate; the LlmArgs deferral is one sentence with no reason. Tests l0_cpu. **Ready with description fixes.**

**02 a2a-ft-kernels** (19 files, +2083/-170, C++)
- Open op-contract bugs: the fake `moe_a2a_dispatch` returns `[ep, runtime_max_tokens_per_rank, ...]`, but with `source_valid_counts`
  the real op returns `[ep, MAX_NUM_TOKENS, ...]` (moeAlltoAllOp.cpp:630); `abort_state` is declared `Tensor?` though the kernels
  write its status (:1203).
- Resolved: capture refusal for unbound comms is back in dispatch. MoeExecutionState's first caller is PR 3 (stated).
- FT-off line correct (incl. the FLAG_VAL 8-byte shift); PERF TBD. **Needs code change** (~4 lines: fake shape, `Tensor(f!)?`) + boundary build.

**03 moe-receive-slots** (18 files, +721/-73, C++)
- One concern (backends read slots); the runner/model_engine binding is runtime code riding along.
- Open: the CuteDSL tuning key is `(unique_id, source_tokens, slot_tokens)` with `source_tokens = runtime_max_tokens_per_rank`
  (nvlink_one_sided.py:1042), so with FT on the class-level cache grows per distinct batch size. The `do_finalize=False` refusal is
  raised after dispatch, so peers hang in combine instead of erroring. Undisclosed FT-off slice `input[:token_selected_experts.size(0)]`.
- **Needs code change** (key on `source_tokens > 0`; move the refusal to bind time).

**04 ulfm-survivors** (12 files, +1813/-22, C++)
- Foundation: survivor APIs (`install_survivors`, `live_ranks`, departed gathers) are first used in 7a; the description does not say so.
- Resolved: PeerFailureWatch self-destruction detaches under the once_flag; `watch_peer_failures`/`barrier_among_survivors` raise
  before setup; mpi4py<4 gives NotImplementedError.
- Stale text: description and commit body name `executor.recovery.mpi_exit`, which does not exist; the exit code is
  `llmapi/mpi_session.exit_fault_tolerant_process` (move reverted). Still open: `print` + `os._exit` in mpi_session; bottom-of-file
  import with `noqa: E402` in mgmn_worker_node.
- **Ready with description fixes** (also reword the commit body). The detach code has never compiled.

**05 recovery-authority** (8 files, +1890/-5)
- Clean scope, CPU-only test (l0_cpu, ipc://). First worker caller is 7a; Status/StatusQuery first used by 10. Neither is stated.
- Open: no protocol version, though PR 9 changes `Stopped`; auth failures classified by exception text (client.py:81-82); reads the
  private `router._last_identity` (authority.py:482). Pickle over HMAC follows the ZeroMqQueue precedent.
- **Needs code change** (version field in Register, refused on mismatch, ~10 lines; name 7a/10 as callers).

**06 eplb-restore-empty-slots** (20 files, +2157/-124, C++)
- Resolved: `restorePlacement` checks FT mode and `mWorkerDoneIterId == mIterId`; 2 gtests added (RestorePlacementNeedsQuiesce,
  RestorePlacementIsRejected; never compiled); `failure_domains` is tested.
- Open: `one_slot_table` is a `from_blob` on `current_device()` (moeLoadBalanceOp.cpp:185). The MoE module lazily imports
  `pyexecutor.peer_loss_recovery` (layering inversion; that file's docstring describes the executor but holds only `failure_domains`).
  gtests sit in the legacy `add_gtest` target. PR 9 rewrites 56 of its lines (restorePlacement/validatePlacement signatures, the rename).
- **Ready with description fixes** (say full /dev/shm copies per node exist until 9). Needs a build.

**07a executor-recovery** (27 files, +3624/-139; product 1863)
- Largest PR (3.76k), over the ~3.5k cap, with a "Why one PR" section; the ADP-absent-ranks part (~0.8k) splits out cleanly.
- Open: `os.kill(getpid(), SIGKILL)` in library code; `discard_interrupted_steps` drops `previous_batch` bookkeeping (no test);
  3 env readers deep in the scheduler; rewrites prmain/04's docstring (14 lines).
- Description: a stray pin bullet sits inside "Why one PR"; the FT-off line matches the code (dict counts, positional fallback);
  the negative-pin test (cef4670789) was never rerun. **Needs code change** (split 7a-0; fold the docstring into prmain/04).

**07b quarantine/503/e2e** (11 files, +1013/-37; product 161)
- Three concerns: quarantine policy, a serve HTTP contract, e2e for 7a.
- Resolved: the 503 gate uses `fault_tolerance_mode()` (cannot raise); progress rides the response gather (rewrites 28 lines of 7a).
- Description errors: the FT-off detail says the paths "read the cached `peer_loss_recovery_enabled()`", but code and header use
  `fault_tolerance_mode()`; the test list cites both `test_without_recovery_...` (gone) and its renamed form. Also: the summary says
  "with recovery on" but the gate fires in detection-only mode too; "two close losses quarantine a whole batch" is not listed;
  `Retry-After: 1` / "about a second" is unsourced.
- **Ready with description fixes**; get a serve-owner sign-off on 400→503.

**10 health-and-guide** (13 files, +758/-3; docs 285)
- Resolved: `unknown` returns 200; the guide documents the fixed 300 s detector; `failed_requests` is defined.
- Open: `RecoveryStatusProbe.close()` has no caller (thread pool and socket never closed); a Status that fails to unpickle returns 500;
  detection-only `/health` changes from an empty 200 to a JSON body.
- Unsourced guide numbers: "16 GiB per rank in our runs" (l.61), "1.4 GiB per GPU" (l.73), "within milliseconds…well under a second"
  (l.189), "GSM8K within two standard errors" (l.204-205), "about a second" (l.240), "64 hex" (the code accepts ≥32).
- **Needs code change** (call close() in lifespan teardown or drop it; cite the numbers or mark them `<PERF_TBD>`).

**08 hang-detection-fencing** (29 files, +2854/-74, C++)
- Resolved: `_shrink` kills and the pre-Propose Fence are gated on `hang_detection` (fence_departed, tests); the description says so.
- Open: the capture guard is an `assert` on private `graphs._capture_allowed` (stripped by `-O`); the fence-command `subprocess.Popen`
  is never reaped; the guide row "Measured … under an earlier rule: 10.6/15.9/36.7 s" is history plus preliminary data; broken
  launcher wrap. Rewrites 27 lines of 05 and 17 of 10. **Needs code change** (small).

**09 ghost-slots-ring-copies** (31 files, +3359/-144, C++)
- Silent behaviour changes: ring copies replace full copies for static placements with no switch; the default goes from
  `extra_slots: "empty"` to `Optional[...] = None`→ghost (the Pydantic guide disfavours Optional-as-auto; use
  `Literal["auto","empty","ghost"]`). Also: `moe_ghost_slot_storage` has no fake or mutation annotation; top-level `import pynvml`.
- Description errors: "its map gets a `shared_mutex`" is false (it exists at the parent, hostAccessibleDeviceAllocator.h:171);
  "Renames … (13 lines)" understates the change to PR 6 (56 lines rewritten, incl. the C++ restore signatures).
- **Needs code change**; consider the prior 9a/9b/9c split.

## Prior maintainer top-10 (redteam2/maintainer_review.md): status now
- **Resolved:** #4 PR 8 PMIx kills/Fence gated (37551d6035); #6 restorePlacement checks (2 gtests never compiled); #9 PeerFailureWatch
  detach, guards, mpi4py<4 (uncompiled, ULFM-only coverage).
- **Partial:** #2 CI skip stated in 4/10 and 7b only, no infra ticket; #5 graph refusal restored, but fake shape and `abort_state`
  schema still wrong; #10 gate cannot raise, unknown→200 and hang_detection_timeout doc fixed; unsourced numbers, probe close(), unpickle→500 open.
- **Open:** #1 build/run/PERF (older tree only, PERF TBD, "Not run yet" inaccurate); #3 config outside LlmArgs (one-line deferral);
  #7 PR 9 silent defaults; #8 size and no in-tree caller (7a 3.76k; 04 and 05 do not name their 7a caller).

## Top 10 items most likely to stall review (ranked), with the cheapest fix
1. **No green build or test of the submitted code.** Fix: one GB200 job building boundaries 02/03/04/06/08/09 (tip-only if slots are
   short), l0_cpu + l0_b200 + gtest on the 09 tip, triage the test_nvlink_one_sided F, replace "Not run yet" with run ids.
2. **e2e proof never runs in CI.** Fix: a "skips in CI (no ULFM image)" line in 8/9/10 with the GB200 run log ids; file the infra
   ticket for a ULFM stage with `TLLM_REQUIRE_ULFM=1`.
3. **Env + side-YAML config instead of LlmArgs.** Fix: 3 sentences in PR 1 (per-process launcher/authority settings that must match a
   GPU-less sidecar, digest-checked) plus a follow-up issue link; a prototype field if llmapi owners insist.
4. **PR 2 op contract** (fake shape with fixed slots; `abort_state` written but declared read-only). Fix: ~4 lines in
   cpp_custom_ops.py and moeAlltoAllOp.cpp, plus a fake-vs-real shape assert in test_moe_a2a_abort.
5. **PR 3 per-iteration tuning key and post-dispatch refusal.** Fix: key on `source_tokens > 0` (1 line); raise `do_finalize=False`
   in `MoeExecutionState.for_model`.
6. **PR 7a size (3.76k) and library SIGKILL.** Fix: split out 7a-0 "ADP absent ranks" (live_ranks/departed gathers, adp_router dict,
   request_utils, test_adp_absent_ranks); move the kv_cache_manager_v2 docstring into prmain/04; one paragraph justifying self-SIGKILL.
7. **PR 9 silent behaviour changes and wrong description.** Fix: `extra_slots: Literal["auto","empty","ghost"]="auto"` from PR 1;
   a `host_copies: all|ring` switch or a bold "default changed"; delete the shared_mutex claim; state the real churn into PR 6.
8. **PR 5 wire protocol without a version.** Fix: `protocol_version` in `Register`, refused on mismatch; a typed auth error from
   ZeroMqQueue instead of matching message text.
9. **Stack shape: numbering and churn.** Fix: renumber the core 1–9 and wave 2 as "wave 2, 1/2 and 2/2" (titles and descriptions
   only); a neutral `extra_slot_count` in PR 6 removes most of the 56-line 09→06 churn; move the launcher `75` exemption into PR 1.
10. **Description slips that cost reviewer trust:** PR 4's `mpi_exit` (description and commit body); PR 7b's FT-off detail vs header
    and stale test name; PR 4/5 not naming their caller; PR 10's unsourced numbers; PR 8's "earlier rule" data.
    Fix: edit the descriptions now; one reword commit for PR 4's body.

## Verdicts
| PR | Verdict | Blocking reason |
|---|---|---|
| 01 | ready with description fixes | LlmArgs justification; name first readers |
| 02 | needs code change | fake shape and `abort_state` schema; boundary build |
| 03 | needs code change | tuning-key growth; post-dispatch refusal hang |
| 04 | ready with description fixes | stale mpi_exit text (also the commit body); name 7a as caller |
| 05 | needs code change | protocol version; name callers |
| 06 | ready with description fixes | disclose /dev/shm cost until 9; gtests need a build |
| 07a | needs code change | split (3.76k); prmain/04 docstring |
| 07b | ready with description fixes | FT-off detail contradiction; stale test id; CI-skip and serve sign-off |
| 10 | needs code change | probe close(); unsourced numbers |
| 08 | needs code change | assert guard; Popen reap; drop "earlier rule" data; launcher wrap |
| 09 | needs code change | silent defaults; false shared_mutex claim; churn disclosure |
