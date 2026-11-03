WideEP FT red team: tests, docs, config, headers, commits. Written 2026-10-20 at about 20:20 PT.
This was a read-only review using only git show, diff and log; no checkout and no edits.
Target: fix-final8-rc1 (78836d1723) + fix-minors (dda2d51984, 16 commits on top of rc1).
Base: merge-base with upstream/main, 0829689c52.
Excluded: anything already in redteam/CODE_REDTEAM_FINAL.txt or MINORS_FIX.txt.
Line numbers are for fix-minors unless stated otherwise.

Format: severity | file:line | issue | fix

== 1. VACUOUS / WEAK TESTS AND SKIPS ==

minor | tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:459-461 (test_hung_rank_fenced)
  Issue: The test asserts that "survived its fence" is not in the authority log, to show that the workers fenced the rank. That line can only be logged when `_command_fence` is set, which happens only when a `--fence-command` is configured (authority.py:284-294, 356-357). The test configures none, so the assertion cannot fail. Nothing in the test positively shows that the worker or runtime fence ended the rank.
  Fix: Assert positive evidence instead. Either look for the worker line "Rank N: the MPI runtime ended failed rank 2" (peer_loss_recovery.py:1147) or the hang_evidence fence log, or pass `--fence-command` that touches a marker file and assert the marker is absent.

minor | tests/unittest/_torch/executor/test_peer_loss_recovery.py:567-570 (test_nothing_is_killed_before_a_shrink_when_no_failure_is_reported)
  Issue: The test makes no assertion. Its only check is the kill_rank Mock(side_effect=AssertionError), and that cannot fail the test: `_end_ranks` calls kill_rank on a daemon "rank_fence" thread, `kill()` catches only PmixUnavailable/RuntimeError, and an AssertionError on that thread is swallowed (pytest only warns). `_end_ranks` reports the rank as ended ("").
  Fix: Record the calls in a list and `assert killed == []`. Also `assert dist.shrink.call_count == 1`.

minor | tests/unittest/_torch/moe/test_moe_a2a_abort.py:249-265 (test_valid_counts_wait_for_a_source_that_arrives_late)
  Issue: "Late" depends on a 50M-cycle `torch.cuda._sleep` on a side stream racing the host launch of the dispatch. If the writes land before the dispatch starts (slow host, or CUDA_DEVICE_MAX_CONNECTIONS=1 serializing the streams), the test still passes with the same counts. So it cannot tell "waited for a late source" apart from "the source was already there".
  Fix: Gate the late writes on a host-written pinned flag (or a stream wait on an event recorded after the dispatch launch). Assert that the dispatch has not completed (`event.query()` is False) before releasing the flag, as test_moe_execution_state.py:151-153 does.

minor | tests/unittest/_torch/executor/test_hang_evidence.py:187-189
  Issue: `pytest.skip("the bindings were built without HeartbeatSender")` is unjustified. HeartbeatSender is compiled and registered unconditionally (nanobind/CMakeLists.txt adds runtime/heartbeatSender.cpp; bindings.cpp calls initHeartbeatSenderBindings). The skip would turn a missing or broken binding, the native heartbeat that the hang design relies on, into a green skip. The importorskip on `tensorrt_llm.bindings.internal.runtime` hides the same failure.
  Fix: Import directly and let a missing attribute fail.

minor | cpp/tests/unit_tests/runtime/moeLoadBalancerTest.cpp:1245-1251, 1365-1371
  Issue: CpuWritesReachResidentAndBackedSlots and CpuCopiesLoadARestoredGhostSlotThroughEachMapping GTEST_SKIP whenever ghost slot memory cannot be host accessible (needs GDRCopy; never where GPU memory is a NUMA node). GB200 is a NUMA-node GPU, and CI GPU containers usually lack the gdrdrv module. So the online-EPLB ghost CPU-copy path is very likely skipped in every CI stage while reporting green.
  Fix: Confirm one CI stage where `hostAccessibleSupported()` is true and list it. Otherwise state in the PR that this path has no CI coverage.

minor | tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:83-84 (_model)
  Issue: Under TLLM_REQUIRE_ULFM=1 a missing ULFM now fails the test, but a missing `DeepSeek-V3-Lite/nvfp4_moe_only` still skips green. The accuracy tests that use this checkpoint (test_llm_api_pytorch.py:320, ...) fail when it is missing.
  Fix: Also call pytest.fail when REQUIRE_ULFM_ENV == "1", or drop the existence check.

minor (doc vs test) | tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:394-410 (test_restore_into_ghost_slots[online]) vs docs/source/features/wide-ep-fault-tolerance.md:79
  Issue: The doc says "Do not set `ghost` with online EPLB where GDRCopy cannot map ghost slot memory", and also that GPUs whose memory is a NUMA node (GB200) get `empty` with online EPLB. The only CI run of [online] is on GB200 with `extra_slots: ghost` forced. So CI checks the configuration the doc tells users not to use. The code accepts it silently: moe_load_balancer.py:1287-1290 switches every online update to GPU memcpy with no warning.
  Fix: Make it consistent. Either log a warning (or refuse) for explicit ghost+online without host access in extra_slots_mode/register_weight_slots_after_to_cuda and keep the test as a fallback check, or soften the doc to "supported, slower".

nit | tests/unittest/_torch/executor/test_hang_evidence.py (whole file)
  Issue: The file has no `pytestmark = cpu_only`, although all but the last test (:306, already skipif no-GPU) are pure CPU. l0_cpu's `unittest/_torch/executor` entry therefore skips it. It runs only through the GPU directory entries (l0_h100.yml:32, l0_b300.yml:55, l0_dgx_b300.yml:34, l0_gb300_multi_gpus.yml:45), on GPU minutes.
  Fix: Mark the CPU tests cpu_only (the module marker plus a non-cpu_only marker for :306, or split the GPU test out).

== 2. CI REGISTRATION AND TLLM_REQUIRE_ULFM ==

Every new test file is registered. Details:
- The cpu_only files are covered by l0_cpu directory entries: _torch/{distributed,executor,moe,modules}.
- l0_cpu also lists explicit entries for unittest/executor/test_recovery_*, apps/test_fault_tolerance_health, apps/test_stream_error_events and test_trtllm_llmapi_launch_exit.
- l0_b200 lists test_moe_a2a_abort, test_moe_execution_state, test_moe_slot_gather (SM100/103 only) and test_kv_cache_revert. test_kv_cache_revert also runs again on the h100/b300 directory entries.
- l0_gb200_multi_gpus has pre-merge test_moe_a2a_late_writes and test_ring_restore_staging (TIMEOUT 10), plus a post-merge block with all 12 wide_ep_ft cases.
- The modified test_nvlink_one_sided runs from l0_dgx_b200:229, and test_cute_dsl_moe from the thop/parallel directory entries.

major (sequencing) | jenkins/L0_Test.groovy:1623-1627 (dddd8de035) + docker/common/install_mpi4py.sh
  Issue: TLLM_REQUIRE_ULFM=1 makes the 12 post-merge FT cases fail where ULFM or mpi4py>=4 is missing. The stack bumps mpi4py only in install_mpi4py.sh. Nothing bumps the image tag, and nothing in docker/ builds Open MPI with ULFM: `ompi_info | grep ftmpi` depends on the base image's Open MPI, which is unverified. So if dddd8de035 merges before a rebuilt image is pinned, GB200-4_GPUs-PyTorch-Post-Merge-1 goes red on all 12 cases, and stays red if the base Open MPI lacks ftmpi.
  Fix: Land the jenkins commit in the same PR as the image-tag bump, after confirming that `ompi_info` lists ftmpi in the new image. Or waive the 12 cases in waives.txt with an NVBug until it lands.

minor | jenkins/L0_Test.groovy:1625
  Issue: `stageName.startsWith("GB200-4_GPUs-PyTorch")` also matches the 5 pre-merge splits and the 5 PerfSanity-Post-Merge stages (lines 6636-6655). It is harmless today because the variable is read only by wide_ep_ft and test_mpi_survivors_ulfm, but it does not match the comment's intent.
  Fix: Match "GB200-4_GPUs-PyTorch-Post-Merge", or say in the comment that every GB200-4 stage gets it.

minor | tests/integration/test_lists/test-db/l0_gb200_multi_gpus.yml:102-113
  Issue: 12 integration cases (TIMEOUT 45 each; each serves DSV3-Lite at EP4 with up to 1800 s serve plus 180 s teardown) are added to a post-merge block whose only stage, GB200-4_GPUs-PyTorch-Post-Merge-1, has splits=1 (L0_Test.groovy:6641). The block also holds about 30 accuracy tests. No duration estimate is given.
  Fix: Measure the 12 cases and add a Post-Merge split (or a dedicated stage) if the stage wall time grows by more than about 1 h.

(Already in MINORS_FIX, not repeated: test_mpi_survivors_ulfm is cpu_only, and the CPU stage does not set TLLM_REQUIRE_ULFM, so it still skips green.)

nit | tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py:41,58-81 and tests/unittest/_torch/distributed/test_mpi_survivors_ulfm.py:26-46
  Issue: The ULFM probe and the "TLLM_REQUIRE_ULFM" constant are copied in two files and have already drifted: one returns a reason, the other a bool.
  Fix: Share one helper (for example in tests/test_common or the integration conftest).

== 3. DOCS ACCURACY ==

Checked against the code, no finding: the env table; config defaults and limits (heartbeat_s <= 2 = 5-1-2; authority_timeout_s < recovery_timeout_s); RUNTIME_FENCE_GRACE_S 1 s / CALL 2 s / COMMAND 5 s; the minimum hang_detection_timeout formula (3 x recovery plus hang resolution); MPI_Finalize grace 30 s; exit code 75; Retry-After 1 and the `peer_loss_retryable` code; the "half the heartbeats" rule; the /health states and their bodies (including detection-only giving `unprotected`); the authority CLI flags and localhost default; the log strings the integration test matches; the examples/wide_ep README shm token, exclusive create and unlink after load.

minor | docs/source/features/wide-ep-fault-tolerance.md:159-160 + tensorrt_llm/executor/utils.py:175-178 + py_executor.py:6943
  Issue: A new request strictly pinned to a lost attention-DP rank fails as `peer_loss_retryable` (HTTP 503, Retry-After: 1), and lines 257-258 tell clients to retry such requests. But a lost rank never rejoins (line 157), so the same retry fails forever on this deployment. A "retryable" 503 for a permanent condition makes OpenAI clients and load balancers spin.
  Fix: Answer a pin to a departed rank as a non-retryable 4xx/503 without Retry-After (a separate code). Or document that the client must drop `attention_dp_rank` or `relax`, or retry on another replica.

minor | docs/source/features/wide-ep-fault-tolerance.md:226-239
  Issue: (a) The section "Not supported together with fault tolerance" opens with "Speculative decoding: every method is accepted". (b) The "Refused at startup" list omits pipeline and context parallelism and MoE TP != 1 as a layout. communicator.py:960-966 refuses any layout other than tp == moe_ep == world with pp == cp == 1, and line 12 implies but does not state this.
  Fix: Move the spec-dec paragraph under its own "Speculative decoding" heading. Add "pipeline or context parallelism (only tp_size == moe_ep_size == world_size)" to the refused list.

nit | docs/source/features/wide-ep-fault-tolerance.md:34 vs communicator.py:968-969, test_peer_loss_recovery.py:66, test_mpi_survivors_ulfm.py:34
  Issue: The doc requires "mpi4py 4.1 or later". The code and both test probes accept any mpi4py>=4 (hasattr(MPI.Comm, "Shrink")).
  Fix: Pick one minimum. If 4.1 is needed (the image pins 4.1.2), check `mpi4py.__version__` in enable_peer_loss_recovery.

nit | tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md:164-166
  Issue: The new "NVLinkOneSided rank mask" section covers only the per-communicator record (`_publish_rank_mask`). Under peer-loss recovery, NVLinkOneSided reads `MoeExecutionState.active_rank_mask` instead (nvlink_one_sided.py:1221-1223, 1385), written by install_mask, together with the abort record and fixed slots. A reader of the guide would not know that the second path exists.
  Fix: Add one sentence on the bound-execution path (MoeExecutionState owns the mask and abort record) and point to execution_state.py.

nit | docs/source/features/feature-combination-matrix.md (not touched)
  Issue: The FT doc lists many refused combinations (guided decoding, beam search, disagg, LoRA, sleep/wake, Ray/RPC, penalties per request), but the feature matrix that AGENTS.md lists has no WideEP FT entry.
  Fix: Add a row or a footnote linking to the FT doc, or say in the PR why it is omitted (experimental).

== 4. ENV VARS AND CONFIG FIELDS ==

New env vars in product code:
- TLLM_FAULT_TOLERANCE_MODE: pre-exists in main, so not new; the C++ reads were already reported.
- TLLM_FAULT_TOLERANCE_AUTHORITY, TLLM_FAULT_TOLERANCE_AUTHORITY_KEY, TLLM_FAULT_TOLERANCE_CONFIG: all three are documented (doc:69-74, config.py docstring) and all are needed. The config-not-env issue and the key in env through mpirun -x were already reported.
New test-only env vars: TLLM_REQUIRE_ULFM, WIDEEP_FT_TEST_KILL_AFTER_STOPPED, PEER_LOSS_DRIVER_KILL / WIDEEP_FT_TEST_SELF_KILL markers, and the RUN_TOKEN env. None is read by product code (git grep over tensorrt_llm/ and cpp/tensorrt_llm/ is empty).
The fault hook lives in tests/integration/defs/wide_ep_ft/worker_hooks/sitecustomize.py and is loaded only through PYTHONPATH in test_rank_lost_during_recovery.
No ForTest/for_test/test-only symbols were added to product code. GhostSlotMemory's public API is used by product code; bookedBytes() is a test-file helper.
No LlmArgs field changed, so the golden manifest and serve schema regeneration are not triggered.

nit | tensorrt_llm/executor/recovery/config.py:73-79
  Issue: Nothing ties hang_suspect_s to hang_confirm_s. With hang_suspect_s >= hang_confirm_s, device waits are reported only after the confirmation window should have elapsed.
  Fix: In the validator, require hang_suspect_s < hang_confirm_s.

== 5. COPYRIGHT HEADERS ==

No finding. Every added code, test and script file carries "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA ..." plus the Apache-2.0 SPDX line. Every modified file that had a header now shows 2026; install_mpi4py.sh and test_executor_request_queue.py gained one. The files without a header follow main's convention: docs .md/.rst, examples README, CMakeLists, test-db yml, and the empty tests/integration/defs/wide_ep_ft/__init__.py.

== 6. COMMIT MESSAGES (git log base..fix-final8-rc1 and ..fix-minors) ==

254 commits, all authored by Andrew Rusli. No Co-Authored-By, Claude, Anthropic, Codex, Copilot or "Generated with" lines. The "OpenAI" hits refer to the OpenAI server API.

minor | commit 2c3af30ce8 (fix-final8-rc1)
  Issue: `Revert "[None][perf] Compile the source-slot input remap out ..."` has no Signed-off-by (it is the only one of 254, so it fails DCO), and its title lacks the [None][type] prefix. It reverts 5663731764, which is in the same stack.
  Fix: Drop both 5663731764 and 2c3af30ce8 when the stack is squashed or split into PRs. Otherwise re-sign with `git commit -s` and retitle "[None][perf] Revert ...".

nit | stack history
  Issue: The stack contains revert churn: 5663731764 and its revert 2c3af30ce8; 881cf4314a and its revert e5c302ee96; ca0b3a52ae reverting 2d1210ae98/a42b33e3cd/3b82bda95d, followed by 911d8f2a1d restoring part of it. 9 subjects are longer than 120 characters (up to 156).
  Fix: Squash the revert pairs out of the PR branches and shorten the subjects (PR title format `[None][type] description`).

== COUNTS ==
major 1, minor 12, nit 7 (total 20).

Scratch note: while checking write access I created an empty file, /tmp/claude-161238/redteam2/.probe. It was not removed, per the no-rm rule; it can be overwritten or queued for deletion.
