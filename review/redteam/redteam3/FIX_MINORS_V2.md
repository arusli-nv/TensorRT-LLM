# fix-minors commits missing from v2 (cef4670789): verdicts

Checked 06:10 PT against /tmp/claude-161238/wt-v2 (v2 cef4670789). Read-only. None of the five changes a measured number
(healthy cost, kills, accuracy): every path they touch is either not on those runs' code path or only changes error reporting.

| commit | subject | verdict |
|---|---|---|
| 1f048122cc | Quiesce static EPLB at once after an interrupted iteration | **Unreachable in v2: port in v3 as hardening.** In v2's `MoeLoadBalancer::quiesce` (moeLoadBalancer.cpp:1384-1398), static EPLB with an interrupted iteration (`mInterruptedIterId >= 0`) waits for `mWorkerDoneIterId == mIterId`. Static EPLB starts no worker (`finalizeModel` starts threads only if `mLayerUpdatesPerIter > 0`, :1202), so that wait would only time out and return false. Recovery never makes this call: `peer_loss_recovery.py:938-939` quiesces only when `not load_balancer.is_static_routing()`. That is why the static-deployment kills passed (w/o EPLB and offline, 12/12, 0.28-1.4 s). The fix (return at once for static, clear `mInterruptedIterId`) and its gtest make the C++ API safe for a direct caller. |
| f499b5ef8d | Check the CUDA errors of the MoE A2A workspace reinitialization | **Missing: port in v3.** v2 moeAlltoAllOp.cpp:317 (`cudaMemsetAsync`) and :323 (`cudaDeviceSynchronize`) are unchecked in `moeA2AInitializeOp`. A failure would surface at the next checked CUDA call with a less specific message. No effect on healthy runs or the passed kills. |
| fc7419efdc | Refuse an EPLB update schedule reset without WideEP fault tolerance | **Hardening: port in v3.** v2 `resetUpdateSchedule` (moeLoadBalancer.cpp:1410-1414) checks quiescence but not `mFaultToleranceMode`. Its only caller is FT recovery, where FT is on. The check guards a misuse of the C++ API, not a reachable path. |
| dddd8de035 | Fail the GB200 WideEP FT tests when the image lacks ULFM | **Missing: port in v3 (CI).** v2's tests already read `TLLM_REQUIRE_ULFM` (test_mpi_survivors_ulfm.py:26, integration wide_ep_ft/test_peer_loss_recovery.py:42), but jenkins/L0_Test.groovy never sets it. A GB200-4_GPUs-PyTorch image without ULFM therefore skips the FT tests instead of failing. This affects CI coverage, not the product. |
| dda2d51984 | Place the one-slot table view on the load balancer's device and state its lifetime | **Hardening: port in v3.** v2 moeLoadBalanceOp.cpp:184-185 labels the `from_blob` view with `c10::cuda::current_device()`. Serving runs one device per rank, with the current device set to the balancer's device, so the label is right there; it would be wrong only if a process called this with another device current. The lifetime comment is documentation only. |

Summary: no correctness bug in the measured configurations, so nothing changes the 07:00 deliverables. All five go into the v3 patch set:
- f499b5ef8d and dddd8de035 are plain gaps;
- 1f048122cc, fc7419efdc and dda2d51984 harden the C++ API for direct callers.

1f048122cc carries a gtest (`StaticQuiesceAfterAnInterruptedIterationReturns`). Port it too.
