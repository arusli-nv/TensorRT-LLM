# Round 2 review of PRs 25 and 26 (current tree `wt-stack`)

I checked the current diffs and the tree. Nothing was run or modified.

## Round 1 findings

| # | Round 1 finding | Status | Why |
|---|---|---|---|
| 25.1 | Blocker: `TLLM_REQUIRE_ULFM=1` set for every `GB200-4_GPUs-PyTorch*` stage | **FIXED** | `jenkins/` no longer mentions `TLLM_REQUIRE_ULFM`. The tests now skip when ULFM is absent. The doc (`wide-ep-fault-tolerance.md:439`) and the PR text both say the images lack ULFM. |
| 25.2 | Major: tests do not set `FORCE_CFT=0` | **FIXED (in the code)** | `nvlink_one_sided.py:143` makes CFT fall back to fence under recovery unless `FORCE_CFT=1` is set. `bind_execution_state` (`:749`) raises only if CFT is still on. The doc at `:55-61` matches. The test env is correct without the variable. |
| 25.3 | Major: weak FT-off coverage | **JUSTIFIED** | The PR text says: "keeps the ULFM transport settings of the helper". It checks that FT-off serving still answers every request. The default UCX launch is covered by the existing suites. The limit is stated rather than hidden. |
| 25.4 | Major: 13 tests of CI cost | **JUSTIFIED, with a caveat** | There is no decision on this and no timing estimate in the PR. The cases differ in mechanism (ghost slots, empty slots, hang fence, rank lost during recovery, authority loss), so I accept the matrix. Item N1 below is the real issue. |
| 25.5 | Minor: unwrapped `_get` in the settle loop, duplicated constants | **STILL OPEN (minor)** | `test_peer_loss_recovery.py:818` has no `URLError` handling, and `WORLD_SIZE` and `KILL_MARKER` are still redefined (`:274,277`). Not blocking. |
| 25.6 | Minor: sitecustomize matches `Stopped` by class name | **JUSTIFIED** | It is test-only and fails loudly. |
| 26.1 | Major: TODOs and stale numbers in the doc | **FIXED** | The doc has no TODO markers. Every measurement now names its configuration (DEP32, DeepSeek-R1 NVFP4, run counts, ranges). The Cost section is filled in. |
| 26.2 | Major: doc contradicts test coverage | **FIXED** | `:437-439` says CI runs only single-node `failure_domain: rank`, and that whole-host loss and `failure_domain: node` are covered only by DEP32 runs, not CI. |
| 26.3 | Minor: huge `extra_slots` table cell | **FIXED** | It now has its own "`extra_slots` in detail" list (`:128`). |
| 26.4 | Minor: page length and redundancy | **JUSTIFIED** | It is long, but it is a single experimental-feature reference. Not a blocker. |
| 26.6 | Doc defaults not cross-checked against code | **Not re-verified here** | `FORCE_CFT` and mpi4py 4.1.2 (`install_mpi4py.sh:11`) are confirmed. The other defaults (`hang_*`, `heartbeat_*`, `recovery_timeout_s`) belong to the config-parsing PRs. |

## New issues

**N1. Major, PR 25: pre-merge and post-merge CI coverage is currently nil.**
- `l0_gb200_multi_gpus.yml:25-26` adds `test_mpi_survivors_ulfm.py` and `[idle-sync]` pre-merge. Lines `106-118` add 13 post-merge tests.
- The doc says the CI images lack ULFM (`wide-ep-fault-tolerance.md:439`). With no `TLLM_REQUIRE_ULFM` anywhere in `jenkins/`, every one of these skips.
- A green stage is therefore indistinguishable from "never ran". The stack also ships a large recovery path whose end-to-end behaviour no CI gate checks.
- This is not a blocker, since the skip is deliberate and disclosed. The fix is one of these:
  - land an image with ULFM, then set `TLLM_REQUIRE_ULFM=1` for the specific stage that runs these tests;
  - or state in the PR that the tests are not yet effective, and make sure someone tracks the follow-up.
- `"the CI test images lack ULFM Open MPI"` in the user doc will go stale. It belongs in the PR text, not in the published guide.

**N2. Major (belongs to PR 10, noted here because it hits the test lists): the fake-communicator unit test is not in any list.**
- `unittest/_torch/distributed/test_mpi_survivors.py` is not in any file under `tests/integration/test_lists`. A grep for `test_mpi_survivors` finds only the `_ulfm` variant.
- Round 1 decided "PR10 `test_mpi_survivors.py` in l0_cpu: FIX". It is not in the tree.
- The fake-communicator suite is the only recovery test that runs without ULFM, so it is what actually runs on current images. Unlisted, it never runs in CI.
- Fix: add it to `l0_cpu.yml`, or to the same GB200 list.

**N3. Minor, PR 25: the pre-merge `test_peer_loss_recovery[idle-sync]` has a 45-minute timeout.**
- That is long for a pre-merge test, on top of a `SERVE_TIMEOUT_S` of 1800 s. It is acceptable only if the test skips. Once an image provides ULFM, re-measure it and consider moving it post-merge.

## Verdicts
- **PR 25: approve with changes.** The blocker and the CFT issue are fixed. Open items: N1 (state the effective CI coverage) and N2 (list the unit test). The minor fragility items are optional.
- **PR 26: approve.** The TODOs are gone, the claims match what CI covers, and the doc matches the code I checked. The line about CI images lacking ULFM should move to the PR text.
