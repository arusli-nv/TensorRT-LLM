Code owner review of PRs 25 and 26. I read both diffs and checked the claims against `/tmp/claude-161238/wt-stack`. Nothing was run, and nothing was modified.

## PR 25: WideEP peer-loss recovery integration tests

**1. Blocker. CI is made to hard-fail on images that may lack ULFM.**
- `jenkins/L0_Test.groovy:1646` sets `TLLM_REQUIRE_ULFM=1` for every stage whose name starts with `GB200-4_GPUs-PyTorch`. `test_peer_loss_recovery.py:344` turns the missing-ULFM case from a skip into `pytest.fail`.
- The stack changes nothing under `docker/` or `jenkins/` to provide ULFM Open MPI. Only mpi4py 4.1.2 is pinned (`docker/common/install_mpi4py.sh:11`). `docker/common/install_base.sh:103,183` installs the distro `openmpi-bin` / `openmpi-devel`.
- I found no `ulfm`, `ftmpi` or `--with-ft` anywhere in `docker/`. Unless the current image already ships ULFM, this PR turns the pre-merge stage `GB200-4_GPUs-PyTorch-1..5` red.
- The prefix match also catches the `GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-1` stage, which runs no FT tests.
- The commit message says only "TLLM_REQUIRE_ULFM=1 is set only for the stages that run these tests". That is inaccurate, because the prefix covers stages that do not run them. It does not say the image provides ULFM.
- Smallest fix: land the image change first, or state which image tag has ULFM. Scope the env var to the stage(s) that actually contain these tests.

**2. Major. The FT tests do not set `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0`.**
- The PR 26 doc says that on CUDA 13.4+, driver 615+ and SM100, CFT counted writes are the default and FT refuses to start unless `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0` is set. The code agrees (`nvlink_one_sided.py:741-745`).
- `_fault_tolerance_env` (`test_peer_loss_recovery.py:387-405`) sets `TRTLLM_FORCE_COMM_METHOD=NVLINK_ONE_SIDED` but not `FORCE_CFT`. GB200 is SM100, so once the image gets a new enough toolkit and driver, all FT tests fail at startup.
- `test_serving_without_fault_tolerance` therefore compares against CFT, not the fence path. That contradicts the PR 26 doc's own "both run the same all-to-all" guidance.
- Fix: add `TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT="0"` to the env helper.

**3. Major. Weak coverage for users who do not enable fault tolerance.**
- `test_serving_without_fault_tolerance` (`:617`) runs under the hardcoded `MPIRUN` (`:309-327`) with ULFM, `pml ob1`, `btl self,tcp` and `coll ^hcoll,ucc`. It also keeps `TLLM_FAULT_TOLERANCE_AUTHORITY_KEY` and `TRTLLM_FORCE_COMM_METHOD` in the env.
- That is not the default configuration (UCX, default collectives), so it does not guard against a regression for FT-off users.
- It is also placed in post-merge only, and it only asserts that requests are answered.
- Fix: at minimum, document that it covers "FT env unset under ULFM transport". Better, add a plain `mpirun` or default-launcher run with no FT variables at all.

**4. Major. CI cost and structure.**
- Post-merge adds 13 parametrized tests. Each loads DeepSeek-V3-Lite NVFP4, captures CUDA graphs and tears down through `MPI_Abort`, with `TIMEOUT (45)` each. That is up to about 10 hours on a 2-way split, and the commit gives no runtime estimate.
- The tests share one driver and differ mostly in flags. Several could share one server launch, or the matrix could be trimmed (for example, run both `[idle-*]` combinations and both `[inflight-*]` combinations only where the overlap path differs).
- Smallest fix: justify the matrix with measured timings or reduce it. Check that the stage split does not push the post-merge stages past their budget.

**5. Minor. Fragility in the test code.**
- `_get` (`:749`) in the `/health` settle loop at `:836` is not wrapped in the `URLError` handling that the startup loop has. A transient connection error while the authority dies fails the test instead of retrying.
- `WORLD_SIZE`, `KILL_MARKER` and `SELF_KILL_MARKER` are redefined in the test (`:292,295,296`) although the driver is already imported from. The hook module's copy is understandable, since it cannot import the test package. The `WORLD_SIZE` and `KILL_MARKER` duplicates in the test file are avoidable.
- `find_worker` in the driver scans `/proc/*/environ` for the run token. That is sound for a single node, but the driver silently assumes all four ranks are local. The doc and `test_two_ranks_lost_together` mention `failure_domain: node` needing multiple hosts, but nothing asserts single-host.

**6. Minor. Test hook.** `worker_hooks/sitecustomize.py` monkey-patches `RecoveryClient.send` by class name (`type(message).__name__ == "Stopped"`), which breaks silently if the message is renamed. It is test-only and `test_rank_lost_during_recovery` would fail loudly, so this is acceptable.

**Test lists.** `test_mpi_survivors_ulfm.py` exists in the tree. The pre-merge list gets 1 FT test and post-merge gets 13, which is reasonable. The unit test already has its own `TLLM_REQUIRE_ULFM` policy, so item 1 applies to it too.

**Verdict PR 25: request changes** (items 1 and 2 must be fixed; 3 and 4 should be addressed).

## PR 26: Document WideEP fault tolerance

**1. Major. The published page is full of unresolved placeholders and stale numbers.**
- There are at least 9 `TODO`s in `docs/source/features/wide-ep-fault-tolerance.md`:
  - `:104` "TODO: cite the run or drop"
  - `:117` "TODO: measure"
  - `:225`, `:227` and `:254` ("TODO: re-measure / re-verify on the current design")
  - `:316` "TODO: measure"
  - `:333` "TODO: re-run on the current design and cite the run"
  - `:370` "TODO: measure"
  - `:391` "TODO: restate from the runs…"
  - `:404` "TODO(measured %, cite the run)"
- The **Cost** section is effectively empty, so the doc advertises a feature with an unquantified throughput cost.
- Line 225 gives timings "under an earlier rule", and line 333 gives GSM8K accuracy from a prior design. Both are stale evidence presented as fact.
- Fix: measure and fill these in, or delete the claims. Do not merge TODOs into user documentation.

**2. Major. The doc contradicts the test coverage it describes.**
- The doc claims "The loss of a whole host" and "hang detection and fencing" are tested only "in single runs". The PR 25 tests cover single-host EP4 only, with `failure_domain: rank` forced in the multi-rank tests.
- The doc's `failure_domain: node` default is therefore not exercised by any integration test in this stack.
- Fix: say so explicitly in "How well each part is tested", or add a 2-node test.

**3. Minor. The `extra_slots` table cell (`:117`) is about 2,000 characters in one cell.**
- It is unreadable in rendered Sphinx/MyST and unreviewable in diffs.
- Fix: move it to a dedicated subsection or bullet list under the table.

**4. Minor. Inconsistent wording on rank 0.**
- `:61` says "rank 0 is never recoverable" and `:117` says "failure_domain: node … every rank of one host but rank 0's". These are consistent in meaning but redundant. The page repeats several statements about what is not recoverable across "What is covered", "Not covered" and "Known limits".
- The page is about 366 lines. A maintainer will struggle to keep it accurate. Consider splitting into a user guide and an operator reference.

**5. Verified, no issue.**
- `docs/source/index.rst:81` places the page in the toctree.
- The feature-matrix addition is consistent with the page. It is a separate table, so it does not break the existing matrix.
- The README claim about `moe_shared_<token>` is backed by `moe_load_balancer.py:208,1074`.

**6. Minor. Doc and code are only partly cross-checked.**
- The page documents many env vars and settings (`hang_*`, `heartbeat_*`, `recovery_timeout_s` and others).
- The review should confirm those defaults and limits against the config parsing in the earlier PRs of the stack. I did not verify each one here.

**Verdict PR 26: approve with changes** (resolve or remove the TODOs and stale measurements; fix the claims that are not backed by tests).

## Overall

| PR | Verdict |
|---|---|
| 25 | **Request changes.** Blocker: CI forced-fail on an image that may lack ULFM. Major: missing FORCE_CFT=0 in test env. |
| 26 | **Approve with changes.** Remove TODOs and stale numbers, and align coverage claims with the tests. |

Neither commit message states a reason for the issues above, so none counts as justified.
