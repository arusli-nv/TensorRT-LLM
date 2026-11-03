Round-2 review of PRs 16–19 (PR_DESCRIPTIONS.md numbers them 10–13). One earlier finding, the 18 #1 shm reload leak, is still unfixed, and the PR text now claims it is fixed. I only read the tree files cited below.

## Earlier findings

### PR 16
- **#1 gate and message — FIXED.** The commit message no longer names `mFaultToleranceMode` and says `isPeerLossRecoveryEnabled()` needs mode plus authority. The code still mixes the cached `mPeerLossRecovery` (`moeLoadBalancer.cpp:592,663,1078,1208`) with live env reads (`:511,868,989`, `hostAccessibleDeviceAllocator.cpp:251`). The env is process-constant, so I am not re-raising it as a blocker. It is a minor tidy-up: read it once at construction.
- **#2 split — JUSTIFIED.** "One PR, not three: a restore maps memory behind a ghost slot… one state machine and are tested together." That is sound.
- **#3 host write of the signal word — JUSTIFIED.** The header and Python docstring state the "device work is done" precondition, and the PR text repeats it.
- **#4 whole-tensor copy into scratch — JUSTIFIED.** The PR text says it is write-tolerant because strays land in shared scratch.
- **#5 stale `mOneReplicaPerExpert` — JUSTIFIED.** The PR text states it is read only with static routing, and `routesEachExpertToOneSlot()` (`:663`) enforces `mLayerUpdatesPerIter == 0`.
- **#6 line wrapping — not re-checked; pre-commit's job.**

### PR 17
- **#1 collective in `extra_slots_mode` — FIXED.**
  - `_agree_ghost_slot_memory_host_accessible()` is the only collective. It runs in `register_weight_slots_after_to_cuda` (`moe_load_balancer.py:1251-1253`).
  - `_ghost_slot_memory_host_accessible()` (`:88`) now only reads the cache and raises `RuntimeError` if the ranks have not agreed.
  - Static routing never reaches it (`:1293`).
  - The module-global cache is per process. That is acceptable because it is a hardware property and the agreement is symmetric.
- **#2 hostname as node key — STILL A PROBLEM (major).**
  - "Kept this way for now" is not a technical reason. `_gather_hosts` (`:1157-1165`) uses `socket.gethostname()`, while host copies are shared over `shared_mpi_comm` (`:1149`). `rank_expert_ids`, `host_copy_ranks` and `copy_nodes` are all derived from the hostname grouping.
  - If the two disagree, ranks of one shared comm compute different splits and read each other's segments at wrong offsets. That is silent corruption, not an error.
  - Smallest fix: in `_gather_hosts`, allgather the shared comm's lowest world rank (or a hash of the sorted member ranks) next to the hostname. Raise `ValueError` if ranks of one shared comm report different hostnames, or if two shared comms report the same one. The failure domain definition stays as documented, and a mismatch fails at startup.
- **#3 `end_iter(interrupted=...)` — JUSTIFIED.** It is `False` without recovery, and the PR text says so.
- **#4 determinism test — not re-verified.** Only a minor suggestion.
- **#6 digest dtype — FIXED.** `placement_digest` casts to `np.int32` (`:1457`), and the PR text says so.

### PR 18
- **#1 reload leaks the old mapping — STILL A PROBLEM (major).**
  - The PR text says: "A weight reload under recovery unlinks the old name, closes the segment this rank owns, and re-imports peers' segments."
  - The code does not do that. `finalize_layer_weights` (`:307-332`) calls `unlink_shared_memory()` (`:312`) and then overwrites `self.own_shm = shm` (`:331`) without calling `close()`. `unlink_shared_memory` (`:425-433`) only removes the name.
  - `finalize_host_tensor_sharing` (`:383`) appends the peers' segments to `imported_shms` again without closing the old ones. `pre_shutdown_cleanup` closes only the latest `own_shm`.
  - Failure scenario: on a second `load_weights` under recovery, the old segment of every layer stays mapped and is no longer reachable by name. The old and new segments coexist, which can hit the `_check_shm_space`/`_reserve_shm` ENOSPC path at the second reload. I did not read `_check_shm_space` to confirm that part.
  - No test covers a second `finalize_layer_weights` (`test_ring_host_copies.py` has none: grep of test names).
  - Smallest fix: before `self.own_shm = shm`, call `self.own_shm.close()` if it is set. Close and clear `imported_shms` before re-import. Add a test that finalizes twice and asserts the old segment is closed.
- **assert → ValueError — FIXED** (`:1484-1488`). `live[turns[node] % len(live)]` (`:1511`) can still divide by zero if `placements` did not come from `recovery_placements`. That is minor, so I am not raising it.
- **#2 fail-late `/dev/shm` sizing — JUSTIFIED.** The PR text now says per-layer sizing is checked at layer creation and per node at the up-front reserve, with sizes in the error, and that "The sizes of all layers are not totalled before weights load". It is honest and the failure is still at startup, not at recovery.
- **#3 hard-coded `MADV_DONTNEED=4` and ctypes — not re-raised.** The PR text justifies the behaviour with measured numbers. `mmap.MADV_DONTNEED` would be a cleaner implementation (minor).

### PR 19
- **#1 numerics test — JUSTIFIED.** The PR text says "The integration restore tests (PR 19) run the real kernels with extra slots, including restored ones", and `tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py` references extra slots. I did not read that file to confirm it is token-exact. The tile-only assumption is documented in `loaded_slot_layout`.
- **#4 unasserted layout assumptions — STILL A PROBLEM, but minor.** The PR text now states the assumption, but `loaded_slot_layout` (`cute_dsl_custom_ops.py:248-263`) still does not assert `num_experts % num_local_experts == 0` or `local_expert_offset % num_local_experts == 0`. A wrong layout would silently produce a wrong cache key instead of failing. Two asserts fix it.
- **#2, #3, #5 — JUSTIFIED.** The PR text lists the paths that do not pass the argument, the trailing default-0 argument, and the shared `tuner_inputs` list. I never confirmed whether `choose_one` mutates `inputs`. That is minor and unchanged.

## New issues in the current diffs
I found no new blocker or major item beyond the two above (17 #2, 18 #1).

## Verdicts
- **PR 16: approve.** The gate message is corrected, and the remaining items are justified.
- **PR 17: approve with changes.** Make `_gather_hosts` check the hostname grouping against `shared_mpi_comm`, and raise on a mismatch.
- **PR 18: request changes.** Close the old `own_shm` and `imported_shms` on reload and add a test. As written, the PR text claims behaviour the code does not have.
- **PR 19: approve.** Add the two asserts in `loaded_slot_layout` if convenient.
