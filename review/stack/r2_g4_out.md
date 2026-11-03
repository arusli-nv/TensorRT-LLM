# Round 2 review: PRs 12, 13, 14, 15, 24 (descriptions 06, 07, 08, 09, 18)

I read the current tree with Read and Grep only and ran nothing. Commits 12, 13, 14, 15 and 24 map to PR descriptions 06, 07, 08, 09 and 18, so the PR numbers below are the commit numbers from round 1.

## Round 1 findings

**PR 12**
- **Op contract break (CPU to CUDA mask, major): FIXED.** PR text 06 now has an "Op contract" design note and says "Callers of the ops with `enable_rank_mask` must change."
- **Local-rank-bit check dropped (major): FIXED.**
  - `execution_state.py:120-121` makes `release_workspaces` raise `RuntimeError` if a departed rank is one of this state's own ranks.
  - `_publish_rank_mask` (`nvlink_one_sided.py:908`) and `install_mask` (`execution_state.py:188-190`) keep their checks.
  - `test_moe_execution_state.py:247` covers the `release_workspaces` case.
  - The op itself still has no local-bit check. That is justified: the PR text says the check lives in the publishers.
- **`resolveAbortState` comment (minor): STILL A PROBLEM, but not blocking.** `moeAlltoAllOp.cpp:171` still says "once per record", while the cache at lines 185-191 is a single-entry `thread_local`. This is a wrong comment, not a bug. Say "cached for the last record".
- **`expert_to_slot` check in the launch (minor): FIXED.** `moeAlltoAllKernels.cu:1283` checks it in `moe_a2a_dispatch_launch`, and `:1353` checks the fence path.
- **8-byte `FLAG_VAL` layout change (minor): JUSTIFIED.** The PR text states it applies to every caller.

**PR 13**
- **Duplicated EPLB stats-copy loop (minor): STILL A PROBLEM, but not blocking.** `moeAlltoAllKernels.cu:676-714` still has two copy bodies. The decisions file did not list it as a fix, so I am not raising it as a blocker.
- **Perf numbers with no reproducer (minor): STILL A PROBLEM, but not blocking.** PR text 07 still quotes 2.1 us, 1.6 us and 0.15 us and says "No test file changes". The functional coverage comes from `test_moe_a2a_abort.py`, which is acceptable. Ask for a benchmark link in the description.

**PR 14**
- **Retry invariant (major): STILL A PROBLEM, but only as a doc item, so minor now.**
  - The PR text says "a retry must be made by every live rank together, because the handle exchange has no round tag."
  - The `remap` docstring (`mnnvl_memory.py:965-978`) says only "a later attempt may remap it". It does not state the all-ranks-together rule.
  - The code comment is where a maintainer will look. Add one sentence to the docstring.
- **Pages unmapped after peers wrote (major): FIXED.** The outcome exchange (`:998-1019`) unmaps on every rank if any rank failed, and the PR text describes this.
- **`pin`/`unpin` lifetime and `_LiveRowsComm` interface (minor): JUSTIFIED.** The PR text says `_LiveRowsComm` implements only the calls the mapping code uses.

**PR 15**
- **CFT auto-selected under recovery (major): FIXED.**
  - `select_cft_counted_writes` (`nvlink_one_sided.py:143-148`) returns False under recovery unless `force_cft is True`, and logs once.
  - `bind_execution_state` still raises, but only when CFT was forced. The error names `FORCE_CFT_ENV`, and the PR text says so.
  - Workspace sizing and construction both go through this function, so they stay consistent.
- **Eager watchdog docstring (major): FIXED.**
  - The docstring at `:447` now says the watchdog observes unbound layers only.
  - The comment at `:1114-1116` is accurate.
  - The PR text says "Eager bound calls skip the host watchdog as well."
  - Dropping the watchdog for eager bound calls is a kept trade-off. The abort record and the device timeout cover it.
- **Detection-only FT with CUDA graphs failing late (major): FIXED.** `wide_ep_ft.py:67-73` raises a `ValueError` at config time when rank-mask mode is on, CUDA graphs are used, and `peer_loss_recovery_enabled()` is false. FT-off users are unaffected, because `rank_mask_enabled` requires `health` or FT mode.
- **Snapshot still captured when bound (minor): not raised again.**
- **`device_visible_host_zeros` (minor): not raised again.**
- **No thread-safety test for `request_abort` against `rearm` (minor): not raised again.**
- **Blanket `cpu_only` mark (minor): not raised again.**

**PR 24**
- **Silent fallback when `ENABLE_RANK_MASK` is false (major): FIXED.** The launch checks it (`:1283`), the op checks it (`moeAlltoAllOp.cpp:808`), and the instantiation comment is accurate.
- **Stale `one_slot_table` cache (minor): JUSTIFIED.** The PR text says "It is safe because the committed one-slot routing is fixed once chosen, and the routing kernel serves every placement." The docstring at `moe_load_balancer.py:957-961` repeats it. A None result keeps the always-correct routing kernel.
- **Table as an ownerless `from_blob` view (minor): JUSTIFIED as documented, weakly.**
  - The PR text says "a captured graph must not outlive it."
  - Nothing enforces this in code. It is acceptable for an experimental feature.
  - Consider a one-line shutdown assertion or having the table's owner hold a reference to the balancer.
- **Scheduler caching `peer_loss_recovery_enabled()` (minor): not raised again.**

## New significant issues

None at blocker or major level in the current diffs. There are three smaller points:
- `release_workspaces` (`execution_state.py:122-123`) does `words[rank >> 6]` on `departed` with no range check. An out-of-range rank raises an unhelpful `IndexError`. This is a minor robustness point, not a correctness issue for valid inputs.
- `moeAlltoAllKernels.cu:1426-1450` still uses a generic lambda inside the `SWITCH_BOOL` macro with a nested `if constexpr`. It works and is commented, but it is harder to read than adding `USE_EXPERT_TO_SLOT` to the switch. This is minor.
- PR text 06 (`PR_DESCRIPTIONS.md`) says "the check sits in `moe_a2a_dispatch_launch`". That is now accurate.

## Verdicts

| PR | Verdict |
|---|---|
| 12 | approve (optionally fix the stale cache comment) |
| 13 | approve (optionally dedupe the loop, link a benchmark) |
| 14 | approve with changes: add the "all live ranks retry together" sentence to the `remap` docstring |
| 15 | approve |
| 24 | approve |
