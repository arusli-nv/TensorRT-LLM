# Hygiene audit: commits, headers, style, lean (v5-pr, read-only, 2026-10-23)

Worktree `/tmp/claude-161238/wt-pr`, branch `v5-pr` = `28ad75b7bb`, 25 commits on `9cc37929e8`, 123 files.
Nothing was edited or committed. Every item below preserves behaviour: it changes only comments, formatting,
commit text or test code, unless it is marked otherwise.
Tools run offline from `$TMPDIR` pip: ruff 0.9.4 (pinned), yapf 0.43.0, isort 5.12.0, autoflake 2.3.1 and
codespell 2.4.1 (all pinned), plus clang-format 16.0.6 (the hook pins 16.0.0). `scripts/legacy_utils.py`
baseline gating was run with `fix=False`.

## Summary

- Commit titles all pass the PR-title regex in `.github/workflows/pr-check.yml`. Every commit has exactly
  1 `Signed-off-by: Andrew Rusli`, author and committer are Andrew Rusli, and there are no Co-authored-by
  lines and no ticket noise. Only the commit bodies need work.
- Headers: 30 of the 38 new source files already use the short SPDX form with 2026. **8 need fixing:** 6 C++
  files and 2 Python tests. No modified file needs a year bump: all of them already say 2026, or have no
  header, as upstream does.
- Style: **7 real pre-commit failures** in 7 files (ruff-format ×4, ruff I001 ×1, yapf ×1, isort ×1). There
  are no codespell, trailing-whitespace, EOF, CRLF, autoflake or new legacy-ruff violations.
- Lean: there are no TODO/FIXME, no debug prints in product code, no info logs in per-step hot paths and no
  new unused imports. The open items are 3 duplicated test helpers, CLEANUP_PLAN §3.2/§3.3 items that are
  still unapplied, and 1 outlier docstring.

## (A) Commit message fixes

All 25 titles match `^\[None\]\[[a-z0-9]+\] ...`. Their types are fix, infra, feat, perf, test and doc, all
of which upstream uses. They are imperative. Lengths run 61-98 characters, against an upstream median of 76
and a p90 of 102.

**A1. Bodies (all 25 commits, behaviour-preserving).** Every body is only a `FT-off guards:` block.
`60fb349ca2` is the only one with a "what" paragraph. "FT" is never defined.
- Proposed form: 1-3 lines on what changes and why, then an optional `With fault tolerance off:` list of
  bullets.
- Commits whose body says nothing about what the change does: `3274ada868`, `6f40128f0f`, `6e29a128fb`,
  `ee2f12c44f`, `3b71abde0e`, `8902d25670`, `f724636c0d`, `ea9822948f`, `28ad75b7bb`.
  - The guard text often implies the "what". Promote it to the first paragraph and drop the "None needed:"
    prefix.

**A2. Branch-relative or PR-description narration in bodies** (move it to the PR text, or reword):

| Commit | Text | Fix |
|---|---|---|
| `de62f48333` | "The TRTLLM_USE_MPI_KVCACHE futures patch moves from _lib.py to _core.py and the setuptools<69 pin is dropped; needs a rebuilt image and full CI." | Keep "Port the TRTLLM_USE_MPI_KVCACHE futures patch to mpi4py 4's `_core.py`; drop the setuptools<69 pin." Move the image/CI note to the PR description. |
| `148ee4311b` | "instead of a truncated stream" | Fine as the why. Optional: "so clients do not see a truncated stream". |
| `ab70b2d3e1` | "it is unrelated to fault tolerance" | This is about the series, not the commit; move it to the PR text. CLEANUP_PLAN R3 already says this commit goes to main on its own. |
| `9b1534068c`, `64058cbdbc`, `ef9fa887a6`, `41eeae07b3`, `11a2d54cfd`, `6ed64f9057`, `60fb349ca2`, `f03e8b3dee` | "main's path / main's dispatch/combine / as on main / main's indexing / main's shm lifecycle / cache keys of main / main's routing kernel" | "main" means nothing once the commit is merged. Write "unchanged" or "the existing ...". |

**A3. Body lines over 100 characters** (rewrap them): `9b1534068c` line 6 (108), `129294d198` line 8 (101),
`11a2d54cfd` line 4 (101).

**A4. Optional title tightening.** Everything conforms already; these only make titles more concise.

| Commit | Current | Proposed |
|---|---|---|
| `8e8a4b0aef` | Report WideEP fault-tolerance state on /health and answer peer-loss failures with 503 | `[None][feat] Report WideEP fault-tolerance state on /health and return 503 on peer loss` |
| `ef9fa887a6` | Restore lost experts into EPLB slots backed by ghost slot memory (C++ load balancer) | `[None][feat] Back empty EPLB slots with ghost memory and restore lost experts into them` |
| `129294d198` | Append extra EPLB slots and restore lost experts into them (Python) | `[None][feat] Add extra EPLB slots for WideEP recovery and restore lost experts into them` |
| `3b71abde0e` | Remap the MNNVL all-to-all workspaces at the same addresses among the survivors | `[None][feat] Remap MNNVL all-to-all workspaces in place among the survivors` |
| `11a2d54cfd` | Tune and key the MoE tactics on the loaded slots under WideEP recovery | `[None][perf] Key MoE tactics on the loaded slots under WideEP recovery` |
| `ee2f12c44f` | Read the NVLinkOneSided rank mask through a pointer and add an abort record | keep it, or `[None][feat] Read the NVLinkOneSided rank mask from a device record and add an abort record` |

## (B) Header fixes

**What the repo actually does.** This compares the 2026 upstream additions on `9cc37929e8` (`git log
--diff-filter=A`, files added since 2026-06-01).
- Python: the short form (2 SPDX lines, no Apache text) and the SPDX-plus-Apache-text form are both
  common. Overall, 746 new files use the short form and 1056 use the long one.
- C++: almost all new files keep the Apache text. The short form exists, for example in
  `cpp/tests/unit_tests/batch_manager/coldPageCodecTest.cpp`, added 2026-09-01.
- `CODING_GUIDELINES.md` §NVIDIA Copyright still shows the old long `Copyright (c) 2025, NVIDIA CORPORATION`
  block.
- The short SPDX form you asked for is accepted upstream, but it is the minority for C++. It is a policy
  choice, not a violation either way.

**Exact header text to use.**

Python:
```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```
C++:
```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

**New files to change** (all of them comment-only):

| File | Problem | Introduced by |
|---|---|---|
| cpp/tensorrt_llm/nanobind/runtime/heartbeatSender.cpp | SPDX + full Apache text (long) | `997a0e6621` |
| cpp/tensorrt_llm/nanobind/runtime/heartbeatSender.h | long | `997a0e6621` |
| cpp/tensorrt_llm/nanobind/runtime/peerFailureWatch.cpp | long | `3f2b431b84` |
| cpp/tensorrt_llm/nanobind/runtime/peerFailureWatch.h | long | `3f2b431b84` |
| cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.cpp | long | `ef9fa887a6` |
| cpp/tensorrt_llm/runtime/moeLoadBalancer/ghostSlotMemory.h | long | `ef9fa887a6` |
| tests/unittest/_torch/test_mnnvl_memory_remap.py | long (Apache text, lines 3-14) | `3b71abde0e` |
| tests/unittest/llmapi/test_trtllm_llmapi_launch_exit.py | long, and the copyright line is wrapped (`...AFFILIATES.` / `# All rights reserved.`) | `9b1534068c` |

- **C++ rebuild cost:** the C++ header edits are comment-only. Do them with the rebuild CLEANUP_PLAN §7 already
  plans, or accept that no rebuild is needed for comment edits.
- **Already correct:** 22 other new `.py` files, which use the short form with 2026.
- **Left as is:**
  - `tests/integration/defs/wide_ep_ft/__init__.py` is empty, like upstream's `kv_cache/`, `llmapi/` and
    `visual_gen/` `__init__.py`. Optional: give it the 2-line header.
  - `docs/source/features/wide-ep-fault-tolerance.md` has no header, like upstream's feature docs.

**Modified files.** No change is needed. Every modified source file already has a header ending in 2026,
for example `2022-2026` and `2025-2026`. The remaining few have no header upstream either:
`nanobind/CMakeLists.txt`, the two `.md` docs and `l0_cpu.yml`.
- **Extra:** `de62f48333` adds a new SPDX header to `docker/common/install_mpi4py.sh`, whose sibling scripts
  have none. That is harmless. Optional: drop the 2 lines to keep the diff minimal.

## (C) Style violations

These are real hook failures. Each one is formatting-only, so a `--fixup` to the commit named in the last
column, followed by autosquash, fixes it.

| # | File:line | Hook | Fix | Commit |
|---|---|---|---|---|
| C1 | tests/unittest/executor/test_recovery_config.py:8-12 | ruff I001 | join to `from tensorrt_llm.executor.recovery.config import CONFIG_ENV, FaultToleranceConfig, failure_domains` | `129294d198` |
| C2 | tests/unittest/_torch/distributed/test_mpi_survivors.py:587 | ruff-format | explode the 3-tuple parametrize list, one tuple per line | `3f2b431b84` |
| C3 | tests/unittest/_torch/test_mnnvl_memory_remap.py:245 | ruff-format | wrap the `def test_a_peer_lost_during_..._exchange(monkeypatch)` signature (it is over 100 characters) | `3b71abde0e` |
| C4 | tests/unittest/executor/test_recovery_hang_monitor.py:419 | ruff-format | wrap the `command_fence(...)(HungRank(...))` call | `6e29a128fb` |
| C5 | tests/unittest/llmapi/test_trtllm_llmapi_launch_exit.py:99 | ruff-format | wrap the `_launch_worker(...)` call | `9b1534068c` |
| C6 | tensorrt_llm/_torch/distributed/communicator.py:974-976 (legacy file, so yapf) | yapf | the closing `)` of `raise RuntimeError(...)` goes on the string line, yapf style. It is a ruff-style leftover, probably from the squashed fixup. | `3f2b431b84` |
| C7 | tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py:27-28 (legacy file) | isort | re-align the continuation `fault_tolerance_config)` under `(failure_domains,`; it is 2 spaces too far right | `129294d198` |

- One-shot check: `ruff format` + `ruff check --fix` on C1-C5, and `yapf -i` + `isort` on C6-C7.
- I confirmed C6 and C7 are new: the same files at base pass yapf and isort.

**Minor, not hook failures:**
- `moe_load_balancer.py:1454`: a 123-character docstring line, over the 120-character comment limit. The
  legacy ruff baseline lets it pass. Wrap it (`129294d198`).
- `cpp/tensorrt_llm/runtime/CMakeLists.txt:30`: `moeLoadBalancer/ghostSlotMemory.cpp` comes before
  `gdrwrap.cpp`. Optional: move it one line down so the `moeLoadBalancer/` entries are alphabetical
  (`ef9fa887a6`).

**Clean:**
- codespell (with the repo's `-L` list), trailing-whitespace, end-of-file, CRLF and autoflake.
- legacy-ruff baseline: 55 violations in the 24 legacy files, all within the baseline, so no regression.
- ruff F401/F811/F841: the counts on legacy files equal base, so no new unused imports.
- clang-format 16.0.6 passes on 17 of the 19 C++ files.
  - The 2 diffs (trailing-comment alignment in `ghostSlotMemory.h:126-129` and `moeLoadBalanceOp.cpp:22-23`)
    also appear on the unmodified base `moeLoadBalanceOp.cpp`. They come from the 16.0.0 vs 16.0.6 version
    difference, not from the series. Re-check them with the pinned 16.0.0 hook (`pre-commit run clang-format`)
    before acting on them.

## (D) Lean items

All of these preserve behaviour. The test-side items touch only tests.

1. **Duplicated test helpers in the MNNVL multi-GPU tests.** Move them into one shared helper module,
   `tests/unittest/_torch/moe/multi_gpu/_mnnvl_recovery_utils.py`, with the 2-line SPDX header. Alternatively,
   merge `test_moe_a2a_late_writes.py` (347 lines) into `test_moe_a2a_recovery.py` (722 lines): both run the
   same 4-GPU fabric recovery setup.
   - `_mnnvl_skip_reason`: byte-identical in `test_moe_a2a_late_writes.py`, `test_moe_a2a_recovery.py` and
     `test_ring_restore_staging.py`.
   - `_StoreComm`: identical in `late_writes` and `recovery`; nearly identical in `ring_restore_staging`.
   - `_offset`: identical in `late_writes`, `recovery` and `test_moe_a2a_abort.py`.
   - `_worker` and `_survivor` have the same names in all three files.
   - Upstream has no existing equivalent besides `test_moe_module.py::_is_mnnvl_supported`, which is a
     different check.
2. **Other test-merge candidates (optional, low value).**
   - `test_moe_a2a_abort.py` + `test_moe_execution_state.py`: both single-GPU, same ops, about 870 lines.
   - `test_resident_placement.py` + `test_ring_host_copies.py`: both cpu_only EPLB placement.
   - Keep the `executor/test_recovery_{config,authority,hang_monitor}.py` split: each file mirrors one module.
3. **CLEANUP_PLAN §3.2/§3.3 items, still unapplied at v5-pr.** I verified each one is present:
   - `_utils.is_mpi_peer_failure_class` has one caller (#11).
   - `_utils.device_visible_host_zeros` has one caller (#12).
   - `hang_monitor.live_helpers(count=2)` is never passed `count` (#13).
   - `mpi_session.exit_fault_tolerant_process(status=0)` is never passed `status` (#14).
   - `protocol.encode_datagram` is called only by tests (#22).
   - The `client.AUTHORITY_ENV` alias is still there (#27).
   - `hang_evidence._FLAG_MASK` duplicates `alltoall_watchdog._COMPLETION_FLAG_MASK` (#28).
   - `communicator._failure_evidence` vs the `mpi_session` `Get_failed` poll loop (#29).
   - `mnnvl_memory` has no shared `_release_mapped` (#31).
   - Classes and tests are as listed in CLEANUP_PLAN; #11-#14, #27 and #28 are P.
4. **Docstrings.** The §3.1 trims landed. One outlier remains: `executor/recovery/authority.py:100`, the
   `RecoveryAuthority` class docstring, is 33 lines.
   - Cut paragraphs 1-2 to the contract, about 6 lines. Keep `Args`.
   - Drop the "(protocol tests)" rationale from `fence_departed`, which is #20's test-only default anyway.
   - Others still at 12-20 lines are contract-level, and I would leave them: `HangMonitor` (20),
     `HangEvidenceReporter` (16), `install_recovery_placement` (15), `serve` (15).
5. **Logging.**
   - No info-level logs in per-step paths. The added `logger.info` calls are in startup, recovery or
     authority paths only.
   - Optional: `peer_loss_recovery.py:1103` `_log_phase` (`WideEP recovery gen=.. phase=.. t=..`, rank 0, once
     per recovery phase) is timing instrumentation; `debug` would fit it better. This changes log output only.
6. **No TODO/FIXME/XXX/HACK/breakpoint/pdb in added lines.** The 3 added `print(..., file=sys.stderr)` calls
   in `mpi_session.py` are deliberate last-gasp exit messages, written after the logger may be gone. Keep them.
