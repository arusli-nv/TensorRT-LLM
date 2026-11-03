# WideEP FT PRs: upstream conventions and a review of our stacks

Read-only research, 2026-10-20 (PT). Repo: /tmp/claude-161238/work/repo, upstream = refs/remotes/upstream/main.
Sizes are `git diff --shortstat` between consecutive refs. "+N/-M" means lines added/removed.

## 1. Upstream practice (MoE / EPLB / all-to-all / executor)

### Sample of merged PRs

| PR | Title | Files | +/- | Notes |
|---|---|---|---|---|
| #19610 | [None][refactor] overhaul NVLink one-sided all-to-all | 53 | +3279/-4489 | Large feature. 59 commits squashed. Body: "What changes / Test Coverage / Checklist" plus CodeRabbit sections. Depended on a separate prerequisite commit (CFT driver detection). A follow-up (#20246) replaced part of it. |
| #16632 | [TRTLLM-14715][feat] preserve native MoE A2A graph VAs across restore | 23 | +4555/-377 | Large feature. A maintainer re-titled it from [None] to a JIRA tag. A reviewer (brnguyen2) asked for Test Coverage to be filled in, flagged hooks with no in-tree caller and an undocumented barrier, and asked for ~30 duplicated lines to be shared. The author split the shared lifecycle into a side PR on the fork (hhzhang16#1) and merged it in. 7x `/bot run --disable-fail-fast`. Multi-GPU stages needed the `ci: full pre-merge approved` label. |
| #17647 | [None][feat] Add CFT counted-write path to MoE all-to-all | 11 | +2681/-451 | Large feature. Body sections: Summary / Risk and gating / Test coverage / Known gaps. It is off by default behind a runtime flag and a compile-time flag. Known gaps were stated openly (no multi-GPU test of the path). |
| #19425 | [None][perf] Publish MoE all-to-all counters before the dispatch data issue | 5 | +253/-132 | Focused kernel perf change. |
| #18800 | [https://nvbugs/6667807][fix] Raise NVLink one-sided MoE all-to-all rank cap to 256 | 6 | +28/-23 | Small fix with an NVBug tag. |
| #17376 | [https://nvbugs/6482566][fix] Make MoE all-to-all completion-flag timeout phase-aware | 5 | +280/-12 | Same area as our watchdog work. |
| #20246 | [None][fix] Merge back #19593 from release/1.3: Make MoE communication timeouts follow the warmup phase | 24 | +2062/-137 | Merge-back; titles name the source PR. |
| #19045 | [https://nvbugs/6607482][logging] Add quiet PyExecutor hang diagnostics | 13 | +544/-25 | Closest to our hang detection. The squash body uses "* Why? / * What?". Opt-in, silent on success, no CUDA events when disabled. |
| #19456 | [None][fix] Tear down cleanly when multi-rank warmup fails | 5 | +362/-10 | Executor failure path. |
| #18457 | [https://nvbugs/6627795][fix] stop charging retiring requests against ADP admission and capacity | 29 | +1997/-164 | Executor/ADP. |
| #18397 | [None][feat] Router Replay (R3): return per-token MoE routing ... | 19 | +1522/-1 | Feature in one PR. |
| #19061 | [TRTLLM-16184][refactor] Consolidate the C++ MoE kernels ... under moe/ | 141 | +270/-205 | Mechanical move, kept separate from logic. |

### Size distribution

Over the last 400 upstream commits touching `_torch/moe`, `kernels/moe`, `_torch/pyexecutor` and `moeLoadBalancer`, lines changed (+ and -) were:

| Percentile | Lines |
|---|---|
| p50 | 620 |
| p75 | 1.75k |
| p90 | 4.8k |
| p97 | 13.7k |

The large ones are mostly mechanical moves or model bring-ups. Feature PRs of 2.5k-4.5k lines merge regularly in MoE comms (#16632, #17647, #19610). They take weeks, many `/bot run` rounds and several approvers.

### Conventions

- **Title.** `[TICKET][type] Imperative summary`, usually 50-95 characters.
  - Types seen: feat, fix, perf, refactor, chore, test, doc, infra, logging. `BREAKING` goes in the title for API breaks.
  - Multi-PR features use either one JIRA per step under an epic (TRTLLM-15760, -15761, -15762, -15766: "Establish ...", "Parameterize ...", "Migrate ...", "Decouple ...") or a suffix ("SID-GR optimization 1/N: ...", "Mooncake store part 1: ...", "Rubin fp4 mla core (P1)", "... - Part 1"/"Part 2").
  - Maintainers re-title `[None]` features to a JIRA tag (#16632).
- **Squash commit body.** Usually only sign-offs and co-authors; the PR page holds the description. A few use "* Why? / * What?" (#19045).
- **PR page.** It follows `.github/pull_request_template.md`:
  - `@coderabbitai summary`, then Description, Test Coverage, PR Checklist (one checkbox), GitHub Bot Help.
  - CodeRabbit adds Summary, Dev Engineer Review and QA Engineer Review, with a merge-risk verdict.
  - Larger feature PRs add their own sections: "Risk and gating", "Known gaps", per-file notes.
- **CI.** Comment `/bot run`, with `--disable-fail-fast` on reruns of large PRs. Multi-GPU stages are UNSTABLE until a maintainer adds the label `ci: full pre-merge approved`. Expect 3-7 runs for a big comms PR.
- **What review pushes back on** (#16632):
  - empty Test Coverage;
  - hooks with no in-tree caller;
  - undocumented barriers or semantics changes;
  - duplicated code;
  - CodeRabbit's docstring-coverage and early-return/sync nitpicks.
- **What reviewers accept:**
  - stated known gaps;
  - off-by-default gating with a description of the default-path delta;
  - a side PR that lands shared lifecycle code first.

## 2. Our stacks against these conventions

### prmain/* (wave 1, 7 PRs, base 0829689c52, each cut from upstream)

| Ref | Files | +/- | Verdict |
|---|---|---|---|
| 01 sse-stream-errors | 3 | +125/-64 | Good. Bug fix, CPU test, standalone value. |
| 02 moe-fake-ops-shared-health | 3 | +36/-0 | OK. It bundles two unrelated fixes; reviewers may ask for a split or a title naming both (it already does). `__deepcopy__` returning self will draw an "is this the right place" question; the draft already lists it as an open item. |
| 03 mpi4py-4.1.2 | 1 | +47/-54 | Infra. Needs an image rebuild and a tag bump, which this PR does not do. infra/oss-compliance will ask who bumps `jenkins/current_image_tags.properties`. Title OK. |
| 04 kv-v2-step-rollback | 3 | +361/-0 | Pushback risk: new public methods with no caller on main ("dead code until PR X"). Mitigation: link the FT PR in the description and keep the GPU test. Owners are kv-cache-manager-devs, who are strict about V2 API surface. |
| 05 eplb-planner-unusable-slots | 3 | +171/-15 | Same "no caller" issue; gtests are present. **Not built.** It must be built and the gtest run before opening. |
| 06 eplb-load-publish | 2 | +22/-31 | Clean refactor. **Not built.** It has no new test, and the CPU-copy path has no test on main; reviewers may ask for one. |
| 07 eplb-host-sharer-expert-lists | 2 | +139/-45 | Good. It includes a real bug fix (more ranks than experts). Lead with the bug fix in the title, not "groundwork". |

General points on prmain:
- Sizes are right for upstream (p50 range).
- No branch has had pytest or a C++ build run (PR_WAVE1 says so). Every branch needs a local build and a test run before opening, or the first `/bot run` will be spent on that.
- 04, 05 and 07 are pure groundwork. Consider stating in each "first caller: <PR link>" and opening them once the caller PR exists as a draft, so reviewers can see the use.

### pr10/* (10 PRs, base 0829689c52, in progress)

| Ref | Files | +/- | Content |
|---|---|---|---|
| pr10/01 | 10 | +308/-77 | Grab-bag: SSE errors, fake ops, impl_blocks slot rows, launcher exit, LB test |
| pr10/02 | 1 | +47/-54 | mpi4py |
| pr10/03 | 5 | +1025/-21 | Ghost slot memory C++ (ghostSlotMemory, hostAccessibleDeviceAllocator), **no test, no caller** |
| pr10/04 | 33 | +3198/-269 | A2A rank mask / abort / receive slots / watchdog / CuteDSL+Cutlass slot reads (~1.7k is tests) |
| pr10/05 | 12 | +2057/-46 | ULFM survivors, MPI session, peerFailureWatch |
| pr10/06 | 8 | +1175/-15 | Recovery authority / protocol / client, launcher |
| pr10/07 | 27 | +2885/-92 | Executor peer-loss recovery, ADP absent ranks, KV revert |
| pr10/08 | 21 | +1518/-295 | EPLB restore / placement (C++ + Python) |
| pr10/09 | 33 | +4634/-279 | Mixed: hang/fence +1.8k, recovery/authority +1.7k, EPLB +0.7k, FT config +0.2k, /health +0.2k |
| pr10/10 | 42 | +3228/-761 | Mixed fix-up layer touching files of 01, 04, 05, 06, 07, 08, 09: recovery +1.4k, EPLB +1.0k, a2a/CuteDSL +0.3k, health +0.2k |
| **Total** | 108 | +18899/-733 | Tests are about half of the added lines (src +9.6k, test +9.3k) |

Problems reviewers would raise:
1. **Branch tips are commits titled "PR NN placeholder".** They are not yet reviewable units: no titles, no bodies.
2. **pr10/10 is a fix-up layer across seven earlier PRs.** It changes peer_loss_recovery.py, authority.py, moe_load_balancer.py, cute_dsl ops and openai_server.py again after earlier PRs add them. Upstream reviewers will not approve PR N while PR N+k rewrites it.
   - Fold every pr10/10 hunk into the PR that introduces that file (`git rebase` with fixup).
   - Treat pr10/09 the same way: its authority (+1.7k) and EPLB (+0.7k) hunks extend pr10/06 and pr10/08.
   - peer_loss_recovery.py and authority.py are each edited in 4 of the 10 PRs.
3. **pr10/09 mixes five concerns** (hang detection, PMIx fence, authority growth, online-EPLB agreement, FT config, /health) at 4.9k lines. That is above upstream p90, and the topics have different owners. Split it.
4. **pr10/03 adds about 1k lines of C++ with no test and no caller** until a later PR.
   - The #16632 review flagged exactly this ("no in-tree caller").
   - Either add a gtest for ghostSlotMemory, or move it into the PR that uses it (ghost extra slots).
5. **pr10 duplicates prmain.** pr10/01 and pr10/02 contain the prmain/01, 02 and 03 content (SSE errors, fake ops / EPGroupHealth, mpi4py). Pick one: land prmain first and rebase pr10 on it, dropping 01/02. Otherwise two PRs carry the same change.
6. **The FT config lands late.** `executor/recovery/config.py` (validated FT config) arrives in pr10/09, but authority (06) and executor (07) already read FT options. Move the config into the authority PR, or into the first FT PR.
7. **Files that are in fix-final8-rc1 but in no pr10 PR** (12 files):
   - `docs/source/features/wide-ep-fault-tolerance.md` (+323), `docs/source/index.rst`, `examples/wide_ep/README.md`;
   - `tests/integration/defs/wide_ep_ft/*` (+838, the GB200 end-to-end tests);
   - `tests/unittest/_torch/moe/multi_gpu/test_ring_restore_staging.py` (+244);
   - `tensorrt_llm/_torch/pyexecutor/llm_request.py` (+49, logits/output checkpoint);
   - `engine/runners/decoder/inputs.py` and `metadata.py`, `thop/moe/utils/moeUtilOp.cpp`.

   The executor PR without the llm_request checkpoint, and the feature without its guide or end-to-end test, would both get pushback.
8. **No tracking ticket.** Every title is `[None]`. A 10-PR, ~19k-line feature will be re-titled to a JIRA (as #16632 was). Get a TRTLLM epic plus one ticket per PR, or use a "WideEP FT k/N:" suffix.

### ftpr/* (older 23-branch restack, base fc52f0adee)

- 23 PRs from +22 to +4683 lines.
- Ten of them are under 400 lines and are test-only or doc-only, such as "[None][test] Cover ...". Several "doc" tip commits are listed as the PR's last commit.
- Too fine-grained at the bottom and too coarse at 13, 14 and 17: ftpr/17 executor recovery is +4.7k and ftpr/14 authority is +3.5k.
- ftpr/23-review-fixes (+796/-572) has the same fix-up-layer problem as pr10/10.
- Superseded by pr10; do not open.

### PR_DESCRIPTIONS_FINAL.md (28 drafts)

Strengths:
- Every FT PR has explicit "FT off", "Tests" (named test functions and their stage) and "Limitations" sections. That matches what #17647 did ("Risk and gating", "Known gaps") and what #16632's reviewer asked for.
- The silent-wrong-output window and the unmeasured CFT path are stated honestly.

Gaps:
1. **It is keyed to the 28-PR plan and to fix-final3 (547a6aa7ed).** pr10 has 10 PRs on fix-final8-rc1. The numbering, dependencies ("PR 6 and PRs 13-18") and file ownership all need remapping.
2. **The header admits it was not aligned** with the restack, and CODEOWNERS was not re-read.
3. **Not in template shape.** It uses Summary / Design notes / FT off / Tests / Limitations / Dependencies / Owners. Upstream needs `@coderabbitai summary`, `## Description`, `## Test Coverage`, `## PR Checklist` with the checkbox, and `## GitHub Bot Help`. Our sections must live inside or after those headings; CI and pre-merge checks look for the template.
4. **Every perf section is `<<PERF: final benchmark pending>>`.** Reviewers for MoE comms changes (#19425, #19610) expect FT-off overhead numbers in the kernel PR at least.
5. **Hash-level history in the body** ("(547a6aa7ed)", "(cafb1f97aa)", "closes the detect-only capture issue in PR_FIX_LIST") refers to private branches and notes. Upstream readers cannot resolve these; drop them.
6. **PR 19 says it is over 2.5k lines "with no clean seam".** Reviewers will push on that line; the pr10 split should address it.

## 3. Recommendations

### 3a. Proposed ~10-PR FT stack (after prmain wave 1 lands)

Prerequisites are prmain/01-07, about 1.1k lines in total, each independent. They should merge first, or at least be open first. Then:

| # | Title (suggested; ticket TBD) | Scope (from pr10) | Approx +lines (src / test) | Depends on | Owners (CODEOWNERS) |
|---|---|---|---|---|---|
| 1 | [TRTLLM-x][feat] WideEP FT 1/10: validated fault-tolerance config and mode switch | `executor/recovery/config.py`, `_utils.fault_tolerance_mode`, `wide_ep_ft.py` options, launcher exit-by-signal (from pr10/01, 06, 09) | ~0.6k (0.3 / 0.3) | prmain | runtime-devs, moe-devs |
| 2 | [TRTLLM-x][feat] WideEP FT 2/10: NVLinkOneSided rank mask, abort record and per-source receive slots | Kernels, thop and `nvlink_one_sided.py`, watchdog (pr10/04 kernel half, plus fold-ins from 10) | ~2.3k (0.9 / 1.4) | 1 | moe-devs, multi-gpu-ci-review (l0_gb200_multi_gpus), runtime-devs (custom_ops) |
| 3 | [TRTLLM-x][feat] WideEP FT 3/10: read all-to-all receive slots in place in Cutlass/CuteDSL MoE | `execution_state.py`, `fused_moe_cute_dsl`, cutlass `moe_kernels`, slot gather (pr10/04 consumer half; CuteDSL extra-slot layout from 10) | ~1.8k | 2 | moe-devs, kernels-devs (cute_dsl ops) |
| 4 | [TRTLLM-x][feat] WideEP FT 4/10: ULFM survivor communicators and degraded MPI session | pr10/05 + communicator bits of 09/10 | ~2.1k | 1 (+ mpi4py) | runtime-devs |
| 5 | [TRTLLM-x][feat] WideEP FT 5/10: external recovery authority and protocol | pr10/06 + authority growth from 09/10 | ~2.5-3k | 1, 4 | runtime-devs |
| 6 | [TRTLLM-x][feat] WideEP FT 6/10: restore a lost rank's experts into EPLB slots | pr10/08 + EPLB parts of 09/10, prmain 05/06/07 consumers | ~2.5k | 2, 5 | moe-devs |
| 7 | [TRTLLM-x][feat] WideEP FT 7/10: recover the executor from a single WideEP peer loss | pr10/07 + recovery parts of 08-10, llm_request checkpoint (+49), decoder inputs, **GB200 integration test** | ~3.5k (cap; split `_recover` phases if larger) | 2-6, prmain/04 | runtime-devs + disagg-devs (py_executor, model_engine, engine/), kv-cache-manager-devs (kv_cache), trt-llm-devs/infra (test-db), multi-gpu-ci-review |
| 8 | [TRTLLM-x][feat] WideEP FT 8/10: confirm and fence hung ranks (heartbeats, hang evidence, PMIx) | Hang/fence half of pr10/09 (+1.8k) + fixes from 10 | ~2.0k | 5, 7 | runtime-devs (nanobind, executor, _torch) |
| 9 | [TRTLLM-x][feat] WideEP FT 9/10: ghost extra EPLB slots and ring host copies | pr10/03 ghost memory C++ **with a gtest** + ring host copies + `test_ring_restore_staging` | ~2.0k | 6 | moe-devs |
| 10 | [TRTLLM-x][feat] WideEP FT 10/10: /health FT states and the WideEP fault-tolerance guide | `fault_tolerance_status.py`, openai_server FT 503 mapping, docs guide, index.rst, wide_ep README | ~0.8k | 7, 8 | runtime-devs, doc-owners |

Rules for the restack:
- **No fix-up PR.** Every hunk lands in the PR that introduces its file. When done, `git diff pr10/10 fix-final8-rc1` limited to FT files should be empty.
- **Cap the size.** Keep each PR at or under ~3.5k lines with tests, and aim for source code under 1.5k where possible.
- **Build each tree.** Each intermediate tree must build (C++) and pass its own unit tests. Upstream bisects on squash commits.
- **FT off.** Each PR states its FT-off delta. Kernel PRs (2, 3) and the executor PR (7) carry measured FT-off overhead.
- **Optional extra split.** If PR 7 exceeds about 3.5k, split it 7a/7b: commit, discard and KV revert first, then request rerun/quarantine and the 503 path.

### 3b. Description template (put this inside upstream's template)

```
Title: [TRTLLM-xxxxx][feat] WideEP FT k/10: <imperative summary, <= 90 chars>

@coderabbitai summary

## Description

**Summary.** <What this PR adds, 3-6 bullets. Link the epic and the stack: "Part k of 10, follows #A, followed by #B.">

**Design.** <Key mechanism and invariants, e.g. fixed slots/graphs, who writes what when, ordering and barriers.
Call out any new barrier/sync and any semantics change to existing code paths.>

**FT-off behaviour.** <Exactly what runs when TLLM_FAULT_TOLERANCE_MODE is unset: new branches, extra kernel args,
moved checks. "No change" only if no line on the default path changed.>

**Performance.**
| Config | Metric | FT off (main) | FT off (this PR) | FT on | Delta |
|---|---|---|---|---|---|
| <model, EP, GPUs, ISL/OSL, concurrency> | <TPOT / tok/s> | <PERF_TBD> | <PERF_TBD> | <PERF_TBD> | <PERF_TBD> |
<Recovery time where relevant: <PERF_TBD>.>

**Limitations / known gaps.** <Stated plainly: unsupported features, unmeasured paths, CI gaps such as no ULFM image.>

## Test Coverage

- `<test file>::<test>`: <what it guards> (<CI stage: l0_cpu / l0_b200 / l0_gb200_multi_gpus pre|post-merge>)
- C++: `<gtest suite.case>` (<ctest label>)
- Manual: <GB200 run, n nodes, result>  (link the job log only if it is public)

## PR Checklist
<keep upstream's list verbatim, and tick the box>

## GitHub Bot Help
To see a list of available CI bot commands, please comment `/bot help`.
```

Notes:
- Keep commit history, private hashes and run IDs out of the body (AGENTS.md: rationale goes in the description, but readers must be able to follow it).
- Put "api-compatible" and the label request in the checklist if any public args change. The FT config moving into LlmArgs is a separate PR, with the golden manifest, the serve schemas and telemetry CODEOWNER approval.

### 3c. Likely reviewer pushback (titles, sizes, splits)

1. `[None]` on a multi-PR feature: get a JIRA epic and per-PR tickets, or use a "k/N" suffix.
2. Fix-up layers (pr10/10, ftpr/23): fold them in.
3. pr10/09 at 4.9k with five concerns: split it.
4. Code with no caller or no test (pr10/03 ghost memory, prmain/04, prmain/05): add a test and link the caller.
5. The guide and end-to-end tests are missing from pr10 entirely: add them, end-to-end in PR 7 and the guide in PR 10.
6. prmain and pr10 duplicate SSE, fake ops and mpi4py: keep one copy.
7. Nothing in the stack has been built or tested. prmain was not built at all, and only fix-final3 or fix-final8 trees were built. Build each intermediate tree before opening.
8. ULFM tests skip in CI (no ULFM image) and the end-to-end tests are post-merge only. Upstream reviewers (#17647) accepted "known gap" statements, but expect a plan: name an infra follow-up.
9. Titles that list many features at once (old PR 24: "Recover with overlap, ADP balancing, MTP/spec-dec, DSA, KV reuse, node loss, later losses") read as scope creep. Keep one concern per title.
10. Perf placeholders: a kernel/comms PR without FT-off overhead numbers will stall. Fill at least the FT-off rows before requesting review.
11. CI: expect `/bot run` then `/bot run --disable-fail-fast`, and ask a maintainer for `ci: full pre-merge approved` for the multi-GPU stages.
