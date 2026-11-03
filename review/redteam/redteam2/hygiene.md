# Branch hygiene audit (read-only)

Generated Tue Oct 20 20:13:40 PDT 2026. Reference final branch: `fix-final8-rc1` @ 78836d1723 (240 non-merge commits on upstream/main 0829689c52).

Repos: **A** = /tmp/claude-161238/work/repo (agent clone); **U** = /lustre/.../users/arusli/src/TensorRT-LLM (user repo).

## Counts

| Repo | local branches | remote-tracking refs | worktrees (incl. main) | detached worktrees | stashes |
|---|---|---|---|---|---|
| A | 174 | 65 (origin 64 incl. HEAD, upstream/main 1; **no `fork` remote configured**) | 81 | 12 | 0 |
| U | 33 | 22 (fork 16 incl. HEAD, origin/main 1, pr/* 5) | 4 | 0 | 1 |

Notes:
- A's `origin` is U's worktree `.wideep-ft-runs/claude/dev`, so A `origin/*` mirrors U's local branches as of A's last fetch. 31 of them (fix-cq4, fix-nr*, fix-prime, fix-sweep, fix-integ, wideep-ft-folded*, ghost, next-build, zmq-queue, ...) no longer exist in U (stale tracking refs); `origin/fix-sort-gate` is stale (89da95d29e vs U 4f0c308b02).
- `git ls-remote fork` from U failed (no DNS for github.com in sandbox), so fork state is from U's `refs/remotes/fork/*` only (may be stale). A has no fork remote, so it has no fork refs.
- Divergence among long-lived branches: U `wideep-ft` e4c6cefa41 vs `fork/wideep-ft` 6f3ad13ab0; U `wideep-ft-all` 800347c9f4 vs `fork/wideep-ft-all` 57fd045971; A `wideep-ft-dev` aae00e6c65 is ahead of U `wideep-ft-dev` 191988d1fb. KEEP all four, but they are not in sync.

## Method

For each ref: non-merge commits not on upstream main (`upstream/main` in A, `origin/main` in U; U's is an ancestor of A's). Each commit is checked against fix-final8-rc1 by ancestry, then by `git patch-id --stable` (same test `git cherry` uses). Refs whose objects exist only in U (fix-sort-gate 4f0c308b02, fork/wideep-ft-all, pr/*, stash) were patch-id-hashed in U and compared with final's patch-ids from A.
- **SUPERSEDED**: no commits outside final by patch-id (what `git cherry fix-final8-rc1 <b>` reports as no '+').
- **SUPERSEDED~** (amended): every commit not patch-equal to one in final has a commit with the **same subject** in final, so it was amended or rebased. Safe to delete if you accept that the final version replaces it.
- **UNIQUE-covered**: has commits with subjects not in final, but each one is either >=90% present in final (its added non-trivial lines appear in final's version of the file) or has the same subject in another FINAL/LONG branch (fix-minors, prmain/*, pr10/*, wideep-ft-dev, ...). Likely safe.
- **UNIQUE**: at least one commit has no subject match in final or any kept branch, and under 90% of its added lines are in final. These need a decision (listed below).
- 'cov' = fraction of a commit's added non-trivial lines that appear verbatim in the fix-final8-rc1 version of the same file (heuristic, line-level).

### FINAL - repo A (20)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| fix-final8-rc1 | 78836d1723 | 240 | 240 / 0 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-final7 |
| fix-minors | dda2d51984 | 254 | 238 / 0 | 16 | 16 | 0 | /tmp/claude-161238/work/wt-minors |
| fix-sse-stream-errors | a9a307a0f4 | 1 | 0 / 0 | 1 | 1 | 0 | /tmp/claude-161238/work/wt-sse |
| pr10/01 | cba25a24ac | 1 | 0 / 0 | 1 | 1 | 0 |  |
| pr10/02 | 71f006fb9f | 2 | 0 / 1 | 1 | 1 | 0 |  |
| pr10/03 | 18ff366878 | 3 | 0 / 1 | 2 | 2 | 0 |  |
| pr10/04 | e3331f42ea | 4 | 0 / 1 | 3 | 3 | 0 |  |
| pr10/05 | 6d6a813728 | 5 | 0 / 1 | 4 | 4 | 0 |  |
| pr10/06 | b359227227 | 6 | 0 / 1 | 5 | 5 | 0 |  |
| pr10/07 | e5f761b4d6 | 7 | 0 / 1 | 6 | 6 | 0 |  |
| pr10/08 | 888ebf9385 | 8 | 0 / 1 | 7 | 7 | 0 |  |
| pr10/09 | de91970a2b | 9 | 0 / 1 | 8 | 8 | 0 |  |
| pr10/10 | b440bf5bdb | 10 | 0 / 1 | 9 | 9 | 0 |  |
| prmain/01-sse-stream-errors | 6ffa651379 | 1 | 0 / 0 | 1 | 1 | 0 |  |
| prmain/02-moe-fake-ops-shared-health | 020889b7ba | 2 | 0 / 0 | 2 | 2 | 0 |  |
| prmain/03-mpi4py-4.1.2 | 94d23962d2 | 1 | 0 / 1 | 0 | 0 | 0 |  |
| prmain/04-kv-v2-step-rollback | 7159fc6b53 | 2 | 0 / 0 | 2 | 2 | 0 |  |
| prmain/05-eplb-planner-unusable-slots | 341d1ac4eb | 1 | 0 / 0 | 1 | 1 | 0 |  |
| prmain/06-eplb-load-publish | ab1578460f | 1 | 0 / 0 | 1 | 1 | 0 |  |
| prmain/07-eplb-host-sharer-expert-lists | 22c11e2a39 | 1 | 0 / 0 | 1 | 1 | 0 |  |

### LONG - repo A (5)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| origin/wideep-ft | e4c6cefa41 | 7 | 0 / 0 | 7 | 7 | 0 |  |
| origin/wideep-ft-all | 800347c9f4 | 36 | 0 / 21 | 15 | 1 | 0 |  |
| origin/wideep-ft-dev | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 |  |
| origin/wideep-ft-pending | de5f7a84b6 | 67 | 0 / 21 | 46 | 23 | 0 |  |
| wideep-ft-dev | aae00e6c65 | 163 | 0 / 154 | 9 | 0 | 0 | /tmp/claude-161238/work/repo |

### LONG - repo U (7)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| fork/wideep-ft | 6f3ad13ab0 | 13 | 0 / 2 | 11 | 11 | 0 |  |
| fork/wideep-ft-all | 57fd045971 | 33 | 0 / 21 | 12 | 1 | 0 |  |
| fork/wideep-ft-dev | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 |  |
| wideep-ft | e4c6cefa41 | 7 | 0 / 0 | 7 | 7 | 0 | /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM |
| wideep-ft-all | 800347c9f4 | 36 | 0 / 21 | 15 | 1 | 0 |  |
| wideep-ft-dev | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 | /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/dev |
| wideep-ft-pending | de5f7a84b6 | 67 | 0 / 21 | 46 | 23 | 0 |  |

### SUPERSEDED - repo A (12)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| WT:/tmp/claude-161238/prmain.wZps/wt | 0829689c52 | 0 | 0 / 0 | 0 | 0 | 0 | /tmp/claude-161238/prmain.wZps/wt |
| fix-correctness | 64afc01969 | 215 | 207 / 8 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-correct |
| fix-final3-rebased | 08ba507229 | 207 | 207 / 0 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-rebase |
| fix-final5 | 24e2320d34 | 225 | 207 / 18 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-final5 |
| fix-final6 | 2f635c962d | 236 | 236 / 0 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-final6 |
| fix-final7 | 2c3af30ce8 | 238 | 238 / 0 | 0 | 0 | 0 |  |
| fix-ghostmap-log | 9642d022fb | 239 | 238 / 1 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-glog |
| fix-onerep-sort | 2c3af30ce8 | 238 | 238 / 0 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-sort |
| fix-p6-split | c3cbb0ae45 | 239 | 238 / 1 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-p6 |
| fix-parity-cpp | cfbdb88fbb | 218 | 207 / 11 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-parity-cpp |
| fix-parity-py | 54e9787b22 | 215 | 207 / 8 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-parity-py |
| fix-r3-port | e31146463e | 237 | 236 / 1 | 0 | 0 | 0 | /tmp/claude-161238/work/wt-r3port |

### SUPERSEDED~ - repo A (59)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| docs-ft | 758cba7957 | 70 | 0 / 64 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-docs |
| feat-mixed | e55961c983 | 69 | 0 / 62 | 7 | 0 | 0 | /tmp/claude-161238/work/wt-mixed |
| feat-sharded | 643ff79f1b | 75 | 0 / 65 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-sharded |
| final-design | 3091edee7e | 83 | 0 / 74 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-final |
| final-design2 | 98632eb787 | 85 | 0 / 76 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-final2 |
| final-design3 | 6e8657d0be | 89 | 0 / 80 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-final3 |
| fix-abortptr | 425bf9b44f | 103 | 0 / 94 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-abortptr |
| fix-conc | 3dc77e11cc | 124 | 0 / 115 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-conc |
| fix-fc | 5f29b0266b | 105 | 0 / 95 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-fc |
| fix-final2 | aae00e6c65 | 163 | 0 / 154 | 9 | 0 | 0 |  |
| fix-final3 | 547a6aa7ed | 207 | 0 / 197 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-final3b |
| fix-ftoff-parity | 547a6aa7ed | 207 | 0 / 197 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-parity |
| fix-ghost-gtest | 6893f9b2e3 | 69 | 0 / 63 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-ghostgtest |
| fix-ghost-tune | 52eac771d8 | 68 | 0 / 61 | 7 | 0 | 0 |  |
| fix-ghost-tune2 | 859be41acf | 68 | 0 / 61 | 7 | 0 | 0 | /tmp/claude-161238/work/wt-ghosttune |
| fix-maint | f6bc16be99 | 153 | 0 / 144 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-maint |
| fix-mtp-inplace | 57f29976c5 | 71 | 0 / 64 | 7 | 0 | 0 | /tmp/claude-161238/work/wt-mtpinplace |
| fix-offline-eplb | 1daac46af2 | 209 | 0 / 199 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-offline |
| fix-ops | c928c58b49 | 118 | 0 / 109 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-ops |
| fix-opt2e | a6ae3053fd | 187 | 0 / 178 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-opt2e |
| fix-perf | aa120be555 | 122 | 0 / 113 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-perf |
| fix-place | b988cc822e | 120 | 0 / 110 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-place |
| fix-req | 9169400b0e | 104 | 0 / 94 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-req |
| fix-review | fe35d48f8c | 165 | 0 / 156 | 9 | 0 | 0 | /tmp/claude-161238/work/wt-review |
| fix-route-skip | 02e0d45e3b | 70 | 0 / 64 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-routeskip |
| fix-sec | 8522b24830 | 125 | 0 / 113 | 12 | 0 | 0 | /tmp/claude-161238/work/wt-sec |
| fix-slim | 36fe8b4eb2 | 205 | 0 / 195 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-slim |
| fix-tests | fa5265ff27 | 109 | 0 / 99 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-tests |
| fix-tunekey | 04e2f68808 | 180 | 0 / 170 | 10 | 0 | 0 | /tmp/claude-161238/work/wt-tunekey |
| fix-ulfm | f387d3c6ca | 125 | 0 / 114 | 11 | 0 | 0 | /tmp/claude-161238/work/wt-ulfm |
| fold-exit0 | 45c7e0cb3b | 66 | 0 / 60 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-exit0 |
| fold-hangev | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-hangev |
| fold-health | f10bee1a0f | 66 | 0 / 60 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-health |
| fold-restore | 1af6bdc9d5 | 66 | 0 / 60 | 6 | 0 | 0 | /tmp/claude-161238/work/wt-restore |
| origin/fix-hang-progress | 45a8e31caa | 53 | 0 / 42 | 11 | 0 | 0 |  |
| origin/fix-hang-rules | 6d448b7105 | 55 | 0 / 46 | 9 | 0 | 0 |  |
| origin/fix-repeat | 1b9310834b | 54 | 0 / 44 | 10 | 0 | 0 |  |
| origin/fix-specdec | 37ae37e2ce | 55 | 0 / 46 | 9 | 0 | 0 |  |
| origin/ghost | 95e58cfd4f | 58 | 0 / 49 | 9 | 0 | 0 |  |
| origin/wideep-ft-consolidated | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 |  |
| origin/wideep-ft-folded | 075653054e | 49 | 0 / 38 | 11 | 0 | 0 |  |
| origin/wideep-ft-folded2 | e898b2c9a0 | 51 | 0 / 41 | 10 | 0 | 0 |  |
| origin/wideep-ft-folded3 | a787a48c43 | 51 | 0 / 41 | 10 | 0 | 0 |  |
| origin/wideep-ft-folded5 | 4c27cd5a98 | 52 | 0 / 41 | 11 | 0 | 0 |  |
| origin/wideep-ft-folded6 | 9fa88aa7f8 | 53 | 0 / 42 | 11 | 0 | 0 |  |
| origin/wideep-ft-folded7 | 10aa913f52 | 53 | 0 / 44 | 9 | 0 | 0 |  |
| origin/wideep-ft-folded8 | 1c0631057a | 53 | 0 / 45 | 8 | 0 | 0 |  |
| origin/wideep-ft-main | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| origin/wideep-ft-node | 10b2849569 | 23 | 0 / 16 | 7 | 0 | 0 |  |
| origin/wideep-ft-phase1 | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| origin/wideep-ft-restore | 960ab25207 | 19 | 0 / 12 | 7 | 0 | 0 |  |
| origin/wideep-ft-restore2 | 2edca79d4c | 22 | 0 / 15 | 7 | 0 | 0 |  |
| origin/wideep-ft-trim | 01ec646c40 | 19 | 0 / 12 | 7 | 0 | 0 |  |
| pr-final/01 | b9089d1264 | 1 | 0 / 0 | 1 | 0 | 0 |  |
| pr-final/02 | 9aabe958bb | 3 | 0 / 2 | 1 | 0 | 0 |  |
| pr-final/03 | 5c7c832543 | 4 | 0 / 2 | 2 | 0 | 0 |  |
| pr-final/04 | 321ebd33b4 | 5 | 0 / 3 | 2 | 0 | 0 |  |
| pr-final/05 | 5280568012 | 6 | 0 / 3 | 3 | 0 | 0 |  |
| pr-final/06 | 35325d44e3 | 7 | 0 / 3 | 4 | 0 | 0 |  |

### SUPERSEDED~ - repo U (13)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| fork/wideep-ft-main | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| fork/wideep-ft-node | 10b2849569 | 23 | 0 / 16 | 7 | 0 | 0 |  |
| fork/wideep-ft-phase1 | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| fork/wideep-ft-restore | 960ab25207 | 19 | 0 / 12 | 7 | 0 | 0 |  |
| fork/wideep-ft-restore2 | 2edca79d4c | 22 | 0 / 15 | 7 | 0 | 0 |  |
| fork/wideep-ft-trim | 01ec646c40 | 19 | 0 / 12 | 7 | 0 | 0 |  |
| wideep-ft-consolidated | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 |  |
| wideep-ft-main | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| wideep-ft-node | 10b2849569 | 23 | 0 / 16 | 7 | 0 | 0 |  |
| wideep-ft-phase1 | 3e9027f4e4 | 18 | 0 / 12 | 6 | 0 | 0 |  |
| wideep-ft-restore | 960ab25207 | 19 | 0 / 12 | 7 | 0 | 0 |  |
| wideep-ft-restore2 | 2edca79d4c | 22 | 0 / 15 | 7 | 0 | 0 |  |
| wideep-ft-trim | 01ec646c40 | 19 | 0 / 12 | 7 | 0 | 0 |  |

### UNIQUE-covered - repo A (15)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| WT:/tmp/claude-161238/work/wt-pr10-1 | cba25a24ac | 1 | 0 / 0 | 1 | 1 | 0 | /tmp/claude-161238/work/wt-pr10-1 |
| WT:/tmp/claude-161238/work/wt-pr10-2 | 6d6a813728 | 5 | 0 / 1 | 4 | 4 | 0 | /tmp/claude-161238/work/wt-pr10-2 |
| WT:/tmp/claude-161238/work/wt-pr10-3 | 888ebf9385 | 8 | 0 / 1 | 7 | 7 | 0 | /tmp/claude-161238/work/wt-pr10-3 |
| ftpr/01-stream-error-events | 7bbbac2a34 | 2 | 0 / 0 | 2 | 2 | 0 |  |
| origin/fix-dr-tests | 0ac4f28c4a | 54 | 0 / 41 | 13 | 2 | 0 |  |
| origin/fix-kv-reuse-test | 60ae484e66 | 57 | 0 / 47 | 10 | 1 | 0 |  |
| origin/wideep-ft-eplb | 4e3b47d51b | 29 | 0 / 19 | 10 | 1 | 0 |  |
| origin/wideep-ft-hang | 3c93d34184 | 26 | 0 / 17 | 9 | 1 | 0 |  |
| origin/wideep-ft-health | df13da0f74 | 27 | 0 / 18 | 9 | 1 | 0 |  |
| origin/wideep-ft-main-pre-final-20261017 | 69763cd9e8 | 15 | 0 / 5 | 10 | 1 | 0 |  |
| origin/wideep-ft-mtp | 3430226df5 | 29 | 0 / 18 | 11 | 1 | 0 |  |
| origin/zmq-queue | 9bf0be43d4 | 59 | 0 / 47 | 12 | 2 | 0 |  |
| pr-final/07 | 35dcbc7aff | 8 | 0 / 3 | 5 | 1 | 0 |  |
| pr-final/08 | 6e609bf41c | 9 | 0 / 3 | 6 | 2 | 0 |  |
| pr-final/09 | 9b693d2af2 | 10 | 0 / 3 | 7 | 2 | 0 |  |

### UNIQUE-covered - repo U (9)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| fork/wideep-ft-eplb | 4e3b47d51b | 29 | 0 / 19 | 10 | 1 | 0 |  |
| fork/wideep-ft-hang | 3c93d34184 | 26 | 0 / 17 | 9 | 1 | 0 |  |
| fork/wideep-ft-health | df13da0f74 | 27 | 0 / 18 | 9 | 1 | 0 |  |
| fork/wideep-ft-mtp | 3430226df5 | 29 | 0 / 18 | 11 | 1 | 0 |  |
| wideep-ft-eplb | 4e3b47d51b | 29 | 0 / 19 | 10 | 1 | 0 |  |
| wideep-ft-hang | 3c93d34184 | 26 | 0 / 17 | 9 | 1 | 0 |  |
| wideep-ft-health | df13da0f74 | 27 | 0 / 18 | 9 | 1 | 0 |  |
| wideep-ft-main-pre-final-20261017 | 69763cd9e8 | 15 | 0 / 5 | 10 | 1 | 0 |  |
| wideep-ft-mtp | 3430226df5 | 29 | 0 / 18 | 11 | 1 | 0 |  |

### UNIQUE - repo A (138)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| WT:/tmp/claude-161238/pc-fix-host | c372c53358 | 181 | 0 / 168 | 13 | 3 | 3 | /tmp/claude-161238/pc-fix-host |
| WT:/tmp/claude-161238/pc-fix-kern | 19ea00bab0 | 183 | 0 / 169 | 14 | 3 | 3 | /tmp/claude-161238/pc-fix-kern |
| WT:/tmp/claude-161238/pc-opt2e | a7dd7c0d7f | 187 | 0 / 172 | 15 | 4 | 4 | /tmp/claude-161238/pc-opt2e |
| WT:/tmp/claude-161238/work/wt-psf-check | 9ae82f698c | 126 | 0 / 54 | 72 | 11 | 7 | /tmp/claude-161238/work/wt-psf-check |
| WT:/tmp/claude-161238/work/wt-psf-check0 | bf5c4d71af | 124 | 0 / 53 | 71 | 10 | 6 | /tmp/claude-161238/work/wt-psf-check0 |
| WT:/tmp/claude-161238/work/wt-psf-check1 | 68f5562af0 | 125 | 0 / 53 | 72 | 11 | 7 | /tmp/claude-161238/work/wt-psf-check1 |
| WT:/tmp/claude-161238/work/wt-psf-check2 | 0e4f56731c | 122 | 0 / 53 | 69 | 10 | 6 | /tmp/claude-161238/work/wt-psf-check2 |
| WT:/tmp/claude-161238/work/wt-psf-check3 | 05e3802db1 | 123 | 0 / 53 | 70 | 10 | 6 | /tmp/claude-161238/work/wt-psf-check3 |
| diag-pin | 47bb38bda8 | 164 | 0 / 154 | 10 | 1 | 1 | /tmp/claude-161238/work/wt-pin |
| feat-all | 2077a4cf31 | 76 | 0 / 65 | 11 | 3 | 3 | /tmp/claude-161238/work/wt-featall |
| feat-all2 | 39b3cd2c29 | 80 | 0 / 65 | 15 | 5 | 5 | /tmp/claude-161238/work/wt-featall2 |
| feat-all3 | 9c618db9e2 | 85 | 0 / 69 | 16 | 6 | 6 | /tmp/claude-161238/work/wt-featall3 |
| feat-phaseb | 1d260bb46f | 73 | 0 / 61 | 12 | 6 | 6 | /tmp/claude-161238/work/wt-phaseb |
| fix-ci | 37c4877fe9 | 188 | 0 / 173 | 15 | 4 | 4 | /tmp/claude-161238/work/wt-ci |
| fix-cost | 19c024b18e | 184 | 0 / 169 | 15 | 4 | 4 | /tmp/claude-161238/work/wt-cost |
| fix-cost-opt2 | c01f782ed2 | 176 | 0 / 161 | 15 | 5 | 5 |  |
| fix-diag-eplb | da13f20b76 | 206 | 0 / 195 | 11 | 1 | 1 | /tmp/claude-161238/work/wt-diag |
| fix-docs2 | 7e9ef22306 | 175 | 0 / 163 | 12 | 2 | 2 | /tmp/claude-161238/work/wt-docs2 |
| fix-exit | 6b4a8dc18f | 171 | 0 / 158 | 13 | 3 | 3 | /tmp/claude-161238/work/wt-exit |
| fix-final3-tier | 88e5302aac | 210 | 0 / 197 | 13 | 3 | 3 |  |
| fix-final4 | d4c8e0a3fd | 208 | 0 / 197 | 11 | 1 | 1 | /tmp/claude-161238/work/wt-final4 |
| fix-getfailed | 5c188e0021 | 181 | 0 / 171 | 10 | 1 | 1 | /tmp/claude-161238/work/wt-gf |
| fix-host | c372c53358 | 181 | 0 / 168 | 13 | 3 | 3 | /tmp/claude-161238/work/wt-host |
| fix-kern | 3358ea468e | 181 | 0 / 167 | 14 | 3 | 3 | /tmp/claude-161238/work/wt-kern |
| fix-ktests | c8768712f9 | 176 | 0 / 164 | 12 | 2 | 2 | /tmp/claude-161238/work/wt-ktests |
| fix-ring | e0ddea0a8e | 172 | 0 / 159 | 13 | 3 | 3 | /tmp/claude-161238/work/wt-ring |
| fix-staging-batch | 7e230b3a21 | 208 | 207 / 0 | 1 | 1 | 1 | /tmp/claude-161238/work/wt-staging |
| fix-static2 | 643c8afdf3 | 207 | 0 / 196 | 11 | 1 | 1 | /tmp/claude-161238/work/wt-static2 |
| fix-sticky | 85719fc28d | 208 | 0 / 197 | 11 | 1 | 1 | /tmp/claude-161238/work/wt-sticky |
| ftpr/02-eplb-shm-names | ada67c31bb | 5 | 0 / 0 | 5 | 5 | 1 |  |
| ftpr/03-eplb-repacked-slots | ccff070544 | 7 | 0 / 0 | 7 | 6 | 1 |  |
| ftpr/04-moe-fake-ops-shared-health | fccdcaa55a | 9 | 0 / 0 | 9 | 8 | 1 |  |
| ftpr/05-autotune-active-slots | 392b11d981 | 11 | 0 / 0 | 11 | 10 | 3 |  |
| ftpr/06-eplb-load-publish | bdacc7a3bd | 12 | 0 / 0 | 12 | 11 | 3 |  |
| ftpr/07-eplb-unusable-slots | 5773fddda9 | 14 | 0 / 0 | 14 | 13 | 3 |  |
| ftpr/08-kv-v2-step-rollback | c938db5bb7 | 16 | 0 / 0 | 16 | 15 | 3 |  |
| ftpr/09-request-output-checkpoint | 04b17589ba | 17 | 0 / 0 | 17 | 16 | 3 |  |
| ftpr/10-mpi4py-4.1.2 | 197620fe38 | 18 | 0 / 1 | 17 | 16 | 3 |  |
| ftpr/11-ft-a2a-rank-mask-abort | 3ab60a3044 | 21 | 0 / 1 | 20 | 19 | 3 |  |
| ftpr/12-ft-moe-receive-slots | d78eeb66e3 | 23 | 0 / 1 | 22 | 21 | 4 |  |
| ftpr/13-ft-ulfm-survivors | c2bd7232de | 25 | 0 / 1 | 24 | 23 | 4 |  |
| ftpr/14-ft-recovery-authority | e4cd65c471 | 27 | 0 / 1 | 26 | 25 | 4 |  |
| ftpr/15-ft-eplb-restore | cd859e2523 | 29 | 0 / 1 | 28 | 27 | 5 |  |
| ftpr/16-ft-adp-absent-ranks | 2374f2261f | 31 | 0 / 1 | 30 | 29 | 5 |  |
| ftpr/17-ft-executor-recovery | 154efa731c | 34 | 0 / 1 | 33 | 32 | 6 |  |
| ftpr/18-ft-ghost-slots | c1203e32a5 | 37 | 0 / 1 | 36 | 35 | 8 |  |
| ftpr/19-ft-ring-host-copies | 97a0f3695f | 40 | 0 / 1 | 39 | 38 | 9 |  |
| ftpr/20-ft-hang-detection | 903bbcb6b0 | 43 | 0 / 1 | 42 | 41 | 9 |  |
| ftpr/21-ft-health-status | ea6e1c22cb | 46 | 0 / 1 | 45 | 44 | 10 |  |
| ftpr/22-ft-route-one-slot | 2632fe6fdf | 48 | 0 / 1 | 47 | 46 | 10 | /tmp/claude-161238/work/wt-ftpr |
| ftpr/23-review-fixes | 8ab8e7c0b2 | 58 | 0 / 1 | 57 | 56 | 20 | /tmp/claude-161238/work/wt-ftpr-fix |
| origin/backup/pre-fold-c168c39961 | c168c39961 | 23 | 0 / 2 | 21 | 21 | 7 |  |
| origin/backup/pre-fold-c706fdfa84 | c706fdfa84 | 27 | 0 / 2 | 25 | 25 | 11 |  |
| origin/backup/prs-pre-squash | 4474b35e00 | 17 | 0 / 1 | 16 | 10 | 6 |  |
| origin/fix-borrow-map | 606399ab52 | 53 | 0 / 41 | 12 | 1 | 1 |  |
| origin/fix-cq4 | 0f244a2c82 | 86 | 0 / 21 | 65 | 42 | 8 |  |
| origin/fix-fence-before-propose | 7fca3df0b8 | 61 | 0 / 47 | 14 | 3 | 1 |  |
| origin/fix-ft-dispatch | 70c69c5a96 | 68 | 0 / 54 | 14 | 4 | 1 |  |
| origin/fix-hang-deadline | 36b4b9804d | 60 | 0 / 47 | 13 | 3 | 1 |  |
| origin/fix-integ | 237b7240ac | 86 | 0 / 22 | 64 | 41 | 8 |  |
| origin/fix-nr10 | 20d5f59bc9 | 79 | 0 / 21 | 58 | 35 | 6 |  |
| origin/fix-nr2 | 7c6e809281 | 78 | 0 / 21 | 57 | 34 | 6 |  |
| origin/fix-nr58 | 874281e687 | 78 | 0 / 21 | 57 | 34 | 6 |  |
| origin/fix-nr6 | a4468f3813 | 76 | 0 / 21 | 55 | 32 | 5 |  |
| origin/fix-nr7 | ad0c5cac02 | 76 | 0 / 21 | 55 | 32 | 6 |  |
| origin/fix-prime | 89e697b6b4 | 89 | 0 / 22 | 67 | 44 | 8 |  |
| origin/fix-rerun-pin | 3953177de9 | 58 | 0 / 47 | 11 | 2 | 1 |  |
| origin/fix-resp-cost | 1fec69ae64 | 68 | 0 / 54 | 14 | 4 | 1 |  |
| origin/fix-reuse | b7cfb40656 | 56 | 0 / 46 | 10 | 1 | 1 |  |
| origin/fix-sort-gate | 89da95d29e | 68 | 0 / 59 | 9 | 2 | 2 |  |
| origin/fix-sweep | d9fff88a4a | 86 | 0 / 22 | 64 | 41 | 8 |  |
| origin/next-build | d1f10ea6f9 | 69 | 0 / 55 | 14 | 4 | 1 |  |
| origin/prs-rebase-trial | d9777a8612 | 10 | 0 / 1 | 9 | 3 | 2 |  |
| origin/trim-vector-47b33f8758 | 47b33f8758 | 19 | 0 / 12 | 7 | 1 | 1 |  |
| origin/wideep-ft-dev-pre-consolidate | 79cdd9f36f | 64 | 0 / 49 | 15 | 4 | 1 |  |
| origin/wideep-ft-dev-prefold | 9ad3d1f8e0 | 89 | 0 / 23 | 66 | 43 | 8 |  |
| origin/wideep-ft-exploration | 474d437ad4 | 29 | 0 / 0 | 29 | 29 | 23 |  |
| origin/wideep-ft-exploration-next | d9084befa9 | 180 | 0 / 6 | 174 | 174 | 104 |  |
| origin/wideep-ft-fix | 625c5e1759 | 17 | 0 / 4 | 13 | 9 | 6 |  |
| origin/wideep-ft-fix-pre-gather | b474200595 | 14 | 0 / 2 | 12 | 12 | 1 |  |
| origin/wideep-ft-folded4 | b83e2088c9 | 52 | 0 / 41 | 11 | 1 | 1 |  |
| origin/wideep-ft-next | a958187bfc | 31 | 0 / 3 | 28 | 20 | 14 |  |
| origin/wideep-ft-prs | 1707fac125 | 13 | 0 / 2 | 11 | 5 | 3 |  |
| pr-final/10 | 6fd0c1913c | 11 | 0 / 3 | 8 | 3 | 1 |  |
| pr-final/11 | 8929b3825c | 12 | 0 / 3 | 9 | 3 | 1 |  |
| pr-final/11z | ac54a97d9e | 115 | 0 / 49 | 66 | 10 | 6 |  |
| pr-final/12 | 4d445f3976 | 14 | 0 / 3 | 11 | 3 | 1 |  |
| pr-final/13 | 53e6057bc3 | 16 | 0 / 4 | 12 | 3 | 1 |  |
| pr-final/13b | be079f7da2 | 89 | 0 / 39 | 50 | 7 | 4 |  |
| pr-final/14 | 4e81ee8766 | 22 | 0 / 5 | 17 | 3 | 1 |  |
| pr-final/14b | dfe128de22 | 91 | 0 / 41 | 50 | 7 | 4 |  |
| pr-final/15 | fdc322efad | 40 | 0 / 16 | 24 | 3 | 1 |  |
| pr-final/15b | 3682b0d00b | 93 | 0 / 41 | 52 | 8 | 4 |  |
| pr-final/16 | 07cf5045c2 | 25 | 0 / 7 | 18 | 3 | 1 |  |
| pr-final/16b | 24c1f180d2 | 94 | 0 / 41 | 53 | 8 | 4 |  |
| pr-final/17 | 126a65b01c | 26 | 0 / 8 | 18 | 3 | 1 |  |
| pr-final/17b | c290549c63 | 58 | 0 / 27 | 31 | 4 | 2 |  |
| pr-final/17c | c9c1b0db25 | 69 | 0 / 30 | 39 | 5 | 2 |  |
| pr-final/17z | 7e73056291 | 119 | 0 / 52 | 67 | 10 | 6 |  |
| pr-final/18 | 3b57603063 | 28 | 0 / 9 | 19 | 3 | 1 |  |
| pr-final/19 | 3e2ba7cc57 | 33 | 0 / 11 | 22 | 3 | 1 |  |
| pr-final/19b | 6c5f77a2ad | 59 | 0 / 28 | 31 | 4 | 2 |  |
| pr-final/19c | dcf8e58820 | 98 | 0 / 41 | 57 | 9 | 5 |  |
| pr-final/19z1 | 6557b05345 | 109 | 0 / 45 | 64 | 10 | 6 |  |
| pr-final/19z2 | 256d145836 | 116 | 0 / 49 | 67 | 10 | 6 |  |
| pr-final/20a | e11b6ff556 | 48 | 0 / 22 | 26 | 3 | 1 |  |
| pr-final/20b | 2f12cfd469 | 64 | 0 / 29 | 35 | 5 | 2 |  |
| pr-final/20z | 68fe820a7b | 113 | 0 / 48 | 65 | 10 | 6 |  |
| pr-final/21a | 41789c5844 | 37 | 0 / 14 | 23 | 3 | 1 |  |
| pr-final/21b | aeef8e85f9 | 50 | 0 / 23 | 27 | 4 | 2 |  |
| pr-final/21c | f918195264 | 61 | 0 / 28 | 33 | 5 | 2 |  |
| pr-final/22 | 0d70cd81c2 | 46 | 0 / 20 | 26 | 3 | 1 |  |
| pr-final/22c | 75d1a1c508 | 100 | 0 / 43 | 57 | 9 | 5 |  |
| pr-final/22z | 369820ca8f | 114 | 0 / 49 | 65 | 10 | 6 |  |
| pr-final/23a | 312adca97a | 47 | 0 / 21 | 26 | 3 | 1 |  |
| pr-final/23b_22b | 87919f1bb0 | 67 | 0 / 30 | 37 | 5 | 2 |  |
| pr-final/23c | 7d699a4e60 | 101 | 0 / 44 | 57 | 9 | 5 |  |
| pr-final/24a | 448472d070 | 35 | 0 / 13 | 22 | 3 | 1 |  |
| pr-final/24b | e65441fffd | 41 | 0 / 16 | 25 | 3 | 1 |  |
| pr-final/24c | 01b0f7d57d | 52 | 0 / 24 | 28 | 4 | 2 |  |
| pr-final/24d | 74d8b0ea3b | 56 | 0 / 25 | 31 | 4 | 2 |  |
| pr-final/24e | 74cd03dd49 | 72 | 0 / 32 | 40 | 5 | 2 |  |
| pr-final/24f | 1e3a6eb0da | 104 | 0 / 44 | 60 | 9 | 5 |  |
| pr-final/24z | 0e4f56731c | 122 | 0 / 53 | 69 | 10 | 6 |  |
| pr-final/25a | cad5c8fa9e | 54 | 0 / 25 | 29 | 4 | 2 |  |
| pr-final/25b | 31b160beb4 | 78 | 0 / 34 | 44 | 6 | 3 |  |
| pr-final/25c | 4653856251 | 105 | 0 / 44 | 61 | 9 | 5 |  |
| pr-final/25z | 68f5562af0 | 125 | 0 / 53 | 72 | 11 | 7 |  |
| pr-final/26 | c713d0de68 | 86 | 0 / 38 | 48 | 6 | 3 |  |
| pr-final/26z1 | 7d4edc8b4d | 117 | 0 / 50 | 67 | 10 | 6 |  |
| pr-final/26z2 | d83a2e43db | 121 | 0 / 52 | 69 | 10 | 6 |  |
| pr-final/27a | e47a0546d7 | 63 | 0 / 29 | 34 | 5 | 2 |  |
| pr-final/27b | cf4bf1e6d4 | 107 | 0 / 44 | 63 | 10 | 6 |  |
| pr-final/27z1 | 4929fe3c7c | 110 | 0 / 45 | 65 | 10 | 6 |  |
| pr-final/27z2 | 98499e34b5 | 120 | 0 / 52 | 68 | 10 | 6 |  |
| pr-final/28 | 11f61041f0 | 82 | 0 / 36 | 46 | 6 | 3 |  |
| pr-stack | 787ed7c59b | 82 | 0 / 43 | 39 | 6 | 3 | /tmp/claude-161238/work/wt-prstack |
| pr-stack-final | 68f5562af0 | 125 | 0 / 53 | 72 | 11 | 7 | /tmp/claude-161238/work/wt-prstack-final |

### UNIQUE - repo U (19)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| backup/pre-fold-c168c39961 | c168c39961 | 23 | 0 / 2 | 21 | 21 | 7 |  |
| backup/pre-fold-c706fdfa84 | c706fdfa84 | 27 | 0 / 2 | 25 | 25 | 11 |  |
| backup/prs-pre-squash | 4474b35e00 | 17 | 0 / 1 | 16 | 10 | 6 |  |
| fix-borrow-map | 606399ab52 | 53 | 0 / 41 | 12 | 1 | 1 |  |
| fix-resp-cost | 1fec69ae64 | 68 | 0 / 54 | 14 | 4 | 1 |  |
| fix-reuse | b7cfb40656 | 56 | 0 / 46 | 10 | 1 | 1 | /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/reuse |
| fix-sort-gate | 4f0c308b02 | 68 | 0 / 59 | 9 | 2 | 2 | /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/sortgate |
| fork/wideep-ft-exploration | 479d51c512 | 142 | 0 / 2 | 140 | 140 | 100 |  |
| prs-rebase-trial | d9777a8612 | 10 | 0 / 1 | 9 | 3 | 2 |  |
| refs/stash | 20788762d5 | 18 | 0 / 0 | 19 | 19 | 17 |  |
| trim-vector-47b33f8758 | 47b33f8758 | 19 | 0 / 12 | 7 | 1 | 1 |  |
| wideep-ft-dev-pre-consolidate | 79cdd9f36f | 64 | 0 / 49 | 15 | 4 | 1 |  |
| wideep-ft-dev-prefold | 9ad3d1f8e0 | 89 | 0 / 23 | 66 | 43 | 8 |  |
| wideep-ft-exploration | 474d437ad4 | 29 | 0 / 0 | 29 | 29 | 23 |  |
| wideep-ft-exploration-next | d9084befa9 | 180 | 0 / 6 | 174 | 174 | 104 |  |
| wideep-ft-fix | 625c5e1759 | 17 | 0 / 4 | 13 | 9 | 6 |  |
| wideep-ft-fix-pre-gather | b474200595 | 14 | 0 / 2 | 12 | 12 | 1 |  |
| wideep-ft-next | a958187bfc | 31 | 0 / 3 | 28 | 20 | 14 |  |
| wideep-ft-prs | 1707fac125 | 13 | 0 / 2 | 11 | 5 | 3 |  |

### EXTERNAL - repo U (5)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| pr/15785 | 1c1f960f0c | 7 | 0 / 0 | 7 | 7 | 0 |  |
| pr/15788 | fe0fceceaf | 1 | 0 / 0 | 1 | 1 | 0 |  |
| pr/15789 | d1c4cebd5e | 5 | 0 / 0 | 5 | 5 | 0 |  |
| pr/15895 | ecdcd76dcd | 6 | 0 / 0 | 6 | 6 | 0 |  |
| pr/19610 | 36a09c1352 | 50 | 0 / 0 | 50 | 50 | 0 |  |

### BASE - repo A (2)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| origin/HEAD | 191988d1fb | 65 | 0 / 59 | 6 | 0 | 0 |  |
| upstream/main | 0829689c52 | 0 | 0 / 0 | 0 | 0 | 0 |  |

### BASE - repo U (3)

| ref | sha | commits over main | in final (anc / patch-eq) | not patch-eq | no-subject-match | undecided | worktree |
|---|---|---|---|---|---|---|---|
| fork/HEAD | fc2f8543e0 | 0 | 0 / 0 | 0 | 0 | 0 |  |
| fork/main | fc2f8543e0 | 0 | 0 / 0 | 0 | 0 | 0 |  |
| origin/main | fc52f0adee | 0 | 0 / 0 | 0 | 0 | 0 |  |

## UNIQUE commits needing a decision

Grouped by the set of refs carrying them; one row per distinct subject (best-coverage copy shown). Refs abbreviated: A:/U: prefixes dropped, `origin/`/`fork/` collapsed onto the branch name.

### G1 U stash@{0} (2026-10-07, 'wideep-final-checkpoint-before-latest-rebase' on wideep-ft) - 17 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 09b9b54c8e | 2026-10-07 | 0.00 | 246 | [None][test] document WideEP process-loss qualification and limits | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 20788762d5 | 2026-10-07 | 0.00 | 407 | On wideep-ft: wideep-final-checkpoint-before-latest-rebase | refs/stash |
| 21d631c31a | 2026-10-07 | 0.00 | 32 | [None][docs] record isolated WideEP recovery prerequisite proofs | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 264f98a017 | 2026-10-07 | 0.00 | 19 | [None][test] record multi-node MPI survivor control and recovery blockers | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 332c9f718e | 2026-10-07 | 0.00 | 1427 | [None][test] define static WideEP deployment evidence | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 352455aab3 | 2026-10-07 | 0.06 | 268 | [None][fix] propagate fatal RPC worker errors to inference clients | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 404a14ac6d | 2026-10-07 | 0.00 | 89 | [None][docs] characterize MPI host failures and WideEP watchdog behavior | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 4e99e36ef2 | 2026-10-07 | 0.00 | 82 | [None][test] record MPI WideEP fault and restart behavior | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 6acec0e3d2 | 2026-10-07 | 0.00 | 12 | [None][docs] record MPI binding assessment and graph reuse limits | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 805990b2fc | 2026-10-07 | 0.00 | 67 | [None][test] guard WideEP fault cleanup against storage failure and PID reuse | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 87a630ac34 | 2026-10-07 | 0.02 | 279 | [None][fix] fail pending Ray requests on confirmed model actor death | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 8d06880cc3 | 2026-10-07 | 0.00 | 1110 | [None][test] run WideEP fault and restart checks through MPI | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| 9f5acdc2e5 | 2026-10-07 | 0.00 | 510 | [None][test] verify WideEP communication faults and interrupted cleanup | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| a21341c3e7 | 2026-10-07 | 0.00 | 1139 | [None][test] consolidate WideEP fault injection and explicit restart checks | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| dd99c07e00 | 2026-10-07 | 0.50 | 4 | [None][refactor] remove Ray-specific actor death monitoring | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| f263042139 | 2026-10-07 | 0.00 | 20 | [None][fix] preserve engine failure causes when rejecting new LLM requests | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |
| f99c7cb21f | 2026-10-07 | 0.00 | 2705 | [None][test] add bounded Ray WideEP process-fault regressions | refs/stash, wideep-ft-exploration, wideep-ft-exploration-next |

### G2 wideep-ft-exploration / exploration-next (pre-v7 history, 10-11..10-15) - 73 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 6f00fb52c7 | 2026-10-07 | 0.00 | 165 | [None][test] harden WideEP regression cleanup and clarify restart timings | wideep-ft-exploration, wideep-ft-exploration-next |
| e98d97b9ce | 2026-10-07 | 0.00 | 22 | [None][docs] record WideEP same-graph abort proof limits | wideep-ft-exploration, wideep-ft-exploration-next |
| f3bb2c78c1 | 2026-10-07 | 0.00 | 375 | [None][test] qualify MPI WideEP fault and restart benchmarks | wideep-ft-exploration, wideep-ft-exploration-next |
| 3e4674c99b | 2026-10-08 | 0.00 | 21 | [None][docs] summarize fence recovery prerequisite proofs | wideep-ft-exploration, wideep-ft-exploration-next |
| 1a8a89c6fb | 2026-10-09 | 0.00 | 7 | [None][docs] record immutable workspace and MPI helper proofs | wideep-ft-exploration, wideep-ft-exploration-next |
| a515aa3628 | 2026-10-09 | 0.00 | 12 | [None][docs] record fixed-slot recovery component evidence | wideep-ft-exploration, wideep-ft-exploration-next |
| 182ce42f98 | 2026-10-11 | 0.70 | 114 | [None][feat] keep the MPI session serving across an admitted peer loss | wideep-ft-exploration, wideep-ft-exploration-next |
| 18d9c77327 | 2026-10-11 | 0.57 | 1219 | [None][chore] WIP: round-3 review fixes, V2 and block-reuse rollback, integration test | wideep-ft-exploration, wideep-ft-exploration-next |
| 2a0a4fca90 | 2026-10-11 | 0.00 | 427 | [None][docs] update WideEP recovery design notes and harness | wideep-ft-exploration, wideep-ft-exploration-next |
| 44eded2a5c | 2026-10-11 | 0.01 | 124 | [None][feat] pause frontend admission during WideEP recovery | wideep-ft-exploration, wideep-ft-exploration-next |
| 474d437ad4 | 2026-10-11 | 0.06 | 8119 | [None][chore] snapshot Codex WideEP recovery work in progress | wideep-ft-exploration, wideep-ft-exploration-next |
| 4e8b440b31 | 2026-10-11 | 0.06 | 8119 | [None][chore] snapshot WideEP recovery work in progress | wideep-ft-exploration, wideep-ft-exploration-next |
| 69703daa56 | 2026-10-11 | 0.00 | 965 | [None][chore] snapshot WideEP recovery harness and design notes | wideep-ft-exploration, wideep-ft-exploration-next |
| 72f22284b0 | 2026-10-11 | 0.32 | 407 | [None][feat] add graph-latched MoE execution state for WideEP recovery | wideep-ft-exploration, wideep-ft-exploration-next |
| 738b46ae9b | 2026-10-11 | 0.00 | 185 | [None][docs] rewrite the WideEP recovery design guide | wideep-ft-exploration, wideep-ft-exploration-next |
| 7f0fbad916 | 2026-10-11 | 0.34 | 908 | [None][feat] add the WideEP recovery authority and protocol | wideep-ft-exploration, wideep-ft-exploration-next |
| a0311cb56a | 2026-10-11 | 0.24 | 220 | [None][feat] use fixed per-source slots in the fence MoE all-to-all descriptor | wideep-ft-exploration, wideep-ft-exploration-next |
| b15c9306e3 | 2026-10-11 | 0.25 | 777 | [None][feat] recover the executor from an idle WideEP peer loss | wideep-ft-exploration, wideep-ft-exploration-next |
| bc998e8c84 | 2026-10-11 | 0.65 | 499 | [None][feat] add ULFM survivor communicators for host collectives | wideep-ft-exploration, wideep-ft-exploration-next |
| c02a6025d3 | 2026-10-11 | 0.43 | 58 | [None][chore] WIP: nullable remove_sequence request, two-process cancel test | wideep-ft-exploration, wideep-ft-exploration-next |
| ed20a243ad | 2026-10-11 | 0.49 | 2262 | [None][chore] WIP: apply round-1 review fixes and simplify the FT kernel path | wideep-ft-exploration, wideep-ft-exploration-next |
| 091199e002 | 2026-10-12 | 0.00 | 3 | [None][docs] rewrap overview | wideep-ft-exploration, wideep-ft-exploration-next |
| 0b00c07a97 | 2026-10-12 | 0.85 | 127 | [None][feat] revert the KV cache allocations of a discarded step | wideep-ft-exploration, wideep-ft-exploration-next |
| 122dde086e | 2026-10-12 | 0.00 | 61 | [None][docs] chart-led summary in the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| 183d3017f4 | 2026-10-12 | 0.00 | 2548 | [None][docs] record the folded series, stress evidence and harness updates | wideep-ft-exploration, wideep-ft-exploration-next |
| 1c0e1dfef7 | 2026-10-12 | 0.63 | 580 | [None][feat] MoE execution state for WideEP recovery | wideep-ft-exploration, wideep-ft-exploration-next |
| 2660e84ae1 | 2026-10-12 | 0.79 | 216 | [None][refactor] let attention-DP routing skip absent ranks | wideep-ft-exploration, wideep-ft-exploration-next |
| 32726e23c2 | 2026-10-12 | 0.00 | 114 | [None][docs] restructure the WideEP FT design doc around failover questions | wideep-ft-exploration, wideep-ft-exploration-next |
| 33b0103ed1 | 2026-10-12 | 0.14 | 223 | [None][feat] resident-replica EPLB placement after a rank loss | wideep-ft-exploration, wideep-ft-exploration-next |
| 35b350ae77 | 2026-10-12 | 0.00 | 93 | [None][docs] plain-language results in the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| 385630faa2 | 2026-10-12 | 0.00 | 11 | [None][docs] clearer dispatch layout diagram | wideep-ft-exploration, wideep-ft-exploration-next |
| 3eba8df683 | 2026-10-12 | 0.00 | 12 | [None][infra] update WideEP FT docs for identical recoverable-rank reports | wideep-ft-exploration, wideep-ft-exploration-next |
| 43f0e5e854 | 2026-10-12 | 0.00 | 13 | [None][docs] use main's names in the dispatch layout | wideep-ft-exploration, wideep-ft-exploration-next |
| 4bf761d63f | 2026-10-12 | 0.00 | 63 | [None][docs] Phase 1 work-item status in the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| 5300c34529 | 2026-10-12 | 0.00 | 19 | [None][docs] code links in the WideEP FT deliverables | wideep-ft-exploration, wideep-ft-exploration-next |
| 551022ac78 | 2026-10-12 | 0.00 | 40 | [None][docs] mpi4py 4.1.2 compatibility evidence | wideep-ft-exploration, wideep-ft-exploration-next |
| 587e193c69 | 2026-10-12 | 0.00 | 220 | [None][docs] WideEP FT design doc in the review format | wideep-ft-exploration, wideep-ft-exploration-next |
| 5bfd7b97b2 | 2026-10-12 | 0.00 | 18 | [None][infra] clarify the #15785 boundary and link the docs page in the WideEP FT design | wideep-ft-exploration, wideep-ft-exploration-next |
| 6052d501f6 | 2026-10-12 | 0.00 | 20 | [None][docs] one-model results in the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| 7468074ed6 | 2026-10-12 | 0.63 | 874 | [None][feat] WideEP recovery authority | wideep-ft-exploration, wideep-ft-exploration-next |
| 87cb72f15b | 2026-10-12 | 0.14 | 129 | [None][feat] resident-replica EPLB placement after a peer loss | wideep-ft-exploration, wideep-ft-exploration-next |
| 8a40bf81d1 | 2026-10-12 | 0.00 | 33 | [None][docs] final results in the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| 8db13cb84e | 2026-10-12 | 0.00 | 710 | [None][docs] update the WideEP FT design doc, results chart and PR drafts | wideep-ft-exploration, wideep-ft-exploration-next |
| a7daf68547 | 2026-10-12 | 0.00 | 115 | [None][docs] tighten the WideEP FT design doc | wideep-ft-exploration, wideep-ft-exploration-next |
| b91b95eaa4 | 2026-10-12 | 0.00 | 8 | [None][infra] list the v7 draft PRs in the WideEP FT design | wideep-ft-exploration, wideep-ft-exploration-next |
| bc7d2b8711 | 2026-10-12 | 0.00 | 29 | [None][infra] update WideEP FT docs to the retitled commits | wideep-ft-exploration, wideep-ft-exploration-next |
| bf3791cb9c | 2026-10-12 | 0.06 | 250 | [None][fix] propagate fatal RPC worker errors | wideep-ft-exploration, wideep-ft-exploration-next |
| dceb6ca0de | 2026-10-12 | 0.33 | 486 | [None][feat] fault-tolerant execution descriptor for the NVLinkOneSided all-to-all | wideep-ft-exploration, wideep-ft-exploration-next |
| e72f2a951e | 2026-10-12 | 0.00 | 5 | [None][docs] WideEP FT design doc: own-crash path and test coverage | wideep-ft-exploration, wideep-ft-exploration-next |
| f947374c24 | 2026-10-12 | 0.00 | 12 | [None][infra] update WideEP FT docs for live-rank admission capacity | wideep-ft-exploration, wideep-ft-exploration-next |
| 1172ac4194 | 2026-10-13 | 0.00 | 289 | [None][doc] add the WideEP FT failover breakdown and apply audit fixes | wideep-ft-exploration, wideep-ft-exploration-next |
| 22766a6629 | 2026-10-13 | 0.00 | 62 | [None][doc] use plain standard charts for the WideEP FT evaluation | wideep-ft-exploration, wideep-ft-exploration-next |
| 27f064d3fb | 2026-10-13 | 0.00 | 154 | [None][doc] condense the WideEP FT late-write diagram | wideep-ft-exploration, wideep-ft-exploration-next |
| 3804fe1c0d | 2026-10-13 | 0.00 | 7 | [None][doc] finalize the WideEP FT diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| 479d51c512 | 2026-10-13 | 0.00 | 54 | Add the prefill-load failover, the cap-128 finding and the #15785 boundary | wideep-ft-exploration, wideep-ft-exploration-next |
| 8f66bed4fa | 2026-10-13 | 0.00 | 56 | Add in-flight kills with a new-request probe to the failover chart | wideep-ft-exploration, wideep-ft-exploration-next |
| 907f07a89b | 2026-10-13 | 0.00 | 8 | [None][doc] clarify the WideEP FT throughput chart labels | wideep-ft-exploration, wideep-ft-exploration-next |
| 9691c19f42 | 2026-10-13 | 0.00 | 16 | [None][doc] tighten the WideEP FT evaluation notes | wideep-ft-exploration, wideep-ft-exploration-next |
| 99a33e1e83 | 2026-10-13 | 0.00 | 12 | [None][doc] clarify that only the payload layout changes for late writes | wideep-ft-exploration, wideep-ft-exploration-next |
| b0342dfe64 | 2026-10-13 | 0.00 | 494 | [None][doc] redraw the WideEP FT architecture and late-write diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| b1d1899497 | 2026-10-13 | 0.00 | 444 | [None][doc] add the WideEP FT overhead and load evaluation | wideep-ft-exploration, wideep-ft-exploration-next |
| b76ab6b893 | 2026-10-13 | 0.00 | 5 | [None][doc] update the WideEP FT architecture caption | wideep-ft-exploration, wideep-ft-exploration-next |
| c070bb1ca4 | 2026-10-13 | 0.00 | 431 | [None][doc] apply review fixes to the WideEP FT diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| cf88faea8d | 2026-10-13 | 0.00 | 367 | [None][doc] add WideEP FT integration and late-write diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| d79ecd39c1 | 2026-10-13 | 0.00 | 471 | [None][doc] add the WideEP FT kernel-path diagram | wideep-ft-exploration, wideep-ft-exploration-next |
| dbff448af7 | 2026-10-13 | 0.00 | 89 | [None][doc] keep the WideEP FT user doc as a draft outside the series | wideep-ft-exploration, wideep-ft-exploration-next |
| e01da9c2ac | 2026-10-13 | 0.00 | 4 | [None][doc] refresh WideEP FT commit links | wideep-ft-exploration, wideep-ft-exploration-next |
| eaf9c5f9db | 2026-10-13 | 0.00 | 14 | [None][doc] use benchmark numbers recomputed from the raw traces | wideep-ft-exploration, wideep-ft-exploration-next |
| ee807bfde6 | 2026-10-13 | 0.00 | 289 | [None][doc] restructure the WideEP FT design doc to mirror the review doc | wideep-ft-exploration, wideep-ft-exploration-next |
| f135dec4ea | 2026-10-13 | 0.00 | 215 | Simplify overhead and late-write diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| f3bf4cb960 | 2026-10-13 | 0.00 | 208 | [None][doc] use a plain palette in the WideEP FT diagrams | wideep-ft-exploration, wideep-ft-exploration-next |
| b474200595 | 2026-10-14 | 0.08 | 13 | [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs | backup/pre-fold-c168c39961, backup/pre-fold-c706fdfa84, wideep-ft-exploration-next, wideep-ft-fix-pre-gather |
| c0ed3f32cd | 2026-10-15 | 0.20 | 10 | WIP gather backoff | wideep-ft-exploration-next |

### G3 ftpr/* PR stack (A) - 20 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 0efadf22ae | 2026-10-19 | 0.00 | 470 | [None][chore] Move peer-loss request bookkeeping and expert restore out of peer_loss_recovery.py (no behaviour | ftpr/23-review-fixes |
| 154efa731c | 2026-10-19 | 0.81 | 187 | [None][doc] Add the WideEP fault-tolerance guide | ftpr/17-ft-executor-recovery, ftpr/18-ft-ghost-slots, ftpr/19-ft-ring-host-copies, ftpr/20-ft-hang-detection, ftpr/21-ft-health-status +2 |
| 21bf366934 | 2026-10-19 | 0.30 | 70 | [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert | ftpr/05-autotune-active-slots, ftpr/06-eplb-load-publish, ftpr/07-eplb-unusable-slots, ftpr/08-kv-v2-step-rollback, ftpr/09-request-output-checkpoint +14 |
| 2e1ff28c6a | 2026-10-19 | 0.81 | 254 | [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place | ftpr/12-ft-moe-receive-slots, ftpr/13-ft-ulfm-survivors, ftpr/14-ft-recovery-authority, ftpr/15-ft-eplb-restore, ftpr/16-ft-adp-absent-ranks +7 |
| 33180826d8 | 2026-10-19 | 0.14 | 28 | [None][fix] Refuse CUDA graph capture of NVLinkOneSided collectives the host watchdog follows | ftpr/23-review-fixes |
| 392b11d981 | 2026-10-19 | 0.08 | 12 | [None][test] Cover the tuning profile with empty local EPLB slots | ftpr/05-autotune-active-slots, ftpr/06-eplb-load-publish, ftpr/07-eplb-unusable-slots, ftpr/08-kv-v2-step-rollback, ftpr/09-request-output-checkpoint +14 |
| 664d6a9917 | 2026-10-19 | 0.87 | 547 | [None][test] Cover ghost EPLB slots and the KV cache memory they borrow | ftpr/18-ft-ghost-slots, ftpr/19-ft-ring-host-copies, ftpr/20-ft-hang-detection, ftpr/21-ft-health-status, ftpr/22-ft-route-one-slot +1 |
| 675f378e04 | 2026-10-19 | 0.57 | 21 | [None][fix] Pre-map peers' EPLB host copies only with peer-loss recovery | ftpr/23-review-fixes |
| 6ac6447b6c | 2026-10-19 | 0.87 | 421 | [None][feat] Restore a lost rank's experts into EPLB slots from host copies | ftpr/15-ft-eplb-restore, ftpr/16-ft-adp-absent-ranks, ftpr/17-ft-executor-recovery, ftpr/18-ft-ghost-slots, ftpr/19-ft-ring-host-copies +4 |
| 7070dc161f | 2026-10-19 | 0.37 | 19 | [None][fix] Keep CutlassMoeFCRunnerInterface::runMoe's signature; set the input source slots through a setter | ftpr/23-review-fixes |
| 7a809fbece | 2026-10-19 | 0.00 | 1 | [None][test] Name the fake tp_allgather_int64 placeholder departed, as MPIDist does | ftpr/23-review-fixes |
| 8394411b45 | 2026-10-19 | 0.33 | 24 | [None][fix] Let a survivor that fails during a recovery exit alone until the install, so the others plan again | ftpr/23-review-fixes |
| 8ab8e7c0b2 | 2026-10-19 | 0.40 | 15 | [None][fix] Discard a step whose MoE collective recorded an abort, even before the watch reports the failure | ftpr/23-review-fixes |
| 97a0f3695f | 2026-10-19 | 0.18 | 11 | [None][doc] Document WideEP ring host copies | ftpr/19-ft-ring-host-copies, ftpr/20-ft-hang-detection, ftpr/21-ft-health-status, ftpr/22-ft-route-one-slot, ftpr/23-review-fixes |
| 9f92c3e9d3 | 2026-10-19 | 0.27 | 15 | [None][fix] Do not send a stream error event after [DONE] | ftpr/23-review-fixes |
| ab336177a6 | 2026-10-19 | 0.00 | 26 | [None][fix] Report unprotected, not starting, under an authority that restarted while serving | ftpr/23-review-fixes |
| ada67c31bb | 2026-10-19 | 0.00 | 4 | [None][doc] Describe the per-job EPLB shared memory names and their cleanup | ftpr/02-eplb-shm-names, ftpr/03-eplb-repacked-slots, ftpr/04-moe-fake-ops-shared-health, ftpr/05-autotune-active-slots, ftpr/06-eplb-load-publish +17 |
| c1203e32a5 | 2026-10-19 | 0.33 | 9 | [None][doc] Document WideEP ghost extra slots | ftpr/18-ft-ghost-slots, ftpr/19-ft-ring-host-copies, ftpr/20-ft-hang-detection, ftpr/21-ft-health-status, ftpr/22-ft-route-one-slot +1 |
| e646fa1f13 | 2026-10-19 | 0.26 | 23 | [None][fix] Answer an attention-DP pin outside the group as a client error, not a retryable peer loss | ftpr/23-review-fixes |
| ea6e1c22cb | 2026-10-19 | 0.86 | 35 | [None][doc] Document the /health fault-tolerance states | ftpr/21-ft-health-status, ftpr/22-ft-route-one-slot, ftpr/23-review-fixes |

### G4 pr-final/* / pr-stack* / psf-check / wt-pr10 scratch worktrees (A) - 7 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 82182e181b | 2026-10-19 | 0.86 | 73 | [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots | pr-final/10, pr-final/11, pr-final/11z, pr-final/12, pr-final/13 +55 |
| 1f10f3bf1f | 2026-10-20 | 0.87 | 99 | [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending | pr-final/11z, pr-final/13b, pr-final/14b, pr-final/15b, pr-final/16b +23 |
| 62cc9ca632 | 2026-10-20 | 0.00 | 2 | [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event | pr-final/11z, pr-final/17z, pr-final/19c, pr-final/19z1, pr-final/19z2 +19 |
| 6d78167fa8 | 2026-10-20 | 0.90 | 58 | [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it | pr-final/11z, pr-final/13b, pr-final/14b, pr-final/15b, pr-final/16b +27 |
| 9ae82f698c | 2026-10-20 | 0.71 | 17 | [None][fix] Default online EPLB to empty extra slots where GPU memory is a NUMA node, and log the BAR1 size | pr-final/25z, pr-stack-final, wt:wt-psf-check, wt:wt-psf-check1 |
| aeef8e85f9 | 2026-10-20 | 0.25 | 8 | [None][feat] Accept online EPLB with WideEP peer-loss recovery | pr-final/11z, pr-final/13b, pr-final/14b, pr-final/15b, pr-final/16b +39 |
| cf4bf1e6d4 | 2026-10-20 | 0.79 | 19 | [None][test] Restore after a peer loss into online EPLB's empty extra slots | pr-final/11z, pr-final/17z, pr-final/19z1, pr-final/19z2, pr-final/20z +14 |

### G5 U older fix/backup/rewrite branches (10-14..10-19) - 29 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 11952e0c2a | 2026-10-15 | 0.88 | 223 | fixup! [None][feat] ULFM survivor communicators for host collectives | backup/pre-fold-c168c39961, backup/pre-fold-c706fdfa84 |
| 162132465b | 2026-10-15 | 0.42 | 99 | fixup! [None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all | backup/pre-fold-c168c39961, backup/pre-fold-c706fdfa84 |
| 19e9d13e95 | 2026-10-15 | 0.67 | 3 | fixup! [None][test] add a WideEP peer-loss recovery integration test | backup/pre-fold-c706fdfa84 |
| 16917b2f3e | 2026-10-16 | 0.14 | 28 | [None][perf] Exchange attention-DP rank state as one int64 allgather | wideep-ft-next |
| 2323f0c18f | 2026-10-16 | 0.55 | 40 | fixup! [None][feat] Support the overlap scheduler with WideEP peer-loss recovery | wideep-ft-next |
| 2828410be6 | 2026-10-16 | 0.59 | 29 | [None][fix] Survive a peer loss mid-iteration of the overlap scheduler | wideep-ft-next |
| 3cf5f06ffe | 2026-10-16 | 0.25 | 8 | fixup! [None][feat] Read the all-to-all receive slots in place under a MoE execution state | backup/prs-pre-squash |
| 48f15a1d8f | 2026-10-16 | 0.69 | 13 | fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery | backup/pre-fold-c168c39961, backup/pre-fold-c706fdfa84 |
| 49867dc4b1 | 2026-10-16 | 0.90 | 145 | [None][feat] Roll back KV Cache Manager V2 allocations of a discarded step | wideep-ft-fix, wideep-ft-next |
| 4bd18f06b0 | 2026-10-16 | 0.30 | 101 | [None][perf] Gather attention-DP responses once per forward pass | wideep-ft-next |
| 4c1cb82e77 | 2026-10-16 | 0.12 | 73 | [None][perf] Revoke host communicators from the native peer-failure watch | wideep-ft-next |
| 4fc207b69f | 2026-10-16 | 0.51 | 61 | fixup! [None][feat] Bind MoE layers for recovery and route a lost rank's experts to their replicas | backup/prs-pre-squash |
| 6e0702df18 | 2026-10-16 | 0.61 | 28 | [None][fix] Clear stale step state when the overlap scheduler discards a step | wideep-ft-next |
| 77b5c2a93d | 2026-10-16 | 0.24 | 126 | [None][feat] Route a lost rank's experts to their replicas after a peer loss | wideep-ft-fix, wideep-ft-next |
| 822eb0f5a7 | 2026-10-16 | 0.52 | 485 | [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all | backup/prs-pre-squash, prs-rebase-trial, wideep-ft-fix, wideep-ft-next, wideep-ft-prs |
| 941228ada7 | 2026-10-16 | 0.71 | 42 | [None][fix] Fix MoE all-to-all issues found while enabling WideEP fault tolerance | backup/prs-pre-squash, prs-rebase-trial, wideep-ft-prs |
| a2669e0880 | 2026-10-16 | 0.57 | 1076 | [None][feat] Bind MoE layers to the all-to-all execution descriptor for recovery | wideep-ft-fix, wideep-ft-next |
| a470d307bb | 2026-10-16 | 0.18 | 164 | [None][perf] Decide the attention-DP CUDA graph lookup from the padding vote | wideep-ft-next |
| ac649fae5f | 2026-10-16 | 0.75 | 64 | [None][fix] Make host votes say how a departed rank counts | wideep-ft-fix, wideep-ft-next, wideep-ft-prs |
| b1d7fa789d | 2026-10-16 | 0.08 | 13 | [None][fix] Use the steady-state all-to-all timeout in captured CUDA graphs | wideep-ft-fix, wideep-ft-next |
| cec575cb70 | 2026-10-16 | 0.50 | 4 | fixup! [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all | backup/prs-pre-squash |
| d8e49882b6 | 2026-10-16 | 0.14 | 7 | fixup! [None][feat] Add ULFM survivor communicators for host collectives | backup/prs-pre-squash |
| df2bf6cca7 | 2026-10-16 | 0.00 | 1 | fixup! [None][fix] Make host votes say how a departed rank counts | wideep-ft-next |
| 3c720e6c56 | 2026-10-18 | 0.40 | 5 | fixup! [None][feat] Configure the WideEP failure domain and find nodes by host name | fix-cq4, fix-integ, fix-nr10, fix-nr2, fix-nr58 +5 |
| 45a5272dd3 | 2026-10-18 | 0.79 | 82 | fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable | fix-cq4, fix-integ, fix-nr10, fix-nr2, fix-nr58 +5 |
| 4cf60b059d | 2026-10-18 | 0.88 | 34 | fixup! [None][feat] Recover a WideEP deployment from further losses | fix-fence-before-propose, fix-ft-dispatch, fix-hang-deadline, fix-resp-cost, next-build +1 |
| 7afa422c67 | 2026-10-18 | 0.38 | 24 | fixup! [None][feat] Read WideEP fault-tolerance settings from one validated config | fix-cq4, fix-integ, fix-nr10, fix-nr2, fix-nr58 +6 |
| ad0c5cac02 | 2026-10-18 | 0.80 | 74 | fixup! [None][fix] End trtllm-llmapi-launch by its worker's signal | fix-cq4, fix-integ, fix-nr10, fix-nr2, fix-nr7 +3 |
| be1038e947 | 2026-10-18 | 0.64 | 11 | fixup! [None][feat] Confirm a hung WideEP rank in the recovery authority | fix-cq4, fix-integ, fix-nr10, fix-nr2, fix-nr58 +5 |

### G7 single-feature and experiment branches in A and U (10-17..10-20) - 28 subjects

| commit | date | cov | +lines | subject | refs |
|---|---|---|---|---|---|
| 47b33f8758 | 2026-10-17 | 0.14 | 63 | [None][perf] compact WideEP receive-slot routing one token row per thread | trim-vector-47b33f8758 |
| 3953177de9 | 2026-10-18 | 0.88 | 34 | [None][fix] Re-run a lost WideEP rank's pinned requests on a survivor | fix-rerun-pin |
| 606399ab52 | 2026-10-18 | 0.08 | 143 | fixup! [None][feat] Add a VMM allocator whose empty EPLB slots hold no device memory | fix-borrow-map |
| b7cfb40656 | 2026-10-18 | 0.20 | 83 | [None][chore] Share WideEP recovery helpers with the executor | fix-reuse |
| 03b60d4c85 | 2026-10-19 | 0.08 | 481 | [None][feat] Continue a lost WideEP rank's streaming requests on a survivor | feat-all, feat-all2, feat-all3, feat-phaseb |
| 539d3c0d9f | 2026-10-19 | 0.29 | 48 | [None][feat] Test that a lost WideEP rank's stream continues with its exact greedy output | feat-all, feat-all2, feat-all3, feat-phaseb |
| 7dcc38fbd2 | 2026-10-19 | 0.02 | 223 | [None][chore] Move the WideEP stream continuation into its own module | feat-all2, feat-all3, feat-phaseb |
| 8b2061cf4b | 2026-10-19 | 0.28 | 345 | [None][perf] Read the WideEP routing in its receive slots instead of compacting it in the dispatch | fix-sort-gate |
| 908128f254 | 2026-10-19 | 0.04 | 97 | [None][feat] Rebuild a lost WideEP stream's tokens from rank 0's kept gathers | feat-all, feat-all2, feat-all3, feat-phaseb |
| e3c2cec484 | 2026-10-19 | 0.14 | 184 | [None][feat] Sort top-K ids held in fixed per-source slots in moe_sort | fix-sort-gate |
| f0a70fe5bf | 2026-10-19 | 0.00 | 69 | [None][chore] Trim the WideEP stream continuation module | feat-all2, feat-all3, feat-phaseb |
| f24dc807c9 | 2026-10-19 | 0.00 | 2 | [None][fix] Copy a WideEP stream continuation's time budget only when the request has one | feat-all3, feat-phaseb |
| 174631c1ef | 2026-10-20 | 0.02 | 201 | [None][test] Check that Renormalize routes the same on the 384- and 512-expert tiers, and time both | fix-final3-tier |
| 19c024b18e | 2026-10-20 | 0.07 | 14 | [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch | fix-ci, fix-cost, fix-host, wt:pc-fix-host, wt:pc-fix-kern +1 |
| 3a663d7150 | 2026-10-20 | 0.25 | 4 | [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the  | fix-ci, fix-cost, fix-kern, wt:pc-fix-kern, wt:pc-opt2e |
| 47bb38bda8 | 2026-10-20 | 0.28 | 18 | [None][test] Diagnostic: pin EPLB host copies in GPU-memcpy mode | diag-pin |
| 5361965826 | 2026-10-20 | 0.19 | 48 | [None][fix] Put ghost slots on host-accessible memory only for online EPLB | fix-cost-opt2 |
| 5c188e0021 | 2026-10-20 | 0.15 | 41 | [None][perf] Check for a failed peer after each host collective in one native call without the GIL | fix-getfailed |
| 62bb00ba01 | 2026-10-20 | 0.25 | 60 | [None][fix] Keep ghost slots in device memory, loaded by GPU copies, where GPU memory is not a NUMA node | fix-cost-opt2 |
| 71513633b8 | 2026-10-20 | 0.19 | 68 | [None][perf] Back ghost slots with memory on the GPU's NUMA node, so restores copy into them from the CPU | fix-cost-opt2, fix-exit, fix-ring |
| 7e230b3a21 | 2026-10-20 | 0.12 | 26 | [None][perf] Copy a staging chunk's received expert pieces into the slots in one batch | fix-staging-batch |
| 80085dc184 | 2026-10-20 | 0.00 | 9 | [None][fix] Name the register-word rank-mask check apart from the pointer overload | fix-ci, fix-cost, fix-cost-opt2, fix-docs2, fix-exit +6 |
| 814a2cad73 | 2026-10-20 | 0.00 | 7 | [None][perf] Add a 384-expert tier to the Renormalize routing policy | fix-final3-tier, fix-static2 |
| 85719fc28d | 2026-10-20 | 0.09 | 92 | [None][perf] Keep EPLB's resident experts in their slots | fix-sticky |
| 88e5302aac | 2026-10-20 | 0.00 | 4 | [None][perf] Add the 384-expert tier to the Renormalize 512-thread cluster routing kernel | fix-final3-tier |
| 9d7b33ecb8 | 2026-10-20 | 0.26 | 53 | [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait | fix-ci, fix-cost, fix-cost-opt2, fix-docs2, fix-exit +7 |
| d4c8e0a3fd | 2026-10-20 | 0.12 | 26 | [None][fix] Default online EPLB's appended extra slots to empty; keep ghost opt-in | fix-final4 |
| da13f20b76 | 2026-10-20 | 0.04 | 77 | [None][chore] DIAGNOSTIC: log per-iteration EPLB update timing under TLLM_EPLB_UPDATE_LOG=1 | fix-diag-eplb |


## Cleanup commands (NOT run; for approval)

Set once:
```bash
A=/tmp/claude-161238/work/repo
U=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM
```
`git worktree remove` without `--force` refuses a worktree with uncommitted or untracked changes; that refusal is the safety check (worktree dirtiness was not inspected, since `git status` is outside the allowed read set). Note: `git branch -D` drops branches U's running jobs might reference by name; jobs run from /tmp copies, but check for a live job on a branch before deleting it.

### Tier 1: SUPERSEDED and SUPERSEDED~ (in final by patch-id or by an amended same-subject commit)

```bash
git -C $A worktree remove /tmp/claude-161238/work/wt-docs
git -C $A branch -D docs-ft   # SUPERSEDED~ 758cba7957
git -C $A worktree remove /tmp/claude-161238/work/wt-mixed
git -C $A branch -D feat-mixed   # SUPERSEDED~ e55961c983
git -C $A worktree remove /tmp/claude-161238/work/wt-sharded
git -C $A branch -D feat-sharded   # SUPERSEDED~ 643ff79f1b
git -C $A worktree remove /tmp/claude-161238/work/wt-final
git -C $A branch -D final-design   # SUPERSEDED~ 3091edee7e
git -C $A worktree remove /tmp/claude-161238/work/wt-final2
git -C $A branch -D final-design2   # SUPERSEDED~ 98632eb787
git -C $A worktree remove /tmp/claude-161238/work/wt-final3
git -C $A branch -D final-design3   # SUPERSEDED~ 6e8657d0be
git -C $A worktree remove /tmp/claude-161238/work/wt-abortptr
git -C $A branch -D fix-abortptr   # SUPERSEDED~ 425bf9b44f
git -C $A worktree remove /tmp/claude-161238/work/wt-conc
git -C $A branch -D fix-conc   # SUPERSEDED~ 3dc77e11cc
git -C $A worktree remove /tmp/claude-161238/work/wt-correct
git -C $A branch -D fix-correctness   # SUPERSEDED 64afc01969
git -C $A worktree remove /tmp/claude-161238/work/wt-fc
git -C $A branch -D fix-fc   # SUPERSEDED~ 5f29b0266b
git -C $A branch -D fix-final2   # SUPERSEDED~ aae00e6c65
git -C $A worktree remove /tmp/claude-161238/work/wt-final3b
git -C $A branch -D fix-final3   # SUPERSEDED~ 547a6aa7ed
git -C $A worktree remove /tmp/claude-161238/work/wt-rebase
git -C $A branch -D fix-final3-rebased   # SUPERSEDED 08ba507229
git -C $A worktree remove /tmp/claude-161238/work/wt-final5
git -C $A branch -D fix-final5   # SUPERSEDED 24e2320d34
git -C $A worktree remove /tmp/claude-161238/work/wt-final6
git -C $A branch -D fix-final6   # SUPERSEDED 2f635c962d
git -C $A branch -D fix-final7   # SUPERSEDED 2c3af30ce8
git -C $A worktree remove /tmp/claude-161238/work/wt-parity
git -C $A branch -D fix-ftoff-parity   # SUPERSEDED~ 547a6aa7ed
git -C $A worktree remove /tmp/claude-161238/work/wt-ghostgtest
git -C $A branch -D fix-ghost-gtest   # SUPERSEDED~ 6893f9b2e3
git -C $A branch -D fix-ghost-tune   # SUPERSEDED~ 52eac771d8
git -C $A worktree remove /tmp/claude-161238/work/wt-ghosttune
git -C $A branch -D fix-ghost-tune2   # SUPERSEDED~ 859be41acf
git -C $A worktree remove /tmp/claude-161238/work/wt-glog
git -C $A branch -D fix-ghostmap-log   # SUPERSEDED 9642d022fb
git -C $A worktree remove /tmp/claude-161238/work/wt-maint
git -C $A branch -D fix-maint   # SUPERSEDED~ f6bc16be99
git -C $A worktree remove /tmp/claude-161238/work/wt-mtpinplace
git -C $A branch -D fix-mtp-inplace   # SUPERSEDED~ 57f29976c5
git -C $A worktree remove /tmp/claude-161238/work/wt-offline
git -C $A branch -D fix-offline-eplb   # SUPERSEDED~ 1daac46af2
git -C $A worktree remove /tmp/claude-161238/work/wt-sort
git -C $A branch -D fix-onerep-sort   # SUPERSEDED 2c3af30ce8
git -C $A worktree remove /tmp/claude-161238/work/wt-ops
git -C $A branch -D fix-ops   # SUPERSEDED~ c928c58b49
git -C $A worktree remove /tmp/claude-161238/work/wt-opt2e
git -C $A branch -D fix-opt2e   # SUPERSEDED~ a6ae3053fd
git -C $A worktree remove /tmp/claude-161238/work/wt-p6
git -C $A branch -D fix-p6-split   # SUPERSEDED c3cbb0ae45
git -C $A worktree remove /tmp/claude-161238/work/wt-parity-cpp
git -C $A branch -D fix-parity-cpp   # SUPERSEDED cfbdb88fbb
git -C $A worktree remove /tmp/claude-161238/work/wt-parity-py
git -C $A branch -D fix-parity-py   # SUPERSEDED 54e9787b22
git -C $A worktree remove /tmp/claude-161238/work/wt-perf
git -C $A branch -D fix-perf   # SUPERSEDED~ aa120be555
git -C $A worktree remove /tmp/claude-161238/work/wt-place
git -C $A branch -D fix-place   # SUPERSEDED~ b988cc822e
git -C $A worktree remove /tmp/claude-161238/work/wt-r3port
git -C $A branch -D fix-r3-port   # SUPERSEDED e31146463e
git -C $A worktree remove /tmp/claude-161238/work/wt-req
git -C $A branch -D fix-req   # SUPERSEDED~ 9169400b0e
git -C $A worktree remove /tmp/claude-161238/work/wt-review
git -C $A branch -D fix-review   # SUPERSEDED~ fe35d48f8c
git -C $A worktree remove /tmp/claude-161238/work/wt-routeskip
git -C $A branch -D fix-route-skip   # SUPERSEDED~ 02e0d45e3b
git -C $A worktree remove /tmp/claude-161238/work/wt-sec
git -C $A branch -D fix-sec   # SUPERSEDED~ 8522b24830
git -C $A worktree remove /tmp/claude-161238/work/wt-slim
git -C $A branch -D fix-slim   # SUPERSEDED~ 36fe8b4eb2
git -C $A worktree remove /tmp/claude-161238/work/wt-tests
git -C $A branch -D fix-tests   # SUPERSEDED~ fa5265ff27
git -C $A worktree remove /tmp/claude-161238/work/wt-tunekey
git -C $A branch -D fix-tunekey   # SUPERSEDED~ 04e2f68808
git -C $A worktree remove /tmp/claude-161238/work/wt-ulfm
git -C $A branch -D fix-ulfm   # SUPERSEDED~ f387d3c6ca
git -C $A worktree remove /tmp/claude-161238/work/wt-exit0
git -C $A branch -D fold-exit0   # SUPERSEDED~ 45c7e0cb3b
git -C $A worktree remove /tmp/claude-161238/work/wt-hangev
git -C $A branch -D fold-hangev   # SUPERSEDED~ 191988d1fb
git -C $A worktree remove /tmp/claude-161238/work/wt-health
git -C $A branch -D fold-health   # SUPERSEDED~ f10bee1a0f
git -C $A worktree remove /tmp/claude-161238/work/wt-restore
git -C $A branch -D fold-restore   # SUPERSEDED~ 1af6bdc9d5
git -C $A branch -D pr-final/01   # SUPERSEDED~ b9089d1264
git -C $A branch -D pr-final/02   # SUPERSEDED~ 9aabe958bb
git -C $A branch -D pr-final/03   # SUPERSEDED~ 5c7c832543
git -C $A branch -D pr-final/04   # SUPERSEDED~ 321ebd33b4
git -C $A branch -D pr-final/05   # SUPERSEDED~ 5280568012
git -C $A branch -D pr-final/06   # SUPERSEDED~ 35325d44e3
```
```bash
git -C $U branch -D wideep-ft-consolidated   # SUPERSEDED~ 191988d1fb
git -C $U branch -D wideep-ft-main   # SUPERSEDED~ 3e9027f4e4
git -C $U branch -D wideep-ft-node   # SUPERSEDED~ 10b2849569
git -C $U branch -D wideep-ft-phase1   # SUPERSEDED~ 3e9027f4e4
git -C $U branch -D wideep-ft-restore   # SUPERSEDED~ 960ab25207
git -C $U branch -D wideep-ft-restore2   # SUPERSEDED~ 2edca79d4c
git -C $U branch -D wideep-ft-trim   # SUPERSEDED~ 01ec646c40
```
### Tier 2: UNIQUE-covered (likely safe; non-final commits are >=90% in final or kept on a FINAL/LONG branch)
Spot-check one or two in the decision tables first if desired.

```bash
git -C $A branch -D ftpr/01-stream-error-events   # UNIQUE-covered 7bbbac2a34
git -C $A branch -D pr-final/07   # UNIQUE-covered 35dcbc7aff
git -C $A branch -D pr-final/08   # UNIQUE-covered 6e609bf41c
git -C $A branch -D pr-final/09   # UNIQUE-covered 9b693d2af2
```
```bash
git -C $U branch -D wideep-ft-eplb   # UNIQUE-covered 4e3b47d51b
git -C $U branch -D wideep-ft-hang   # UNIQUE-covered 3c93d34184
git -C $U branch -D wideep-ft-health   # UNIQUE-covered df13da0f74
git -C $U branch -D wideep-ft-main-pre-final-20261017   # UNIQUE-covered 69763cd9e8
git -C $U branch -D wideep-ft-mtp   # UNIQUE-covered 3430226df5
```
Detached worktrees whose HEAD is the tip of, or contained in, a kept or Tier 1/2 branch (no commits lost):
```bash
git -C $A worktree remove /tmp/claude-161238/prmain.wZps/wt
git -C $A worktree remove /tmp/claude-161238/work/wt-pr10-1
git -C $A worktree remove /tmp/claude-161238/work/wt-pr10-2
git -C $A worktree remove /tmp/claude-161238/work/wt-pr10-3
```
Detached worktrees on pr-stack-final / pr-final/24z / pr-final/25z commits (UNIQUE, Tier 3); remove together with those branches:
```bash
git -C $A worktree remove /tmp/claude-161238/work/wt-psf-check0
git -C $A worktree remove /tmp/claude-161238/work/wt-psf-check1
git -C $A worktree remove /tmp/claude-161238/work/wt-psf-check2
git -C $A worktree remove /tmp/claude-161238/work/wt-psf-check3
```

### Tier 3: UNIQUE (ONLY after deciding on the commits listed above)
Each branch has at least one commit with under 90% of its lines in final and no same-subject commit on any kept branch.

```bash
git -C $A worktree remove /tmp/claude-161238/work/wt-pin
git -C $A branch -D diag-pin   # UNIQUE 47bb38bda8
git -C $A worktree remove /tmp/claude-161238/work/wt-featall
git -C $A branch -D feat-all   # UNIQUE 2077a4cf31
git -C $A worktree remove /tmp/claude-161238/work/wt-featall2
git -C $A branch -D feat-all2   # UNIQUE 39b3cd2c29
git -C $A worktree remove /tmp/claude-161238/work/wt-featall3
git -C $A branch -D feat-all3   # UNIQUE 9c618db9e2
git -C $A worktree remove /tmp/claude-161238/work/wt-phaseb
git -C $A branch -D feat-phaseb   # UNIQUE 1d260bb46f
git -C $A worktree remove /tmp/claude-161238/work/wt-ci
git -C $A branch -D fix-ci   # UNIQUE 37c4877fe9
git -C $A worktree remove /tmp/claude-161238/work/wt-cost
git -C $A branch -D fix-cost   # UNIQUE 19c024b18e
git -C $A branch -D fix-cost-opt2   # UNIQUE c01f782ed2
git -C $A worktree remove /tmp/claude-161238/work/wt-diag
git -C $A branch -D fix-diag-eplb   # UNIQUE da13f20b76
git -C $A worktree remove /tmp/claude-161238/work/wt-docs2
git -C $A branch -D fix-docs2   # UNIQUE 7e9ef22306
git -C $A worktree remove /tmp/claude-161238/work/wt-exit
git -C $A branch -D fix-exit   # UNIQUE 6b4a8dc18f
git -C $A branch -D fix-final3-tier   # UNIQUE 88e5302aac
git -C $A worktree remove /tmp/claude-161238/work/wt-final4
git -C $A branch -D fix-final4   # UNIQUE d4c8e0a3fd
git -C $A worktree remove /tmp/claude-161238/work/wt-gf
git -C $A branch -D fix-getfailed   # UNIQUE 5c188e0021
git -C $A worktree remove /tmp/claude-161238/work/wt-host
git -C $A branch -D fix-host   # UNIQUE c372c53358
git -C $A worktree remove /tmp/claude-161238/work/wt-kern
git -C $A branch -D fix-kern   # UNIQUE 3358ea468e
git -C $A worktree remove /tmp/claude-161238/work/wt-ktests
git -C $A branch -D fix-ktests   # UNIQUE c8768712f9
git -C $A worktree remove /tmp/claude-161238/work/wt-ring
git -C $A branch -D fix-ring   # UNIQUE e0ddea0a8e
git -C $A worktree remove /tmp/claude-161238/work/wt-staging
git -C $A branch -D fix-staging-batch   # UNIQUE 7e230b3a21
git -C $A worktree remove /tmp/claude-161238/work/wt-static2
git -C $A branch -D fix-static2   # UNIQUE 643c8afdf3
git -C $A worktree remove /tmp/claude-161238/work/wt-sticky
git -C $A branch -D fix-sticky   # UNIQUE 85719fc28d
git -C $A branch -D ftpr/02-eplb-shm-names   # UNIQUE ada67c31bb
git -C $A branch -D ftpr/03-eplb-repacked-slots   # UNIQUE ccff070544
git -C $A branch -D ftpr/04-moe-fake-ops-shared-health   # UNIQUE fccdcaa55a
git -C $A branch -D ftpr/05-autotune-active-slots   # UNIQUE 392b11d981
git -C $A branch -D ftpr/06-eplb-load-publish   # UNIQUE bdacc7a3bd
git -C $A branch -D ftpr/07-eplb-unusable-slots   # UNIQUE 5773fddda9
git -C $A branch -D ftpr/08-kv-v2-step-rollback   # UNIQUE c938db5bb7
git -C $A branch -D ftpr/09-request-output-checkpoint   # UNIQUE 04b17589ba
git -C $A branch -D ftpr/10-mpi4py-4.1.2   # UNIQUE 197620fe38
git -C $A branch -D ftpr/11-ft-a2a-rank-mask-abort   # UNIQUE 3ab60a3044
git -C $A branch -D ftpr/12-ft-moe-receive-slots   # UNIQUE d78eeb66e3
git -C $A branch -D ftpr/13-ft-ulfm-survivors   # UNIQUE c2bd7232de
git -C $A branch -D ftpr/14-ft-recovery-authority   # UNIQUE e4cd65c471
git -C $A branch -D ftpr/15-ft-eplb-restore   # UNIQUE cd859e2523
git -C $A branch -D ftpr/16-ft-adp-absent-ranks   # UNIQUE 2374f2261f
git -C $A branch -D ftpr/17-ft-executor-recovery   # UNIQUE 154efa731c
git -C $A branch -D ftpr/18-ft-ghost-slots   # UNIQUE c1203e32a5
git -C $A branch -D ftpr/19-ft-ring-host-copies   # UNIQUE 97a0f3695f
git -C $A branch -D ftpr/20-ft-hang-detection   # UNIQUE 903bbcb6b0
git -C $A branch -D ftpr/21-ft-health-status   # UNIQUE ea6e1c22cb
git -C $A worktree remove /tmp/claude-161238/work/wt-ftpr
git -C $A branch -D ftpr/22-ft-route-one-slot   # UNIQUE 2632fe6fdf
git -C $A worktree remove /tmp/claude-161238/work/wt-ftpr-fix
git -C $A branch -D ftpr/23-review-fixes   # UNIQUE 8ab8e7c0b2
git -C $A branch -D pr-final/10   # UNIQUE 6fd0c1913c
git -C $A branch -D pr-final/11   # UNIQUE 8929b3825c
git -C $A branch -D pr-final/11z   # UNIQUE ac54a97d9e
git -C $A branch -D pr-final/12   # UNIQUE 4d445f3976
git -C $A branch -D pr-final/13   # UNIQUE 53e6057bc3
git -C $A branch -D pr-final/13b   # UNIQUE be079f7da2
git -C $A branch -D pr-final/14   # UNIQUE 4e81ee8766
git -C $A branch -D pr-final/14b   # UNIQUE dfe128de22
git -C $A branch -D pr-final/15   # UNIQUE fdc322efad
git -C $A branch -D pr-final/15b   # UNIQUE 3682b0d00b
git -C $A branch -D pr-final/16   # UNIQUE 07cf5045c2
git -C $A branch -D pr-final/16b   # UNIQUE 24c1f180d2
git -C $A branch -D pr-final/17   # UNIQUE 126a65b01c
git -C $A branch -D pr-final/17b   # UNIQUE c290549c63
git -C $A branch -D pr-final/17c   # UNIQUE c9c1b0db25
git -C $A branch -D pr-final/17z   # UNIQUE 7e73056291
git -C $A branch -D pr-final/18   # UNIQUE 3b57603063
git -C $A branch -D pr-final/19   # UNIQUE 3e2ba7cc57
git -C $A branch -D pr-final/19b   # UNIQUE 6c5f77a2ad
git -C $A branch -D pr-final/19c   # UNIQUE dcf8e58820
git -C $A branch -D pr-final/19z1   # UNIQUE 6557b05345
git -C $A branch -D pr-final/19z2   # UNIQUE 256d145836
git -C $A branch -D pr-final/20a   # UNIQUE e11b6ff556
git -C $A branch -D pr-final/20b   # UNIQUE 2f12cfd469
git -C $A branch -D pr-final/20z   # UNIQUE 68fe820a7b
git -C $A branch -D pr-final/21a   # UNIQUE 41789c5844
git -C $A branch -D pr-final/21b   # UNIQUE aeef8e85f9
git -C $A branch -D pr-final/21c   # UNIQUE f918195264
git -C $A branch -D pr-final/22   # UNIQUE 0d70cd81c2
git -C $A branch -D pr-final/22c   # UNIQUE 75d1a1c508
git -C $A branch -D pr-final/22z   # UNIQUE 369820ca8f
git -C $A branch -D pr-final/23a   # UNIQUE 312adca97a
git -C $A branch -D pr-final/23b_22b   # UNIQUE 87919f1bb0
git -C $A branch -D pr-final/23c   # UNIQUE 7d699a4e60
git -C $A branch -D pr-final/24a   # UNIQUE 448472d070
git -C $A branch -D pr-final/24b   # UNIQUE e65441fffd
git -C $A branch -D pr-final/24c   # UNIQUE 01b0f7d57d
git -C $A branch -D pr-final/24d   # UNIQUE 74d8b0ea3b
git -C $A branch -D pr-final/24e   # UNIQUE 74cd03dd49
git -C $A branch -D pr-final/24f   # UNIQUE 1e3a6eb0da
git -C $A branch -D pr-final/24z   # UNIQUE 0e4f56731c
git -C $A branch -D pr-final/25a   # UNIQUE cad5c8fa9e
git -C $A branch -D pr-final/25b   # UNIQUE 31b160beb4
git -C $A branch -D pr-final/25c   # UNIQUE 4653856251
git -C $A branch -D pr-final/25z   # UNIQUE 68f5562af0
git -C $A branch -D pr-final/26   # UNIQUE c713d0de68
git -C $A branch -D pr-final/26z1   # UNIQUE 7d4edc8b4d
git -C $A branch -D pr-final/26z2   # UNIQUE d83a2e43db
git -C $A branch -D pr-final/27a   # UNIQUE e47a0546d7
git -C $A branch -D pr-final/27b   # UNIQUE cf4bf1e6d4
git -C $A branch -D pr-final/27z1   # UNIQUE 4929fe3c7c
git -C $A branch -D pr-final/27z2   # UNIQUE 98499e34b5
git -C $A branch -D pr-final/28   # UNIQUE 11f61041f0
git -C $A worktree remove /tmp/claude-161238/work/wt-prstack
git -C $A branch -D pr-stack   # UNIQUE 787ed7c59b
git -C $A worktree remove /tmp/claude-161238/work/wt-prstack-final
git -C $A branch -D pr-stack-final   # UNIQUE 68f5562af0
```
```bash
git -C $U branch -D backup/pre-fold-c168c39961   # UNIQUE c168c39961
git -C $U branch -D backup/pre-fold-c706fdfa84   # UNIQUE c706fdfa84
git -C $U branch -D backup/prs-pre-squash   # UNIQUE 4474b35e00
git -C $U branch -D fix-borrow-map   # UNIQUE 606399ab52
git -C $U branch -D fix-resp-cost   # UNIQUE 1fec69ae64
git -C $U worktree remove /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/reuse
git -C $U branch -D fix-reuse   # UNIQUE b7cfb40656
git -C $U worktree remove /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/sortgate
git -C $U branch -D fix-sort-gate   # UNIQUE 4f0c308b02
git -C $U branch -D prs-rebase-trial   # UNIQUE d9777a8612
git -C $U branch -D trim-vector-47b33f8758   # UNIQUE 47b33f8758
git -C $U branch -D wideep-ft-dev-pre-consolidate   # UNIQUE 79cdd9f36f
git -C $U branch -D wideep-ft-dev-prefold   # UNIQUE 9ad3d1f8e0
git -C $U branch -D wideep-ft-exploration   # UNIQUE 474d437ad4
git -C $U branch -D wideep-ft-exploration-next   # UNIQUE d9084befa9
git -C $U branch -D wideep-ft-fix   # UNIQUE 625c5e1759
git -C $U branch -D wideep-ft-fix-pre-gather   # UNIQUE b474200595
git -C $U branch -D wideep-ft-next   # UNIQUE a958187bfc
git -C $U branch -D wideep-ft-prs   # UNIQUE 1707fac125
```
Detached worktrees whose HEAD is on **no branch** (removing them orphans the commit; it is then recoverable only by SHA until gc):
```bash
git -C $A worktree remove /tmp/claude-161238/pc-fix-kern   # 19ea00bab0, unreachable from any branch
git -C $A worktree remove /tmp/claude-161238/pc-opt2e      # a7dd7c0d7f, unreachable from any branch
git -C $A worktree remove /tmp/claude-161238/work/wt-psf-check  # 9ae82f698c, unreachable from any branch
git -C $A worktree remove /tmp/claude-161238/pc-fix-host   # c372c53358 = fix-host tip
```
Stash (G1: 17 subjects, 5 files, +432/-219 on tests/integration/defs/wide_ep_ft and friends, based on wideep-ft dd99c07e00 from 2026-10-07):
```bash
git -C $U stash show -p stash@{0}   # review first
git -C $U stash drop stash@{0}
```

### Other ref cleanup
```bash
# A: drop 31 stale origin/* tracking refs whose branches U no longer has (and refresh origin/fix-sort-gate)
git -C $A fetch --prune origin
# U: external PR snapshots (other authors' PRs; re-fetchable from upstream)
git -C $U update-ref -d refs/remotes/pr/15785
git -C $U update-ref -d refs/remotes/pr/15788
git -C $U update-ref -d refs/remotes/pr/15789
git -C $U update-ref -d refs/remotes/pr/15895
git -C $U update-ref -d refs/remotes/pr/19610
```

### Fork branch deletion (SUGGESTION ONLY; remote state unverified because ls-remote failed)
Keep: fork/main, fork/wideep-ft, fork/wideep-ft-all, fork/wideep-ft-dev.
```bash
git -C $U push fork --delete wideep-ft-eplb   # UNIQUE-covered 4e3b47d51b
git -C $U push fork --delete wideep-ft-exploration   # UNIQUE 479d51c512
git -C $U push fork --delete wideep-ft-hang   # UNIQUE-covered 3c93d34184
git -C $U push fork --delete wideep-ft-health   # UNIQUE-covered df13da0f74
git -C $U push fork --delete wideep-ft-main   # SUPERSEDED~ 3e9027f4e4
git -C $U push fork --delete wideep-ft-mtp   # UNIQUE-covered 3430226df5
git -C $U push fork --delete wideep-ft-node   # SUPERSEDED~ 10b2849569
git -C $U push fork --delete wideep-ft-phase1   # SUPERSEDED~ 3e9027f4e4
git -C $U push fork --delete wideep-ft-restore   # SUPERSEDED~ 960ab25207
git -C $U push fork --delete wideep-ft-restore2   # SUPERSEDED~ 2edca79d4c
git -C $U push fork --delete wideep-ft-trim   # SUPERSEDED~ 01ec646c40
```
