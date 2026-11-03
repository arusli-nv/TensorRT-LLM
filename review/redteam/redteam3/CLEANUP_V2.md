# Branch and worktree cleanup against v2 (read-only audit)

Generated Wed Oct 21 06:06:47 PDT 2026. Reference: **v2 @ cef4670789** (302 non-merge commits on upstream/main 0829689c52), repo A = /tmp/claude-161238/work/repo, repo U = /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM.

## Method (same as redteam2/hygiene.md, reference switched to v2)

1. Per ref: non-merge commits not on main (`upstream/main` in A, `origin/main` fc52f0adee in U, an ancestor of A's).
2. A commit is in the reference if it is an ancestor of v2 or in the ftcore/*+ftwave2/*+prmain/*+prmain-all stack, or `git patch-id --stable`-equal to one there (U-only commits hashed in U).
3. Otherwise: same subject in the reference = amended; same subject in any kept branch (fix-final8-lean, ft-final2, wideep-ft*, fork/wideep-ft*, origin/wideep-ft*) = covered; else covered if >=90% of its non-trivial added lines appear verbatim in the v2 version of the file.
4. **SUPERSEDED**: every commit in the reference. **SUPERSEDED~**: the rest are amended (same subject in v2/stack/prmain). **UNIQUE-covered**: the rest are covered. **UNIQUE**: at least one commit is none of these.
5. Kept set (not graded): v2, fix-final8-lean, ft-final2, ftcore/*, ftwave2/*, prmain/*, prmain-all, wideep-ft-dev (A and U), wideep-ft-all, wideep-ft-pending, wideep-ft, main/master, and the worktree /tmp/claude-161238/wt-v2 (the v2 worktree). Detached worktrees are graded by their HEAD.

In-use worktrees: the sandbox cannot see other sessions' processes, so "in use" could not be checked directly. Kept-branch worktrees are excluded. Commands for worktrees whose index was written after 2026-10-20 22:00 PT carry a `# recent:` comment.

## Counts

| repo | KEEP | SUPERSEDED | SUPERSEDED~ | UNIQUE-covered | UNIQUE | total graded |
|---|---|---|---|---|---|---|
| A | 23 | 55 | 52 | 13 | 129 | 249 |
| U | 4 | 0 | 7 | 5 | 17 | 29 |

Rows are local branches plus detached worktrees (shown as `WT:<path>`). A has 239 local branches and 120 worktrees, U has 33 and 4.

## Sample check with `git cherry`

| ref | command | result |
|---|---|---|
| fix-final8 | `git cherry v2 fix-final8` | no output (all 275 commits are ancestors of v2) |
| fix-parity-cpp | `git cherry v2 fix-parity-cpp` | 11 `-`, 0 `+` |
| rt2-trial | `git cherry v2 rt2-trial` | 14 `-`, 0 `+` |
| v2-cpp | `git cherry v2 v2-cpp` | 5 `-`, 0 `+` |
| fix-correctness | `git cherry v2 fix-correctness` | 8 `-`, 0 `+` |
| ftf2/01-ft-config | `git cherry ftwave2/09-ghost-slots-ring-copies ftf2/01-ft-config` | 1 `-`, 0 `+` (against v2 alone it shows 8 `+`: 7 prmain commits plus 1, because the prmain fixes sit under the stack, not in v2) |

All six agree with the script.

## Commands for the user (SUPERSEDED and SUPERSEDED~ only)

### SUPERSEDED

#### Repo A (55)

```bash
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-correct
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-rebase
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final5
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final6
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final8
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final7
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-glog
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-sort
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-p6
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-parity-cpp
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-parity-py
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-r3port
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-rt2c
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-rt2pin
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/wt-v2cpp  # recent: index written 10-20 23:15
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/wt-v2py  # recent: index written 10-20 23:50
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/prmain.wZps/wt
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-E  # DIRTY: check first (76 changed paths); recent: index written 10-20 23:31
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-I  # DIRTY: check first (112 changed paths); recent: index written 10-21 00:38
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-K  # DIRTY: check first (76 changed paths); recent: index written 10-20 23:27
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R  # DIRTY: check first (115 changed paths); recent: index written 10-20 22:09
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-c01  # DIRTY: check first (7 changed paths); recent: index written 10-20 23:02
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-c02  # DIRTY: check first (22 changed paths); recent: index written 10-20 23:32
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-c04  # DIRTY: check first (49 changed paths); recent: index written 10-20 23:32
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-c05  # DIRTY: check first (54 changed paths); recent: index written 10-20 23:32
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-w08  # DIRTY: check first (73 changed paths); recent: index written 10-20 23:32
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-R-w09  # DIRTY: check first (80 changed paths); recent: index written 10-20 23:32
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-asm-X  # DIRTY: check first (112 changed paths); recent: index written 10-20 23:55
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-pm
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-src2  # recent: index written 10-20 22:53
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/restack/wt-v2  # recent: index written 10-21 00:19
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/repo/restack/wt-asm-R-c01  # recent: index written 10-20 23:01
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/repo/restack/wt-asm-R-c04  # recent: index written 10-20 23:01
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/repo/restack/wt-asm-R-c05  # recent: index written 10-20 23:01
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/repo/restack/wt-asm-R-w08  # recent: index written 10-20 23:01
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/repo/restack/wt-asm-R-w09  # recent: index written 10-20 23:01
git -C /tmp/claude-161238/work/repo branch -D fix-correctness  # 64afc01969, 215 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final3-rebased  # 08ba507229, 207 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final5  # 24e2320d34, 225 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final6  # 2f635c962d, 236 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final7  # 2c3af30ce8, 238 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final8  # 4de25459ce, 275 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final8-rc1  # 78836d1723, 240 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ghostmap-log  # 9642d022fb, 239 commits
git -C /tmp/claude-161238/work/repo branch -D fix-onerep-sort  # 2c3af30ce8, 238 commits
git -C /tmp/claude-161238/work/repo branch -D fix-p6-split  # c3cbb0ae45, 239 commits
git -C /tmp/claude-161238/work/repo branch -D fix-parity-cpp  # cfbdb88fbb, 218 commits
git -C /tmp/claude-161238/work/repo branch -D fix-parity-py  # 54e9787b22, 215 commits
git -C /tmp/claude-161238/work/repo branch -D fix-r3-port  # e31146463e, 237 commits
git -C /tmp/claude-161238/work/repo branch -D fix-recompute-discard  # 38c63f8732, 281 commits
git -C /tmp/claude-161238/work/repo branch -D ft-final  # f8b425ea0a, 283 commits
git -C /tmp/claude-161238/work/repo branch -D ft-src  # ec87a5de98, 279 commits
git -C /tmp/claude-161238/work/repo branch -D ft-src2  # 3bb877e24f, 280 commits
git -C /tmp/claude-161238/work/repo branch -D ft-src3  # c3601e76c0, 281 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/01-ft-config  # 4d4cf3a2b3, 8 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/01-ft-config  # f46f30f63f, 8 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/02-a2a-ft-kernels  # af1f012297, 9 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/03-moe-receive-slots  # 4b3c78b25d, 10 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/04-ulfm-survivors  # df059d0b5f, 11 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/05-recovery-authority  # 016fd00c19, 12 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/06-eplb-restore-empty-slots  # 8478c17e38, 13 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/07a-executor-recovery  # b8d020d5ac, 14 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/07b-executor-recovery-requests-e2e  # d24c2cbbc7, 15 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/08-hang-detection-fencing  # 1bfb5d1eba, 17 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/09-ghost-slots-ring-copies  # f1a810619e, 18 commits
git -C /tmp/claude-161238/work/repo branch -D ftv2/10-health-and-guide  # 264c357baf, 16 commits
git -C /tmp/claude-161238/work/repo branch -D rt2  # 626f39b405, 279 commits
git -C /tmp/claude-161238/work/repo branch -D rt2-pin  # c8bc678ef1, 253 commits
git -C /tmp/claude-161238/work/repo branch -D rt2-trial  # 1455b78463, 266 commits
git -C /tmp/claude-161238/work/repo branch -D v2-cpp  # c011a935f2, 292 commits
git -C /tmp/claude-161238/work/repo branch -D v2-py  # 66d54cadd7, 296 commits
```

Note: `restack/wt-v2` is a detached copy of v2; the kept v2 worktree is /tmp/claude-161238/wt-v2. The `restack/wt-asm-*` worktrees sit at prmain-all with uncommitted changes, so check them before removing.

### SUPERSEDED~ (amended)

Every commit not patch-equal has a commit with the same subject in v2 or the stack, so it was amended or rebased; delete them only if you accept that the v2 version replaces them. (fix-final2 points at the same commit as A wideep-ft-dev, fold-hangev and U wideep-ft-consolidated at U wideep-ft-dev's commit, so deleting them loses nothing.)

#### Repo A (52)

```bash
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-docs
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-mixed
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-sharded
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final2
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final3
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-abortptr
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-conc
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-fc
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-final3b
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-parity
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-ghostgtest
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-ghosttune
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-maint
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-mtpinplace
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-offline
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-ops
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-opt2e
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-perf
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-place
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-req
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-review
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-routeskip
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-sec
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-slim
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-sse  # DIRTY: check first (1 changed paths)
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-tests
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-tunekey
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-ulfm
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-exit0
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-hangev
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-health
git -C /tmp/claude-161238/work/repo worktree remove /tmp/claude-161238/work/wt-restore
git -C /tmp/claude-161238/work/repo branch -D docs-ft  # 758cba7957, 70 commits
git -C /tmp/claude-161238/work/repo branch -D feat-mixed  # e55961c983, 69 commits
git -C /tmp/claude-161238/work/repo branch -D feat-sharded  # 643ff79f1b, 75 commits
git -C /tmp/claude-161238/work/repo branch -D final-design  # 3091edee7e, 83 commits
git -C /tmp/claude-161238/work/repo branch -D final-design2  # 98632eb787, 85 commits
git -C /tmp/claude-161238/work/repo branch -D final-design3  # 6e8657d0be, 89 commits
git -C /tmp/claude-161238/work/repo branch -D fix-abortptr  # 425bf9b44f, 103 commits
git -C /tmp/claude-161238/work/repo branch -D fix-conc  # 3dc77e11cc, 124 commits
git -C /tmp/claude-161238/work/repo branch -D fix-fc  # 5f29b0266b, 105 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final2  # aae00e6c65, 163 commits
git -C /tmp/claude-161238/work/repo branch -D fix-final3  # 547a6aa7ed, 207 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ftoff-parity  # 547a6aa7ed, 207 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ghost-gtest  # 6893f9b2e3, 69 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ghost-tune  # 52eac771d8, 68 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ghost-tune2  # 859be41acf, 68 commits
git -C /tmp/claude-161238/work/repo branch -D fix-maint  # f6bc16be99, 153 commits
git -C /tmp/claude-161238/work/repo branch -D fix-mtp-inplace  # 57f29976c5, 71 commits
git -C /tmp/claude-161238/work/repo branch -D fix-offline-eplb  # 1daac46af2, 209 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ops  # c928c58b49, 118 commits
git -C /tmp/claude-161238/work/repo branch -D fix-opt2e  # a6ae3053fd, 187 commits
git -C /tmp/claude-161238/work/repo branch -D fix-perf  # aa120be555, 122 commits
git -C /tmp/claude-161238/work/repo branch -D fix-place  # b988cc822e, 120 commits
git -C /tmp/claude-161238/work/repo branch -D fix-req  # 9169400b0e, 104 commits
git -C /tmp/claude-161238/work/repo branch -D fix-review  # fe35d48f8c, 165 commits
git -C /tmp/claude-161238/work/repo branch -D fix-route-skip  # 02e0d45e3b, 70 commits
git -C /tmp/claude-161238/work/repo branch -D fix-sec  # 8522b24830, 125 commits
git -C /tmp/claude-161238/work/repo branch -D fix-slim  # 36fe8b4eb2, 205 commits
git -C /tmp/claude-161238/work/repo branch -D fix-sse-stream-errors  # a9a307a0f4, 1 commits
git -C /tmp/claude-161238/work/repo branch -D fix-tests  # fa5265ff27, 109 commits
git -C /tmp/claude-161238/work/repo branch -D fix-tunekey  # 04e2f68808, 180 commits
git -C /tmp/claude-161238/work/repo branch -D fix-ulfm  # f387d3c6ca, 125 commits
git -C /tmp/claude-161238/work/repo branch -D fold-exit0  # 45c7e0cb3b, 66 commits
git -C /tmp/claude-161238/work/repo branch -D fold-hangev  # 191988d1fb, 65 commits
git -C /tmp/claude-161238/work/repo branch -D fold-health  # f10bee1a0f, 66 commits
git -C /tmp/claude-161238/work/repo branch -D fold-restore  # 1af6bdc9d5, 66 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/02-a2a-ft-kernels  # 18c2cbe3bb, 9 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/03-moe-receive-slots  # ca1c52f356, 10 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/04-ulfm-survivors  # 3c2643c066, 11 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/05-recovery-authority  # dddbec7290, 12 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/06-eplb-restore-empty-slots  # 3d8ee31b3a, 13 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/07a-executor-recovery  # 16c05f5ef7, 14 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/07b-executor-recovery-requests-e2e  # a0ff2f2f48, 15 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/08-hang-detection-fencing  # 5b07fd75eb, 17 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/09-ghost-slots-ring-copies  # 9e05b3e1e8, 18 commits
git -C /tmp/claude-161238/work/repo branch -D ftf2/10-health-and-guide  # e889864d2a, 16 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/01  # b9089d1264, 1 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/02  # 9aabe958bb, 3 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/03  # 5c7c832543, 4 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/04  # 321ebd33b4, 5 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/05  # 5280568012, 6 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/06  # 35325d44e3, 7 commits
git -C /tmp/claude-161238/work/repo branch -D pr-final/07  # 35dcbc7aff, 8 commits
```

#### Repo U (7)

```bash
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-consolidated  # 191988d1fb, 65 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-main  # 3e9027f4e4, 18 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-node  # 10b2849569, 23 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-phase1  # 3e9027f4e4, 18 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-restore  # 960ab25207, 19 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-restore2  # 2edca79d4c, 22 commits
git -C /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM branch -D wideep-ft-trim  # 01ec646c40, 19 commits
```

## UNIQUE-covered (no commands)

- A `ft-final2-lean` e477fa6c4b: 1 commit(s) not in the reference, each covered (1x subject in kept branch): [None][test] Trim the WideEP FT tests to one focused test per behaviou
- A `ftpr/01-stream-error-events` 7bbbac2a34: 2 commit(s) not in the reference, each covered (2x cov>=0.90 in v2): [None][test] Cover stream error events and the 503 retryable mapping; [None][fix] End a failed OpenAI stream with an SSE error event and ans
- A `pr-final/08` 6e609bf41c: 1 commit(s) not in the reference, each covered (1x cov>=0.90 in v2): [None][feat] Let the EPLB planner leave slots marked -1 empty
- A `pr-final/09` 9b693d2af2: 1 commit(s) not in the reference, each covered (1x cov>=0.90 in v2): [None][feat] Let the EPLB planner leave slots marked -1 empty
- A `pr10/01` 4e53bd293e: 1 commit(s) not in the reference, each covered (1x cov>=0.90 in v2): PR 01 placeholder
- A `pr10/02` 7592ca6f0c: 1 commit(s) not in the reference, each covered (1x cov>=0.90 in v2): PR 01 placeholder
- A `pr10/03` c9b0cb6829: 2 commit(s) not in the reference, each covered (2x cov>=0.90 in v2): PR 03 placeholder; PR 01 placeholder
- A `pr10/04` 8195b0769a: 3 commit(s) not in the reference, each covered (3x cov>=0.90 in v2): PR 04 placeholder; PR 03 placeholder; PR 01 placeholder
- A `pr10/05` b533532537: 4 commit(s) not in the reference, each covered (4x cov>=0.90 in v2): PR 05 placeholder; PR 04 placeholder; PR 03 placeholder ...
- A `prmain-all-old` 112316707f: 4 commit(s) not in the reference, each covered (4x cov>=0.90 in v2): [None][feat] Let the EPLB host-copy sharer hold an explicit list of ex; [None][test] Cover KV Cache Manager V2 step rollback; [None][test] Check that a quantization override shares the EP group he ...
- A `WT:/tmp/claude-161238/restack/wt-asm-test` 112316707f: 4 commit(s) not in the reference, each covered (4x cov>=0.90 in v2): [None][feat] Let the EPLB host-copy sharer hold an explicit list of ex; [None][test] Cover KV Cache Manager V2 step rollback; [None][test] Check that a quantization override shares the EP group he ...
- A `WT:/tmp/claude-161238/work/wt-pr10-1` 8195b0769a: 3 commit(s) not in the reference, each covered (3x cov>=0.90 in v2): PR 04 placeholder; PR 03 placeholder; PR 01 placeholder
- A `WT:/tmp/claude-161238/work/wt-rs` 112316707f: 4 commit(s) not in the reference, each covered (4x cov>=0.90 in v2): [None][feat] Let the EPLB host-copy sharer hold an explicit list of ex; [None][test] Cover KV Cache Manager V2 step rollback; [None][test] Check that a quantization override shares the EP group he ...
- U `wideep-ft-eplb` 4e3b47d51b: 1 commit(s) not in the reference, each covered (1x subject in kept branch): [None][feat] Report WideEP hang evidence from the workers and fence hu
- U `wideep-ft-hang` 3c93d34184: 1 commit(s) not in the reference, each covered (1x subject in kept branch): [None][feat] Report WideEP hang evidence from the workers and fence hu
- U `wideep-ft-health` df13da0f74: 1 commit(s) not in the reference, each covered (1x subject in kept branch): [None][feat] Report WideEP hang evidence from the workers and fence hu
- U `wideep-ft-main-pre-final-20261017` 69763cd9e8: 1 commit(s) not in the reference, each covered (1x cov>=0.90 in v2): [None][feat] Keep attention-DP routing and the MPI session serving wit
- U `wideep-ft-mtp` 3430226df5: 1 commit(s) not in the reference, each covered (1x subject in kept branch): [None][feat] Report WideEP hang evidence from the workers and fence hu

## UNIQUE: decisions for the user (no commands)

Format: ref sha; then each unique commit as `sha cov/lines subject` (cov = share of its added lines found in v2).

- **A `diag-pin`** 47bb38bda8 (worktree /tmp/claude-161238/work/wt-pin): 1 unique of 164. *Diagnostic-only commit; delete.*
  - 47bb38bda8 0.22/18 [None][test] Diagnostic: pin EPLB host copies in GPU-memcpy mode
- **A `feat-all`** 2077a4cf31 (worktree /tmp/claude-161238/work/wt-featall): 3 unique of 76. *Stream continuation (a Phase 2 feature), not in v2. Keep one copy (feat-all3 or feat-phaseb, same 6 subjects) and delete the rest.*
  - 908128f254 0.05/98 [None][feat] Rebuild a lost WideEP stream's tokens from rank 0's kept gathers
  - 710410c4ce 0.31/49 [None][feat] Test that a lost WideEP rank's stream continues with its exact greedy output
  - 79be07323f 0.08/482 [None][feat] Continue a lost WideEP rank's streaming requests on a survivor
- **A `feat-all2`** 39b3cd2c29 (worktree /tmp/claude-161238/work/wt-featall2): 5 unique of 80. *Stream continuation (a Phase 2 feature), not in v2. Keep one copy (feat-all3 or feat-phaseb, same 6 subjects) and delete the rest.*
  - 11910d015f 0.00/69 [None][chore] Trim the WideEP stream continuation module
  - 7dcc38fbd2 0.01/223 [None][chore] Move the WideEP stream continuation into its own module
  - 58180ffdcf 0.05/98 [None][feat] Rebuild a lost WideEP stream's tokens from rank 0's kept gathers
  - b9478246d9 0.31/49 [None][feat] Test that a lost WideEP rank's stream continues with its exact greedy output
  - 910258907e 0.08/482 [None][feat] Continue a lost WideEP rank's streaming requests on a survivor
- **A `feat-all3`** 9c618db9e2 (worktree /tmp/claude-161238/work/wt-featall3): 6 unique of 85. *Stream continuation (a Phase 2 feature), not in v2. Keep one copy (feat-all3 or feat-phaseb, same 6 subjects) and delete the rest.*
  - f24dc807c9 0.00/2 [None][fix] Copy a WideEP stream continuation's time budget only when the request has one
  - 3a3936f447 0.00/69 [None][chore] Trim the WideEP stream continuation module
  - 34251ed013 0.01/223 [None][chore] Move the WideEP stream continuation into its own module
  - a942923fa8 0.05/98 [None][feat] Rebuild a lost WideEP stream's tokens from rank 0's kept gathers
  - 539d3c0d9f 0.31/49 [None][feat] Test that a lost WideEP rank's stream continues with its exact greedy output
  - 03b60d4c85 0.08/482 [None][feat] Continue a lost WideEP rank's streaming requests on a survivor
- **A `feat-phaseb`** 1d260bb46f (worktree /tmp/claude-161238/work/wt-phaseb): 6 unique of 73. *Stream continuation (a Phase 2 feature), not in v2. Keep one copy (feat-all3 or feat-phaseb, same 6 subjects) and delete the rest.*
  - 1d260bb46f 0.00/2 [None][fix] Copy a WideEP stream continuation's time budget only when the request has one
  - f0a70fe5bf 0.00/69 [None][chore] Trim the WideEP stream continuation module
  - 3a8aeae9c6 0.01/223 [None][chore] Move the WideEP stream continuation into its own module
  - 163282d1c0 0.05/98 [None][feat] Rebuild a lost WideEP stream's tokens from rank 0's kept gathers
  - 33c38e0fe8 0.31/49 [None][feat] Test that a lost WideEP rank's stream continues with its exact greedy output
  - 7afab83f66 0.08/482 [None][feat] Continue a lost WideEP rank's streaming requests on a survivor
- **A `fix-ci`** 37c4877fe9 (worktree /tmp/claude-161238/work/wt-ci): 4 unique of 188. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 19c024b18e 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 1ef1f5e80f 0.25/4 [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the payload copy
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-cost`** 19c024b18e (worktree /tmp/claude-161238/work/wt-cost): 4 unique of 184. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 19c024b18e 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 1ef1f5e80f 0.25/4 [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the payload copy
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-cost-opt2`** c01f782ed2: 5 unique of 176. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 5361965826 0.19/48 [None][fix] Put ghost slots on host-accessible memory only for online EPLB
  - 62bb00ba01 0.26/61 [None][fix] Keep ghost slots in device memory, loaded by GPU copies, where GPU memory is not a NUMA node
  - 80085dc184 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 71513633b8 0.19/68 [None][perf] Back ghost slots with memory on the GPU's NUMA node, so restores copy into them from the CPU
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-diag-eplb`** da13f20b76 (worktree /tmp/claude-161238/work/wt-diag): 1 unique of 206. *Diagnostic-only commit; delete.*
  - da13f20b76 0.04/76 [None][chore] DIAGNOSTIC: log per-iteration EPLB update timing under TLLM_EPLB_UPDATE_LOG=1
- **A `fix-docs2`** 7e9ef22306 (worktree /tmp/claude-161238/work/wt-docs2): 2 unique of 175. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-exit`** 6b4a8dc18f (worktree /tmp/claude-161238/work/wt-exit): 3 unique of 171. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 80085dc184 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 71513633b8 0.19/68 [None][perf] Back ghost slots with memory on the GPU's NUMA node, so restores copy into them from the CPU
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-final3-tier`** 88e5302aac: 3 unique of 210. *384-expert Renormalize tier, not in v2. Delete if it was dropped on purpose, else port it as its own PR.*
  - 88e5302aac 0.00/4 [None][perf] Add the 384-expert tier to the Renormalize 512-thread cluster routing kernel
  - 174631c1ef 0.01/199 [None][test] Check that Renormalize routes the same on the 384- and 512-expert tiers, and time both
  - 814a2cad73 0.00/7 [None][perf] Add a 384-expert tier to the Renormalize routing policy
- **A `fix-final4`** d4c8e0a3fd (worktree /tmp/claude-161238/work/wt-final4): 1 unique of 208. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - d4c8e0a3fd 0.12/26 [None][fix] Default online EPLB's appended extra slots to empty; keep ghost opt-in
- **A `fix-getfailed`** 5c188e0021 (worktree /tmp/claude-161238/work/wt-gf): 1 unique of 181. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - 5c188e0021 0.15/40 [None][perf] Check for a failed peer after each host collective in one native call without the GIL
- **A `fix-host`** c372c53358 (worktree /tmp/claude-161238/work/wt-host): 3 unique of 181. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - c372c53358 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-kern`** 3358ea468e (worktree /tmp/claude-161238/work/wt-kern): 3 unique of 181. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - dbddc4d9f2 0.25/4 [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the payload copy
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-ktests`** c8768712f9 (worktree /tmp/claude-161238/work/wt-ktests): 2 unique of 176. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-minors`** dda2d51984 (worktree /tmp/claude-161238/work/wt-minors): 5 unique of 254. *Was a FINAL branch in hygiene.md. Its 5 small fixes are not in v2. Check whether v2 dropped them on purpose before deleting.*
  - dda2d51984 0.00/8 [None][fix] Place the one-slot table view on the load balancer's device and state its lifetime
  - 1f048122cc 0.50/14 [None][fix] Quiesce static EPLB at once after an interrupted iteration
  - fc7419efdc 0.33/3 [None][fix] Refuse an EPLB update schedule reset without WideEP fault tolerance
  - f499b5ef8d 0.00/3 [None][fix] Check the CUDA errors of the MoE A2A workspace reinitialization
  - dddd8de035 0.00/4 [None][infra] Fail the GB200 WideEP fault-tolerance tests when the image lacks ULFM
- **A `fix-populate-join`** 6bdb664dde (worktree /tmp/claude-161238/work/wt-pop): 1 unique of 253. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - 6bdb664dde 0.79/19 [None][perf] Finish prefaulting the EPLB host copies before the load balancer is final, so the first requests do not stall
- **A `fix-populate-join2`** 4f879c3a31 (worktree /tmp/claude-161238/work/wt-pop2): 1 unique of 262. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - 445a9c8f42 0.00/4 [None][infra] Fail the GB200 WideEP fault-tolerance tests when the image lacks ULFM
- **A `fix-ring`** e0ddea0a8e (worktree /tmp/claude-161238/work/wt-ring): 3 unique of 172. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 80085dc184 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 71513633b8 0.19/68 [None][perf] Back ghost slots with memory on the GPU's NUMA node, so restores copy into them from the CPU
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `fix-staging-batch`** 7e230b3a21 (worktree /tmp/claude-161238/work/wt-staging): 1 unique of 208. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - 7e230b3a21 0.12/26 [None][perf] Copy a staging chunk's received expert pieces into the slots in one batch
- **A `fix-static2`** 643c8afdf3 (worktree /tmp/claude-161238/work/wt-static2): 1 unique of 207. *384-expert Renormalize tier, not in v2. Delete if it was dropped on purpose, else port it as its own PR.*
  - 643c8afdf3 0.00/7 [None][perf] Add a 384-expert tier to the Renormalize routing policy
- **A `fix-sticky`** 85719fc28d (worktree /tmp/claude-161238/work/wt-sticky): 1 unique of 208. *Single perf/default experiment, not in v2. Delete if the experiment was rejected.*
  - 85719fc28d 0.09/92 [None][perf] Keep EPLB's resident experts in their slots
- **A `ftpr/02-eplb-shm-names`** ada67c31bb: 1 unique of 5. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/03-eplb-repacked-slots`** ccff070544: 1 unique of 7. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/04-moe-fake-ops-shared-health`** fccdcaa55a: 1 unique of 9. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/05-autotune-active-slots`** 392b11d981: 3 unique of 11. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/06-eplb-load-publish`** bdacc7a3bd: 3 unique of 12. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/07-eplb-unusable-slots`** 5773fddda9: 3 unique of 14. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/08-kv-v2-step-rollback`** c938db5bb7: 3 unique of 16. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/09-request-output-checkpoint`** 04b17589ba: 3 unique of 17. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/10-mpi4py-4.1.2`** 197620fe38: 3 unique of 18. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/11-ft-a2a-rank-mask-abort`** 3ab60a3044: 4 unique of 21. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/12-ft-moe-receive-slots`** d78eeb66e3: 5 unique of 23. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/13-ft-ulfm-survivors`** c2bd7232de: 5 unique of 25. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/14-ft-recovery-authority`** e4cd65c471: 5 unique of 27. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/15-ft-eplb-restore`** cd859e2523: 6 unique of 29. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/16-ft-adp-absent-ranks`** 2374f2261f: 7 unique of 31. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/17-ft-executor-recovery`** 154efa731c: 8 unique of 34. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/18-ft-ghost-slots`** c1203e32a5: 11 unique of 37. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - c1203e32a5 0.33/9 [None][doc] Document WideEP ghost extra slots
  - 664d6a9917 0.87/547 [None][test] Cover ghost EPLB slots and the KV cache memory they borrow
  - 52586bb577 0.89/717 [None][feat] Back WideEP extra EPLB slots with ghost memory lent by the KV cache
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/19-ft-ring-host-copies`** 97a0f3695f: 12 unique of 40. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 97a0f3695f 0.18/11 [None][doc] Document WideEP ring host copies
  - c1203e32a5 0.33/9 [None][doc] Document WideEP ghost extra slots
  - 664d6a9917 0.87/547 [None][test] Cover ghost EPLB slots and the KV cache memory they borrow
  - 52586bb577 0.89/717 [None][feat] Back WideEP extra EPLB slots with ghost memory lent by the KV cache
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ada67c31bb 0.00/4 [None][doc] Describe the per-job EPLB shared memory names and their cleanup
- **A `ftpr/20-ft-hang-detection`** 903bbcb6b0: 13 unique of 43. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - 903bbcb6b0 0.86/59 [None][doc] Document WideEP hang detection
  - 97a0f3695f 0.18/11 [None][doc] Document WideEP ring host copies
  - c1203e32a5 0.33/9 [None][doc] Document WideEP ghost extra slots
  - 664d6a9917 0.87/547 [None][test] Cover ghost EPLB slots and the KV cache memory they borrow
  - 52586bb577 0.89/717 [None][feat] Back WideEP extra EPLB slots with ghost memory lent by the KV cache
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - 21bf366934 0.30/70 [None][perf] Profile the CuteDSL NVFP4 MoE with tokens only on EPLB slots that hold an expert
  - ... and 1 more
- **A `ftpr/21-ft-health-status`** ea6e1c22cb: 14 unique of 46. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - ea6e1c22cb 0.74/35 [None][doc] Document the /health fault-tolerance states
  - 903bbcb6b0 0.86/59 [None][doc] Document WideEP hang detection
  - 97a0f3695f 0.18/11 [None][doc] Document WideEP ring host copies
  - c1203e32a5 0.33/9 [None][doc] Document WideEP ghost extra slots
  - 664d6a9917 0.87/547 [None][test] Cover ghost EPLB slots and the KV cache memory they borrow
  - 52586bb577 0.89/717 [None][feat] Back WideEP extra EPLB slots with ghost memory lent by the KV cache
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - 392b11d981 0.08/12 [None][test] Cover the tuning profile with empty local EPLB slots
  - ... and 2 more
- **A `ftpr/22-ft-route-one-slot`** 2632fe6fdf (worktree /tmp/claude-161238/work/wt-ftpr): 15 unique of 48. *Older PR stack, replaced by ftcore/ftwave2. Delete with the rest of ftpr/*.*
  - bac62fd860 0.85/157 [None][perf] Route one-replica EPLB placements in the NVLinkOneSided dispatch under WideEP fault tolerance
  - ea6e1c22cb 0.74/35 [None][doc] Document the /health fault-tolerance states
  - 903bbcb6b0 0.86/59 [None][doc] Document WideEP hang detection
  - 97a0f3695f 0.18/11 [None][doc] Document WideEP ring host copies
  - c1203e32a5 0.33/9 [None][doc] Document WideEP ghost extra slots
  - 664d6a9917 0.87/547 [None][test] Cover ghost EPLB slots and the KV cache memory they borrow
  - 52586bb577 0.89/717 [None][feat] Back WideEP extra EPLB slots with ghost memory lent by the KV cache
  - 154efa731c 0.74/187 [None][doc] Add the WideEP fault-tolerance guide
  - 6030a6bc8f 0.89/116 [None][feat] Keep attention-DP routing and balancing serving without a departed rank
  - 6ac6447b6c 0.85/423 [None][feat] Restore a lost rank's experts into EPLB slots from host copies
  - 2e1ff28c6a 0.81/255 [None][feat] Let the CuteDSL NVFP4 and Cutlass MoE read NVLinkOneSided receive slots in place
  - 3ab60a3044 0.50/2 [None][doc] Describe the NVLinkOneSided device rank mask
  - ... and 3 more
- **A `ftpr/23-review-fixes`** 8ab8e7c0b2 (worktree /tmp/claude-161238/work/wt-ftpr-fix): 25 unique of 58. *Older 23-PR stack top. Its review fixes partly differ from v2. Skim the list, then delete the whole ftpr/* stack.*
  - 8ab8e7c0b2 0.40/15 [None][fix] Discard a step whose MoE collective recorded an abort, even before the watch reports the failure
  - 8394411b45 0.33/24 [None][fix] Let a survivor that fails during a recovery exit alone until the install, so the others plan again
  - 0efadf22ae 0.00/477 [None][chore] Move peer-loss request bookkeeping and expert restore out of peer_loss_recovery.py (no behaviour change)
  - ab336177a6 0.04/26 [None][fix] Report unprotected, not starting, under an authority that restarted while serving
  - 675f378e04 0.52/21 [None][fix] Pre-map peers' EPLB host copies only with peer-loss recovery
  - 7a809fbece 0.00/1 [None][test] Name the fake tp_allgather_int64 placeholder departed, as MPIDist does
  - e646fa1f13 0.30/23 [None][fix] Answer an attention-DP pin outside the group as a client error, not a retryable peer loss
  - 9f92c3e9d3 0.27/15 [None][fix] Do not send a stream error event after [DONE]
  - 7070dc161f 0.37/19 [None][fix] Keep CutlassMoeFCRunnerInterface::runMoe's signature; set the input source slots through a setter
  - 33180826d8 0.18/28 [None][fix] Refuse CUDA graph capture of NVLinkOneSided collectives the host watchdog follows
  - bac62fd860 0.85/157 [None][perf] Route one-replica EPLB placements in the NVLinkOneSided dispatch under WideEP fault tolerance
  - ea6e1c22cb 0.74/35 [None][doc] Document the /health fault-tolerance states
  - ... and 13 more
- **A `pr-final/10`** 6fd0c1913c: 1 unique of 11. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/11`** 8929b3825c: 1 unique of 12. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/11z`** ac54a97d9e: 6 unique of 115. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/12`** 4d445f3976: 1 unique of 14. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/13`** 53e6057bc3: 1 unique of 16. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/13b`** be079f7da2: 4 unique of 89. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/14`** 4e81ee8766: 1 unique of 22. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/14b`** dfe128de22: 4 unique of 91. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/15`** fdc322efad: 1 unique of 40. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/15b`** 3682b0d00b: 4 unique of 93. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/16`** 07cf5045c2: 1 unique of 25. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/16b`** 24c1f180d2: 4 unique of 94. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/17`** 126a65b01c: 1 unique of 26. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/17b`** c290549c63: 2 unique of 58. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/17c`** c9c1b0db25: 2 unique of 69. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/17z`** 7e73056291: 6 unique of 119. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/18`** 3b57603063: 1 unique of 28. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/19`** 3e2ba7cc57: 1 unique of 33. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/19b`** 6c5f77a2ad: 2 unique of 59. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/19c`** dcf8e58820: 5 unique of 98. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/19z1`** 6557b05345: 6 unique of 109. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/19z2`** 256d145836: 6 unique of 116. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/20a`** e11b6ff556: 1 unique of 48. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/20b`** 2f12cfd469: 2 unique of 64. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/20z`** 68fe820a7b: 6 unique of 113. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/21a`** 41789c5844: 1 unique of 37. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/21b`** aeef8e85f9: 2 unique of 50. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/21c`** f918195264: 2 unique of 61. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/22`** 0d70cd81c2: 1 unique of 46. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/22c`** 75d1a1c508: 5 unique of 100. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/22z`** 369820ca8f: 6 unique of 114. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/23a`** 312adca97a: 1 unique of 47. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/23b_22b`** 87919f1bb0: 2 unique of 67. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/23c`** 7d699a4e60: 5 unique of 101. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24a`** 448472d070: 1 unique of 35. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24b`** e65441fffd: 1 unique of 41. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24c`** 01b0f7d57d: 2 unique of 52. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24d`** 74d8b0ea3b: 2 unique of 56. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24e`** 74cd03dd49: 2 unique of 72. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24f`** 1e3a6eb0da: 5 unique of 104. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/24z`** 0e4f56731c: 6 unique of 122. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/25a`** cad5c8fa9e: 2 unique of 54. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/25b`** 31b160beb4: 3 unique of 78. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/25c`** 4653856251: 5 unique of 105. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/25z`** 68f5562af0: 7 unique of 125. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 68f5562af0 0.71/17 [None][fix] Default online EPLB to empty extra slots where GPU memory is a NUMA node, and log the BAR1 size
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/26`** c713d0de68: 3 unique of 86. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/26z1`** 7d4edc8b4d: 6 unique of 117. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/26z2`** d83a2e43db: 6 unique of 121. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/27a`** e47a0546d7: 2 unique of 63. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/27b`** cf4bf1e6d4: 6 unique of 107. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/27z1`** 4929fe3c7c: 6 unique of 110. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/27z2`** 98499e34b5: 6 unique of 120. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-final/28`** 11f61041f0: 3 unique of 82. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-stack`** 787ed7c59b (worktree /tmp/claude-161238/work/wt-prstack): 3 unique of 82. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - ac29766dd0 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - b8d41b82f7 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 82182e181b 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr-stack-final`** 68f5562af0 (worktree /tmp/claude-161238/work/wt-prstack-final): 7 unique of 125. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 68f5562af0 0.71/17 [None][fix] Default online EPLB to empty extra slots where GPU memory is a NUMA node, and log the BAR1 size
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `pr10/06`** 2f22236789: 1 unique of 6. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `pr10/07`** 44de695c8a: 2 unique of 7. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `pr10/08`** 0f5643ffc2: 3 unique of 8. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `pr10/09`** 2f9340bc25: 4 unique of 9. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 2f9340bc25 0.89/3767 PR 09 placeholder
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `pr10/10`** eca27d44da: 4 unique of 10. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 2f9340bc25 0.89/3767 PR 09 placeholder
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `pr10/11`** f904f98222: 4 unique of 11. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 2f9340bc25 0.89/3767 PR 09 placeholder
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `rsg-v2/E`** 72c4b4ff38 (worktree /tmp/claude-161238/work/wt-rsg-E): 1 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - dcbf713e5d 0.82/1406 stage c06
- **A `rsg-v2/K`** dad3a6809a (worktree /tmp/claude-161238/work/wt-rsg-K): 1 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - 4f5dff490f 0.88/1657 stage c02
- **A `rsg-v2/R`** 5fff4bb44e (worktree /tmp/claude-161238/work/wt-rsg-R): 2 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - fa7f2a02c2 0.88/1405 stage w08
  - c2f0150310 0.89/185 stage c01
- **A `rsg-v2/X`** 3137fcbfab (worktree /tmp/claude-161238/work/wt-rsg-X): 2 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - a694e791bc 0.86/2702 stage c07a
  - 212aa4c901 0.88/33 stage c06
- **A `rsg/E`** 631b689798: 1 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - 217c191f8e 0.82/1316 stage c06
- **A `rsg/K`** 0bfc2aed73: 1 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - eb307ca457 0.88/1645 stage c02
- **A `rsg/R`** 7626afd348: 3 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - d9eae44817 0.87/1387 stage w08
  - 87c8f45af0 0.80/5 stage c02
  - 4cb023d9c1 0.89/185 stage c01
- **A `rsg/X`** ca202b5e07: 2 unique of 18. *Restack "stage" snapshots (cov 0.80-0.89) from building v2. Delete once v2 is accepted.*
  - 0b45cc2973 0.86/2683 stage c07a
  - d07ed1b681 0.88/33 stage c06
- **A `rt2-correct`** 3ff5d44558 (worktree /tmp/claude-161238/work/wt-rt2a): 1 unique of 261. *Red-team branch whose one unique commit (fail GB200 tests without ULFM) is also on fix-minors. Decide once for both.*
  - 445a9c8f42 0.00/4 [None][infra] Fail the GB200 WideEP fault-tolerance tests when the image lacks ULFM
- **A `rt2-hotpath`** eb2fab6c64 (worktree /tmp/claude-161238/work/wt-rt2b): 1 unique of 257. *Red-team branch whose one unique commit (fail GB200 tests without ULFM) is also on fix-minors. Decide once for both.*
  - 445a9c8f42 0.00/4 [None][infra] Fail the GB200 WideEP fault-tolerance tests when the image lacks ULFM
- **A `rt2-org`** 628a7ad713: 1 unique of 257. *Red-team branch whose one unique commit (fail GB200 tests without ULFM) is also on fix-minors. Decide once for both.*
  - 445a9c8f42 0.00/4 [None][infra] Fail the GB200 WideEP fault-tolerance tests when the image lacks ULFM
- **A `rt2-tests`** c8e3ce182f (worktree /tmp/claude-161238/work/wt-rt2d): 1 unique of 262. *Red-team doc tweak, not in v2. Port it if wanted, else delete.*
  - daeebd41bb 0.00/3 [None][doc] Say a retry of a request pinned to a lost attention-DP rank fails until the deployment restarts
- **A `WT:/tmp/claude-161238/pc-fix-host`** c372c53358: 3 unique of 181. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - c372c53358 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `WT:/tmp/claude-161238/pc-fix-kern`** 19ea00bab0: 3 unique of 183. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 19ea00bab0 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 3a663d7150 0.25/4 [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the payload copy
  - 7862ceee68 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `WT:/tmp/claude-161238/pc-opt2e`** a7dd7c0d7f: 4 unique of 187. *Rank-mask/ghost-slot variants with low cov in v2. Delete unless you still want that alternate approach.*
  - 19c024b18e 0.07/14 [None][fix] Reject a CUDA active-rank mask that only a copy kernel could write ahead of a PDL dispatch
  - 1ef1f5e80f 0.25/4 [None][perf] Reload the dispatch rank mask for the last token's publish instead of keeping it live across the payload copy
  - 9c84c39c59 0.00/9 [None][fix] Name the register-word rank-mask check apart from the pointer overload
  - 9d7b33ecb8 0.26/53 [None][perf] Read the active-rank mask and the expert -> slot table before the dispatch grid dependency wait
- **A `WT:/tmp/claude-161238/work/wt-pr10-2`** 0f5643ffc2: 3 unique of 8. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `WT:/tmp/claude-161238/work/wt-pr10-3`** f904f98222: 4 unique of 11. *"PR NN placeholder" squashes from the pr10 stack (cov 0.70-0.89). Delete.*
  - 2f9340bc25 0.89/3767 PR 09 placeholder
  - 0f5643ffc2 0.81/1203 PR 08 placeholder
  - 44de695c8a 0.89/2309 PR 07 placeholder
  - 2f22236789 0.70/928 PR 06 placeholder
- **A `WT:/tmp/claude-161238/work/wt-psf-check`** 9ae82f698c: 7 unique of 126. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 9ae82f698c 0.71/17 [None][fix] Default online EPLB to empty extra slots where GPU memory is a NUMA node, and log the BAR1 size
  - 60e9f0724e 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 6d96250e7b 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - 1f10f3bf1f 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 5dbd123ec1 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - 32da1ab60f 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 24066d2af9 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `WT:/tmp/claude-161238/work/wt-psf-check0`** bf5c4d71af: 6 unique of 124. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `WT:/tmp/claude-161238/work/wt-psf-check1`** 68f5562af0: 7 unique of 125. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - 68f5562af0 0.71/17 [None][fix] Default online EPLB to empty extra slots where GPU memory is a NUMA node, and log the BAR1 size
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `WT:/tmp/claude-161238/work/wt-psf-check2`** 0e4f56731c: 6 unique of 122. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **A `WT:/tmp/claude-161238/work/wt-psf-check3`** 05e3802db1: 6 unique of 123. *pr-final restack trial; its unique commits are near-v2 drafts. Delete.*
  - cf4bf1e6d4 0.89/19 [None][test] Restore after a peer loss into online EPLB's empty extra slots
  - 62cc9ca632 0.00/2 [None][doc] List /v1/responses among the WideEP streams that end with an SSE error event
  - f66b8272cd 0.87/100 [None][fix] Check each MoE dispatch source's token count, trapping on a corrupt one unless an abort is pending
  - 6d78167fa8 0.90/59 [None][feat] Let EPLB's updates fill a WideEP ghost slot once a restore loaded it
  - aeef8e85f9 0.25/8 [None][feat] Accept online EPLB with WideEP peer-loss recovery
  - 6fd0c1913c 0.86/74 [None][feat] Let the Cutlass MoE read its input rows from fixed per-source slots
- **U `backup/pre-fold-c168c39961`** c168c39961: 7 unique of 23. *Pre-fold/pre-squash backup of fixups. Delete.*
  - c168c39961 0.00/3 fixup! [None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 12b982aeef 0.51/92 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 5f8f89b539 0.50/10 fixup! [None][feat] ULFM survivor communicators for host collectives
  - 736445f280 0.00/2 fixup! [None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 11952e0c2a 0.87/224 fixup! [None][feat] ULFM survivor communicators for host collectives
  - ce4fc082f0 0.12/16 fixup! [None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 66244a1ead 0.08/13 [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs
- **U `backup/pre-fold-c706fdfa84`** c706fdfa84: 11 unique of 27. *Pre-fold/pre-squash backup of fixups. Delete.*
  - c706fdfa84 0.00/7 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 48f15a1d8f 0.69/13 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - bf8f4b98e6 0.00/8 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - f2215465ff 0.00/8 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 6d0aa8d6c9 0.00/1 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 19e9d13e95 0.67/3 fixup! [None][test] add a WideEP peer-loss recovery integration test
  - d53b44738c 0.67/3 fixup! [None][feat] ULFM survivor communicators for host collectives
  - 0420e30dc8 0.40/5 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 55eb526851 0.27/359 fixup! [None][feat] bind MoE layers to the all-to-all execution descriptor for recovery
  - 162132465b 0.41/99 fixup! [None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - c13cda2721 0.08/13 [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs
- **U `backup/prs-pre-squash`** 4474b35e00: 6 unique of 17. *Pre-fold/pre-squash backup of fixups. Delete.*
  - 4fc207b69f 0.51/61 fixup! [None][feat] Bind MoE layers for recovery and route a lost rank's experts to their replicas
  - d8e49882b6 0.14/7 fixup! [None][feat] Add ULFM survivor communicators for host collectives
  - 3cf5f06ffe 0.25/8 fixup! [None][feat] Read the all-to-all receive slots in place under a MoE execution state
  - cec575cb70 0.50/4 fixup! [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 390f120486 0.51/484 [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 941228ada7 0.71/42 [None][fix] Fix MoE all-to-all issues found while enabling WideEP fault tolerance
- **U `fix-borrow-map`** 606399ab52: 1 unique of 53. *Single experiment or fixup, not in v2. Delete if the experiment was rejected (fix-reuse and fix-sort-gate have worktrees under .wideep-ft-runs).*
  - 606399ab52 0.08/143 fixup! [None][feat] Add a VMM allocator whose empty EPLB slots hold no device memory
- **U `fix-resp-cost`** 1fec69ae64: 1 unique of 68. *Single experiment or fixup, not in v2. Delete if the experiment was rejected (fix-reuse and fix-sort-gate have worktrees under .wideep-ft-runs).*
  - fcc86b10b2 0.88/34 fixup! [None][feat] Recover a WideEP deployment from further losses
- **U `fix-reuse`** b7cfb40656 (worktree /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/reuse): 1 unique of 56. *Single experiment or fixup, not in v2. Delete if the experiment was rejected (fix-reuse and fix-sort-gate have worktrees under .wideep-ft-runs).*
  - b7cfb40656 0.19/83 [None][chore] Share WideEP recovery helpers with the executor
- **U `fix-sort-gate`** 4f0c308b02 (worktree /lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/arusli/src/TensorRT-LLM/.wideep-ft-runs/claude/wt/sortgate): 2 unique of 68. *Single experiment or fixup, not in v2. Delete if the experiment was rejected (fix-reuse and fix-sort-gate have worktrees under .wideep-ft-runs).*
  - 471b3d0db5 0.28/345 [None][perf] Read the WideEP routing in its receive slots instead of compacting it in the dispatch
  - eec07f3cec 0.13/183 [None][feat] Sort top-K ids held in fixed per-source slots in moe_sort
- **U `prs-rebase-trial`** d9777a8612: 2 unique of 10. *Pre-consolidation line whose commits were rewritten into v2. Delete unless you want the old review history.*
  - 822eb0f5a7 0.52/486 [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 941228ada7 0.71/42 [None][fix] Fix MoE all-to-all issues found while enabling WideEP fault tolerance
- **U `trim-vector-47b33f8758`** 47b33f8758: 1 unique of 19. *Single experiment or fixup, not in v2. Delete if the experiment was rejected (fix-reuse and fix-sort-gate have worktrees under .wideep-ft-runs).*
  - 47b33f8758 0.16/64 [None][perf] compact WideEP receive-slot routing one token row per thread
- **U `wideep-ft-dev-pre-consolidate`** 79cdd9f36f: 1 unique of 64. *Pre-fold history of wideep-ft-dev (fixups). Delete.*
  - 4cf60b059d 0.88/34 fixup! [None][feat] Recover a WideEP deployment from further losses
- **U `wideep-ft-dev-prefold`** 9ad3d1f8e0: 9 unique of 89. *Pre-fold history of wideep-ft-dev (fixups). Delete.*
  - edadd55835 0.20/5 fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable
  - e90db1b05f 0.79/82 fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable
  - e5fd733c4b 0.80/74 fixup! [None][fix] End trtllm-llmapi-launch by its worker's signal
  - cf36fc9386 0.50/2 fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable
  - 7afa422c67 0.38/24 fixup! [None][feat] Read WideEP fault-tolerance settings from one validated config
  - 997e9cb11a 0.33/6 fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable
  - 3c720e6c56 0.40/5 fixup! [None][feat] Configure the WideEP failure domain and find nodes by host name
  - be1038e947 0.64/11 fixup! [None][feat] Confirm a hung WideEP rank in the recovery authority
  - 895202bf8b 0.45/20 fixup! [None][fix] Keep a serving WideEP deployment healthy when its authority is unreachable
- **U `wideep-ft-exploration`** 474d437ad4: 23 unique of 29. *Early exploration line (fork/wideep-ft-exploration holds a copy). Keep as history or delete and rely on the fork.*
  - 474d437ad4 0.06/8162 [None][chore] snapshot Codex WideEP recovery work in progress
  - 1a8a89c6fb 0.00/7 [None][docs] record immutable workspace and MPI helper proofs
  - a515aa3628 0.00/12 [None][docs] record fixed-slot recovery component evidence
  - 3e4674c99b 0.00/21 [None][docs] summarize fence recovery prerequisite proofs
  - e98d97b9ce 0.00/22 [None][docs] record WideEP same-graph abort proof limits
  - 6f00fb52c7 0.00/165 [None][test] harden WideEP regression cleanup and clarify restart timings
  - f3bb2c78c1 0.00/376 [None][test] qualify MPI WideEP fault and restart benchmarks
  - 6a90526f90 0.50/4 [None][refactor] remove Ray-specific actor death monitoring
  - f64167f520 0.00/12 [None][docs] record MPI binding assessment and graph reuse limits
  - 21d631c31a 0.00/32 [None][docs] record isolated WideEP recovery prerequisite proofs
  - 404a14ac6d 0.00/89 [None][docs] characterize MPI host failures and WideEP watchdog behavior
  - 805990b2fc 0.00/67 [None][test] guard WideEP fault cleanup against storage failure and PID reuse
  - ... and 11 more
- **U `wideep-ft-exploration-next`** d9084befa9: 104 unique of 180. *Early exploration line (fork/wideep-ft-exploration holds a copy). Keep as history or delete and rely on the fork.*
  - c0ed3f32cd 0.20/10 WIP gather backoff
  - 47f94f7bf9 0.08/13 [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs
  - b474200595 0.08/13 [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs
  - 479d51c512 0.00/54 Add the prefill-load failover, the cap-128 finding and the #15785 boundary
  - 8f66bed4fa 0.00/56 Add in-flight kills with a new-request probe to the failover chart
  - f135dec4ea 0.00/215 Simplify overhead and late-write diagrams
  - 1172ac4194 0.00/289 [None][doc] add the WideEP FT failover breakdown and apply audit fixes
  - 9691c19f42 0.00/16 [None][doc] tighten the WideEP FT evaluation notes
  - 22766a6629 0.00/62 [None][doc] use plain standard charts for the WideEP FT evaluation
  - b1d1899497 0.00/444 [None][doc] add the WideEP FT overhead and load evaluation
  - d79ecd39c1 0.00/471 [None][doc] add the WideEP FT kernel-path diagram
  - e01da9c2ac 0.00/4 [None][doc] refresh WideEP FT commit links
  - ... and 92 more
- **U `wideep-ft-fix`** 625c5e1759: 6 unique of 17. *Pre-consolidation line whose commits were rewritten into v2. Delete unless you want the old review history.*
  - 14ea37ad82 0.69/71 [None][fix] Make host votes say how a departed rank counts
  - b1d7fa789d 0.08/13 [None][fix] Use the steady-state all-to-all timeout in captured CUDA graphs
  - 49867dc4b1 0.89/145 [None][feat] Roll back KV Cache Manager V2 allocations of a discarded step
  - 77b5c2a93d 0.24/126 [None][feat] Route a lost rank's experts to their replicas after a peer loss
  - a2669e0880 0.57/1076 [None][feat] Bind MoE layers to the all-to-all execution descriptor for recovery
  - 2fbad354a2 0.51/484 [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all
- **U `wideep-ft-fix-pre-gather`** b474200595: 1 unique of 14. *Pre-consolidation line whose commits were rewritten into v2. Delete unless you want the old review history.*
  - b474200595 0.08/13 [None][fix] use the steady-state all-to-all timeout in captured CUDA graphs
- **U `wideep-ft-next`** a958187bfc: 15 unique of 31. *Pre-consolidation line whose commits were rewritten into v2. Delete unless you want the old review history.*
  - cfcd67b577 0.81/99 [None][fix] Make object gathers say what a departed rank counts as
  - 4c1cb82e77 0.12/73 [None][perf] Revoke host communicators from the native peer-failure watch
  - 2323f0c18f 0.53/40 fixup! [None][feat] Support the overlap scheduler with WideEP peer-loss recovery
  - df2bf6cca7 0.00/1 fixup! [None][fix] Make host votes say how a departed rank counts
  - 6e0702df18 0.61/28 [None][fix] Clear stale step state when the overlap scheduler discards a step
  - 4eadb37b11 0.66/74 [None][fix] Make host votes say how a departed rank counts
  - 2828410be6 0.62/29 [None][fix] Survive a peer loss mid-iteration of the overlap scheduler
  - a470d307bb 0.18/164 [None][perf] Decide the attention-DP CUDA graph lookup from the padding vote
  - 4bd18f06b0 0.30/101 [None][perf] Gather attention-DP responses once per forward pass
  - 16917b2f3e 0.14/28 [None][perf] Exchange attention-DP rank state as one int64 allgather
  - b1d7fa789d 0.08/13 [None][fix] Use the steady-state all-to-all timeout in captured CUDA graphs
  - 49867dc4b1 0.89/145 [None][feat] Roll back KV Cache Manager V2 allocations of a discarded step
  - ... and 3 more
- **U `wideep-ft-prs`** 1707fac125: 3 unique of 13. *Pre-consolidation line whose commits were rewritten into v2. Delete unless you want the old review history.*
  - ac649fae5f 0.75/64 [None][fix] Make host votes say how a departed rank counts
  - 822eb0f5a7 0.52/486 [None][feat] Add a run-time rank mask and abort to the NVLinkOneSided all-to-all
  - 941228ada7 0.71/42 [None][fix] Fix MoE all-to-all issues found while enabling WideEP fault tolerance

## Remote-tracking refs (information only, no commands)

Repo A (`origin` = U's worktree `.wideep-ft-runs/claude/dev`; A has no `fork` remote):
- 30 `origin/*` refs whose branch no longer exists in U: fix-cq4, fix-dr-tests, fix-fence-before-propose, fix-ft-dispatch, fix-hang-deadline, fix-hang-progress, fix-hang-rules, fix-integ, fix-kv-reuse-test, fix-nr10, fix-nr2, fix-nr58, fix-nr6, fix-nr7, fix-prime, fix-repeat, fix-rerun-pin, fix-specdec, fix-sweep, ghost, next-build, wideep-ft-folded, wideep-ft-folded2..8, zmq-queue. A `git fetch --prune` would drop them.
- `origin/fix-sort-gate` 89da95d29e differs from U's fix-sort-gate 4f0c308b02.
- `origin/wideep-ft-dev` 191988d1fb is behind A's wideep-ft-dev aae00e6c65 (kept, so this is expected).

Repo U (from the local `refs/remotes`; the fork was not contacted):
- `fork/wideep-ft` 6f3ad13ab0 vs local e4c6cefa41, `fork/wideep-ft-all` 57fd045971 vs local 800347c9f4, `fork/wideep-ft-exploration` 479d51c512 vs local 474d437ad4: these differ.
- fork copies of branches graded SUPERSEDED~ here: fork/wideep-ft-main, fork/wideep-ft-phase1, fork/wideep-ft-node, fork/wideep-ft-restore, fork/wideep-ft-restore2, fork/wideep-ft-trim. Fork copies of UNIQUE-covered ones: fork/wideep-ft-eplb, fork/wideep-ft-hang, fork/wideep-ft-health, fork/wideep-ft-mtp. They match their local branches, which are stale against v2.
- `pr/15785`, `pr/15788`, `pr/15789`, `pr/15895`, `pr/19610`: upstream PR heads with no local branch.
- `origin/main` fc52f0adee is behind A's upstream/main 0829689c52.

