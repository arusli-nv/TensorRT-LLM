# WideEP FT -> main: PR stack plan (2026-10-19; restacked on local branch pr-stack, not pushed)
Source: wideep-ft-dev fc52f0adee..6e8657d0be (89 commits, 111 files, +18,211/-577); every commit is in exactly one PR. Sizes: summed
per-commit numstat, P(roduct)/T(ests incl. cpp/tests)/D(ocs/*.md); later commits rewrite earlier lines, so sums overstate net diffs.
Owners (CODEOWNERS, last match): moe=trt-llm-moe-devs, rt=trt-llm-runtime-devs, dis=trt-llm-disagg-devs (py_executor/model_engine/engine),
kv=trt-llm-kv-cache-manager-devs, kern=trt-llm-kernels-devs, inf=trt-llm-infra-devs, doc=trt-llm-doc-owners, mg=trt-llm-multi-gpu-ci-review
(l0_gb200_multi_gpus.yml), itg=trt-llm-devs+qa-function+infra (integration defs). Clean on fc52f0adee (git merge-tree): 1, 2, 4, 5; 6 conflicts only in l0_b200.yml; the rest need a rebase.
## Part A: main-side PRs, no dependency on FT (FT-off column: useful on main / neutral)
| # | Title | Commits | P / T / D | Owners | Reviewers check | FT off |
|---|---|---|---|---|---|---|
| 1 | [None][fix] End a failed chat-completions stream with an error event | c59b62a064 | +22/-16, +19, 0 | rt | SSE error-event shape for clients | useful (bug) |
| 2 | [None][fix] Fix the fake moe_a2a op args and share EPGroupHealth across ConfigurableMoE copies | 9285afdfb2 921e7ebef0 | +8, +28, 0 | rt, moe | fake-op signatures = real ops; one health object | useful (bugs) |
| 3 | [None][fix] Load EPLB slots of MoE weights repacked into non-contiguous layouts | 507f137d47 (borrowed-op dense-block check moved to 12) | +37/-10, +17, 0 | moe | slot_rows on DEEPGEMM FP8 Blackwell; dense-block refusal | useful (bug) |
| 4 | [None][infra] Upgrade mpi4py to 4.1.2 | 520db51eb3 | +47/-54, 0, 0 | inf | image rebuild; MPI futures runtime 3.1.5 -> 4.1.2 for every user (API-rename patch updates only) | neutral |
| 5 | [None][fix] End trtllm-llmapi-launch by its worker's signal | c5abacd2d1 | +22/-4, +79, 0 | rt, inf (l0_cpu) | exit status by signal for every job | useful |
| 6 | [None][feat] Roll back a discarded step's allocations in KV Cache Manager V2 | 54bc5b85a2 (KV hunks of 2e06/29fc not moved: they change revert_prepare_resources' signature, whose caller lands in 19) | +28, +152, 0 (+~54 P/~98 T) | kv | revert semantics, reuse never offers reverted KV | neutral (no caller yet) |
| 7 | [None][chore] Share EPLB's slot load and routing publish (loadAndPublishPlacement) | d4e4935657 (60bba main part; installPlacement part = f481701f9b in 21c) | +23/-47, 0, 0 | moe | same calls, same order on the healthy update | neutral |
| 8 | [None][feat] Let the EPLB planner leave slots marked -1 empty | 7e33244c20 (738a planner part; restore part = ac29766dd0 in 25b) | +75/-21, +190/-7, 0 | moe | empty mask = planner unchanged | neutral |
| 9 | [None][perf] Tune the CuteDSL NVFP4 MoE without tokens on empty EPLB slots | 76a23f300f (ported to main: no source_tokens) | +73/-22, +16, 0 | rt, moe | tuning keys/shapes unchanged; zero empty slots = same profile | neutral |
| 10 | [None][feat] Let the Cutlass MoE read its input rows from receive slots (MoeInputSourceSlots) | 82182e181b (b5b3 kernel/op; wiring = a5be97f8e0 in 14, doc folded into 300ed063be) | +101/-25, +77/-5, +2/-2 | moe, rt | rejected combos (LoRA, scales, W4AFP8, min-latency); OSS runner only | neutral |
| 11 | [None][feat] Let the EPLB host tensor sharer keep a subset of the experts | 9152a10a1a (ported to main: updates_enabled, not host_copies) | +48/-26, +60, 0 | moe | default split unchanged; /dev/shm preallocation fails fast | useful (clear error) |
| 12 | [None][feat] Ghost EPLB slot memory on CudaVirtualMemoryManager | a64de181e0 (3664 + b347 dense-block check); d0a8+8c1b = 32011c6bf6 in 25b; C++ rename not split | ~+700 net (sum +1,004/-286), +253, 0 | moe, rt (VMM) | scratch-swap configurator, graph-safe mapping, free() sync | dead code until 25: send as RFC or with 25 |
## Part B: FT core (each keeps main's kernels/routing/executor loop with FT off)
| # | Title | Commits | P / T / D | Owners | Deps | Reviewers check |
|---|---|---|---|---|---|---|
| 13 | [None][feat] NVLinkOneSided: rank mask by pointer, abort record, fixed source slots | 634dfe1335 (23ce+2945+6e86) 95113f8f63; 3b3f moved to 14 | +311/-160, +427/-22, +4 | moe, rt | main | dead sender's region never reused; FT-off kernels identical |
| 14 | [None][feat] Read all-to-all receive slots in place; compact routing lists and copy | e856848a5a 0e3dd7e765 9066b4517e 45fdca335f a5be97f8e0 d9db732e0a | +860/-191, +807/-62, +7/-4 | moe, kern, rt, doc | 13 | execution-state seam, backend contract, +28 lines in _utils.py |
| 15 | [None][perf] Route one-replica EPLB placements in the dispatch (skip the route kernel) | 0fc62c6dbd 4994bf3378 9683de0777 (stacked after 21a: 3024 needs the restore) | +177/-24, +225/-2, 0 | moe, rt | 13, 14 | one-slot latch vs later replicas; graph-captured table address |
| 16 | [None][feat] ULFM survivor communicators and an MPI session that serves after a peer loss | 155e13aee4 15169e85c3 de9f843d6e (f4af+703c+2e19, doc hunk folded into 300ed063be); ada8 = f6d791c408 in 19 | +907/-57, +992/-26, +10/-3 | rt, doc | 4 | ob1/tcp requirement, finalize timer, +55 lines in _utils.py |
| 17 | [None][feat] External recovery authority on ZeroMqQueue with a validated FT config | 445923a93d; config = 17b 761af2d8d6 cbcdabcc9e; ZeroMqQueue = 17c e2d8f2ba70 (both unsplit) | +1,146/-315, +1,027/-104, +16/-10 | rt, doc | main | HMAC before unpickle, session/sequence replay guard, ipc.py options |
| 18 | [None][feat] Keep attention-DP routing serving with an absent rank; bind MoE layers for recovery | 5b9a8fd830 1e44a675fa | +202/-36, +318/-29, 0 | rt, dis, moe | main | no change with all ranks present |
| 19 | [None][feat] Recover the executor from a single WideEP peer loss | f43a037ede 341d3b45e0 (d803+996 driver hunk) 300ed063be (015a+doc hunks of 996/f4af/b5b3) f6d791c408 935076e8ae; 19b 5fb5dadad4 (8080); 1919 -> 24e; 1a49 -> 28 | +1,061/-72, +1,444/-20, +114/-4 | rt, dis, itg, mg, doc | 6, 13-18 | step commit/discard, redelivery, per-response cost; >2.5k, no clean seam |
| 20 | [None][feat] Report the WideEP FT status on /health | 20a 7357625056 (0552 alone); 20b c0afa87e69 (cde1+1fe1; cde1 needs 17b, 1fe1 needs 27a) | +311/-47, +200/-12, +20/-9 | rt, doc | 17, 19 | body only when FT on; healthy if authority unreachable |
| 21 | [None][feat] Restore a lost rank's experts into EPLB slots; agree on online-EPLB placement | 21a 51f5607c50 be97ca91dc; 21b 9258e3086d b8d41b82f7 (fe38 wiring); 21c 8c0767fbb3 f481701f9b | +620/-184, +526/-57, +1 | moe, rt | 7, 8, 19 | restore = EPLB placement update; placement among survivors |
| 22 | [None][feat] Confirm hung WideEP ranks from worker evidence and native heartbeats | 583adc151a b60a30114a 58f6787e4e 2c78f41f39 a4245d2344; 22b 23dec565a5 (4a53, unsquashed) | +1,207/-133, +811/-146, +31/-12 | rt, doc | 17, 19 | squash 4a53 rule rewrite; one poller; heartbeat C++ in nanobind |
| 23 | [None][feat] Fence hung or departed WideEP ranks (self, node-mate, PMIx) | 23a a176d0d6d6; 23b c4b266a639; 3efd f921 -> 28 | +493/-88, +448/-35, +34/-9 | rt, doc | 16, 22 | every SIGKILL path (0.2); PMIX_JOB_CTRL_KILL on one proc only |
| 24 | [None][feat] Recover with overlap, ADP balancing, MTP/spec-dec, DSA, KV reuse, node loss, later losses | 24a 93eb815413 44837b9770; 24b d08481e571; 24c b2bf7c7695 85cfd57eba; 24d 1871cf2d5f 4f12c35a05; 24e 87ad2b3b92 1e6b12ceed 574db394f5 | +742/-279, +989/-199, +60/-33 | rt, dis, kv, itg, mg, doc | 19, 21, 23 | re-plan during recovery; could split node-loss + further losses (~0.9k) |
| 25 | [None][feat] Back WideEP extra EPLB slots with ghost memory and lend it from the KV cache | 25a 89b14c8e68 e64e28f73a; 25b ee4e8564f0 80890a2d78 32011c6bf6 ac29766dd0 62605e8c40 4248c8ba09; b054 -> 26 | +533/-209, +387/-107, +7/-6 | moe, rt, dis, mg, doc | 12, 21 | KV quota shrink only while idle; ghost default; online EPLB fills ghosts first |
| 26 | [None][feat] Keep WideEP host copies in a ring of nodes | 75dce6b88e 87cc267dec (9ad5+1269+aa70+2286+3ef9+432b) 787ed7c59b (b054) | +522/-334, +536/-82, +10/-5 | moe, rt, mg, doc | 11, 21, 25 | squash trim/always into 9ad5; restore via a2a workspace |
| 27 | [None][test] Extend the WideEP peer-loss integration tests and finish the guide | 27a 2ec35068f5 bc5a915110; af28 = 62605e8c40 in 25b; e5ac -> 28 | 0, +407/-97, +93/-39 | itg, mg, doc | all | GB200 test-list cost |
## Mixed or misplaced commits (restack work). Split: fe3892a02b (kernel stats -> 14, EPLB wiring -> 21); b5b3d3188c (Cutlass kernel/op -> 10, slot wiring + doc -> after 14);
  2e06fe68f1, 29fc79acb3 (kv_cache_manager_v2 hunks -> 6); db2c096076, a3000c8132 (config -> 17, worker reads -> 19/21/22);
  3c28d56067 (ipc.py options + transport -> 17 from the start; its heartbeatSender/fault_tolerance_status hunks -> 22/20);
  e3542e071d (rename: C++ -> 12, Python/docs -> 25; the "borrowed" name never ships); 4a5350727f (rule rewrite squashed
  into 22's earlier commits; native heartbeat could be its own commit).
- Move: 996bc538fe's peer_loss_driver.py hunk after d803966b37; 3efdf3fc11, f92138aa3e after the PMIx fence (23);
  60bba5c490 rebased onto main's update only (installPlacement half -> 21); 83b96f52a4 rides 13/14's fixed-slot payload.
- Squash: 2945b0f5bb 6e8657d0be->23cec4ded7, 8c1b0afd1d->d0a8b9e0c9, 703c763be2 2e19c3b6d2->f4afe429ba, cde13f30ec 1fe19e354a->0552c9fab7,
  aa70da37dc 126976a33b 228625a11a 3ef90d7f2b 432b8907db->9ad5ac0d27.
- Over ~2k after squash: 17, 19 (e528dd8faa alone +1,834), 22. Could merge: 7+8, 13+15, 22+23. Config stays a file (no LlmArgs
  field: no manifest/schema regen yet). Send first: 1, 2, 3.
## Restack result (branch pr-stack in the /tmp/claude-161238/work/repo clone, worktree /tmp/claude-161238/work/wt-prstack; 82 commits, tree == wideep-ft-dev)
Stack order (bottom first; a suffix marks a plan PR split into non-contiguous segments): 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 |
24a overlap+ADP | 21a restore | 15 | 24b node loss | 22 | 23a | 20a | 21b | 24c MTP+DSA | 25a | 24d reuse+re-plan | 17b config | 19b re-run |
21c | 27a tests | 20b (cde1+1fe1) | 23b+22b (e8ff, 4a53) | 17c transport | 24e | 25b | 28 (new: 1a49 3efd e5ac f921) | 26.
Why the Part B tail follows the original order: later commits were written on earlier ones (cde1 uses db2c's config digest, so it left 0552 for 20b; 0552 needs ab50/4aab-0820/2c87;
b020 needs 0552 and 3572 needs b020; 128b, db2c, a300, 8080, bea4, e8ff, 4a53, 3c28, 9f5f, 1919, 29fc need 278a; a040 needs 8080; 1fe1,
e354, 1a49 need a040; 3efd needs 3c28 and 1a49; 9ad5 needs 1a49), so plan PRs 15 and 19-27 cannot each be contiguous without
re-authoring code. Probe: doc/test-list-free merge-tree replay (scripts in /tmp/claude-161238/ps). Splits done: fe38, b5b3, 996b, 60bb, f4af
(doc), b347 (borrowed-op hunk), 738a (restore hunk). Skipped: 2e06/29fc KV (signature change, caller in 19), db2c/a300 config
(its fields and authority hunks cover 21-23 features), 3c28 (rewrites authority/client/protocol code of 20-24), e3542 C++
(d0a8 rewrites the allocator on 278a's API; renaming early would rewrite 278a and its Python callers), 4a53 rule squash.
Ported to main (manual resolution): 3, 6 (l0_b200), 8, 9, 11, 12 tests; FT commits re-resolved on top: 0758, 380e, a886, 278a.
Known: 01ff/ada8 (docs) sit in 19 and describe hang/fence timeouts before 22/23 land, as the plan placed them.
Checks: pre-commit clean on all 82 commits; pyflakes finds no undefined names in any commit's changed files; tree diff vs wideep-ft-dev empty.

## prmain/08-adp-gate-quiet (c7740d7e83 on upstream/main 0829689c52; 2026-10-22 07:55 PT)
Main-side attention-DP scheduler fix, user-approved as in scope. The same change is on v5 as 2e8d6d7da1, which uses
FT's live-rank count. Main uses the gathered rank count `len(responses_list)`.
- What it does: after a burst of arrivals, with free context slots on every rank and an empty waiting queue, the gate
  releases a held context after 4 quiet decisions instead of waiting out timeout_iters.
- Evidence: GATEAB (CLAIMS 166) shows 0 timeout releases at 32/GPU and a 4-20x cut of the 8/GPU bistable mode. GUARD2
  (CLAIMS 168) shows no regression at 128/512 (whole-window -0.18/-0.82 %, client +0.42/+0.33 %, n=4).
- Residual, for the PR description: a straggler cohort that formed mid-wave can persist, at a lower rate. In one 8/GPU
  cell, 2.7 timeout releases per wave remained.
- Tests: 6 in TestAdpBalanceReleasesAfterBurst. They ran on v5 (35 Adp tests passed in GUARD2's gate) but not yet on a
  main wheel. Never open the PR; the user decides.

## Upstream fix (separate from FT): memory clobber on the A2A fence asm (2026-10-22)
Removed from v5 (v5-zero c0b36968a4) so FT-off kernels stay byte-identical to main; send to main as its own [None][fix].
- Where (upstream/main 0829689c52): cpp/tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.cu:602/604 (dispatch,
  before the completion-flag publish) and :1709/1711 (combine, after the completion-flag waits).
- Why: `asm volatile` without a `"memory"` clobber orders only against other volatile asm. The compiler may still move
  plain loads/stores across the fence: payload stores could sink past fence.release (peer sees the flag before the data),
  and combine's payload loads could hoist above fence.acquire. The fabric fence at :1907 already carries the clobber.
- Diff:
```diff
-            asm volatile("fence.release.sys;");
+            asm volatile("fence.release.sys;" ::: "memory");
 #else
-            asm volatile("fence.acq_rel.sys;");
+            asm volatile("fence.acq_rel.sys;" ::: "memory");
@@ combine
-        asm volatile("fence.acquire.sys;");
+        asm volatile("fence.acquire.sys;" ::: "memory");
 #else
-        asm volatile("fence.acq_rel.sys;");
+        asm volatile("fence.acq_rel.sys;" ::: "memory");
```

## Reviewer-facing changes to main's MoE all-to-all kernel interface (v5-zero; FT off runs these too)
1. Kernel-argument structs: active_rank_mask passed by device pointer (was by-value [4]); new fields expert_to_slot,
   num_routed_experts, abort. FT-off's parameter block differs in size and layout.
2. FLAG_VAL slot reserved as 8 bytes (released round), shifting later workspace offsets by 4 bytes in every mode.
3. FT-only branches compiled into shared kernels (released check, abort poll, combine aborted path; constant-folded when FT
   is off) and a USE_EXPERT_TO_SLOT template flag doubling the dispatch instantiations.
4. Op schema: new optional arguments abort_state, expert_to_slot, barrier=True; mask as device pointer.
Decision rule (monitor, 11:46): C30 (VAL-P, main wheel vs our FT off, same nodes, ABBA at 32/128). FT-off - main within
noise (<= 0.05 ms trim / <= 0.4 %) -> keep the shared kernels; otherwise split the FT entry points.
