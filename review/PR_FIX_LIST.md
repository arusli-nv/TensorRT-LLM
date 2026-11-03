# PR-prep fix list (late findings on the measured build 793ffd906d; follow-up layer, not in the measured build)
Started 23:15 PT 2026-10-19. Sources: /tmp/claude-161238/redteam/{maintainer,fixes,harness,scenarios}.md (design-doc session) and review/RED_TEAM_QUEUE.md.

## Behaviour (follow-up commits, each flagged in PR_PROPOSAL.md)
- Detect-only FT mode arms the watchdog during CUDA graph capture (wide_ep_ft.py:60-65, nvlink_one_sided.py:1219,1374): fix it or re-reject capture.
- A second loss that is a GPU error (ECC/Xid) rather than a process death, during a recovery, ends the job: survives_own_loss is False during recovery (peer_loss_recovery.py ~528/557).
- Authority restart while serving: only documented ("restart only with the job"). Workers never re-register, so /health reports 503 starting and every later loss is terminal. Fix: re-register, or report unprotected.
- The native heartbeat carries the last committed iteration, not GPU progress: a GIL-holding stall >30 s with hang detection on gets the rank fenced. Consider a GPU-side progress counter.
- A few-ms unwatched window between Commit and the peer watch restarting; _GATHER_HISTORY=8 with no startup check.
- The poison-request quarantine has a gap without attention DP; a per-step Python loop runs on ranks that never serve rerun requests (fixes.md items 2, 4).
- The resumed line's failed count F has no owner split (lost-rank vs interrupted twice vs other), so the kill verdict can only bound F by the lost ranks' load.
- FT-off changes to gate behind FT or call out:
  - the EPLB shm name is randomized;
  - stale segments now raise;
  - the reservation and pre-map thread;
  - the SSE error event + [DONE] can end twice;
  - the C++ local-rank-active check was removed;
  - ADP unservable gives 503 where it should be 400.
- CutlassMoeFCRunnerInterface::runMoe signature: keep the old one working.

## Decide in PR descriptions
- hang_detection defaults to False (config.py:68): should it default on?
- FT config is env/YAML (experimental), with an LlmArgs follow-up PR.
- mpi4py upgrade as its own infra PR with the image tag bump; ULFM tests skip in CI (no ULFM image), and the GB200 cases are post-merge only.

## GPU (only if slots are free after every final pack)
- Kill a rank during an online-EPLB weight update at R1, with an accuracy/exact check; a failure there would be silent.
- (23:21 PT) A dispatch source released as aborted has its tokens dropped in that step. If this rank's own abort lands only after the step commits, that step's outputs could be wrong; it is a narrow race, needing a lost rank that already delivered its data for the step. This exposure predates tonight (the count used to be zeroed). Follow-up: when a kernel releases a source, record that the step was aborted (status), and have commit_step refuse such a step.
- (23:28 PT, generalizes the entry above) A rank that observes its abort stops WAITING but keeps PUBLISHING rows and flags. A peer that has not yet seen its own abort can complete and commit a step using that rank's partial expert work, with no local signal: silent wrong output in a few-ms window. The fix is sender-side: after a kernel observes an abort, it withholds (or taints) its completion flags so receivers block until their own abort, and commit_step refuses a tainted step. This is a kernel protocol change, needs its own tests, and is stated as a limitation in the design doc.
- (00:25 PT, REGRESSION, fail-closed) A loss during a ring-copy restore can end the job: after f7020e68ae every survivor quarantines the ranges the newly lost rank sourced in every plan proposed earlier in the recovery, and on a small workspace (EP4 V3-Lite, 10.5 MB rows) that plus the lost ranks' write ranges leaves [0, 0] bytes per half. integration test_rank_lost_during_recovery on build-final-aae00e6c65 (passed at b18f0b64ed). Fix candidates: quarantine only the plans actually staged (agreed via the authority), or reuse those ranges once the fence has completed. Not covered by the finals (no loss during recovery there).
- (01:18 PT, perf) Rank 0 maps ghost memory (_restore_placements) before it services the authority's Fence (_wait_for_proposal), so fence and propose wait on the mapping: 0.65-0.96 s of a ~1.8 s static-layout recovery. Service Fence first, or map while the authority fences; that keeps "memory mapped before Ready" and should roughly halve recovery. Why the mapping itself takes ~0.9 s is TBD (see the rank-0 log timestamps).
- (02:05 PT, PERF, user-visible) Deployment throughput after a recovery is 20-35% below live/32 on the static layout (kill OFF rank5 .63-.74 vs .969; node1 .64-.92 vs .875) and ~5-10% below on online EPLB (kill ON rank5-3 .89, twokill .86-.91 vs .938). Measured from client per-request timings in short windows (8-25 s). Rank 0's step time rose 18.5 -> 24 ms. Hypothesis: on the static layout, restored experts in ghost slots give 8 ranks 9 experts, and the MoE step is bound by the most-loaded rank; no rebalancing follows. Follow-ups: measure steady state with a longer post-recovery load; place restored experts on the least-loaded ranks, or let EPLB re-plan after recovery.
- (02:09 PT, EXPLAINED) The post-recovery throughput drop on the static layout (0.63-0.74x) is CUDA-graph batch padding, not FT. The closed-loop client spreads 3972 requests over 31 ranks (128.1 each), so the per-rank batch goes 128 -> 129 and ADP pads every rank to the next captured size, 192: step 18.0 -> 24.3 ms. At equal rows the step is +4% (expert imbalance). Node loss kept 128/rank and stayed flat. Follow-ups: add captured sizes just above 128 (136/144) for wide-EP configs, and count only real rows in the harness's KILL_FINAL tok/s fields (they count padded rows).
- (02:09 PT, DOC NOTE for FT users) A loss of K of N ranks raises the per-rank load by N/(N-K). With attention-DP CUDA graphs, capture a batch size a few percent above the healthy per-rank maximum; otherwise every rank pads to the next captured size (128 -> 192 here) and loses about 25% of throughput on top of the lost capacity.
- (07:25 PT 2026-10-20, ROBUSTNESS, FT startup) A false positive from the ULFM failure detector killed an FT startup (GHOST-COST-8088395 tune-ft-ghost320): the KV host tier's auto sizing pinned 70 GiB per rank (about 280 GiB per node). The resulting memory-pressure stall made rank 9 miss heartbeats for more than mpi_ft_detector_timeout (5 s); it was declared dead and aborted itself, and the startup watch then ended the job as designed. The passing runs pinned 53-58 GiB per rank and still saw 23-35% memory-pressure peaks. Fix: detector timeout 30 s (harness and docs; note the hang-detection trade-off), and cap or chunk the auto host-tier pinning (kv_cache_manager_v2.py ~1677).
- (07:25 PT, BUG, FT exit) With FT on, a graceful SIGTERM stop hangs in MPI_Finalize under ULFM mpirun, both clean and after a loss: SHUTDOWN-R1-8089163 ft-clean/ft-loss, where no 'MGMN ... exit code' line appears and _leave_degraded_world's Finalize never returns. FT off exits correctly (rc 143 = SIGTERM, workers rc 0); the harness's 'expect rc 0' is wrong for SIGTERM. Next: py-spy/gdb stacks at the graceful timeout; a noft-ulfm graceful case to separate the ULFM launcher from FT code; a bounded Finalize on the clean path.

## Limitation (2026-10-20 08:15 PT): bounded MPI_Finalize under FT is a mitigation
- Under ULFM, MPI_Finalize hangs after a clean FT-on stop and after a loss (SHUTDOWN-R1 #125). The root cause is not pinned: either the detector thread's finalize wait (ompi comm_ft_detector.c:208-225) or the job-wide PMIx fence that includes the dead rank.
- fix-cost a53d73604b + c01f782ed2: under peer-loss recovery each process gives Finalize 30 s, then os._exit(status) after flushing the logs. The timeout prints "MPI_Finalize (<process end|degraded world>) did not end within 30 s". The SHUTDOWN verdict reports the shutdown duration, so a 30 s stall stays visible.
- To pin the cause: take py-spy/gdb stacks of a hung rank, and run one ft-clean case with WIDEEP_NO_DETECTOR=1. Write this up in the doc/PR as a limitation.

## Limitation (2026-10-20): CFT path unmeasured
- This cluster's driver 580.126.20 is below 615, so CFT counted writes are disabled on both arms: "CFT counted writes disabled" in main's rank logs, and the harness also forces FORCE_CFT=0. On drivers >= 615, the cost of FT on vs a CFT-enabled FT off, and the CFT kernels with R1/R2 prefetch, rank mask and expert_to_slot, have no GPU coverage or cost data. State this in the PR.

## Limitation (2026-10-20): ULFM host-collective transport cost
- ULFM needs pml ob1. On these nodes the usable BTL is tcp on eth0/eth1, which are 200 Gb/s RoCE netdevs; the NDR IB ports have no IP netdev (HOSTCOLL-8028647 nics-*.txt).
- Measured 32-rank latency (T2-UCT-8017010), ob1/tcp vs UCX: allgather 434 vs 211 us, gather_1kb 466 vs 245 us, bcast_1kb 122 vs 43 us. That is per-message TCP latency against RDMA, not link speed.
- Rejected levers:
  - btl sm hangs MPI_Comm_shrink at 32 ranks (DECISIONS W1 T1, T1-SM-8013748/-8014143);
  - btl uct hangs the server (T2);
  - a FT-only fixed-size gather was dropped as an inelegant protocol special case (user, 2026-10-20).
- It is visible only when rank 0's host step reaches the device step: online EPLB is near that limit at 32/GPU (host 15.66 vs device 15.80 ms); static is not (12.69 vs 12.77). Report the online 128/512 cost from the full benchmark as the ULFM transport cost.

## Coverage notes (2026-10-20, tuning-key red-team)
- Tuning-key normalization (fix-slim 13b0c3a72e) counts only trailing empty slots per rank. Hand-written static placements with middle or uneven empty slots fall back to 10-slot keys (perf only; deployed configs give E=2 / E=1 uniformly).
- DeepEP direct / count-native paths key with E=0 (not the FT path: FT uses NVLinkOneSided). Nothing explicitly refuses DeepEP with FT on: state it, or add a check.
- 17 odd token counts (2^j-15 / 2^j-31, E=2) bucket differently from FT off; T is even on the NVLinkOneSided paths, so unreachable there.

## Limitations (2026-10-20, Option 2 review)
- NUMA path (GPU memory is a NUMA node): ghost and resident slot memory are HOST_NUMA allocations that the GEMMs read every step. This is untested on NUMA-mode hardware (this cluster is CDMM). The start-up trial falls back to empty if the driver rejects it.
- GDR path: the trial pins one 2 MiB chunk. allocate() then pins every resident run. A later pin failure (e.g. BAR1 exhausted) fails startup, or a restore, hard, as main's gdrCudaMalloc does for migrate_to_host_accessible. Rank 0 logs the host-mapped bytes at startup.
- Restore pins pieces one at a time, under the GhostSlotMemory mutex and the gdr lock; the kill smoke reports this time.

## Environment note (2026-10-20): test_mpi_session.py in this container
- The same 9 tests fail on main's wheel (build-pr0-fc52f0adee) and ours (build-final-183d390691), run back to back on one node in job 8097161 with full checkouts:
  - test_remote_mpi_session[6 params]
  - test_llmapi_launch_multiple_tasks[2]
  - test_mpi_pool_session_does_not_resurrect_unset_forwarded_env (timeout)
- They spawn nested MPI/launch processes, which this Slurm container cannot run. Every FT and finalize test passes. CI is expected to run them properly.

## Pending decision (2026-10-20): 384-expert routing tier
- Static FT (320 slots) runs the routing kernel's 512-expert tier where FT off (256) runs the 256 tier: +39 us/step (~0.3%), measured in 8099338-B. Online is unaffected: 288 and 320 slots both run the 512 tier.
- The fix is one shared line, Tier<384,8> in PolicyTraits<NoOpPreprocess, SoftmaxPostprocess>. It is kept only if a microbench shows no regression for any shape, including main's Renormalize models with 257-384 experts at 1 token through decode sizes. Otherwise it is dropped and listed as a known ~0.3% static cost, with the 384 tier as an upstream follow-up.
- Upstream suggestion, not in scope: slot-sticky EPLB placement, so that updates don't re-copy experts already resident on the rank. That causes the step stalls in every online arm.
- 2026-10-20 15:00 decision: the 384 tier is DROPPED. At 32/GPU the FT path routes about 32-64 tokens per rank through the 256-thread cluster kernel, which cannot take a 384 tier (static_assert MaxNumExperts <= NumThreads or a multiple). Fixing it would need a policy-specific dispatch change (rejected as patchy). Known static cost of about +39 us/step (~0.3%): the 320-slot layout runs the 512-expert routing tier. Upstream follow-up: a 384 tier in the routing kernels. The work is kept on branch fix-final3-tier (identity test plus microbench).
- 2026-10-20 (user-approved): fix B "keep resident experts in their slots" (branch fix-sticky). This is a separate upstream PR for main's shared EPLB, not part of the FT PRs. It keeps main's rank assignment and relabels slots within each rank, so updates copy only experts that moved between ranks. It is kept only if FT off and FT on at 32/GPU (n>=3) show fewer stalls and less throughput loss; otherwise it is undone.
