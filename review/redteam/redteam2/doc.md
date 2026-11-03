# DESIGN_DOC.md fact-check (red team 2, 2026-10-20 ~22:40 PT)

Sources: code ft-src2 3bb877e24f (/tmp/claude-161238/wt-ftsrc2, vs upstream/main), CLAIMS.md (#n = row), runs/,
HANDOFF.md tail, NOT_CLEAN.md, and the in-tree guide docs/source/features/wide-ep-fault-tolerance.md ("guide").
"Lnn" = DESIGN_DOC.md line. Code paths are relative to wt-ftsrc2.

## 1. False statements

1. **L216 "the offline layout's 320 slots push the routing kernel into its 512-slot tier (~0.3%)".** Wrong layout.
   Offline EPLB is already 288 slots with FT off (runs/STD-8110144/std_layout.txt), which is past 256 and so
   already in the 512 tier (RoutingCustomKernels.cuh:1538, tiers 256 then 512). Only **w/o EPLB** jumps tiers
   (256 -> 320). ~0.3% has no CLAIMS row; NOT_CLEAN B1 says +39 us.
   Fix: "Without EPLB, FT's 320 slots (256 + 64 ghost) move the routing kernel from its 256-slot tier to its 512-slot
   tier (+39 us/step, <row>); a 384 tier is an upstream follow-up."
2. **L221 "losing the recovery authority while serving ... end[s] the deployment".** False. Serving continues and
   /health returns 200 `unprotected`; only a later rank loss ends the job (fault_tolerance_status.py:24-39; guide
   L152-153). The integration test test_authority_lost_while_serving (test_peer_loss_recovery.py:496) asserts
   exactly this. Fix: "Losing the authority leaves the deployment serving but unprotected (/health 200
   `unprotected`); a later rank loss then ends the job. Nothing restarts the authority."
3. **L221 "losing both ring copies of an expert".** Each expert has exactly one ring copy, on the previous node
   (eplb_recovery.py:106-115). Fix: "losing a node together with the node that holds its host copies, or a later
   loss on a node whose copy node is already gone".
4. **L91-92 "Without online EPLB: each node keeps the next node's experts".** False for offline EPLB. Ring copies
   are used only for a static placement with one slot per expert (moe_load_balancer.py:1077-1082). Any placement
   with replicas, offline included, keeps every expert on every node (moe_load_balancer.py:1062-1063; guide
   L22-24). Fix: "With replicas (online or offline EPLB): every node keeps every expert. Without replicas
   (256 static slots): each node keeps the next node's experts."
5. **L103-104 "With online EPLB, if ghost slots run out ... take slots of experts that still have a second copy".**
   This applies to every placement with replicas, offline included (eplb_recovery.py:24-29, 50-53). Fix: "With
   replicas (online or offline EPLB), ...".
6. **L223-225 "Narrow wrong-output window ... closing it needs the all-to-all kernels to withhold completion".**
   Probably stale. commit_step now also rejects a step whose MoE abort record is set (peer_loss_recovery.py:683
   `moe_execution_state.aborted()`), and its docstring (L663-669) argues both survivor decisions are correct.
   This comes from commits 8d49b0bf80 and beacc41778 (abort record, fixed source slots). NOT_CLEAN C2
   predates them. Fix: drop the bullet, or have the authors state the residual window and the test that covers it.
7. **L145-146 /health.** The code has six states (fault_tolerance_status.py:28-39): healthy, degraded and
   unprotected return 200; recovering, terminal and unknown return 503. "so orchestrators don't restart a job
   that is only recovering (recovering, 503)" is backwards: a liveness probe that fails on 503 *does* restart it.
   The 503 is there for load balancers. Fix: "200 with healthy/degraded/unprotected; 503 with recovering
   (~1 s, at most authority_timeout_s 20 s), terminal or unknown. Load balancers stop routing on 503; liveness
   probes must tolerate 503 for authority_timeout_s."
8. **L169 `figures/draft/fig6_recovery.png` does not exist.** figures/draft has only fig4_throughput and
   fig5_cost. Add the figure or remove the reference.

## 2. Numbers with no source or from a superseded build (no line in the doc cites a CLAIMS row)

9. **L8 "regroup (~1 s)"** holds only for online EPLB: online 0.92-1.31 s (#121, n=3, build aae00e6c65); w/o
   EPLB rank 1.25-1.79 s, node 2.02-2.64 s (#119, n=3; ghost mapping 0.95 s of it); final build 0.94-2.69 s (#135,
   n=1). Fix: "~1-3 s (#119/#121), from the kill to the first committed step".
10. **L48 / L130 "frozen rank ... detected in ~5 s".** This covers only a whole-process freeze (SIGSTOP), caught
    by Open MPI's detector. It needs the user-set launch flags `--mca mpi_ft_detector true ...
    mpi_ft_detector_timeout 5` (guide L54-56). Measured: detection 4.2-4.8 s, serving again 5.5-6.8 s
    (#119/#121). A rank that hangs while alive (GPU hang, deadlock, GIL or host stall) is recovered only with
    `hang_detection`, which defaults to **False** (recovery/config.py:76-80). With it on, recovery takes about
    5 + 30 + fence s, 36.7 s measured (guide L174). Otherwise only the 300 s executor hang detector ends the job.
    A 45 s host or GIL stall was not recovered (#97). Fix: "A frozen process: detected in ~5 s
    (mpi_ft_detector_timeout), serving again in ~6-7 s. A rank that hangs while alive needs hang_detection
    (default off), ~35-40 s."
11. **L154 "preliminary (n = 3 per arm, recovery n = 1)".** n=3 is from the superseded build 183d390691
    (#132/#133). The final build has n=1 w/o EPLB and n=2 online (#134/#135, A10). The reviewed tree 3bb877e24f
    has no healthy-cost or kill data at all. Fix: state the build and the CLAIMS row behind each plotted point,
    or hold Figure 5.
12. **L165 "empty-slot fallback ~1.4 GiB per GPU".** No CLAIMS row. 1 slot x 58 layers x 24.77 MB (#84) =
    1.44 GB = 1.34 GiB. Fix: "~1.3 GiB (1.44 GB) per GPU" + source.
13. **L213-215 "host collectives take ~2x as long as with UCX".** No CLAIMS row. The only measurement, #98, has
    ob1/TCP vs UCX at -0.25% step time (CI -2.23..1.74) at 128/GPU. Cite a row, or say the step cost was within
    noise (#98).
14. **L180-181 "in simulation it roughly doubled peak expert load".** No CLAIMS row; it appears only in an
    analysis note. Cite the file and the setup.
15. **L121 "~64 bytes" routing row.** Unsourced. Plausibly top-8 x (int32 slot + fp32 scale). Say how it is derived.
16. **L13-16 Dynamo numbers / ref [1].** Cannot be checked here. Ref [2] (TENT) is never cited in the text:
    cite it or drop it.

## 3. Over-claims

17. **L77-78 "no memory and no extra GEMM work" + L95-96 "FT picks the same kernels as FT off".** The GEMM
    autotuner does key on loaded slots, but FT is not kernel-neutral: the w/o-EPLB routing tier changes (item 1);
    FT requires and switches on the load-balancer path that FT-off w/o EPLB never runs (peer_loss_recovery.py:405,
    B2); online ghost slots hold 13.09 GiB of GDRCopy BAR1 mappings on rank 0 (KILL-FINAL-ON-A-8102532). Fix: "no device memory and no GEMM work while healthy", and list these three costs.
18. **L121-122 "new membership starts only once no failed process can still write".** This contradicts L209-210:
    a kill does not guarantee in-flight NVLink writes have stopped. Fix: "... once every failed process is killed;
    in-flight writes are contained by the mailboxes, not by the kill."
19. **L49 "Anything else ... fails closed".** Not guaranteed: host/PRRTE death "depends on Open MPI's
    runtime" (guide L131-133, C5); a live hang with hang_detection off ends only via the 300 s executor detector;
    silent wrong results are never detected (guide L170-172). Fix: "Anything else ends the job as without FT, possibly only after the 300 s hang timeout; silent corruption
    is not detected."
20. **L28 / L143-144 "rerun or end with a retryable error".** Incomplete: a request a second loss interrupts always fails
    (retryable); a new request strictly pinned to a lost ADP rank gets non-retryable HTTP 400 (guide L7-8, L160-163).
21. **L203 "an agreement that survives a second loss mid-recovery".** Only before install; a loss during install
    fails closed (L142). The R1 evidence is n=1 per load on an older build (#72). Add "before install".
22. **L222 "losses of rank 0 or during install are designed but not yet tested on GPUs".** Rank 0 is never
    recoverable by design (guide L19). Fix: "rank 0 loss and a loss during install fail closed by design;
    neither path has run on GPUs."
23. **L153 "same build and nodes in alternating rounds".** Arms also differ in transport (UCX vs ULFM
    ob1/TCP + detector, C4); share pinned/transplanted tactics (A4); MTP acceptance is forced (A2); random 1k/1k only
    (A3); max_tokens fixed 18688 (A5); backlog 16384, kill graph sizes 136/148 (A8); online has MTP, others not (A1);
    FT-off vs main not re-measured on the rebased build (A15). Put this list in Setup.
24. **L111-114 "In a packed receive buffer ... FT instead gives each sender a fixed mailbox".** Upstream
    NVLinkOneSided already has per-sender regions [ep_size, max_tokens_per_rank, ...] (upstream
    moeAlltoAllOp.cpp:504, nvlink_one_sided.py:1048). Upstream sets their stride per step
    (runtime_max_tokens_per_rank); FT fixes it. Fix: "Upstream gives each sender a region whose stride follows
    that step's largest sender; FT fixes the stride, so ...".

## 4. Names, config, requirements (none stale; mostly missing)

25. **No config, env var or module is named anywhere.** Add a Configuration section: env TLLM_FAULT_TOLERANCE_MODE /
    _AUTHORITY / _AUTHORITY_KEY / _CONFIG; YAML extra_slots, failure_domain `node`, recovery_timeout_s 30,
    authority_timeout_s 20, hang_detection False, hang_confirm_s 30, heartbeat_s 1, heartbeat_timeout_s 5
    (recovery/config.py:33-97); env + YAML, not LlmArgs (C6); modules executor/recovery/{config,authority,
    authority_server,client,protocol,hang_monitor,mpi_exit}.py, peer_loss_recovery.py, eplb_recovery.py; how to
    launch the authority. No stale `failure_domains` path appears in the doc.
26. **L40 "Open MPI >= 5.0 ... mpi4py >= 4".** Neither the code nor the guide sets a version floor; the guide says
    "built with ULFM (ompi_info | grep ftmpi)". Missing: required flags `--mca pml ob1 --mca btl self,tcp
    --mca coll ^hcoll,ucc` + mpi_ft_detector flags; container mpi4py 3.1.5 -> 4.1.2 for everyone
    (install_mpi4py.sh); CI image has no ULFM so FT tests skip unless TLLM_REQUIRE_ULFM=1 (test_peer_loss_recovery.py
    :41-81, C7); an auto-sized KV host tier stalls startup >5 s and the detector kills the rank (guide L64-67).
27. **L32-40 "supports".** Change to "requires" for KV Cache Manager V2 and an EPLB load_balancer config. FT refuses
    to start (peer_loss_recovery.py:369-406) with guided decoding, beam search, KV transceiver (disagg), KV connector,
    PEFT, sleep/wakeup, LM-head TP in ADP, SWA scratch reuse, KV pool rebalance, a non-default ADP router, or
    hang_detection_timeout <= 90 s.
28. **L211-212 CFT** goes further than "not supported". With CUDA >= 13.4 and driver >= 615, CFT is selected
    automatically and FT binding raises "Bound execution requires the fence path; CFT is enabled"
    (nvlink_one_sided.py:750-751). Fix: "set TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT=0; startup fails otherwise".
    Confirm on a 615 node.

## 5. Omissions a reviewer will ask about

29. **FT-off changes**: mpi4py 4.1.2 in the container; Cutlass runMoe and moeAlltoAllKernels signature/argument
    changes (C2/P6 residual, FT-off cost unmeasured); SSE stream-error handling in openai_server.py; ADP router pin
    handling. FT-off vs main is not re-measured on the rebased build (A15).
30. **Host memory**: full copies are 368 GB of /dev/shm per node (#84), a new cost for offline EPLB + FT (FT off
    keeps none); ring copies (no EPLB) ~1/N of that; ghost mapping and prefault add startup time.
31. **Shutdown**: 43-45 s with FT vs 16.7 s without (#135). This exceeds Kubernetes' 30 s default grace.
32. **Recovery breakdown**: ghost mapping dominates. 0.65-1.45 s on the final build (HANDOFF 17:55); 0.95 of
    1.79 s in #119; bound by bytes mapped per node.
33. **Post-loss throughput** is TBD, and the old est_tok_s method is invalid (A13).
    **Accuracy after recovery is omitted**: GSM8K on 31 survivors, -1.44/-0.61 pt, within 2 SE, PASS (#124, older
    build). Add it with that caveat.
34. **Tests**: 12 integration cases on V3-Lite EP4, one node; last pass 12/12 on 547a6aa7ed
    (INTEGRATION-...-8101557/test.log); the final tree has no integration run (rc1 skipped it). DEP32 evidence is
    harness runs, not CI.
35. **Scope**: tested only on R1 NVFP4, DEP32, GB200 (C9). Of the speculative-decoding methods, only one-model MTP
    with a linear draft is tested; the others warn at startup.
36. **Online cost at 128/GPU**: +~300 us from the compact-copy revert (#136). The R3 port is in the tree but
    unmeasured, so do not state an online cost yet.
