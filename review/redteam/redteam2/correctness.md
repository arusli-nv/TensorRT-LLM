WideEP FT correctness review (fresh eyes), 2026-10-20 ~22:10 PT. Read-only, static: nothing built or run.
Tree: /tmp/claude-161238/wt-ftsrc2 (ft-src2 @ 3bb877e24f); paths relative to it, lines at that commit.
Labels: CONFIRMED = traced in code and re-checked by me. PLAUSIBLE = mechanism traced, a precondition unverified or not run.

SUMMARY
- No path found where a request is silently dropped (client hangs forever) or answered twice, and none that hangs
  unbounded: every failure ends in authority Abort, worker RecoveryTimeout, or the hang detector, each hard-killing the job.
- Most serious: two step-discard bugs in the overlap loop (S1, S2). Both bite only under the standard recipe
  (overlap + MTP + V2 KV); no test covers either.
- The protocol fails closed, but more often than the docs imply (P1-P3).
- Nothing checks restored weights on the device or byte for byte. The only output check after a restore is GSM8K in
  the out-of-tree harness (section 4).

== 1. RECOVERY PROTOCOL: RACE WINDOWS AND MID-PHASE DEATHS ==
P1 MEDIUM, CONFIRMED: peer_loss_recovery.py:525-532, 1010-1044.
- A rank lost after Commit but before the survivors resume ends the job instead of starting a new recovery.
- `_recover` runs inside run()'s `except HostCollectiveInterrupted`. The serving watch is stopped at :889 and re-armed
  only at :1044. The post-Commit allgather (:1019) and broadcast (:1038) are interruptible, so a loss there escapes the
  handler and the job is hard-killed. The authority stays RECOVERED and never sees a Stopped.
- Scenario: a second rank dies within one allgather of Commit, or a rank dies between sending Installed and resuming.
- The docs say a loss after Install ends the job, so this is fail-closed, but they don't say the window runs past Commit.
P2 MEDIUM, CONFIRMED: authority_server.py:120-135 (`_send`: ROUTER put_nowait; a ZMQError is only logged, never resent).
- How a message gets lost: a route reconnecting, EAGAIN, or a half-open TCP inside the 10 s ZMTP heartbeat
  (client.py:47-51). The authority's state moves on anyway.
- Lost Commit: the other survivors resume; the one that missed it waits recovery_timeout_s (30 s), raises
  RecoveryTimeout and kills the job. For up to 30 s the survivors are split. No corruption: the next ADP allgather blocks.
- Workers cannot resend (a duplicate Stopped/Installed is a violation, authority.py:595-598). Fix: resend an
  idempotent Commit until acked, or let a worker query "state of generation g".
P3 MEDIUM, PLAUSIBLE: peer_loss_recovery.py:986-990, client.py:226-233, `_shrink_again`.
- A further loss just as Install goes out: survivor A already has Install queued and installs. Survivor B sees
  further_loss first and blocks in comm.Shrink() on the old communicator A has left, so it cannot read the Abort.
- Only the hang detector bounds this (>= shrink + 2x recovery, at least 90 s), not the authority's 20 s.
- Same for a peer that hangs without dying inside barrier/broadcast_among_survivors (:1228, :1327): hang
  monitoring is off during recovery (authority.py:266, 292). Outcome: the job ends after 90 s or more, no corruption.
P4 LOW, CONFIRMED: authority.py:434, 522-543, worker :953-954. A replan does not extend the deadline. A second loss at t=18 s of the 20 s window leaves ~2 s, so it almost always
  aborts. test_rank_lost_during_recovery covers only an early second loss.
P5 LOW, CONFIRMED: authority.py:120, 207-210. The authority persists no state. A restarted one stays in REGISTERING and drops every worker message, so the next
  loss ends the job. Mitigations: /health reports `unprotected`; SIGTERM mid-recovery sends Abort.
P6 LOW:
- CONFIRMED: `_command_fence` holds one entry (authority.py:253-265, 325). A second rank confirmed hung within the
  grace period drops the first rank's fence, and the monitor never re-confirms that rank.
- PLAUSIBLE: Fence has no generation (hang_evidence.py:241-271) and kills by rank, not pid. A rank that was only slow
  becomes a further loss; after Install that is P1 or P3.
Checked OK:
- Stale Propose skipped (:932-937); stale Ready/Stopped dropped (authority.py:363-365); queued messages discarded
  before Stopped (:924); Install/Commit generation-checked; sequence numbers stop replays; Shrink gives one failed
  set; authority_timeout < recovery_timeout enforced; a loss between Install and Installed aborts within 20 s.
- Online EPLB, when one survivor commits step N and another discards it: quiesce, rank 0's placement as the base,
  and reset_update_schedule reconcile them (:907-913, 1004). The load balancer's own iter_id is not realigned
  (harmless today).

== 2. REQUEST ACCOUNTING AND KV REVERT ==
S1 MEDIUM-HIGH: py_executor.py discard_interrupted_steps (~:8686-8700), scheduler_v2.py:1852.
- CONFIRMED (ordering): the discard pauses recompute victims first (reset_for_recompute: position 0, KV freed). Only
  then does it restore each step's (context_current_position, state, draft) snapshot.
- PLAUSIBLE (precondition): a victim of step N+1 can be a request in the still-uncommitted step N. The overlap loop
  records N+1's victims in `_recompute_unpaused` (:5849); `_is_recompute_pause_candidate` accepts started requests;
  `_add_inflight_ids` is PP only.
- Scenario: V runs a non-first context chunk in N (or ends prefill over a reused prefix); scheduling N+1 picks V as
  a victim and frees its KV; commit_step(N) raises; the discard pauses V (pos 0), then restores N's snapshot (pos > 0,
  CONTEXT_INIT); revert_prepare_resources(N) sees "not first chunk" + empty kv_cache_map and does nothing.
- Result: V re-runs with position > 0 and no prefix KV: an assertion, or silently wrong output for V.
- Fix: pause victims after the restore loop. Test: a discard whose victim is also in the older step.
S2 MEDIUM with MTP, CONFIRMED control flow / PLAUSIBLE effect: py_executor.py overlap loop ~:5995-6090.
- Order: commit_step(N) (:5995), _update_requests(N), _flush_pending_transfer_responses (:6032, a tp_gather every
  ADP iteration and the likeliest place to block next), _sample_async(N+1), then _process_previous_batch (:6090).
- A loss raised in the flush or the sampling goes to the discard, which sets previous_batch = None (:8711). Committed
  step N's V2 generation-resource update (kv_cache_manager_v2.py ~:6590-6650) then never runs.
- Effects: `_inflight_draft_lens` keeps a permanent one-entry lag; rejected-draft pages are not rewound (capacity
  leak per recovery, ValueError near max_seq_len).
  - With dynamic draft length or CUDA-graph padding (extend_capacity_for_tokens rewrites inflight[-1]), later
    rewinds use the wrong width; per :6612 they can free pages a step is still writing.
  - Harmless without spec-dec.
- Fix: the discard runs N's generation-resource update when N committed but was not processed.
S3 LOW, CONFIRMED: the discard does not restore set_exclude_last_generation_logits, py_last_context_chunk, or
generation requests' py_draft_tokens. Verification keeps the output exact; only acceptance drops.
S4 LOW (UX), CONFIRMED: openai_server.py _end_stream_on_error. A streaming request failed by recovery gets HTTP 200 + SSE `error` (peer_loss_retryable), even with zero tokens
  streamed. Clients that retry on status codes won't retry these.
S5 LOW (docs), CONFIRMED: wide-ep-fault-tolerance.md:264 says "Greedy outputs after a recovery match a server that
never used the lost rank's experts". Nothing tests this, and test_peer_loss_recovery.py:319-321 says exact tokens
change (reduction order). Reword.
Checked OK:
- Departed rank's requests (peer_loss_recovery.py:1494-1531): delivered nothing -> re-run from the prompt (same id,
  pin relaxed); streamed or interrupted before -> retryable error; cancel broadcast -> CANCELLED.
- "Delivered" = rank 0 processed a non-final response, so re-runs never resend tokens; responses are built only
  after commit (no token twice); queued/partially broadcast/never-received requests are re-admitted; gather
  redelivery is consistent; a second interruption without progress quarantines.
- KV revert: a first chunk (also over a reused prefix) frees and rewinds; later chunks shrink to the captured
  capacity; reverts run newest first; draft KV is reverted too.
- Reuse-commit deferred to commit_step (no garbage reuse blocks); _join_device_work before any free;
  py_decoding_iter only after commit; penalties/logits PP/guided/beam/disagg/PEFT refused.

== 3. OUTPUT CORRECTNESS AFTER A RESTORE ==
Guards:
- Ready carries the placement + copy-plan digest; the authority compares them (authority.py:550).
- Install -> Installed (all ranks) -> Commit -> survivor allgather before the first forward: no rank routes on the
  new placement while a peer is on the old one.
- Placement before mask (:1613-1617); rearm() before Installed; install_mask refuses a dirty abort record.
- commit_step syncs, then discards on peer_failed/aborted(), so an aborted (zeroed) combine is never committed.
- Fixed per-source slots + source_valid_counts + released marker: a released source contributes no rows.
  fence.acquire.sys runs before reads.
- C++ requires a surviving replica per expert (all layers); FP4/FP8 scales/alphas are slot weights (restored too);
  ring restores only into empty slots; mask/abort/table/slot VAs fixed (CUDA-graph safe).
R1 LOW-MEDIUM, PLAUSIBLE: moe_load_balancer.py:1407-1488 restore_copy_plan, :1490-1520 stage_restored_slots.
- Staging halves avoid only the dead ranks' a2a write ranges, so they sit inside survivors' dispatch/combine rows.
  rearm() resets flags, not payload bytes.
- Scenario: a source hangs (frozen, not dead) with a host-to-peer copy_ in flight; its bytes land after Commit in a
  live source's slot.
- Mitigated (sources sync before the barrier; dead contexts torn down; rank 0 kills failed ranks first, :1080), unbounded.
R2 LOW, CONFIRMED by design: moeAlltoAllKernels.cu:362-365.
- Routes to a masked rank, and table entries == num_experts, are dropped silently. Correct only while every table is
  updated before the mask (it is today).
- Cheap add: a debug counter for dropped valid routes, checked in commit_step.
R3 LOW, PLAUSIBLE: placement_digest (:1394) does not cover the restored bytes; no slot-vs-host-copy check.
- A wrong-source or partial copy goes undetected.
- Cheap add: a per-slot checksum exchanged in the existing staging barrier (ms over ~10 GB).
R4 LOW (latent), CONFIRMED: moeLoadBalancer.cpp:976-985 (cpp.md A2, still open). FT accepts empty host weights. An online-EPLB layer without them would read out of bounds at :1090.

== 4. ACCURACY CHECKS TODAY ==
In tree: tests/integration/defs/wide_ep_ft/test_peer_loss_recovery.py is a smoke test, not an accuracy check.
- EP4 V3-Lite, 4 "capital of X" prompts: expected substring, exact token count, a printable/no-repeat check, and
  healthy-prefix equality for survivors.
- Placement keeps 2 replicas of every expert (peer_loss_driver.py:44), so only the ghost/appended tests do a real restore.
- PLAUSIBLE: they run in no CI stage. They are post_merge in l0_gb200_multi_gpus.yml:102-113 and skip without ULFM
  unless TLLM_REQUIRE_ULFM=1, which no CI file sets; the image installs distro openmpi (install_base.sh:103).
Out of tree: harness/job.sh ACCURACY-KILL-ON/OFF (:3849).
- What it runs: lm_eval gsm8k, 5-shot, greedy, all 1319 questions through trtllm-serve. Rank 5 gets SIGKILL in a
  decode step under a warm load, then the eval runs on 31 survivors vs FT-off; PASS within 2 combined stderr.
- CLAIMS row 124 (job 8089163, build-final-aae00e6c65, R1 NVFP4 DEP32, n=1/arm): static 0.9386 / 0.9469 vs 0.9530;
  online 0.9454 vs 0.9454; FT on no kill 0.9484. A ~1.7 pt band cannot catch one wrong expert or per-request corruption.
- Gaps: (a) The eval starts after recovery, so re-runs, discarded steps and requests in flight across the kill are never
    graded. That is exactly where S1 and S2 live. The warm load checks only completion counts.
  (b) bench.py already implements --kill-during-eval (:405-429), but job.sh never schedules it.
  (c) --accuracy-max-retries 3 retries failed requests silently. The verdict counts them but does not fail on them.
Cheapest addition for tonight's finals (no code; one case in the ACCURACY-KILL group, reusing acc-ref-final):
  acc_go acc-killdur-final-1 && (export BENCH_CONFIG=dep32-final.yaml WIDEEP_VICTIM=5 TLLM_TEST_KILL_RANK=5 \
      TLLM_TEST_KILL_TRIGGER=$out/trigger-acc-killdur-final-1 TLLM_TEST_KILL_IN_DECODE=$out/trigger-acc-killdur-final-1 \
      TLLM_AUTOTUNER_CACHE_PATH=$out/autotuner/ft-dep32-final.json; unset TLLM_FAULT_TOLERANCE_EXTRA_SLOTS
    run_case acc-killdur-final-1 ft --mode accuracy --accuracy-max-retries 3 --kill-during-eval --kill-window-s 15,30)
  acc_verdict acc-killdur-final-1 dep32-final.yaml 1
- Budget ~12 min. dep32-final is the overlap+MTP recipe, where S1 and S2 apply.
- PASS: strict within 2 combined stderr of acc-ref-final; n=1319; rc 0; one resume without [5]; reruns > 0 and
  eval_after_kill_s < 0 (kill landed mid-eval); report `retried` (loss-failed requests answered on retry) separately.
- Sharper, nearly free: log_samples=True in accuracy_client.py, then diff per-question answers vs acc-ref-final.
  Compare the flip rate of questions in flight at kill_wall_s with the rest. This catches per-request corruption
  (S1) that the aggregate misses.
- Also run 2 reps of the static arm: at n=1, -1.44 pt is not separable from a small real loss.
Next in tree (not tonight): peer_loss_driver inflight mode with ~32 GSM8K questions per rank, killed mid-decode. Assert survivors equal the
  healthy round, and re-runs reach healthy accuracy within 1 question.
- Set TLLM_REQUIRE_ULFM=1 in the gb200 stage so these tests cannot skip silently.
