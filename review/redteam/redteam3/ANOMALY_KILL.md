# ANOMALY_KILL: why KILL-FINAL-OFF-A-8115497 rank5-3 took 2785 ms (vs 450 and 429 ms)

Sources: `$R=.../runs/KILL-FINAL-OFF-A-8115497/{rank5-1,rank5-2,rank5-3,node1-*}`: rank-*.log, authority.log, timeline,
hostpressure-*.log, result.json. Code: branch v2 (cef4670789): peer_loss_recovery.py `_recover`, moe_load_balancer.py
`stage_restored_slots`, communicator.py `PeerFailureWatch`. Harness: mvp/harness/bench.py `wait_resumed`, rank.sh.

## 1. Where the time went (rank 0 phases, ms after detect; authority stages)
| case | det->stop | stop->ready (r0) | ready->Install | installed->Commit | total |
|---|---|---|---|---|---|
| rank5-1 / rank5-2 | 26 / 25 | 401 / 382 | 3 / 3 | 1 / 0 | 450 / 429 |
| node1-1..3 | 25 | 1323-1371 | 4-7 | 0-1 | 1379-1428 |
| **rank5-3** | **985** | 555 | **719** | **507** | **2784** |
- The extra ~2.35 s sits in exactly three waits, each a point where every survivor waits for the slowest one:
  (a) detect->stopped = the ULFM `shrink()` among survivors; (b) agree = the authority waits for the last `Ready`;
  (c) install = it waits for the last `Installed`. Rank 0's own work was normal: ghost mapping 148 ms (vs 150),
  Install arrived at 526.093, the same ms the authority entered INSTALLING. The authority was not slow: it logs nothing
  between PREPARING and INSTALLING and sent Install/Commit the same ms it moved state.
- All 31 survivors behaved as one group: all were in `tp_gather` when detected, all logged "interrupted" before 01:48:44.000
  (<=165 ms after rank-0 detect), all `Stopped` reached the authority within 1.4 ms, and all resumed "2785 ms after".
  So one late survivor held up the shrink. It was not device work: `_join_device_work` runs after the shrink.
- Rank 0's staging, 394 ms vs 220 ms, is also a wait. In rank5-3, ghost+staged ~= 540 ms on 29 ranks (370 in rank5-2).
  **Rank 26 staged in 17.9 ms** (rank5-2 minimum: 112 ms). Each chunk ends in an MPI barrier, so rank 26 can only finish
  in 18 ms if everyone was already waiting for it. It began staging ~120-180 ms after its own ghost mapping ended (~525.01).
  Rank 0's +174 ms is therefore time spent waiting for rank 26.

## 2. The straggler: rank 26 (node 6, local rank 2), log-emit stall
- Rank 26 is the only survivor in rank5-1, -2 or -3 whose watch thread wrote "Peer failure watch fired" (01:48:43) and
  "a peer failed; aborting" (01:48:44) in different seconds. The only work between those two records is the first
  record's emit (write+flush of the rank log) plus `Event.set()`. So that log write (or the handler lock behind the main
  thread's "interrupted" write) took >=160 ms. This happened while the main thread was on its way into the shrink.
- Its late start in staging falls right after its "Mapped ... ghost" log line. Every slow wait (a-c) also comes
  right after a survivor's logger call: "interrupted" before shrink, "Staged" before Ready, "Restored" before Installed.
  The durations measured in code (Mapped, Staged, Restored 3.8 ms) all look normal because they are measured before the write.
- Rank logs are line-buffered files on Lustre (rank.sh: `stdbuf -oL ... > $run/rank-$rank.log`; its comment for
  TLLM_TEST_LOCAL_LOGS says "line-buffered logs on Lustre can block a rank"). During recovery,
  bench.py `wait_resumed` reads all 32 rank logs every 0.5 s from node 0. Those cross-client reads conflict with the
  writers' Lustre locks at exactly the moment the recovery logs. The stalls (~0.96/0.72/0.51 s) are spaced like that poll.
- Host IO PSI corroborates: of the 10 cases in job 8115497, io-PSI avg10 is >0 within +-12 s of detect only in rank5-3, and
  only on node 6 (0.54, ranks 24-27) and node 2 (0.36, ranks 8-11). Every other node in every other case reads 0.
  Caveat: earlier jobs (8084821-8102532) often show io-PSI >0 without stalls, so on its own PSI does not predict a stall.

## 3. Ruled out
- Host memory/compaction: pgmigrate_success, compact_stall and compact_migrate_scanned did not change on any node from -6 to +6 s;
  memory PSI 0, CPU ~99% idle, no python in D-state samples (2 s sampler, so a short stall can be missed).
- Kill landing mid-step / long collective: all ranks were in `tp_gather`, 1 step discarded, as in rank5-2.
- EPLB/populate: static EPLB (num_slots 256/320, no update lines); no online quiesce.
- MoE communication timeouts / `_join_device_work`: these run after the shrink and per rank, while the Stopped spread was 1.4 ms.
- Authority stall: see 1. Victim teardown: RANK_EXIT lands +0.81 s after detect in all three rank5 cases.
- Case order / warm state: each case relaunches the server. The kill came 100 s after healthy (94-105 s in the others).
  node1-3 (6th case) ran normally.
- GC: rank 0 shows periodic ~200-250 ms host-step stalls in decode, never ~1 s. Three stalls of 0.5-1 s on one rank
  within 2.7 s do not fit gen-2 GC. Not excluded, though: about 25% likely.

## 4. Verdict
- Most likely cause, about 60% confidence: one survivor (rank 26) stalled repeatedly in synchronous rank-log writes to
  Lustre on the recovery's critical path. The harness's 0.5 s polling of all 32 logs during recovery made it worse.
  Alternatives: a GIL/GC stall on rank 26 (~25%), or something else (~15%). This is mainly a test-infrastructure tail, but the product
  does log on the critical path before each Stopped/Ready/Installed, so a slow log sink delays recovery anywhere.
- Frequency: 1 of 10 cases in job 8115497. Across all ~55 KILL-FINAL cases, a gap of >=300 ms in shrink/Ready/Installed shows up 4 times,
  and only this case went past 1 s. Read it as a systematic tail of about 2-5% per kill, with occasional seconds-long excursions; not a v2 code regression.
  The p50 (rank5-1/-2: 450/429 ms) stands.

## 5. Cheapest checks for the remaining kill cases
1. Zero GPU cost, on existing logs: for each case, compute per-rank ghost+staged and the minimum staged value (a straggler
   enters staging late), plus fired/aborting second splits. This names the late rank in every slow case.
2. Run the remaining kill cases with rank logs node-local (`-loc` arm, TLLM_TEST_LOCAL_LOGS=1). Also make `wait_resumed`
   poll the `/fault_tolerance` endpoint or rank-0's log only. If the >=300 ms gaps disappear, the cause is confirmed.
3. Optional diagnostic: let `_log_phase` record on every rank (local logs only), so the authority's slowest Ready/Installed
   is attributed directly.
Side note: OFF-B-8115497 rank5-2 had trigger->detect = 6676 ms (reruns=5, lost_held=123). That is a separate kill-landing anomaly.
