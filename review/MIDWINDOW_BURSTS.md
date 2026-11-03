# Mid-window rank-0 stall bursts in online-EPLB pairs (log analysis, read-only)

Scope: every arm in STD{,-G1,-G2,-G3}-{8202865,8206860,8215947,8244021,8245492,8245498,8239874} (50 arms).
Online arms: 12 FT-on (10 std-online-ft + 2 std-online-ft-empty), 10 FT-off std-online, 4 std-online-main.
Times are PT. Scripts were scratch-only ($TMPDIR); nothing was written to the run dirs.

Method
- Rank-0 burst: in each measured window (healthy-8/32/128/512, from bench.log), 10-s buckets of
  sum(host_step_time for steps > 40 ms); burst = bucket > window median + 1.5 s (first and last buckets dropped).
  Typical load-32 median is 2.5-3.0 s per 10 s. At load 128/512 the detector fires every ~50 s in
  every arm, FT on and off (the context phase of each bench iteration), so those are workload structure, not stalls.
- Node episode: vmstat samples with b >= 8 or wa >= 5, merged when < 20 s apart.

## 1. Rank-0 bursts in measured windows (load 8/32)
| arm | window | burst | length | excess over median | min steps/10 s (typ ~480) |
|---|---|---|---|---|---|
| STD-8202865/std-online-ft-1 | healthy-32 (15:55:38) | 15:59:28-15:59:58 | 30 s | 10.5 s | 198 |
| STD-8206860/std-online-ft-1 | healthy-32 (18:12:47) | 18:13:07-18:13:37 | 30 s | 11.4 s | 158 |
| STD-8245492/std-online-ft-1 | healthy-32 (15:00:59) | 15:04:49-15:06:09 | 80 s | 17.9 s (19.2 s for all buckets >1 s over median) | 219 |
All other online arms: no bucket more than 1.1 s over the median at load 8/32 (FT-off max 0.9 s; main max 0.7 s).
Count: FT-on 3/12 arms (3/10 excluding ft-empty), FT-off 0/10, main 0/4.

## 2. Node state during each burst
| burst | node with D-state/iowait | host | ranks | max b | max wa | bo during burst |
|---|---|---|---|---|---|---|
| 8202865 FT | node6 (15:59:31-38) | nvl72028-T15 | 24-27 | 48 | 7 | background only |
| 8206860 FT | node6 (18:13:09-31) | nvl72122-T15 | 24-27 | 4 | 7 | background only |
| 8245492 FT | node1 (15:05:03-06:07, 21 samples) | nvl72167-T02 | 4-7 | 64 | 11 | mean 3.5 MB/s vs 3.5 MB/s before the burst |
Every burst has exactly one node with an episode, and it was never rank 0's node (node 0 is flat, as you saw).
The three are different physical hosts on different racks, so it is not one bad node. Node indices were 6, 6 and 1;
with n=3 and 8 nodes, two matches on node6 is weak evidence.
Common signature on the affected node: from 0 to tens of D-state tasks almost every second, wa 1-11%, sy 1-3% (flat), and `free`
swinging by ~0.4-0.5 GB every few seconds. The free swing alone is not specific: it appears in quiet windows
of both arms too. The D-state count is what separates these buckets.

## 3. FT-off arms
- Inside measured windows, a node episode occurs only in FT-on arms: the 3 above plus single-sample blips in
  8244021 FT (node1, b=12) and 8245498 FT (node0, b=33). Neither blip produced a rank-0 burst. FT-off: 0 episodes in 10 std-online
  arms. main: 0, apart from 1 bucket with b summed to 5 or more. Offline and no-EPLB arms (FT on and off): 0 episodes in any measured window.
- In burn-in (the first ~15-105 s after the page-cache drop), episodes happen in both arms, more often with FT:
  FT 6 arms (8206860, 8245498, G1-8215947, G1-8244021, 8239874 ft-empty), off 2 arms (G1-8206860, G1-8244021).
  Model-load episodes before the bench run on all 8 nodes in every arm, FT and off alike.
- Burst timing does not line up with the drop: drop+689 s, +174 s and +374 s (window +230 s, +20 s, +230 s). FT
  ran first in 8202865 and second in 8206860 and 8245492, so arm order does not explain it either.
- The no-prefault overlay (fix-nopopulate-d71f169e4b) was absent in 8202865/8206860 and present in 8245492.
  Removing the prefault did not remove the mid-window burst.

## 4. What wrote to disk and what blocked
- Not local block I/O: bi = 0, and bo stays at the ~1.4 MB/s floor with periodic 10-50 MB/s spikes that show up on every node
  in every arm. The burst's bo is no higher than the minutes before it.
- Not swap: swpd, si and so are 0 throughout. No new dmesg lines (kernel-N-before/after) on the affected nodes: no
  hung-task warning, OOM or Xid. No rank-log, agent-log or authority-log lines during the bursts. Rank 0 is the only
  rank that logs every iteration.
- The logs cannot name the blocked tasks. Missing: per-task D-state sampling (stat, wchan, comm), the /proc/vmstat
  counters (pgmigrate_*, compact_*, thp_*, pgfault, nr_writeback), /proc/pressure/{memory,io}, and Lustre
  client stats. wa together with D-state and no block I/O fits waits on memory (page or folio locks, migration,
  compaction) or on Lustre. It does not fit core dumps, journald, cache writes or swap.

## Conclusion
- The burst looks FT-specific: online-EPLB with FT on, 3/12 arms against 0/14 FT-off and main arms, and
  in-window D-state episodes only in FT arms. n is small. Assuming independence, the chance that all three bursts land in the FT
  group is about 0.06-0.09, so this is suggestive, not proven.
- It is not node-specific: three different hosts. It is random in time.
- One affected node stalls everyone through the all-to-all.
- Cheap check that would settle it: in the next scheduled online pair (no extra slot), add a 1-s sampler on every node that records
  `ps -eLo pid,tid,stat,wchan:32,comm` (D rows only) plus /proc/vmstat and /proc/pressure/memory deltas. It would show which
  process and threads block and in which kernel function (for example migrate or folio-lock waits versus Lustre). That tells FT
  threads or peer-shm mappings apart from something unrelated.
