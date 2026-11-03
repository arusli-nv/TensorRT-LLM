# Burst forensics: the 4 caught mid-window bursts (read-only log analysis, 2026-10-24 21:25 PT)

Inputs: mvp/runs/STD-{8202865,8206860,8245492}/std-online-ft-1, STD-8246041/std-offline-ft-1 (SLOTS-AB G0), their
FT-off arms, logs/slurm/slurm-<job>.out, logs/bundle/<job>-*.log, and for calibration the DIAG jobs
STD{,-G1}-{8252991,8266440} (which have dstate/hostpressure samplers). Build in every burst arm and in DIAG:
build-freeze-58caa0fde6. 8245492 also had overlay fix-nopopulate-d71f169e4b (fix 1). **No burst arm ran fix 2 (27442bd91d).**
Scripts and the full episode list: review/diag/burst_forensics_scripts/ (runs_main.txt = every free-swing run per node per arm).
Times PT. "Swing" = a 1-s change of vmstat `free` larger than 300 MB.

## 1. Per-burst facts
| burst | node (host) | rank-0 slow period | node signature (same seconds) | FT-off arm, same node, same drop offset |
|---|---|---|---|---|
| 8202865 FT online | node6 (nvl72028-T15), ranks 24-27 | 15:59:29-15:59:58: gen-only steps 200-375 ms, 2-3 per s (normal 18 ms) | free -0.45..-0.5 GB swings 15:59:25-15:59:58 (14 swings), b up to 48, wa up to 7 | 0 swings, b 0, wa 0 |
| 8206860 FT online | node6 (nvl72122-T15), ranks 24-27 | 18:13:10-18:13:32 (starts after the window's first context wave) | swings 18:13:09-18:13:31 (6), b up to 4, wa up to 7 | 0 swings, b 1, wa 0 |
| 8245492 FT online (fix 1) | node1 (nvl72167-T02), ranks 4-7 | 15:04:53-15:06:12 | swings 15:04:54-15:06:10 (23), b up to 64, wa up to 11 | 0 swings, b 1, wa 0 |
| 8246041 FT offline | node6 (nvl72163-T17), ranks 24-27 | 16:41:00-16:41:40, excess 2.3/1.3/0.4/1.9 s per 10 s bucket (smaller burst) | swings 16:40:50-16:41:40 (13), b up to 2, wa up to 1 | 0 swings, b 0, wa 0 |

- Onset and end match within the 1-s sampling at both ends in all three online bursts. Example (8202865): node6 free
  -470 MB at 15:59:29, when rank 0's first 336 ms gen step lands. Free is back on its trend at 15:59:59, when steps return to 18 ms.
  Every other node keeps its steady drift of about +7 MB/s throughout (FT and off alike).
- In the slow steps, prev_device_step_time is about host_step_time. Rank 0's GPU itself is waiting, so the cause is a peer in
  a collective or all-to-all. dmon (one sample per ~2 s per node) shows all 32 GPUs going idle or spinning together, so it
  cannot name the first rank to stall. Non-zero ranks write no timestamped lines between startup and teardown, so the
  rank logs cannot name it either.
  **The first rank or thread to stall cannot be determined from these logs.** The first node can: it is the node with the swings.
- No FT process logs anything in any window: agent-*.log are 0 bytes in all 4 arms, and authority.log ends at
  REGISTERING->SERVING. No hang/suspect/DeviceWait/ZeroMQ/heartbeat lines, and no rank-0 warnings. kernel-N-before/after
  differ only in the header epoch, so there is no new dmesg line (no Xid, OOM or hung-task; hung-task needs 120 s anyway).
- dmon: no power or thermal violations (pviol/tviol) on the burst node, and clocks are flat. File growth: only rank-0.log is
  written per iteration, and the other files are fixed-rate samplers. Mtimes are end-of-run only, so file growth carries no signal.
- Bundle/slurm logs: no events from this group in any window. In 8245492 the other 8-node group (G1) started
  std-noeplb-1 at 15:04:50, 3 s before the burst. In 8246041, G1 was loading lay320. In 8202865 and 8206860, the other group
  was mid-benchmark at the burst. So a Lustre-load coincidence holds in at most 2 of 4.

## 2. The swing signature is kcompactd page migration (calibrated on the DIAG jobs)
- In DIAG FT burn-in episodes, the same vmstat pattern lines up with kcompactd migration second by second. Example:
  8252991 ft-4 node2, 18:00:49-18:02:16. Free swings of ±0.3-0.6 GB coincide with pgmigrate_success +40-85k pages per 2 s
  (64 KiB pages, about 1.5-2.5 GB/s), compact_stall +0 (kcompactd, not direct compaction), and PSI io rising to 5%.
  The python3 / cuda-EvtHandlr threads sit in `migration_entry_wait_on_locked`, with b up to 26. Swings stop in the
  same sample in which pgmigrate stops.
- Over all DIAG serving time (ready to bench end, nodes 1-7, 15,404 samples of 2 s): 84% of swing samples
  (425/504) fall during migration (>1000 pages per sample). P(swing | migrating) = 26% per sample; P(swing | not) = 0.6%.
  A sustained swing run therefore marks a kcompactd migration episode.
- The swing size, about 0.45-0.5 GB, is close to one pageblock on this 64K-page arm64 kernel (512 MiB, inferred from the
  page size, not read from /proc/pagetypeinfo). It fits compaction isolating free pages one pageblock at a time.
- Inference: each burst is a kcompactd migration episode on one node, lasting 23-76 s at about 2 GB/s, i.e. tens to about
  150 GB moved. Ranks on that node block in migration-entry waits, and everyone else waits for them in the all-to-all.
  This is the same mechanism HANDOFF 12:00 PT established for the post-drop startup stall, occurring late on a single node.

## 3. FT-specificity and placement (86 arms: the 50 MIDWINDOW arms + 8246041 G0/G1 + DIAG)
- Sustained swing runs (>=30 s, >=6 swings) on a non-zero node inside a healthy-8/32 window: exactly 4, and they are
  exactly the 4 bursts. All 4 are in FT arms (4/34 FT, 0/52 off or main). Every burst has one, and no window run lacks a burst.
- Burn-in (just after the drop): FT arms average 3.0-3.5 nodes with an episode per arm (online and offline), and online FT
  has b>=16 in 12/20 arms. Off arms average 0.8-1.0 nodes per arm, and online off has b>=16 in 2/16. Same signature, larger with FT.
- Burst node versus burn-in: in 3 of 4 arms the burst node had **no** post-drop episode while other nodes did (8206860:
  nodes 2,3; 8245492: nodes 3-7; 8202865: none anywhere). In 8246041, node6 had both. This is weak support for "the late
  episode is that node's deferred or second compaction pass".
- Placement: nodes 6, 6, 1, 6 (plus a burn-in run on node3 in 8246041 that raised rank-0 stall to 3.7-6.1 s per 10 s).
  These are four different hosts on four racks. The authority runs on node0 (TLLM_FAULT_TOLERANCE_AUTHORITY=…-T01). Every node
  runs an agent, and ring host copies are symmetric. **No node-specific FT role explains node6/node1.** Treat placement as random (n=4).
- Timing: drop+156 to +682 s, no fixed offset. Onset is not tied to context waves: 8245492 starts 4 s before one.

## 4. Memory facts relevant to the mechanism
- Offline: after the drop, FT keeps about 390-430 GB of `cache` (the /dev/shm host copies; free about 340-376 GB), while
  FT-off drops to about 70 GB (free about 690 GB). Online: both arms hold about 390 GB in /dev/shm (hostmem-N), because online
  EPLB needs host copies with or without FT. So the volume of shm alone does not separate the arms online. Whether ranks
  **map** those pages does (FT: own copies stay mapped before fix 2; peers prefaulted before fix 1).
- No swap, bi = 0, bo at its background level, sy flat at 1-3%. This matches migration waits, not I/O.

## 5. Conclusion
**Most likely mechanism:** a late kcompactd (proactive) compaction pass on one node migrates movable pages, including
pages mapped into the FT ranks. Before fix 2, these include each rank's own EPLB host copies. The node's ranks block in
migration-entry and folio waits, which shows as D-state and iowait with no block I/O. Through the all-to-all and the
collectives, every rank's gen steps go from 18 ms to 200-375 ms for the 25-80 s the pass takes.
Evidence:
- The burst signature equals the DIAG-verified migration signature.
- Start and end coincide to the second.
- It occurs only on FT arms, both inside windows and (much more) in burn-in.
- It hits a different random node each time.

**Not established:**
- Why a pass starts mid-window, and only with FT. One hypothesis: FT-mapped pages make the post-drop pass incomplete or
  deferred, so the fragmentation score crosses the proactive threshold again later. No /proc/vmstat or fragmentation data
  exists for any burst arm.
- Which pages were migrated (FT shm versus rank anon heap).
- Which rank stalls first.
- Whether fix 2 removes the late passes. No arm with fix 2 has been examined for mid-window bursts.

**Most decisive next measurement:** run the next online FT pair on the fix-2 build (27442bd91d) with the DIAG
hostpressure sampler on every node: pgmigrate_success, compact_daemon_*, nr_free_pages, plus
`/sys/kernel/debug/extfrag/unusable_index` or /proc/buddyinfo every 2 s. Compare against an identical arm with
`vm.compaction_proactiveness=0` set on the nodes, if ops allows.
- If proactiveness=0 eliminates both burn-in and mid-window episodes, kcompactd is confirmed as the trigger.
- If fix 2 alone removes the stalls while migration still happens, the FT-mapped pages are confirmed as what turns
  migration into stalls.

Cheapest code read in the meantime: check whether anything in the FT path holds extra references on, or pins, the shm
pages (cudaHostRegister, ATS/UVM registration of the host copies). That would make them unmigratable and keep the
fragmentation score high, which would explain the repeat passes.
