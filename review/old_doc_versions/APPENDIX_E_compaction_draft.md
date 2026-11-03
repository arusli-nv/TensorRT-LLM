## Appendix E. Host copies and kernel memory compaction

The issue. Kernel memory compaction (kcompactd, proactive by default) moves movable pages to build huge pages.
The /dev/shm EPLB host copies are movable. While a page moves, any thread that touches it waits
(migration_entry_wait_on_locked, D state). FT made this worse in two ways: every rank mapped every local
rank's copy (prefault), and each rank kept its own copy mapped.

When it hits:
- Startup on a used node. A large free (page-cache drop, previous server's teardown) starts a 45-90 s
  compaction pass that moves ~100 GB per memory node. With the prefault, every rank's steps stall. CLAIMS 183.
- Mid-window, online EPLB. Online EPLB reads the host copies every step. Before the fixes, 3 of 12 online FT
  windows had a 30-80 s burst. FT off was exposed too (one burst in a diagnostic run). CLAIMS 188, 192.

The fixes (in the PRs):
1. No peer prefault: a rank maps a peer's copy only when a restore reads it. Startup stall 21-46 s ->
   6-11 s (n=3 each). CLAIMS 184.
2. The owner drops its own mapping after writing its copy (MADV_DONTNEED; the shm pages stay). Startup stall
   16.8 / 15.0 s -> 2.7 / 5.7 s, FT off 2.2 / 3.2 s. CLAIMS 187.

On the final code: 0 mid-window bursts in every window checked (cost runs 0/4 FT + 0/4 off; long online
windows 0/2 FT + 0/3 off; B-AB2 0/2 FT + 0/2 off). Restore and accuracy unchanged (CLAIMS 193).

What did not work, and why:
- Pinning (cudaHostRegister) cuts compaction stalls ~5x but does not remove them: kernel 6.8 still isolates and
  unmaps pinned shmem pages before it gives up on them. Pinning every copy starved nodes in an earlier run (job 8121589).
  CLAIMS 192.

Next step: driver-allocated host memory (option B).
- Host copies in cuMemCreate host-NUMA pinned memory, owned by the driver, so compaction does not move them.
- 1-node probe: ~10x less slow-copy time than /dev/shm, with residual waits (22 vs 110 / 65 samples), so the
  result is inconclusive by the rule set before the probe. Part of the gap is less compaction work during the
  driver-memory blocks; in the one block with heavy migration, its slow copies (8.9 s) were within the /dev/shm range (3.8-11.6 s). CLAIMS 195.
- At full scale it needs reserving before weight loading. Then main's loader skips read-ahead, adding ~200 s
  to startup. A partial read-ahead in the loader fixes that; it changes main's loader.
- Full-scale A/B (2 groups of 8 nodes, one run per arm): no steady gain over the final code (output tok/s/GPU
  -1.1% to +0.3% at 32/GPU, -2.5% to -0.8% at 128/GPU), higher TTFT, 0 bursts in either. Not adopted; it is
  the next step if bursts reappear. CLAIMS 196.
