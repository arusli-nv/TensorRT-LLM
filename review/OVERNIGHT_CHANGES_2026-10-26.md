# What changed overnight (2026-10-25 23:25 -> 10-26 morning)

The doc is review/DESIGN_DOC_final.md. Backups of each step are in review/diag/DESIGN_DOC_final.pre-*.md.

## Decisions (user, 10-26 ~07:05)
1. REMAP-PEER: the doc keeps "consistent with a teardown stall, not proven by the pre-set rule" (no decision needed).
2. nvl72046-T15: fine to leave (user).
3. Fix candidates (/health cache, stats-loop ballot, 8-byte flag): later, only if clearly clean and worth it (user).

## Design doc changes
- **First-reader pass:**
  - "fence" has two defined meanings; the "frozen build" is renamed "pre-fix build";
  - terms are defined at first use; units are µs; CIs use [lo, hi];
  - a citation key is under the header;
  - stale figures A3/A4 are dropped (their data is in Baseline);
  - Fig 1/2/3/8 takeaways now match the figures;
  - recovery times use one basis everywhere (rank 0.8-1.6 s, node 1.4-2.8 s);
  - new text on how to turn FT on (env vars + mpirun --with-ft ulfm);
  - new Limits bullets (launch requirement; late-write tests last run on v5-zero);
  - a new Alternatives row (orchestrator failover).
- **TTFT +7% (corrected by the soak test, CLAIMS 223):** an FT effect in the synchronized-start benchmark, not only a
  benchmark artifact. FT starts locked more often (14/18 vs 6/18, p = 0.018); locked FT windows are +161 ms; steady
  windows and realistic traffic show no change. Cause not known.
- **Healthy hang soak (CLAIMS 222):** 0 hangs in 7.1 / 7.2 loaded h; rate < 0.42 per hour.
- **Online restart slowness:** main's online-EPLB host-copy build (~250 s extra model load on every launch), not page
  cache (CLAIMS 211 corrects 198).
- **All-to-all ablation (CLAIMS 210):** the FT kernel cost is fully attributed.
  - With stats, 2.13 of 3.00 µs is the per-peer mask check.
  - Without stats, the cost is the release-flag read (0.45 µs) and the lane-parallel checks (0.49 µs).
  - Serving cannot resolve it (CLAIMS 212).
- **Recovery reruns (THIN-CELLS, CLAIMS 213, 8/8 PASS):**
  - offline rank and node kills are now n=3;
  - frozen rank and two losses are now n=2 per mode;
  - every value is inside the existing ranges;
  - the slow first step after a second loss did not recur.
- **No-EPLB remap two-speed (CLAIMS 214, 215):**
  - the extra ~0.5 s is one stall importing the dead GPU's node-mate's fresh memory;
  - per-call times are otherwise the same as online;
  - online never stalls.
- **/health can miss a recovery** (new Limits bullet): a 1 s answer cache vs a ~0.7-0.9 s recovering window.
- **test_moe_module:** one CUDA illegal memory access, unreproduced in 16 reruns on other nodes, including main's code.

## Product stress (S2 8299473, R1 32 GPUs, random kills under traffic)
- Both kills recovered in 1.03 s. Only the lost rank's requests failed (21, 19), and throughput stayed at ~1.0x.
- One health-check FAIL: the /health cache issue above.

## Healthy realistic traffic (S1 8310100)
- At ~78% load, FT on vs off is indistinguishable: TTFT p50 within 5 ms, TPOT p50 within 0.1 ms, throughput within
  0.2% (n=2 pairs). No stalls, no client failures.
- One FAIL: host memory on node 0 fell 6.9 GB in an FT-off arm; it is not in the serving processes. Being attributed.

## Second loss later (S3 8314070, CLAIMS 218)
- 2/2 servers pass: 4 losses, resume 0.87-1.03 s, only the lost ranks' requests failed; 30 ranks serve at the end.

## Still running or pending at the time of writing
- **Soak:** 3 jobs done, ~3.6 loaded h per arm, 0 events in either arm (interim); SOAK-4 8318007 queued.
- **S2 job 2 (8318006):** more random kills under traffic, including a node kill; queued.
