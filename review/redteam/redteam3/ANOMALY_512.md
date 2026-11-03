# ANOMALY_512: offline-FT low attainment at 512/GPU, and offline-FT stalls at 8/GPU (STD-8110144)

R = .wideep-ft-runs/claude/mvp/runs/STD-8110144. Online rows: STD-G1-8110144. Only rank 0 writes iteration lines
(rank 1-31 logs: 0 matches for `num_active`), so the iteration-log view is rank 0 only. Its line also carries
`currank_total_requests = A/B`, where B is the global count of requests popped from the executor waiting queue.
That gives a system-wide per-wave count (point 2). The healthy-512 client runs in lock-step waves: 8 clients x 2048 slots, ISL=OSL=1024, ignore_eos,
`force_close=True` (one new TCP connection per request to 127.0.0.1:8000). Log timestamps are PDT.

## Evidence table (healthy-512 row; "decode steps" = rank-0 steps with no ctx and num_active>=100)
| arm (job) | out tok/s | med num_active | share >=461 | KV util max | steps num_unsched>0 | ctx-step share | rank-0 admitted per wave | TTFT p50/p99 s | TTFT >=55 s |
|---|---|---|---|---|---|---|---|---|---|
| offline off (8110144) | 240842 | 512 | 0.90 | 0.58 | 0.18 | 0.07 | 512/512/513 | 14.4/26.0 | 198 |
| offline FT (8110144)  | 224265 | 412 | 0.22 | 0.53 | 0.22 | 0.08 | 464/410/431 + tail wave 227 | 14.3/68.9 | **7226 (14.7%)** |
| noeplb off (8110144)  | 185786 | 512 | 0.90 | 0.57 | 0.16 | 0.08 | full | 25.3/47.7 | 0 |
| noeplb FT (8110144)   | 183915 | 512 | 0.90 | 0.57 | 0.16 | 0.07 | full | 22.9/47.8 | 0 |
| online off (G1-8110144) | 241954 | 512 | 0.90 | 0.58 | 0.16 | 0.08 | full | 19.7/32.6 | 0 |
| online FT (G1-8110144)  | 245285 | 512 | 0.90 | 0.58 | 0.16 | 0.08 | full | 16.1/29.2 | 0 |

Only offline-FT has low attainment. Rank-0 median num_active per 60 s bin since ready: off 660 s 512, 720 s 512,
780 s 406 (wave tail); FT 720 s 382, 780 s 411, 840 s 228, 900 s 227.
Rank-0 decode host_step medians are 68.3 ms (off) and 61.7 ms (FT); device step medians are 78.3 and 70.1 ms. Step
times are not the cause.

## 512/GPU: cause
**Requests reach the executor about 63 s late, so attainment falls in each wave. The executor admits all it receives.**
1. The executor holds no requests back.
   - `_fetch_and_enqueue_requests` drains the request queue every iteration; `_pop_from_waiting_queue` pops up to
     `32*512 - total_active`; every popped request is routed (v2 `py_executor.py` ~6583-6880, `request_utils.get_from_waiting_queue`,
     `adp_router.DefaultADPRouter`).
   - FT adds no gate there: `peer_loss_recovery.broadcast_requests` and `split_unservable_requests` only matter
     for pinned or lost ranks.
   - In FT wave 1, rank 0 sat at 464 active with free slots, KV at 0.34-0.47 and num_unscheduled at 0. Yet
     `B` grew only 14,907 (off: exactly 16,384), and then by about 3 per 10 s. The missing requests were never in
     the executor queue.
2. Every rank is short by the same amount, so this is not a few-rank or ADP imbalance. The per-wave global
   shortfall is 1,477, 3,178 and 2,584, which is 46, 99 and 81 per rank. Rank 0 was short 48, 102 and 81.
   Offline FT then needs a 4th tail wave of 7,237 requests (rank 0: 227) that offline off does not have.
3. Client side: the late requests have TTFT in a tight band of 62-68 s. Nothing falls in the 30-55 s band of this
   cluster (FT: 266 requests at 30-55 s, mostly normal).
   - 62-68 s is Linux's 6th SYN retransmit (1+2+4+8+16+32 = 63 s) plus a normal 1-5 s prefill.
   - Within each client process, the late requests are contiguous blocks in issue order. Examples: part6 requests
     830-2047 (the tail of the wave-1 burst); part0 requests 1611-2047 and 2731-3600.
   - That pattern fits connections dropped at the listener during a burst. It does not fit queueing inside the
     engine, which would give a spread TTFT of wave length plus 2-25 s.
   - No failures or resets (49152/49152). FT part5 had no late requests and finished in 190 s vs 224 s.
4. Listener headroom is zero. The harness raises uvicorn's backlog to 16384 (`WIDEEP_BACKLOG backlog=16384` in all
   arms; testhooks.py:65-67), and somaxconn is 16384. That equals the 16,384 fresh connections each wave boundary
   opens. Offline-off shows the same signature at a small scale: 198 requests at about 66 s, in parts 5 and 7.
   Offline-FT's frontend also ingests a burst more slowly: warm-healthy-512 TTFT p50 13.2 s vs 7.4 s, and wave-1
   pops took about 30 s vs about 20 s. That is enough to push it past the edge.
5. The effect is not specific to FT. FT-off runs show the same 60-70 s cluster: in LADDER-8099447 healthy-512,
   noft-1 had 169 and noft-3 had 1965 of 32768 requests in 60-70 s. Other runs had none, so it is intermittent.

**Confidence: high** that the 512 deficit happens before the executor and that the 63 s cluster causes the
attainment and the -6.9%. **Medium** that the mechanism is accept/SYN-queue overflow (no TCP counters were
recorded). **Low** for why offline-FT's frontend admits a burst more slowly; this may be
run-to-run variance. Online-FT and noeplb-FT use the same FT frontend with no effect: not a systematic FT code path.

**Ruled out for 512:**
- (a) Host memory pressure. Node 0 had 381 GB free, PSI memory 0, no D-state samples, and 0 pgmigrate pages in the
  512 window on nodes 0, 1, 4, 5 and 7. Nodes 2, 3 and 6 migrated only about 15k 64 KiB pages each.
- (b) KV capacity / KV host cache: both arms 58.5 GiB, 35136 B/token; KV peaks 0.53 (FT) vs 0.58 (off).
- (c) Populate. The last WIDEEP_POPULATE `done` came 308 s before ready.
- (d) ADP routing and per-rank skew (points 1-2); step-time regression.

**Cheapest confirming checks (no GPU):**
- On node 0 during any 512 row in the finals, run `nstat -az TcpExtListenOverflows TcpExtListenDrops
  TcpExtTCPReqQFullDrop` before and after the row, and sample `ss -ltn 'sport = :8000'` Recv-Q. A non-zero delta
  in offline-FT confirms the mechanism.
- Data the finals already produce: the healthy-512 per-request TTFTs. The 60-70 s count is the tell, so report it
  per arm.
- Possible fix to evaluate: set `TLLM_TEST_SERVE_BACKLOG` and somaxconn to 32768 or more.
- Until fixed, treat any 512 A/B with a 60-70 s TTFT cluster as invalid.

## 8/GPU: offline-FT stalls (a separate cause)
**kcompactd page migration of the FT host copies.** Offline-FT holds 390 GB of /dev/shm per node (76%); offline-off
holds 0. Page size is 64 KiB. In the healthy-8 window, offline-FT migrated pgmigrate pages on these nodes:
| node | 0 | 1 | 2 | 3 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| pages | 1.42M | 2.61M | 0.21M | 0.54M | 0.51M | 0.83M | 0.24M |
That is about 0.4 TB in total. compact_stall stayed 0 and numa_pages_migrated 0, while kcompactd scanned pages
heavily. Offline-off, noeplb-FT and online show 0.

The stall share follows the migration in 20 s bins from row start:
| bin | +0 s | +20 s | +40 s | +60 s | +80 s | +100 s | +120-180 s |
|---|---|---|---|---|---|---|---|
| migrated pages (all nodes) | 3.34M | 2.08M | 0.79M | 16k | 15k | 0.51M | about 9k per bin |
| stall share | 0.97 | 0.70 | 0.39 | 0.04 | 0.06 | 0.48 | 0.04-0.06 |
`burst_counts --since-ready 60` gives offline-FT 50.7% / 18.7% / 16.9% / 0.9%, against offline-off 0.7-10.2%.
Populate threads are ruled out because they finished 308 s before ready. **v2's populate-join is not expected to clear
this stall**: compaction runs after populate. Confirming checks:
- Rerun with `TLLM_TEST_PIN_HOST_COPIES=1` (testhooks.py:68-70) or `vm.compaction_proactiveness=0`.
- Compare the pgmigrate deltas in hostpressure-*.log over the finals' first row. Code pointer: v2 88860b1d6b
  (background prefault of EPLB host copies).
**Confidence: high** on correlation, **medium** on causation.
