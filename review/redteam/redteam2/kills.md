# Red team: WideEP FT kill-test methodology (R1 NVFP4 DEP32, KILL-FINAL*)
H/RUNS/CLAIMS = .../claude/mvp/{harness,runs,CLAIMS.md}; all KILL-FINAL-* verdicts read ~22:00 PT (8110820 running).

## Ranked issues (worst first)
1. **"Node kill" is not a node loss.** node1 SIGKILLs the 4 worker processes; the host, prted, NIC, IMEX/NVLink
   fabric and driver all stay up (job.sh:3612-3614, testhooks.py:114-117; CLAIMS #96 says "prted alive"). Detection
   is then ULFM's immediate proc-failed from the live daemon: rank 0 "Peer failure watch fired with MPI error class
   76" ~80 ms after the kill (RUNS/KILL-FINAL-ON-A-8102532/rank5-1/rank-0.log; detect 0.01-0.09 s, CLAIMS #119/#121).
   A real host/power/NVSwitch-tray loss goes through the heartbeat detector (period 1 s, timeout 5 s, timeline
   MPIRUN_START mca) plus PRRTE daemon-loss handling, which was only shown on a 3-node probe (CLAIMS #105/#111).
   The stop row (detector path) costs 5.0-6.8 s vs 0.9-2.6 s.
   **Publishing "node loss recovers in ~1-2.6 s" overstates it by ~5 s and has never been tested at DEP32 with
   a dead host.** Fix: label the row "process-group kill (host alive)", quote the stop row as the detector-path
   bound, and add at least one real host loss (power-off via BMC, `echo b > /proc/sysrq-trigger`, or kill prted
   + workers + block the node's eth0/IB) before saying "node".
2. **The post-loss throughput the harness prints is not capacity, and its expected value is wrong.**
   est_tok_s = median(rank-0 gen_tokens/host_step_time) x live ranks (job.sh:3527-3540). Problems:
   (a) closed-loop load with fixed concurrency (bench.py:167-191, 128 x 32 = 4096 streams): after a loss the
   same 4096 streams are spread over 31/28 ranks (132/146 per rank), with max_batch_size 512 (config-0.yaml).
   That is a step-time comparison at equal offered concurrency, so the right expectation is
   step(128)/step(132 or 146) from the healthy ladder (about 0.95-1.0), not live/32 = .97/.875 as #120 assumes.
   Neither number is "capacity after N-1/N-4".
   (b) Rank 0's host_step_time and batch stand in for all ranks, and no batch normalisation is done: the >= load/2
   filter (job.sh:3530) mixes batch 64-146 steps. Result: physically impossible "after > before" values in 25 of
   the 43 verdicts (e.g. ON-A-8086324 rank5-3 354547 vs 296896; stop rows +30-90%). #120 already says the field
   "must not be used", yet every verdict and KILL_FINAL_ROW (job.sh:3653) still print est_tok_s medians.
   (c) Windows too short: 2 waves (job.sh:3603); after-windows 112-863 steps, node1 often <200 (~5 s) incl. online's
   ~3 s post-resume transient (#120). KF_ITERATIONS unset in kills_final8_cmd.txt.
   Fix: measure capacity saturated (512/GPU = max_batch_size: survivors cap, excess queues): client output_throughput
   over >= 60 s from resume+10 s vs a healthy window; expect <= live/32. Call 128/GPU "TPOT impact at fixed load".
3. **CUDA graph sizes 136/148 make the result valid for 128/GPU only, and they change the config.**
   KF_GRAPH_SIZES=136,148 (kills_final8_cmd.txt; job.sh:3453-3455) is tuned to 4096/31 = 132 and 4096/28 = 146.
   At any other load, padding still rounds up to 192/256 (+22-30% step time, #120). The healthy-cost runs forbid
   extra sizes (job.sh:2684-2686), so the published healthy cost and the published after-loss throughput come from
   different graph sets (extra capture time and graph memory reduce KV capacity). Fix: either ship the extra sizes
   as part of the FT config and include them in the healthy-cost arm, or publish both numbers, the padded one
   (default sizes) and the one with extra sizes, and state which load each applies to. A saturated 512/GPU
   measurement avoids the problem, because 512 is a captured size.
4. **Recovery time is server-internal. The client-visible gap is not computed by the harness.**
   - recovery_s = victim's WIDEEP_SELF_KILL stamp -> rank 0 `phase=first_step` (job.sh:3567). It excludes the
     trigger->kill lag (0-8.3 s, correct to exclude), the first step's own duration, the overlap pipeline, the
     detokenizer and the HTTP flush. It is measured on rank 0 only: the other survivors' first_step is not used.
   - resume_s = harness polling 32 logs every 0.5 s (bench.py:356-374): poll lag, not a server metric. Don't publish.
   - probe_ttft_s = new request after wait_resumed, queued behind 4096 streams (bench.py:559): 40 s at 512/GPU. Not recovery.
   - The number a user experiences: for every surviving stream that spans t0, the inter-token gap across t0
     (max ITL), from bench --save-detailed itls, reported as median/p99/max. Add TTFT for requests that arrive
     during recovery and the end-to-end delay of re-run requests. #120's kill_tput.py does a related replay, but it
     lives in $TMPDIR (not in the harness, not reproducible). Move it into kf_verdict.
5. **"0 client failures" is misleading.** client_failed=0 in every PASS, but 121-128 (rank) / 484-512 (node) /
   254-260 (twokill) streams per kill are cut off. Per #10, the server ends them *normally*, and the client only
   detects them because of --ignore-eos length (bench.py:213-216). A real client without ignore_eos would get a
   truncated completion and no error, which is silent corruption unless finish_reason marks it. Check what
   finish_reason the server sends on a cut stream. Publish "lost-rank requests failed: N (= that rank's in-flight
   share; 3.1% / 12.5% of in-flight)" and "requests re-run transparently: reruns (0-35)".
   no_survivor_failed (job.sh:3574-3576) compares a client count with the server's own failed count, so it is
   circular: if survivors' requests were also failed and counted, it would still pass. Cross-check request ids
   against their owning rank.
6. **Provenance.** The run-dir kill_final.txt files keep the pre-fix verdicts. For example OFF-A-8084821
   recovery_s 8.57/4.58/3.95 conflicts with CLAIMS #119's recomputed 1.79/1.25/1.34. ON-A-8084812 rank5-1 is a FAIL
   with no result.json and is not in the ledger. Write recomputed verdicts next to the originals
   (kill_final.recomputed.txt) and add a FAIL/void row to the ledger for every discarded case.
7. **Builds and n.** final8 is n=1 per row (KF_REPEATS=1) on the rc1 wheel + a Python overlay (KF_ALLOW_OVERLAY).
   The n=3 rows (#119/#121) come from build-final-aae00e6c65, and online recovery changed ~4x between builds
   (0.92-1.31 s on aae00 vs 0.22-0.36 s on 066b, #130). Do not pool across builds. With n=3, report the 3 values
   (or min/median/max) and no CI. The spread is large (static rank5 1.0-1.8 s), and the tail matters
   (recovery_timeout_s=30 in authority.log).

## (1) Which time to publish
Publish three times, all from the victim's kill line: (a) detection, kill -> `phase=detect`; (b) recovery,
kill -> first committed step on the *slowest* survivor (rank 0's line is only one of 31); (c) client gap, kill ->
next token on survivor streams, p50/p99/max from ITLs. Also give the trigger->kill lag separately, so nobody reads
it as recovery. FT-off's down_s (#117) is injection -> first token, so (c) is the time to compare with it, not (b).

## (2) Client impact accounting
Per case, give one table: in flight at t0 = survivors' streams (delayed by the gap, none failed), lost-rank streams
that had already streamed (failed: short), lost-rank requests that had not streamed (re-run: count and added
latency), and requests that arrived during recovery (TTFT). For 512/GPU, client_reset (343/849, ON-A-8094116) was
labelled "client-side". Show it is zero in a healthy 512 run on the same build (backlog 16384, #135) before
excluding it; otherwise it counts as kill-induced.

## (3) Throughput after a loss: issues 2-3. Ghost-slot expert imbalance (+0-5%, #120) and the online ~3 s
transient wash out in a >= 60 s window; forced MTP acceptance (job.sh:3451) means the tok/s is not real-acceptance tok/s.
## (4) n and noise: issue 7. Spread ~= median (static rank5 1.0-1.8 s); #120 ratios (.63-.79, 8-25 s windows) too noisy.

## (5) Coverage gaps (in the final matrix vs anywhere)
| Scenario | Final matrix | Elsewhere | Gap |
|---|---|---|---|
| Rank SIGKILL at decode step boundary | yes | - | kill lands right before graph replay, after host collectives (testhooks.py:14-15): one point; never mid-kernel / mid-a2a write (DR-H2 #100 separate) |
| Node kill, host alive | yes | #80, #96 | see issue 1 |
| Host/fabric actually lost | no | 3-node MPI probe #105/#111 | **untested at scale** |
| Frozen rank (SIGSTOP) | -B rows, not in final8 | #102, #119, #121 | fence command empty (job.sh:3618): SIGSTOPped process is never fenced/killed; survivors rely on mapping isolation |
| GPU hang, host alive (stall/sticky error) | no | #102/#103/#109 stall controls | the realistic XID case (TLLM_TEST_STALL/STICKY_IN_DECODE exist) is not in finals; its detection is hang_confirm_s 30 s |
| Kill during prefill | no | #96 n=1 (old build) | add or scope out |
| Kill during online EPLB weight update | no | none found | online mode's riskiest window |
| Kill during recovery/restore (re-plan) | no | #17 (Lite: FAILS e2e), #72 (R1, n=1/load) | needs a final-build run or explicit scope-out |
| Second failure after commit | -B twokill | #94/#99/#121 | not in final8 |

## (6) FT-off cold-restart fairness (#117, COLD-RESTART-A/B-8082563)
It is mostly *favourable to FT-off*, which makes the FT advantage conservative. That is fine, but say it:
relaunch on the same 8 nodes (with a real node loss FT-off needs a spare node or a scheduler wait; DEP28 is not a
valid fallback), scheduler wait and harness gap_s excluded, autotuner/Triton/container caches warm, and the Lustre
server cache possibly warm. The local page cache is dropped (serve.sh:117-121, 268-272), which is the correct
model for a replacement node. Unfair points: it ran on an older wheel (b18f0b64ed) with the online
dep32-final.yaml, while the FT-on finals now use std-*.yaml on rc1. The comparison counts first token, not
recovery of steady throughput, on both sides. FT-off fails *all* 4096 in-flight requests (stop: 2048 + 2048
unsent), against FT-on's 3%/12.5%. Publish the ratio as ">= ~100x less downtime (FT-on recovery vs FT-off down_s)"
with both definitions stated, not as a precise factor.

## (7) Proposed final kill matrix (one 16-node job, 2 x 8-node groups, 115 min group limit, 15-min start guard)
Settings: one build, no overlay, std configs as deployed. Graph sizes: decide (issue 3) and keep them the same in
the healthy-cost runs. KF_LOAD=128 and KF_ITERATIONS=4, so the after-window is >= 60 s. Add bench --save-detailed
ITL gap extraction to kf_verdict. Order rows rank/node alternately so drift is spread over both.
| Group (8 nodes) | Cases (order) | Est. min |
|---|---|---|
| G0 | noeplb rank x3, node x3 interleaved (6 x 10) -> online rank x3 (3 x 14) | 60 + 42 = 102 |
| G1 | offline rank x3, node x3 interleaved (6 x 10) -> online node x3 (3 x 14) | 60 + 42 = 102 |
- 18 cases, n=3 per cell, ~13 min of slack per group. That slack is not enough for a 19th case (static 10 min plus
  the 15-min guard), so stop/twokill/hang are **out of this job**. Cite #119/#121 for them, with the build
  caveat, or put a stop x1 as the tail case of each group so the guard drops it if time is short.
- Online runs last in each group: if a group runs over, the guard drops one online repeat (n=2), not a static cell.
  Alternatively, put online first if online is the headline.
- Add a separate 512/GPU saturated case per mode (capacity, issue 2) if a slot allows: n=1 each, ~15 min.
- Required before calling anything "node loss": one real host-loss case (issue 1), even at n=1.
- Pre-write launch/kf_rows-<job>-* before submitting (not "right after"), so the first case cannot read stale rows.

## Publishable today
Recovery kill -> first committed step (rank 0) for rank and process-group kills, n=3 per build (#119/#121, build
named); failed/re-run counts with the circularity caveat. Not publishable: est_tok_s, resume_s, probe_ttft_s,
"node loss" without a dead host, the #120 ratios as capacity.
