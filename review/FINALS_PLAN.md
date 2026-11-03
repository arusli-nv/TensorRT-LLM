# Finals plan v6 (user: overnight, stakeholder-grade trends): data and figures (monitor, red-teamed for consistency and bias, 2026-10-22 night)

## Fixed for every run (no per-arm differences)
DeepSeek-R1 NVFP4, DEP32 on 8x GB200, attention DP, overlap, MTP (same acceptance setting in every arm), CUDA graphs, fixed
1k/1k, the refrozen build only, fence a2a, FT arms with hang detection on (30 s confirm) in healthy AND recovery runs,
warm restarts, identical burn-in (past the first online-EPLB update cycle), identical measured windows per load, one
tactic table per mode tuned once on the freeze build and pinned for every arm (PIN_CHECK OK, cache_miss equal), offline
EPLB from one placement file for all arms.

## Bias controls
- Replication unit = 8-node group. Every mode runs on >= 4 different groups; no mode is tied to one node set.
- Each group runs one off/on pair per mode (adjacent, same server-start conditions); which arm goes first is balanced 2/2 per mode across groups and split over G0/G8; mode order rotates across jobs.
- One launch per arm sweeps all loads (8/32/128/512), so loads share the server start and order effects are equal.
- Kills: fixed trigger (decode steady, after N steps, same N for all); victim rotated across repeats (no-EPLB ranks 5/22, nodes 1/5; online ranks 13/29,
  nodes 3/7; never rank 0's node; n=2 per row per mode, one per victim; stop stays rank 2, a harness limit); every kill row runs in both modes.
- Precision from existing pairs (6 same-group off/ft pairs @32, CLAIMS 171/173 + ABEXP): decode trim gap SD 0.20%,
  client tok/s gap SD 1.43% (190 s windows, stall-driven). 95% CI half-width: trim n=4 +/-0.31%, n=8 +/-0.17%;
  client n=4 +/-2.3%, n=8 +/-1.2%, n=16 +/-0.8%. So client tok/s at n=4 cannot support a "<1%" claim.
- Metrics: see PRE-REG below.

## 1. Data (consistent n)
| Block | Modes | Loads | Arms | n |
|---|---|---|---|---|
| H Main healthy set (one method for every bar) | online EPLB 288, offline EPLB, no EPLB | 32, 128 (300 s steady windows; parity arm main 9cc in the same groups for online + no EPLB; static-LB floor arm for no EPLB) | FT off, FT on | 4 pairs each |
| HX Extended loads (appendix, same method, labelled) | all 3 modes | 8, 512 (300 s windows) | FT off, FT on | 2 pairs |
| P Main parity | online, no EPLB | 32/128 | main 9cc wheel vs our FT off, same groups as H | 4 |
| C Prefill / whole-window | from H (ctx_classify + ctx_stall_cost) | 32/128 | off, on | 4 |
| R Recovery | online, no EPLB | 128 (+ rank kill at 32) | FT on: rank kill, node kill, two sequential losses, frozen rank, kill in prefill, kill during EPLB update (online only); rank 0 and deadline (must end the job) | 2 per row per mode (victims rotated) |
| A Accuracy (appendix) | online | - | GSM8K 1319: FT off, FT on healthy, FT on after a rank kill | 2 each |
| D Downtime baseline | all 3 | - | warm restart (FT off) vs FT recovery | 2 per mode |
| O Overheads | all 3 | - | KV tokens per rank, ghost-slot memory, /dev/shm host copies, startup time | from H |

Budget: H 24 sweeps + floor 4 + P 16 -> ~5 jobs; R ~52 cases -> ~5 jobs; A+D 1 job => ~11 jobs x <=1.9 h on 2 slots
(~10-11 h wall).

## 2. Figures
1. Headline: FT cost % +/- 95% CI for all 3 modes x 32/128 (one method, n=4 pairs each, every pair as a dot), per-step (trim + full mean) and end-to-end (client tok/s), with the 1% line.
2. All three modes: throughput, TPOT, TTFT at 32/128, FT off vs on (small multiples, same axes); appendix adds 8/512.
3. No-EPLB decomposition: off -> static-LB floor -> FT on.
4. Main parity: our FT off vs main (state the gate-fix effect).
5. Recovery timeline: tok/s over time through a rank kill, node kill and two sequential losses, phase markers.
6. Resume time per failure type and mode (frozen rank = confirm + resume).
7. Post-loss throughput vs ideal live/total.
8. Request fate per failure type (completed / retryable / failed). Appendix: GSM8K off / on / after kill.
9. Downtime and GPU-hours: FT recovery vs warm restart; break-even failure rate.
10. Design: architecture, recovery sequence (stop -> agree -> remap -> restore -> resume), restored-expert placement,
    late writes contained by the remap (replaces the old fixed-regions figure).
11. Limits: frozen rank ~30 s detection; second loss inside the remap window ends the job; KV-cache events and CFT
    unsupported; untested failure types (power loss, NVLink fault); disaggregated serving out of scope.

v6 sizing (user 00:00): target <= ~8 h wall overnight, 4 pairs per cell (one per group), 300 s windows.
Expected precision (prior SDs): per-step trim gap upper bound ~ mean + 0.24%; client tok/s CI ~ +/-1.8% (reported as estimate +/- CI,
no <1% claim on client at n=4).

PRE-REG (fixed before the first finals job; any change logged in DECISIONS before readout):
1. Unit = one 8-node group; pair = off and on arms adjacent in that group, order per launch (2/2 off-first per mode).
2. Per-pair cost % = (on - off)/off for: trim-mean decode step, full mean step (trim + stall), stall ms/step,
   client tok/s/GPU. Cell = mode x load; report mean, 95% t-CI (df=n-1), every pair as a dot; n stated.
   Stall threshold = 1.5x median everywhere (trim_readout and ctx_stall_cost aligned).
3. "FT cost < 1%" on a metric only if every cell's one-sided 95% upper bound < 1% (IUT). Claimed for the per-step
   metric only if the full-mean bound also passes. Client tok/s is reported as estimate +/- CI, no <1% claim at n=4.
4. 32 and 128 are not independent; any per-mode summary averages them within a pair first. Modes are never pooled.
5. A case is invalid (decided from logs, blind to the cost, before any pair is computed) if: PIN_CHECK fails or
   cache_miss differs between arms; any ListenOverflows increase; result row missing/not DONE; < 20 steady steps;
   a node/rank fault, Xid or hang-detector trigger in a healthy arm; window shorter than planned.
6. An invalid arm drops its whole pair. The lost pair is rerun once on a new group appended at the end; no other
   rerun or exclusion. Stall-heavy but valid runs are kept (stalls are part of the cost).
7. Order effect (off-first vs ft-first pairs) and G0 vs G8 are reported as checks, not used to adjust.
8. Recovery rows: outcome counts (k/n) and per-run times only; no CIs or between-mode claims at n=2; pooled k/N across
   all recovery cases is reported. Startup-time (block O) only as order-balanced means. Parity/floor (3-arm cyclic order)
   labelled descriptive.

v6.1 (2026-10-23, monitor, before any finals data): off-first 2/2 per mode (4 pairs); kill victims per mode as above (n=2);
GSM8K appendix runs dep32-final.yaml with victim rank 5 (labelled as such); PRE-REG 5 cache_miss equality is within an
off/on pair; a main/LB/ulfm mismatch voids only its parity/floor row. Readout rules (a)-(f): mvp/DECISIONS.md 09:3x entry.
