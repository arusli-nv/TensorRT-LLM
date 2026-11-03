# Pre-registered outcomes -> actions (monitor, 2026-10-22 12:10 PT)

Written before the results arrive, so readouts are not reasoned post hoc. Metric: decode trim mean (tools/trim_readout.py),
same-group deltas, both groups must agree in sign; stall ms/step and client tok/s reported beside.

## ZERO-AB 8162956 (off | lb | ft-zero | ft-zero-noghost | ft-v5land @32/128)
Validity first: off G0 vs off G8 within 0.1 ms; cache_miss equal across arms (192 baseline); PIN_CHECK OK; noghost ACTIVE
line shows num_slots=256. Any failure -> fix the harness, don't read the numbers.
| Outcome @32 (both groups) | Reading | Action |
|---|---|---|
| ft-zero - off <= 0.05 ms and 128 <= 0.4%, AND stall ms/step <= lb's + off-vs-off spread (G0 vs G8), AND client tok/s + whole-window within 1% | design goal met | stop perf work on decode; C++ batch rebuild; go to TRANSPORT + VAL |
| trim passes, stall fails | decode shift fixed, extra stalls still open | report stalls as a separate open item, not "goal met" |
| trim + stall pass, client/whole-window miss | n=1 noise (client moves with stalls) | second round on the same nodes, not a FAIL |
| ft-zero ~ lb (+0.10-0.15) and noghost ~ ft-zero | residual = indirection (expert_to_slot in no-EPLB) | accept per user ("as good as main"); confirm online EPLB arm shows ~0 in VAL-H |
| noghost < ft-zero by >= 0.04 | ghost-slot layout still costs | land G2 384 routing tier (patch ready) in the C++ batch |
| ft-zero between lb and v5land, noghost ~ ft-zero | unexplained FT path cost | use the monitor's pre-staged diagnostic table (zpred); 1 job of hook arms |
| ft-zero >= v5land | regression in v5-zero | bisect G1b vs G1a vs G4 with overlays (Python) before any rebuild |
| groups disagree in sign | noise > effect | second round on the same nodes, n=2 before any call |

## KILL-SMOKE 8162957 (G0 rank5/node1/rank0; G8 stop + rank5acc; gate runs test_moe_a2a_recovery x5)
| Outcome | Action |
|---|---|
| all PASS, resume <= low single digits, post-loss tput >= 0.8 x live/total | remap validated at scale; proceed |
| recovery test flakes in gate (any of 5) | treat as real until the failing assertion is explained; no 16-node rerun before a 1-node repro |
| rank5/node1 FAIL on remap (cuMem* error, timeout) | read 'Remapped ... in X ms' + error; fix in Python if possible; KILL rerun 8 nodes |
| resume > 10 s with remap dominating | profile remap phases (create/export/import/KV resize); options: pre-reserved spare row, parallel imports |
| post-loss tput < 0.8 x live/total | per-rank load readout -> hot rank? -> even-spread restore fix (postf agent) |
| stop: victim replays after SIGCONT | window real -> remap is load-bearing; document. No replay -> document as defense in depth |
| rank5acc GSM8K < 0.90 | correctness blocker; stop perf work, root-cause first |

## TRANSPORT (online 128, ctx steps; off UCX | off-ulfm eth0 | 2nic | uct | ft-zero arms)
| Outcome | Action |
|---|---|
| off-ulfm ~ off | transport is not the cost; if ft-zero ~ off too -> v5-zero fixed it (bound path); else profile ctx step |
| off-ulfm >> off, 2nic or uct closes it | make that the launcher default (config, no code), smoke + kill before finals |
| off-ulfm >> off, nothing closes it | stated limitation (ULFM needs ob1); quantify per cell; consider bcast size reduction only if main-compatible |

## ZERO-AB prediction (monitor agent, code read of v5-zero 7a03252e92; registered before readout)
Point estimate @32: ft-zero - off ~ +0.03 ms (range +0.01..+0.07); ft-zero - lb ~ -0.09 ms (lb pays route(), ft-zero uses
the dispatch table lookup). P(A <= 0.05) ~55%, P(C) ~25%. Pre-staged next steps per bucket:
- B (~lb): table not in effect (route() still runs) -> arm TLLM_TEST_FT_NO_SLOT_TABLE=1; equal to ft-zero = table unused.
- C: host FT work exposed / abort polls in skew waits -> arms TLLM_TEST_FT_OFF=resp,step,coll,watch (+COLL/PHASE timers),
  pml=ucx healthy-only transport A/B, TLLM_TEST_FT_UNBOUND=1; optional new Python hook TLLM_TEST_FT_NO_ABORT_POLL=1.
- D: check cache_miss/tactics first (free), then nsys_ab ft-zero vs off + FT_UNBOUND.
- E (noghost << ft-zero): diff tactics free; FT-off TLLM_TEST_EMPTY_EXTRA_SLOTS=1 vs lb; nsys MoE kernel diff.
Measurement-problem signals: off G0 vs G8 > 0.03 ms; lb - off not ~ +0.11; cache_miss/tactics differ; noghost proof line
missing or route() visible in ft-zero; trim moving with stalls. Do NOT use TLLM_TEST_FT_NO_INPLACE_SLOTS on v5-zero (REFUSED).
