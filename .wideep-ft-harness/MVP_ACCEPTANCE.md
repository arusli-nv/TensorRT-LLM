# v7 MVP acceptance status

Each row is a v7 MVP item and its acceptance criterion, with the evidence we have.

| Item | Acceptance (v7) | Evidence | Status |
|---|---|---|---|
| 1a.8 / 1a.8b abort | bounded cooperative abort, survivor context preserved | kernel abort tests; EP32 kill mid-dispatch/combine | done: EP32 R1 kill right before a graph replay; all 31 survivors aborted in-flight MoE collectives, exact tokens, 2.6 s (Kimi-K2 2.4 s). The abort record is in pinned host memory, so it also reaches eager kernels: EP32 kill at the victim's first eager prefill forward under full load, 2/2 passed (248 survivor requests, 0 failures) |
| 1a.8a late writes | issued-before-death writes cannot contaminate reused storage | fixed slots + gather by construction; single-process quarantine test; 2-process SIGKILL reproducer | done: 2-process SIGKILL reproducer passes at three kill timings |
| 1a.11 graph reuse | same graph exec, zero recapture/compile/tune, all-active perf measured, captured-collective audit | EP4 + EP32 runs, no capture after kill; perf A/B; nsys kernel audit | done: perf 1.4%, FT-off kernels opcode-identical to main; nsys on rank 5 shows every decode all-to-all inside graphs (139,374 graph instances of each A2A kernel = 58 layers x 2,403 replays; eager only in warmup and context steps, 58 x 47) |
| 1b.2a admission | reject insufficient coverage; admit resident-replica configs | covers_loss_of, authority admission, create() checks | done |
| 1b.4a mapping | sparse membership maps consistently to logical slots | logical gathers with fill values, unit tests, EP32 | done |
| 1c.3a / 1c.4a survivor collectives | remove a middle rank without poisoned-world calls | ULFM Revoke/Shrink, real 4-rank ULFM test, EP4/EP32 rank 2 | done |
| 1c.4b coordinator | fault injection at every transition: no mixed state or early resume; coordinator loss fails closed | authority phase/violation/deadline tests; worker transition matrix test | done (tests to run) |
| 1c.4 / 1c.4c disposition | no partial logits, duplicate streamed tokens, uncertain KV reuse; explicit error | commit gate, numbered resend, reuse-safe rollback test, retry error | done (reuse test to run) |
| 1d.1 lifecycle | process death does not kill healthy workers; unsupported configs rejected; bounded teardown; unified deadlines | ULFM launch, create(), HangDetector, Abort teardown (launcher must stop mpirun on multi-node) | done with documented caveat |
| 1d.4 physical | real model, kill during dispatch/combine, bounded resume, no late-write contamination, correct requests, zero recapture | EP32 R1 idle + in-flight exact; EP4 integration test; mid-MoE run | done: EP32 idle (0.8 s), in-flight (1.4 s), mid-MoE (2.6 s), all exact; EP4 CI test idle + in-flight |
| 1d.5 perf gate | all-active overhead, degraded goodput, recovery stage timings | FT on/off bench; post-loss bench; kill/commit/first-token timestamps | done: all-active overhead 1.4% (14,488 vs 14,689 tok/s); degraded per-rank goodput 99.4% (31 ranks 13,960 tok/s); kill to commit 0.14 s, to first token 0.16 s, requests done 0.71 s (R1 EP32, idle). Without FT a loss restarts the job: 166-213 s startup plus relaunch |

Models: DeepSeek-R1 NVFP4 and Kimi-K2-Thinking NVFP4 at EP32 (idle, in-flight, mid-MoE, block reuse on and off). Stress on the fold tip (wheel 1976684d4b, jobs 7889828, 7890410, 7890795, 7890796):
- R1 EP32 mid-MoE kill x5: all exact, 2.59-2.64 s.
- Victims 1 and 30 (rank 30 on another node) in-flight/mid-MoE: exact, 3.1 and 3.2 s.
- Kill under full chunked-prefill load: 248 survivor requests, 0 failures; at the host collective (0.22 s) and at an eager prefill forward (4.2 and 4.6 s).
- Uncovered victim (rank 5, no replicas): fails closed; the authority refuses the loss and every request gets an explicit error within 0.04 s.
- EP16 (4 nodes, 2 spares per rank): idle exact, first token 0.29 s; mid-MoE exact, 2.95 s.
