# Overnight plan, 2026-10-25 -> 10-26 (monitor repo-72, dev repo-f0)

The user asked at about 23:25 PT: "run helpful stuff overnight, mainly focused on helpful stuff for the presentation / design doc."
- The presented data stays frozen.
- Overnight results are diagnostics or product data. Each goes into the doc only as a labelled, verified addition with its
  job and CLAIMS row, and the user sees it before presenting.
- Every GPU test has its rule written before data. New tests run on 1 node first.

## GPU queue: 2 slots, ordered by value to the doc

### Slot A (16 nodes)
1. **S2 job 1, 8299473** (running): random kills under realistic traffic. Feeds the doc's "early user" section.
2. **SERVE-ABLATE** (DROPPED 10-26 ~00:10: existing nsys data shows the a2a floor is mostly peer wait; FT on-off -1.7 us [-116,+101]/step) (new, highest doc value, if the ablation build has a full wheel): one same-node serving set,
   online EPLB, 32/GPU, 1k/1k, pinned tactics. Arms:
   - FT off;
   - FT `TRTLLM_A2A_FT_ABLATE=0`;
   - FT `=S1`;
   - FT `=ALL`.

   It answers the doc's biggest open "why": how much of the ~51-85 us/step steady cost is the all-to-all, and what
   the ballot fix (S1) is worth in serving.
   - Rule (set before data): the a2a share = (ft0 - ftALL) / (ft0 - off) on the trimmed step.
   - ALL is safe only because healthy runs never take an FT branch; outputs were checked bit-identical in CLAIMS 210.
   - Same nodes, reversed order on the second group, n >= 2 per arm.
3. **S1**: realistic healthy traffic, FT on vs off, with memory sampling. Gives the doc product latency/throughput
   for mixed lengths.
4. **S3**: a second loss after 15 min degraded.
5. **S2 job 2**: only if time is left.

### Slot B (small jobs, then the soak)
1. **MOEMODULE-IMA 8300459** (running). Then the prefix run, contingent on its rule. Feeds the doc's Limits section.
2. **Soak smoke** (1 node), then the **soak** (several hours, FT vs FT off on the same ob1/TCP). Answers the healthy
   hang question (CLAIMS 204): is it the transport or FT?

## Job-free work, run by agents while the GPUs work
- **Log analysis on existing runs**, for the doc's three open "why"s:
  - why no-EPLB remaps are ~2x slower than online;
  - why FT batches prefill more (the TTFT +7%);
  - why online restarts are slower.
- **Presentation readiness:** a 10-minute read of DESIGN_DOC_final.md as a first-time reader on the team, listing
  terms that are undefined, figures without takeaways, and jumps in order.
- **Morning:** fold verified results into the doc, summary and notes, and give the user a short list of what changed.

## Update ~23:35 PT: the user relaxed the freeze
User: "presented data need not stay frozen if it's helpful to rerun/change."
- **SERVE-ABLATE** gets a 5th arm, v5-final FT, as the reference. The a2a share is (v5f - ftALL) / (v5f - off). The S1
  gain is ft0 - ftS1, on the same wheel. ft0 - v5f measures the getenv host cost on eager steps.
- **THIN-CELLS** goes in slot A after SERVE-ABLATE and before S1. It reruns the n=1 recovery cells on v5-final with
  the KILL-FINAL harness and the KILL-FINAL rules unchanged:
  - offline rank and node kills, to n=3;
  - frozen rank, no EPLB and online, to n=2;
  - two losses, both modes, to n=2.
  The doc's numbers get replaced with the larger n, and the change is disclosed.
- **Code fixes** (S1 ballot, F8): proposed to the user in the morning if SERVE-ABLATE shows a serving gain.
- **Slot B:** IMA 8300459 gave outcome (3): the failing test passes in all arms when run alone. The ordered-prefix run
  MOEMODULE-PREFIX 8301335 is next, then the soak smoke, then the soak.
- **Morning ask (user):** a pinned job or GPU health check on nvl72046-T15 (the test_moe_module crash node); the broker cannot pin nodes.
- **Morning ask (user):** REMAP-PEER 8310851: accept the post-hoc reading (28/31 survivors that stall end within 3-22 ms -> teardown-type stall) or keep 'no conclusion by rule' (all-31 end spread 117-640 ms > 50 ms).
