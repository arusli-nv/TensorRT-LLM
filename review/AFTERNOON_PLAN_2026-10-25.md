# Afternoon plan, 2026-10-25 (until ~18:00 PT)

User (12:20 PT): use free GPUs to make the analysis stronger and clearer. Repeat runs where people will ask "is this
noise?". Red-team continuously, with the dev agent on the same plan.

Ground rules, written before any new data:
- The presented numbers (CLAIMS 193-202) stay as they are. New runs are a pre-registered replication on the same code
  (v5-final 27442bd91d overlay), same method, same knobs. They are reported next to the original, never instead of it.
- A pooled estimate (original + replication) is shown only as declared here, with the timing disclosed: "replication
  decided on 10-25 after the original n=4 was seen".
- No pass/claim rule changes. Each job's launch file carries its rule before submit; the monitor reviews it.

Priority (value per GPU-hour):
1. REPL-COST (questions it answers: "are the wide CIs / the offline 32 -2% just noise?").
   - Healthy-cost pairs, 32/GPU only (the cells reviewers will ask about): offline, online and no EPLB.
   - +4 pairs per cell, ABBA, same harness.
   - Report: the replication's own mean and 95% CI, plus the pooled n=8 mean and CI per cell.
   - Rule for the offline 32 statement: if the pooled CI excludes 0, keep "significant"; if it includes 0, say "the
     original -2.0% did not replicate; pooled X% [CI]".
2. REPL-REC (question: "n=1-2 per recovery row").
   - Rank and node kills, no EPLB and online, +2 repeats each on v5-final; the standard KILL_FINAL rule.
3. MASK-SPILL check (question: "why is FT's per-step cost ~70-100 us?"). Explain, not ship:
   - build the kernel variant with the mask in registers / __shared__ on a side branch (never v5-pr);
   - cuobjdump -res-usage (24 B stack expected);
   - one EP32 a2a bench, stats and no stats. Prediction: +3.0 -> ~+1.0 us and +1.0 -> ~+0.4 us.
   - Confirms or refutes the mechanism in CLAIMS 200b.
4. Continuous red team (CPU):
   - a reviewer Q&A pass on the final doc every ~2 h;
   - the monitor and the dev agent each check the other's readouts.

Order: 1 and 2 in the two slots first; 3 when a slot frees (the C++ build is the long pole; start it on CPU now if the
build can run outside a job).
