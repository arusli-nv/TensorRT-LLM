# Monitor session handoff (written 2026-10-22 ~08:20 PT)

Role: monitor / logistics / red team / first-principles reviewer / problem solver for the WideEP FT project.
The dev agent (currently `repo-7e`, runs in /tmp/claude-161238/work/repo) codes, builds and submits jobs; the
monitor verifies every result from run files, steers the critical path, and reports to the user.

Source of truth: review/EXEC_PLAN.md, review/FREEZE_CHECKLIST.md, review/HANDOFF.md (monitor log entries are
"- HH:MM PT monitor: ..."), mvp/CLAIMS.md, DECISIONS.md. Memory (shared) holds the user's rules: see MEMORY.md,
especially final-rerun-protocol, high-yield-work, question-the-measurement, overnight-retro-2026-10-22.

State at handoff:
- v5 (branch of record) = v3 + fence-always + design C (per-signature dispatch regions) + exec launcher with TCP
  default (5ebf7f72a2) + ADP gate fix (2e8d6d7da1). Unit gates green; C43 build-sequence hang = harness bug, fixed.
- Main-side PR branch prmain/08-adp-gate-quiet (c7740d7e83) for the gate fix; no PR opened (user opens PRs).
- Open P0: no-EPLB 16-48 req/GPU healthy cost ~0.25 ms/step (~1.8% trim @32), localized to FT's bound a2a path.
  Receive-warp fix, B' and the 384 tier recovered only ~0.04 ms (ABFIX, RECVAB) and stay off v5.
- Running/queued: ABEXP 8158388 (experimental token-major kernel v5-exp, net -112 lines; off | ft-v5 | ft-exp
  @24/32, mirrored; rule: >=2/3 recovery -> bisect the 3 commits; 1/3-2/3 -> bisect anyway; neutral perf ->
  propose landing for design simplicity). Then SPLIT2 8158390 (off/ft/noghost/unbound/noslottable @32).
- Pending user decision: if ABEXP doesn't close the gap, freeze with it documented vs hold for more kernel work
  (user: "give it your best shot then get back to me"; monitor leans freeze-and-document).
- After that: VAL-V5 (validation on the freeze candidate, incl. hang detection on in the healthy FT arm, KV
  capacity, retryable-error path, token-divergence check, main parity) -> CODE FREEZE -> final runs (one job per
  mode on fixed nodes, ABBA n=4, 95% CI) -> figures/deck (mvp/analysis/slides_deck.py) -> doc/PRs.

Tools: stall.py / stepdist.py / ctxdist.py in /tmp/claude-161238 (decode-only rank-0 trimmed mean and stall
ms/step). Slurm via the broker MCP; 2 concurrent jobs (QOS short).

On start: re-create the loops: `/loop 10m <overnight monitor prompt>` and a 3-hour report cron; ListAgents to get
the dev agent's current name; message it once that the monitor moved sessions.
