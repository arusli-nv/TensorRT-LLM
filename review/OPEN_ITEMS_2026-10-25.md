# Open items, and the plan to close each (2026-10-25 ~16:35 PT)

User: "figure out everything; use the GPUs". Ordered by value. Every GPU test carries a decision rule written before
the data.

1. **What in FT's all-to-all kernels costs ~0.85 us per round trip (~50 us/step)?**
   - CPU: diff FT vs main kernels and list the per-call FT-only work (agent, review/diag/a2a_ft_work_coderead.md).
   - GPU: one side-branch build with compile-time ablation variants, each removing ONE FT item (mask checks, 8-byte
     flags, abort-flag check, stats-loop mask). Benchmark them in one 8-node microbench job against the control.
   - Rule: an item "explains" the cost if removing it cuts the no-stats delta by >= 0.3 us at EP32 T64.
2. **Stats-path growth with EP (~3 us at EP32, hidden in serving).** Covered by item 1's variants (stats-loop mask
   removed; stats writes skipped).
3. **Healthy hang (1/36, CLAIMS 204): FT transport or chance?**
   - GPU: a healthy soak, online EPLB 32/GPU for several hours. FT (ob1/TCP) and FT off forced onto the same
     ob1/TCP transport, on 2 groups.
   - Rule: hangs in both -> the transport; FT only -> FT; neither -> rare, bound the rate.
   - Runs overnight in the slot freed after the replication.
4. **Rest of the steady cost (~20-40 us/step not in the a2a microbench).** After item 1: one serving pair, same
   nodes, with the best ablation variant vs v5-final; nsys only if still needed.
5. **Test infra:** test_moe_comm's first test times out at setup under --timeout (both kernels); test_moe_module
   never ran to completion. Cheap 1-node runs: test_moe_comm without --timeout, then test_moe_module in full.
6. **Small:** 7 of 33 FT far stalls not covered by another rank's prefill (CLAIMS 201); the frozen-vs-v5-final node
   kill server-side gap of ~0.1-0.5 s. Log reading only.
7. **Out of reach today (stated as untested):** two frozen ranks at once, CFT (driver 580), host death, NVLink faults,
   64+ GPU scale.
