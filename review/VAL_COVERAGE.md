# VAL-V5 coverage vs FREEZE_CHECKLIST (monitor, 2026-10-22 08:36 PT)

Read-only subagent over a copy of mvp/harness (job.sh/serve.sh byte-identical to Lustre, cmp checked). Monitor verified
items marked [V]. a = in launch/val_v5_cmd.txt now; b = harness supports it, launch file doesn't use it; c = harness can't.

| Row | Requirement | St | How / what's missing | Evidence |
|---|---|---|---|---|
| C7 | SIGSTOP, hang detection on | a | `stop` row in group 8; KILL_FINAL_VERDICT | job.sh:3783-3785 |
| C38 | kill matrix detector off + SHUTDOWN trio | a | rank5/node1/stop + SHUTDOWN-R1 (SHUTDOWN_VERDICT) | val_v5_cmd.txt:20-21 |
| C18/H6 | twokill | a | `twokill` row, KILL_FINAL_VERDICT incl. ft_watch | job.sh:3774-3782 |
| C25 | SIGSTOP near watchdog bound | a partial | `stop` hard-codes hang_confirm_s 30; near-bound variant needs an edit | job.sh:3783 |
| C19 | KV capacity on vs off; kill at full KV | a+b | capacity = manual read of STD pair (no rule); full-KV kill = KF_LOAD=512 KF_ROWS=rank5 (KV actually full UNVERIFIED) | job.sh:3734-3739 |
| C8 | kill during online EPLB update | b | KF_ROWS=rank5upd + online config; `WIDEEP_KILL_PHASE eplb_update` | job.sh:3771-3773; testhooks.py:586-593 |
| C14 | PIN_CHECK OK with a v5 seed | b | drop FT_COST_E2E_SEED -> tuning pass writes pinned.json; launch still pins v2 std-seed2.json | job.sh:3116-3137,3167-3171 |
| C30 | main parity 32+128, ABBA | b + BUG [V] | `std-noeplb-main` sets main's dist but keeps exported MVP_OVERLAY -> v5 Python overlaid on main's wheel. Needs `export MVP_OVERLAY=` as noft-main does | job.sh:2993-2995 vs :2030 |
| C17 | token check per mode, divergence <= A/A | c [V] | EXACT-FINAL refuses MVP_OVERLAY, runs dep32-final/b2-320 not std-*, requires exact equality after the kill, no A/A pair | job.sh:2264, 2323-2332 |
| C10 | recovery-deadline kill | c | no row/verdict; pieces exist (RECOVERY_TIMEOUT_S, TLLM_TEST_DELAY_ABORT_S) | serve.sh:198; testhooks.py:431 |
| C23 | canary late-write kill | c | hooks exist, no canary write/check | testhooks.py:40-45 |
| H4 | fixed kill step | b | WIDEEP_KILL_AFTER_STEPS=N; twokill/stop rows never pass it | job.sh:3738-3740,3781,3785 |
| H5 | print_iter_log A/B | b | arm suffixes -noil / -loc | job.sh:2930-2946 |
| H7 | page-cache drop before burn-in | a data / c rule | a 2nd drop starts in background right after server_ready, i.e. during burn-in | serve.sh:131-134,283-286 |
| H8 | node health fails the group | b partial | preflight fails only with WIDEEP_POLLUTION_CHECKS=hard (launch uses 1); no /usr/bin tool check | serve.sh:100-128 |
| H9/H31 | client tok/s fixed windows; prefill kill | b data / c readout | WIDEEP_TIMED_CLIENT=1 writes tokens.jsonl nobody reads; KF_ROWS=rank5pf exists | bench.py:177-180; job.sh:3767 |
| H11 | ListenOverflows -> INVALID | a data / c rule | counters logged per case, no readout rule | job.sh:35-55 |
| H15 | kill runs node-local logs | b | TLLM_TEST_LOCAL_LOGS=1 in K | rank.sh:38-41 |
| H16 | kill-trigger polling cost | c | Lustre trigger at every decode replay; no node-local trigger / no-trigger control | testhooks.py:157,180 |
| H17 | pre-registered headline | b/c | finals_readout has rank-0 steady median; client tok/s fixed window missing | finals_readout.py:7-12 |
| H19 | hang detection on in healthy FT arm | b | TLLM_FAULT_TOLERANCE_HANG_DETECTION=1 + HANG_CONFIRM_S in E | serve.sh:199-200 |
| H26 | FT off on ULFM launcher | b | arm std-noeplb-ulfm | job.sh:2991 |
| H23 | perf in kill verdict | c | no after/before ratio or resume bound | job.sh:3711-3720 |

Stale in val_v5_cmd.txt (and a/b): (1) header + `O=` name f96a8c1a53/v5f; candidate is build-v5-2ccc3e8e1c + v5h-2e8d6d7da1
(or v5-land's build). (2) transport text "ob1 + sm" wrong: launcher picks ob1 self,tcp, sm opt-in (job.sh:3018 -doc comment
stale too). (3) gate g() uses point-units-f96a8c1a53 / v5f tests and lacks test_py_executor.py, test_recovery_authority.py,
test_moe_a2a_late_writes.py, test_nvlink_one_sided.py (split2 has them); keep test_mpi_survivors_ulfm.py. (4) STD pins the
v2 seed (blocks C14), n=1 ROUND_MIN 25, no hang detection (H19). (5) plan items missing: hang detection in healthy arm, KV
capacity rule, retryable-error path, token-divergence check, main parity. "rank 0's leader is not exec'd" UNVERIFIED on 2e8d6d7da1.

Wall time (estimates from harness comments): all a+b in one 16-node job ~235 min (healthy group) / ~165 min (kill group),
over the 7000 s bundle and 2 h QOS -> two jobs: VAL-H (healthy: main parity group | attribution + hang-det + v5 seed group)
and VAL-K (A: rank5/node1/rank0/rank5@512; B: twokill/stop/rank5upd/SHUTDOWN-R1). c-rows need harness work or a user drop.
