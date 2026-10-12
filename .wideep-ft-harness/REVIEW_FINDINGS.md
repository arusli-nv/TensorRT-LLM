# Pre-PR review findings (2026-10-11) — tracking

Legend: [ ] open, [x] done, [~] documented/deferred with reason.

## Kernels / descriptor (delegated to kernel agent)
- [ ] K1 timeout in FT mode must fail-stop (trap), only `requested` cancels
- [ ] K2 remove test-only `abort_state` op arg and legacy mode
- [ ] K3 drop latch/published/slot; one fixed descriptor with inline abort record
- [ ] K4 gather: PDL, no zero-fill, fold sanitize, bound count; params only in FT branch
- [ ] K5 remove finalize-abort-output kernel
- [ ] K6 unused template param, ENABLE_DESCRIPTOR naming, redundant triggers, naming conventions
- [ ] K7 fixed-slot sizing keyed on recovery_enabled
- [ ] K8 tests lean (~250 lines), timeout regression, count bound
- [ ] K9 callers outside kernel files (runner.py latch call, ...)

## MPI adapter / session
- [ ] M-B1 `_interruptible` converts only when recovery enabled (`_host_world`); guard defer_terminations
- [ ] M-B2 mpi4py version: align with security scanning (>=4.1.2), re-port patch; image rebuild is an [infra] PR; ULFM capability gate
- [ ] M-S1 watcher handle with own stop + Cancel/Free; stop old watcher at recovery start
- [ ] M-S2 free revoked comms / watcher dup+request
- [ ] M-S3 assert MPI_THREAD_MULTIPLE
- [ ] M-S4 watcher non-peer error -> log + fail closed
- [ ] M-S5 one shared peer-failure classifier (torch-free)
- [ ] M-S6 BrokenExecutor counts as departure only if a rank failed
- [ ] M-S7 `_num_failed_ranks` guarded (returns 0 without ULFM)
- [ ] M-S8 grace starts at shutdown; no error logs for departed futures; flush before Abort; document
- [ ] M-S9 EPLB shared comm barriers tolerate departed rank (no /dev/shm leak)
- [ ] M-S10 real ULFM CPU test with mpirun (skip if unavailable)
- [ ] M-S11 gather fill_value audit / explicit
- [ ] M-N1..N9 nits (one dead-rank pattern, Dict-only, second-failure message, header year, weak test, test hygiene, MPI None, poll contention, env validated once)

## Authority
- [x] A-B1 Registered handshake; startup fails without it
- [x] A-B2 stable ROUTING_ID + ROUTER_HANDOVER
- [ ] A-B3 CI list (l0_cpu.yml)
- [x] A-S4 drop torch-free claim (test imports normally)
- [x] A-S5 single recovery deadline < worker deadline
- [x] A-S6 duplicates are violations
- [x] A-S7 drop role/generation param/dead code
- [x] A-S8 type-driven decode + __all__
- [x] A-S9 client errors one family (RecoveryTimeout)
- [x] A-S10 load_key with min length
- [x] A-S11 threat model documented
- [x] A-S12 MAXMSGSIZE, debug-level drop log
- [ ] A-S13 client cached per process (rank only)
- [x] A-S14 warn on empty admission
- [ ] A-S15 tests: ipc endpoint, daemon threads
- [ ] A-S16 tests: wrong key drop, malformed abort, Installed wrong gen, timeout, out-of-range rank, rank 0 never admitted, client rejects unauthenticated; split combined test

## Executor
- [ ] E-B1 create() rejects spec decode/MTP, block reuse, guided decoding, KV connector, beam>1, gather_all_responses; document sampler features
- [ ] E-B2 ADP dummy rollback via `_finalize_adp_dummy_allocation(False)` first; test loss in `_can_queue`
- [ ] E-S3 numbered gathers; resend all unacknowledged contributions
- [ ] E-S4 terminations deferred in every gather path (central)
- [ ] E-S5 broad handlers re-raise as interruption when peer failed
- [ ] E-S6 gather_all_responses; cache client per process
- [ ] E-S7 commit gate bound documented
- [ ] E-S8 narrower executor boundary (adapter methods; unbroadcast hook)
- [ ] E-S9 owner map n>1 children
- [ ] E-S10 split unrelated rpc/llm/_update_requests changes into separate PRs
- [ ] E-N nits: unused imports, import order, stale comment, blank lines, perf events, redundant check, TypeAlias, create() tests, KV round-trip GPU test, gate-in-loop test, dead test side_effect
- [ ] X request_abort() at start of recovery (from kernel review)
