# Code-owner review round 1 (7 agents, 2026-10-30): decisions
Raw reviews: review/stack/r_g{1..7}_out.md. FIX = code change; WHY = keep, reason goes in the PR text.
- PR03 error text to clients: WHY (completions already sent str(e); OpenAI clients need the cause); add a completions
  multi-prompt failure test if cheap. PR04: WHY + data (CLAIMS 166/168) + rank-identical inputs (replicated queue,
  broadcast requests) in the message. PR06: FIX verify install_mpi4py.sh in the container (job).
- PR07 HMAC string match: FIX (typed exception). Restricted unpickler: FIX. Outbound replay: WHY (localhost default,
  same trust model as TRT-LLM's HMAC ZMQ IPC). PR08 fence_departed / fence target: WHY (zombie-rank safety; key
  holders are trusted). Command-fence single slot, _failed_during_recovery reset, Abort.reason size, serve-loop
  isolation: FIX. PR09 test lists, long-key HMAC test, _kill_rank IndexError: FIX.
- PR10 test_mpi_survivors.py in l0_cpu: FIX. Native watch test: check ULFM test. Per-collective Get_failed: WHY (in
  the measured healthy cost). PR11 stop every watch before finalize, rename shadowed method: FIX. Abort on failed
  survivor task: WHY (exits_alone handles the recoverable cases).
- PR12 local-bit check in release_workspaces, expert_to_slot check in the launch: FIX. PR14 retry invariant doc: FIX.
  PR15 CFT auto-selected under recovery: FIX (fall back to fence). Eager watchdog docstring: FIX. Detection-only +
  CUDA graphs config-time error: FIX.
- PR16 message (mFaultToleranceMode): FIX. Split: WHY. PR17 extra_slots_mode collective: FIX; digest dtype: FIX;
  hostname node key: WHY for now. PR18 shm reload leak: FIX; assert->ValueError: FIX; fail-late sizing: WHY.
  PR19 numerics: WHY (integration restore tests run the real kernels with extra slots, token-exact).
- PR20 duplicated revert: check/FIX. PR21 live-rows helper: FIX; fakes without departed: FIX. PR22
  gather_all_responses requirement: FIX; exits_alone test: check. PR23 health probe off the default pool: FIX;
  retryable gate on recovery: FIX; body detail: WHY.
- PR25 TLLM_REQUIRE_ULFM scope: FIX; FORCE_CFT: covered by PR15 fix. PR26 TODOs/stale numbers: FIX (CLAIMS or drop).
