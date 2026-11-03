# Round 2 review: PRs 20–23 (commits 20–23, i.e. PRs 14–17 in the series numbering)

I read all four diffs and the surrounding tree. For PR 22 I read the whole of `peer_loss_recovery.py` plus the executor hooks. I did not read the test files in full. I grepped them for specific scenarios and read the cited tests. Bash was unavailable, so nothing was run.

## Round-1 findings

**PR 20 (KV Cache Manager V2 step revert)**
- **1. Duplicated generation shrink: FIXED.** `_give_back_generation_step` is now shared by `revert_allocate_generation` and `revert_prepare_resources`, and the `reverted_cap < 0` handling is no longer duplicated.
  - The new branch still does not decrement `py_helix_decode_group_index`. That is JUSTIFIED: CP/helix layouts are refused under recovery ("PP/CP/expert TP refused", enforced in `MPIDist.enable_peer_loss_recovery`).
- **2. Test gaps: STILL OPEN, minor.** I found no test for the `if not inflight: continue` path or for the `kv_cache.resize` failure raising `RuntimeError`. Not blocking.
- **3. Ordering requirement: FIXED in the PR text** ("must run after the executor has restored `context_current_position`"). The method's docstring still omits it, which is minor.

**PR 21 (attention-DP with absent ranks)**
- **1. Copy-pasted live-rows filter: FIXED.** `live_rows()` in `communicator.py:201` is used at all three sites (`cuda_graph_runner.py:394`, `:1612`, `py_executor.py:7780`). Its precondition (one row per logical rank) is now stated in the PR 15 description.
- **2. Fakes without `departed`: FIXED.**
  - `fake_dist.py:134,137`, `test_adp_host_exchange.py:263,278` and `test_adp_absent_ranks.py` now accept the keyword.
  - Out-of-tree `Distributed` subclasses must now accept `departed=`. That is JUSTIFIED: the PR 15 and PR 04 descriptions state it, and it follows from the ABC contract.
- **2b. Test lambdas: not re-raised** (minor).
- **3. Bug-compat pin handling: JUSTIFIED.** The commit message and PR text say "as on main", and the behaviour is gated by `peer_loss_recovery_enabled()`.
- **4. Other routers still use lists: FIXED.**
  - The `_assign_explicit_dp_ranks` docstring no longer claims it is shared with ConversationAware.
  - Only `DefaultADPRouter` calls it (`adp_router.py:696`).
  - `_requirements` enforces `type(adp_router) is DefaultADPRouter` (`peer_loss_recovery.py:198`).
- **5, 6 (`_fill_admit_cap`, `qsize` with the wake item): minor, not re-raised.**

**PR 22 (recovery and executor hooks)**
- **1. `gather_all_responses` silently disabling recovery: FIXED enough.** The estimation phase runs on the same executor, and `run()` returns early while `gather_all_responses` is set. During that phase `watch_startup_peer_failures` stays armed, so a peer loss ends the job rather than being reported as recoverable. The `run()` docstring and PR text now say so. A `_requirements` entry would still be cleaner, but that is minor.
- **2. Size and structure: STILL A PROBLEM, minor-to-major.**
  - The file is 1846 lines with 91 defs, and the decision was "Split: WHY".
  - No reason appears in the PR 16 text, so I cannot call it justified. Either add one sentence of reasoning (a single state machine whose stages share `_executor`, `_client` and `_kv_lent`) or move `_deliver_responses`/request redelivery out.
  - Gating is now uniform: `.enabled` guards every per-step hook call. The unguarded `defer_terminations` and `peer_failed` calls are on exception paths and are the base no-ops. Fine.
- **3. SIGKILL on a string-matched error: FIXED.**
  - `exits_alone` requires `survives_own_loss`.
  - `test_peer_loss_recovery.py:1251` and `:1292` (parametrized, asserts `SIGKILL` for classified and unclassified messages) cover it.
- **4–7: minor, not re-raised.** Item 6 (PP) is covered by "PP is refused" in the PR text.

**PR 23 (`/health` and 503 on peer loss)**
- **1. Topology in the unauthenticated body: JUSTIFIED.** The body carries "what operators alert on", and the abort reason (ranks, pids, hosts) is withheld. The residual leak is rank counts and ids only. The schema is now a public contract, which the PR text should state.
- **2. Probe on the default thread pool: PARTLY FIXED.**
  - `health()` now awaits `asyncio.wrap_future(probe.status_future())` on the probe's own thread, so the default pool is no longer starved.
  - It introduces a new queueing problem; see N2 below.
- **3. Retry gate on message text: JUSTIFIED.**
  - The gate is still `fault_tolerance_mode()`, not `peer_loss_recovery_enabled()`.
  - The test docstring gives a sound reason: `peer_loss_recovery_enabled()` raises `ValueError` when the authority is set without the mode, and an error handler must not raise.
  - A tagged substring in user text can only make the requester's own error a 503. That is acceptable.
- **4, 5: documentation points, unchanged.**

## New issues

**N1. major. PR 22. A non-peer forward error leaves a stale prepared step, and the next commit fails with `RuntimeError`.**
- Where: `peer_loss_recovery.py:659` and `:709-712`; `py_executor.py:5528`, `:5635-5641`, `:6565`, `:6670-6693`, `:9393-9413`.
- With recovery on, `step_prepared` runs for every V2 step. It is undone only by `step_released()`, which is called only when `can_queue` is false before the forward.
- If `_forward_step` raises something that is not a peer failure, `_raise_if_peer_failed` passes it through and `_handle_errors` fails the requests. `batch_outputs` is `None` and the step is neither released nor committed (`commit_step` runs only when `batch_outputs is not None`, or for `previous_batch` in the overlap loop).
- The stale `PreparedStep` stays at `_uncommitted_steps[0]`. The next commit raises `RuntimeError("Steps must be committed in the order they were prepared")`. That error is not a `HostCollectiveInterrupted`, so it ends the job.
- A request-scoped or budgeted forward error (which main survives) becomes a job-ending crash with fault tolerance on.
- If a peer loss arrives first, `discard_interrupted_steps` calls `revert_prepare_resources` on already-terminated requests.
- Test coverage: the only forward-failure tests in `test_peer_loss_recovery.py` are the peer-failed ones; none covers a non-peer forward error.
- Fix: call `step_released()` in both `batch_outputs is None` branches (non-overlap `:5655`, overlap `:6670`). Also release on a `_sample_async` failure. Add an executor-level test that a non-peer forward error followed by a normal step commits cleanly.
- The commit message does not address this.
- Confidence: I traced it by reading the code and did not run it.

**N2. major, low likelihood. PR 23. A queued `/health` poll re-queries the authority when the authority is connected but unresponsive.**
- Where: `fault_tolerance_status.py:152-157`.
- `_cached_status` stamps `_status_time = now` using the time taken before the query. After a 0.5 s timeout the cache entry is already older than `max_age_s` (0.1 s), so every queued task re-queries.
- With the authority stopped or hung (so the message is sent, and `SNDHWM:1` does not short-circuit), N concurrent polls take about N × 0.5 s on the single worker.
- A load balancer with a 1–5 s timeout then marks a server that is serving fine as unhealthy. The "authority down" test passes only because of the HWM fast-fail.
- Fix: stamp `_status_time = time.monotonic()` after `_query()`. Also cap the queue, or skip the query if a newer result arrived while the task waited. Add a test with a router that accepts and never answers.

**N3. minor. PR 21 scope.** The commit adds `MoeExecutionState.for_model` wiring in `DecoderRunner` and `ModelEngine.close()`. The message "Keep attention-DP routing serving with absent ranks" does not mention it, and it belongs with PR 22's `moe_execution_state` requirement. This is non-blocking, but move it or mention it.

## Verdicts

| PR | Verdict | Blocking items |
|----|---------|----------------|
| 20 | **approve** | None. Optionally add the two small tests. |
| 21 | **approve** | None. Mention the `DecoderRunner` wiring in the commit message. |
| 22 | **request changes** | N1. Also state the reason for the module size, or split it. |
| 23 | **approve with changes** | N2 (the timestamp fix is one line plus a test). |
