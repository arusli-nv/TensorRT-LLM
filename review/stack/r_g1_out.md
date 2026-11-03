# Review of PRs 01–06 (WideEP fault-tolerance stack)

Review is by reading only (`/tmp/claude-161238/wt-stack`); nothing was run. `commits/` holds 26 diffs and I reviewed 01–06.

## PR 01: fake `moe_a2a_*` registrations accept the real args
No issues found.
- The fake argument order matches the real schema (`cpp/tensorrt_llm/thop/moe/communication/moeAlltoAllOp.cpp:1135-1156`).
- The test is a CPU-only FakeTensorMode trace.
- Minor, not blocking: nothing enforces that fake and real signatures stay in sync, since the real schema also has `abort_state` and `expert_to_slot`, which the stack adds to the fakes in later PRs. A schema-vs-fake signature comparison test would stop this recurring.

**Verdict: approve.**

## PR 02: `EPGroupHealth.__deepcopy__` returns `self`
No blocking issues found.
- **minor** `ep_group_health.py:103`: this applies to every deepcopy of the class, not only the ConfigurableMoE path.
  - Today that is intended: the object is held in `extra_attrs[_HEALTH_KEY]` (`wide_ep_ft.py:35`), and the commit says "Nothing changes where nothing reads it".
  - Any future snapshot-by-deepcopy use would silently alias the state.
  - The docstring covers this, so I'm only flagging it.
- The test is adequate: it asserts identity and goes through `_create_and_sync_backend` with a quant override.

**Verdict: approve.**

## PR 03: end failed OpenAI streams with error event and one `[DONE]`
The commit message is clear and says "This applies to every user".

1. **major** `openai_server.py:326-342`, `_end_stream_on_error`: raw `str(exception)` is sent to every client.
   - The `/v1/chat/completions` stream previously only logged and re-raised.
   - The stack already treats upstream error text as sensitive. `openai_disagg_server.py:187-211` (`_upstream_error_message`) says it avoids "leaking the worker".
   - Failure scenario: a CUDA, NCCL or ZMQ error whose text contains hosts, paths or ranks now reaches the HTTP client.
   - The completions path already sent `str(e)`, so that part is not new. Chat and the Anthropic route (`anthropic_adapter.py:1250-1258`) now forward it too.
   - Smallest fix: send a generic message for non-`OpenAIError`/`ValueError` exceptions and log the detail.
   - Not justified by the message.
2. **minor** `openai_server.py:326-342`: the wrapper swallows every exception after `[DONE]`.
   - A failure in `_extract_metrics` (`:2158`, `:3260`) is now only logged. Before, it propagated and broke the connection.
   - That is sensible, and the docstring says so, but the message does not mention that metrics failures are now silent.
3. **minor** `openai_server.py:3064-3079` and `:3100-3107`: error termination is split across two places.
   - `completion_generator` yields the error event and relies on `generator_wrapper` for `[DONE]`.
   - It calls `_stream_error_events(e)[0]`, which builds a list only to index it.
   - That is coupling by comment ("generator_wrapper ends the stream with [DONE]").
   - A `_stream_error_event(e)` helper returning a single string would be cleaner.
4. **minor** `anthropic_adapter.py:1250`: `"error" in payload` is checked before the `ChatCompletionStreamResponse` parse, so any chunk with an `error` key is treated as fatal.
   - `ChatCompletionStreamResponse` has no `error` field, so this is safe today.
   - Only the `str`/`dict` case is tested. A malformed `error` value (list, int) falls back to `str(message or ...)`, which is fine.
5. **minor, test coverage**: there is no test that a failure before the first chunk (e.g. `anext(promise)` raising) yields error plus `[DONE]` with no prior data, and none for the multi-prompt completions path.
   - The commit message claims "one error event per failed prompt, while the other prompts keep streaming, and a single [DONE]".
   - No test drives `merge_generators` with a failing producer.
   - The wrapper test uses a single generator.
   - Add one test with two prompts where one fails.

**Verdict: approve with changes** (items 1 and 5).

## PR 04: release ADP context hold after a burst ends
This changes scheduling for every ADP-balance user with no opt-out. The message says "This applies to every user" and "[perf]", but states no measurement.

1. **major** `py_executor.py:7833-7839` (diff `:75-82`): there is no benchmark or data, and the heuristic is hard-coded.
   - `ADP_BURST_QUIET_ITERS = 4` is a class constant, not a config field. `timeout_iters` and `low_occupancy_timeout_iters` are user knobs.
   - `burst_ended` releases `context_requests` on ranks that have context while other ranks have none. That is exactly the imbalance the balancer exists to prevent.
   - It can only trade throughput for latency, and the claim is "closed loop at low load", which is the test docstring's rationale only.
   - Failure scenario: a steady trickle of two requests per window, then a quiet gap of four decisions. Context is released unbalanced although more arrivals are imminent.
   - Fix: attach benchmark numbers to the commit, and gate it behind an `attention_dp_config` field, defaulting off until it is measured.
   - Not justified.
2. **minor** `py_executor.py:7619`, `adp_routed_since_balance += len(all_new_flat)`: the claim "Every rank routes the same requests, so these agree across ranks" holds only if `all_new_flat` is rank-identical.
   - The comment asserts it without a test. Divergence would make ranks disagree on `burst_ended` and release on different iterations, which are ADP-synchronised decisions.
   - Please state which collective guarantees it (the allgather at `:7524`?).
3. **minor** `py_executor.py:7835`: `len(self.waiting_queue) == 0` is rank-local, and so is `adp_quiet_iters`.
   - `waiting_queue` may differ per rank (it is checked locally at `:5881` and `:2549`).
   - If it can differ, `burst_ended` can differ across ranks, again desynchronising the release.
   - Verify that the queue is rank-identical, or include it in the gathered counts.
4. **minor**, state hygiene: the state in the diff (`adp_routed_since_balance`, `adp_recent_arrivals`, `adp_quiet_iters`) is only reset implicitly.
   - Nothing clears it when `attention_dp_enable_balance` is off, and `adp_routed_since_balance` still accumulates.
   - It is bounded, so this is harmless.

**Verdict: request changes** (items 1–3). Making it config-gated and giving data would turn this into approve.

## PR 05: `"memory"` clobber on release/acquire fences
Correct, and it matches the already-clobbered fabric fences (`moeAlltoAllKernels.cu:828`, `:837`). The fix is at `:732-734` and `:1896-1898` in the tree, and the commit message is sound.

- **minor, residual**: other flag/payload-ordering asm in the same file has no `"memory"` clobber:
  - `:274`: `ld.relaxed` of `abort->requested`.
  - `:1146`: `ld.relaxed` of `recvCountPtr`.
  - `:1234`: `ld.relaxed` of the data counter.
  - `:2090`: `ld.relaxed.u64` (uses a clobber-less form).
  - If the later payload reads in those paths rely on ordering after the loop, the same hoisting hazard applies.
  - Ordering at those sites needs to come from a fence, and I did not verify that. Please confirm in the PR description, or add the clobber or an explicit fence.
- **minor**: there is no test and no repro of the bug. This is acceptable for a compiler-barrier fix, but the message states a failure ("a peer could see the flag before the data") without evidence such as SASS before and after.

**Verdict: approve** (with a request to audit the sibling asm sites).

## PR 06: mpi4py 3.1.5 → 4.1.2 in the image
The commit message says "This changes the image for every user".

1. **major** `docker/common/install_mpi4py.sh:23-71` (patch applied via `git apply`): the patched code runs inside mpi4py's own `_comm_executor_helper`, and I could not verify that it works.
   - It uses `os` and the names `rank` and `comm`. I cannot confirm that `os` is imported in `src/mpi4py/futures/_core.py` at 4.1.2, or that the hunk anchor `@@ -538,5` matches (that is why this is unverified).
   - If either is wrong, `git apply` fails at image build, or the patched code raises `NameError` at runtime under `TRTLLM_USE_MPI_KVCACHE=1` only.
   - Fix: have the script verify the patch applied (`git apply --check` first), and add a smoke import test, or add a CI check for MPI-KV-cache mode.
2. **major, for users who do not enable FT**: this is a major-version bump of the MPI binding for the whole image.
   - There is 3.x → 4.x API and behaviour change for the 95 `mpi4py` uses across 30 files (`llmapi/mpi_session.py` has 31). Examples are `MPICommExecutor` semantics and pickling-by-default changes.
   - The message gives no test evidence beyond "needs ULFM".
   - Provide a pre-merge run of the existing MPI test lists (`tests/unittest/llmapi` with `mpi_session`, the disagg MPI-KV tests).
3. **minor**: version pins disagree.
   - `requirements.txt:18` is unpinned.
   - `ATTRIBUTIONS-Python.md:16915` says 4.1.1.
   - `security_scanning/pyproject.toml:21` says `>=4.1.2`.
   - Pip installs from source, so 4.1.2 is fixed only by the image script. A pip-installed wheel, or a non-image install, gets any version.
   - ATTRIBUTIONS should be updated to 4.1.2.
4. **minor**, `install_mpi4py.sh:67`: the carried-over `print(...)` debug line stays, and its message now says "in mpi4py _manager" (it was `_manager_split` before). It prints on every worker process.
   - That is pre-existing, but you are re-porting the patch and could drop it.
5. **minor**: the removal of `Cython==0.29.37` on aarch64 and of the setuptools<69 pin goes without verification on aarch64.
   - `cuda_libs` or other installs may still need the pin.
   - The message says only "drop the setuptools<69 pin".
   - This needs an aarch64 build log.
6. **minor**: `curl -L ... | tar` has no checksum verification.
   - Pre-existing, but this is a version change on a supply-chain-sensitive path.
   - Add a sha256 for 4.1.2.

**Verdict: request changes** (items 1 and 2).

## Summary

| PR | Verdict |
|---|---|
| 01 | approve |
| 02 | approve |
| 03 | approve with changes (generic error text; multi-prompt failure test) |
| 04 | request changes (data, config gating, rank-sync claims) |
| 05 | approve |
| 06 | request changes (patch verification; regression evidence for the 4.x bump) |
