# Round 2 review: PRs 07, 08, 09 (PR 01, 02, 03 in the descriptions)

I read the current code and tests in the stack tree. I did not run anything. With `TLLM_FAULT_TOLERANCE_MODE` unset I found no behaviour change, same as in round 1.

## Round 1 findings

### PR 07 (01)

| # | Finding | Status |
|---|---|---|
| 1 | Error-string matching decides authentication | **FIXED** |
| 2 | Authority unpickles any class | **FIXED** |
| 3 | Server-to-worker messages have no freshness | **JUSTIFIED** |
| 4 | Protocol and client tests land one PR late | **JUSTIFIED** |
| 5 | `lru_cache` on one env accessor only | Unchanged, minor, not re-raised |
| 6 | `load_key` not thread-safe, `extra_slots` docs | Unchanged, minor, not re-raised |
| 7 | Heavy import for a "standalone" process | Unchanged, minor, not re-raised |

- **#1 FIXED.** `ipc.py` still raises the string-based `RuntimeError`, but `MessageQueue._parse_data` (`client.py:136`) now raises typed `Unauthenticated` and `MalformedMessage`, and the ROUTER branch (`client.py:126-134`) types the frame-count error too.
  - The `RuntimeError` and `ValueError` cases are pinned by `test_serve_raises_errors_other_than_authentication_failures`.
  - The unpickle exception tuple at `client.py:145` is still incomplete; see N1.
- **#2 FIXED.** `_ProtocolUnpickler.find_class` (`client.py:104-110`) allows only `protocol.*` names, resolved through the `_CLASSES` allowlist. `is_message` then checks the field types exactly.
  - The module test is `module != protocol.__name__`, and `protocol.py` has no `from __future__ import annotations`, so `field.type` holds real types and `_typed` works.
- **#3 JUSTIFIED.** The stated reason is the localhost default and the same trust model as TRT-LLM's HMAC ZMQ IPC.
  - The module docstring tells operators to bind a private interface and use a per-job key.
  - The residual risk is that an injected `Restart` ends every worker with `os._exit(75)`. That is acceptable for an experimental feature.
- **#4 JUSTIFIED.** The description says "Protocol and client wire tests ship with the authority tests in PR 02". That is acceptable inside one stack.

### PR 08 (02)

| # | Finding | Status |
|---|---|---|
| 1 | `fence_departed=True` hard-coded, no opt-out | **JUSTIFIED**, with one open evidence gap |
| 2 | Fence target taken from the worker | **JUSTIFIED** |
| 3 | Single slot for the command fence | **FIXED** |
| 4 | `_failed_during_recovery` never cleared | **FIXED** |
| 5 | `Abort.reason` exceeds `MAX_MESSAGE_BYTES` | **FIXED** for `abort()`, **partial** elsewhere (minor) |
| 6 | No failure isolation in `serve` | **STILL A PROBLEM**, partial |
| 7, 8 | Unbounded send queue, docstring leftovers | Minor, not re-raised |

- **#1 JUSTIFIED.** Quote: "A rank declared failed can still hold its GPU and its mappings of the survivors' memory … the recovery aborts if the runtime refuses or does not answer within 2 s." Failing closed is sound, and the guide (lines 115-121) documents it.
  - Gap: the guide says the kill is "a no-op for a process that already ended". That is the node-loss case, the headline use case. Nothing in the tree shows PRRTE returns success for a rank on a dead node.
  - If it returns an error, every node loss aborts.
  - Smallest fix: put the measured result of a real node kill in the PR text. If it fails, add the opt-out.
  - I am not re-raising the missing opt-out itself.
- **#2 JUSTIFIED.** The description states that key holders are trusted and the values are only checked to be plain. The command has no shell and receives no key (`authority.py:818-841`).
- **#3 FIXED.** `_command_fences` is now a list (`authority.py:172`, `:354`), drained per rank in `expire` (`:282-294`). `HangMonitor._confirmed` keeps one entry per rank.
- **#4 FIXED.** `_failed_during_recovery` is reset at `authority.py:475`.
- **#5.** `abort()` truncates to 4096 characters (`:377`), but two paths do not:
  - `reject()` (`:369`) and the serving-phase `Abort(str(violation))` (`:215`).
  - Both can carry a `repr` of a worker-sent message.
  - The only consequence is that ZeroMQ drops the sender's own connection. That is minor, so I am not blocking on it.
- **#6.** The decision was FIX. `serve` (`authority.py:655-686`) handles only `MalformedMessage` and `Unauthenticated`. Nothing wraps `authority.receive`, `handle`, `expire` or `_send`.
  - A bug in a state handler, such as a `KeyError`, ends the single point of failure. Workers then time out.
  - The test at `test_recovery_authority.py:1067` pins the opposite: `serve` re-raises anything else.
  - The PR text claims only "keeps serving" for malformed messages, so it does not state a reason for skipping the rest.
  - Fix: catch `Exception` around the per-iteration body, log it, and return `authority.abort(...)`. Then change that test.

### PR 09 (03)

| # | Finding | Status |
|---|---|---|
| 1 | `test_hang_evidence.py` not in a test list | **JUSTIFIED** (weakly) |
| 2 | HMAC key longer than one block untested | **FIXED** |
| 3 | `_kill_rank` can raise `IndexError` | **FIXED** |
| 4 | Single-slot runtime-fence thread | **JUSTIFIED** |
| 5 | Fragile libpmix discovery | Unchanged, minor, not re-raised |
| 6, 7 | Threading contract, C++ details | Unchanged, minor, not re-raised |

- **#1 JUSTIFIED, weakly.** The description says "Listed in `l0_b200.yml` (the entry arrives with PR 06)". I confirmed `l0_b200.yml:193` is added by commit 12, not 09.
  - PR 03 alone ships tests CI never runs. The order is stated and the stack merges together, so I accept it.
  - Preferred fix: add the entry in PR 03 itself.
- **#2 FIXED.** The test at `test_hang_evidence.py:192` is parametrized over `b"k"*32` and `bytes(range(100))` (`block_key`, `long_key`).
- **#3 FIXED.** `hang_evidence.py:302` checks `0 <= rank < len(self._pmix_ranks)` before indexing.
- **#4 JUSTIFIED.** The description states "a later fence is skipped with a warning while one is outstanding, and the authority's command fence is the next fallback". Command fences are per rank now, so the fallback is sound.

## New issues in the current diffs

### N1. [major] PR 09 (03), `pmix_fence.py:79-112`: the fence fails open on any exception it does not list

- `kill_rank` documents only `PmixUnavailable` and `RuntimeError`.
- Its caller in `peer_loss_recovery.py:1212-1232` treats a thread that ended with no recorded error as "ended", so `outcome[rank] = errors.get(rank, "")` is `""`.
- Rank 0 then sends `Fenced(error="")`, and `_on_fenced` lets the authority propose, with the failed process possibly still running beside the survivors.
- This is the safety gate behind the `fence_departed` justification above.
- Failure scenarios:
  - `lib.PMIx_Error_string(status)` returns NULL for an unknown status. `.decode()` then raises `AttributeError` at `pmix_fence.py:111`.
  - A ctypes `ArgumentError`, or any other `Exception`.
- The thread dies with a traceback and the worker prints "ended failed rank N" at `:1234-1236`.
- `peer_loss_recovery.py:1214` does call `runtime_ranks[rank]`, and `IndexError` there leaves the same gap. That file belongs to a later PR in the stack.
- Smallest fix: in `kill()` use `except Exception as error: errors[rank] = repr(error)`, or make `kill_rank` convert every failure to `RuntimeError`.
  - Pre-initialise `errors[rank]` to a failure string and clear it only after `kill_rank` returns.
  - Add a test with a fake libpmix whose `PMIx_Error_string` returns `None`.
- The same hole exists, with a smaller consequence, in `hang_evidence.py:307`. There the thread only dies, and the authority's command fence remains.

### N2. [minor-to-major] PR 07 (01), `client.py:145`: incomplete unpickle exception tuple

- The tuple is `(UnpicklingError, AttributeError, EOFError, ValueError, TypeError)`.
- The C unpickler raises `KeyError` for a missing memo entry, and can raise `IndexError`, `OverflowError`, `RecursionError` and `MemoryError`.
- Together with N3, any such exception from an authenticated message ends the authority instead of being refused to the sender.
- Smallest fix: catch `Exception` there and re-raise it as `MalformedMessage`. The message is already authenticated, so nothing is lost.

### N3. Same as 08 #6 above

I have counted the missing top-level guard once, under PR 08 #6. N2 and N3 together are why I rate that item major rather than minor.

## Verdicts

- **PR 07: approve.** #1 and #2 are fixed and tested. Make N2 a one-line follow-up in the same PR.
- **PR 08: approve with changes.** The remaining blocker-level gap is #6, the serve-loop guard. Add the measured node-loss fence result (#1 evidence) to the PR text.
- **PR 09: approve with changes.** Fix N1 (the fence must fail closed) and add its test. Moving the test-list entry into this PR would be better, but it is not required.
