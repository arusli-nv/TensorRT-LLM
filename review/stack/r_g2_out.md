# Review of PRs 07, 08, 09 (WideEP FT: protocol/client, authority, hang detection)

Findings are from reading the diffs and the stack tree. I could not run anything. I read 08's production code and only grepped its tests (the diff view cut off partway through).

With `TLLM_FAULT_TOLERANCE_MODE` unset, I found no behaviour change in any of the three PRs:
- **07:** `socket_options=None` is a no-op, and the new env helpers are inert.
- **08:** a standalone module, not imported by the default path.
- **09:** the only unconditional change is an extra native source compiled into the nanobind module (`heartbeatSender.cpp`, `sha256.h` already on the include path). The commit message's "start only with peer-loss recovery" is accurate for runtime behaviour.

---

## PR 07 — protocol, config, client

**1. [major] Error-string matching decides whether a message is unauthenticated (`recovery/client.py:200-208`)**
- `unauthenticated()` tests `str(error) == "HMAC verification failed"` and `"values to unpack" in str(error)`. Those strings come from `ipc.py:441` and from CPython's text for `identity, data = recv_multipart()` (`ipc.py:475`).
- `serve()` (`authority.py:683-685`) re-raises anything else. If `ipc.py` rewords its message, or a short or odd frame raises a different exception, any peer that can reach the port crashes the authority loop with one bad packet.
- The authority is the single point of failure and has no try/except around the loop.
- **Fix:** add `class HMACVerificationError(RuntimeError)` and a `MalformedFrameError` in `ipc.py`, raised explicitly, and have `unauthenticated()` use `isinstance`. Also wrap the `serve` body so an unexpected exception is logged and answered with `Abort`, not a crash.
- The commit message gives no reason for the string matching, so this is not justified.

**2. [major, security] The authority unpickles messages from every worker (`client.py:211-232`, `protocol.py:582`)**
- The HMAC key is shared by all workers and the authority. The authority is a long-lived process outside the job's termination group. It may also hold ssh credentials, since `--fence-command` is `ssh {host} kill ...`.
- Any worker that holds the key, or any process that reads it before `load_key` scrubs it, gets code execution in the authority through `pickle.loads`.
- The protocol is only dataclasses of int, str, bool and tuple, so a restricted `find_class` allowlist is cheap. `MessageQueue._parse_data` already overrides the parse step.
- **Fix:** subclass `pickle.Unpickler` in `MessageQueue._parse_data`, allowing only `protocol.*` classes and `builtins.tuple`/`frozenset`. Doing this in a single place covers both ends.
- The docstring covers disruption by a peer without the key, but nothing addresses a compromised key-holder. Not justified.

**3. [minor] Server-to-worker messages have no freshness (`client.py:327-345`, `authority.py:763-771`)**
- The docstring states the replay checks "cover only inbound messages". The worker accepts any validly signed message: `Abort`, `Restart`, `Install`, `Commit`.
- A network attacker who can capture an authority-to-worker `Restart` and inject it into the TCP stream kills every worker with `os._exit(75)` (`hang_evidence.py:657`).
- **Fix:** have the authority generate a random session id at start, put it in `Registered`, and include it in every outbound message. Have workers drop a mismatched one.
- The limitation is acknowledged ("Experimental"), so this is only partly justified.

**4. [minor] Tests for risky logic land one PR late**
- 07 ships `is_message`/`_typed`, `decode_datagram`, `load_key`, `unauthenticated`, `MessageQueue._parse_data` and `RecoveryClient.wait`. Its only test file is `test_recovery_config.py`.
- The tests that exist in the stack tree are in 08's files. As a stack that is acceptable, but 07 alone is untested for the HMAC/typing boundary.
- **Fix:** move the protocol and client tests into 07.

**5. [minor] `peer_loss_recovery_enabled` is `lru_cache`d but `fault_tolerance_mode` is not (`_utils.py:516-541`)**
- The two env accessors behave inconsistently, and tests must call `cache_clear()`.
- It is also an import-time-order hazard if the env is set after the first call.
- **Fix:** document it, or cache neither.

**6. [minor] `load_key` is not thread-safe, and the `config.py` docs are slightly off (`client.py:182-197`)**
- The check-then-set on `_KEY` is racy between the main thread and the reporter thread. `load_key` is called from `from_env` and from `start_native_heartbeat`.
- `config.py:467` documents `extra_slots` as "Default: ghost…" while the field default is `None`. The resolved default lives elsewhere.
- **Fix:** guard with a lock and fix the docstring.

**7. [minor] Heavy import for a "standalone" process (`config.py:444`)**
- `StrictBaseModel` comes from `tensorrt_llm.llmapi.utils`, so `python -m tensorrt_llm.executor.recovery.authority` imports the whole package and torch.
- That is acceptable only if documented: the authority needs the full install and the same version.
- The version is enforced by `PROTOCOL_VERSION`, which is good.

**PR 07 verdict: approve with changes.** #1 and #2 should be fixed before merge. #4 should move into this PR.

---

## PR 08 — authority and HangMonitor

**1. [major] `fence_departed=True` is hard-coded in `main()` with no opt-out (`authority.py:902`)**
- `RecoveryAuthority` defaults to `False`, but `main` forces `True`, and there is no CLI flag.
- Rank 0 then has to PMIx-kill every failed rank. See `peer_loss_recovery.py:1196-1229`, whose `_end_ranks` is bounded by `RUNTIME_FENCE_CALL_S=2s`. On any `Fenced(error=...)` the authority aborts (`_on_fenced`).
- On launchers without PMIx (Slurm PMI2, Ray, containers without libpmix), or when the runtime daemon on a dead node cannot answer, every recovery aborts.
- A node loss is the headline use case, and that is exactly when the PRRTE daemon on the lost node cannot answer. I could not verify what PRRTE returns for that case, so treat it as an open question.
- **Fix:** add `--no-fence-departed` (or a config field, included in the digest) and document the PMIx requirement.
- The commit message does not say this. The docstring says "a rank that cannot be ended aborts". That is the intent, but the missing opt-out is a design gap.

**2. [major, security] Fence command target comes from the worker (`authority.py:804-859`, `hang_monitor.py:1023-1026`)**
- `hung.host` and `hung.pid` are taken from the worker's own `Heartbeat`, and `fence` runs `ssh {host} kill -KILL {pid}` as the authority's user.
- `_plain_host` and `pid > 0` only prevent option injection. A compromised or buggy rank can name any resolvable host and any pid (`pid 1` of a neighbouring node), then look hung. The authority kills it with its own credentials. The blast radius goes beyond the rank's own process.
- **Fix:** record each worker's host at `Register` (or from an allowlist of job hosts) and accept a fence only for hosts in that set. Never take a pid from a rank other than the one being fenced.
- The docstring only claims option-injection safety, so this is not justified.

**3. [major] A single slot for the command fence (`authority.py:372`)**
- `_on_hung` overwrites `self._command_fence`. If a second rank is confirmed hung inside the 5 s grace, the first rank's last-resort fence is silently dropped.
- The first rank stays hung, because `_confirmed` never retries.
- **Fix:** keep a list of `(deadline, hung)`.

**4. [minor] `_failed_during_recovery` is never cleared (`authority.py:410`)**
- It is not reset when a later recovery starts (`_on_stopped`, SERVING/RECOVERED branch). A departed rank's `Stopped` or `Ready` is then silently dropped instead of raising "departed rank sent …".
- The effect is a weakened violation check on later recoveries.
- **Fix:** `self._failed_during_recovery = frozenset()` where `_replans = 0` is set.

**5. [minor] `Abort.reason` can exceed `MAX_MESSAGE_BYTES` (`authority.py:458`, `client.py:290`)**
- `_admit` raises with `{reported}`, the full set of divergent domain sets. At hundreds of ranks with differing reports that is hundreds of KB.
- ZeroMQ closes the connection on an oversized message (`MAXMSGSIZE`), so workers never see the `Abort`. They reconnect and time out at `recovery_timeout`.
- The same applies to `f"…{message}"` reasons.
- **Fix:** truncate the reason to about 1 KB in `abort()`.

**6. [minor] The serve loop has no failure isolation (`authority.py:669-702`)**
- It is a single thread with no top-level try/except. A bug in a state handler (for example a `KeyError` on `_evidence` or `_routes`) kills the authority. Workers then see timeouts and take the terminal path.
- **Fix:** log and `abort(...)`.

**7. [minor] `_send` does not give failures a clear signal (`authority.py:708-723`)**
- `ROUTER_MANDATORY` plus `put_nowait` raise `zmq.ZMQError` for an unroutable peer, which `_send` catches. That is fine.
- `abort()` broadcasts to departed ranks too, so every abort logs failures, rate-limited. This is a log-noise nit.
- SNDHWM is `0` from `set_hwm(0)`, so a stalled peer's outbound queue grows without bound. Cap SNDHWM.

**8. [minor] Docstring and leftover issues**
- `command_fence`'s docstring has a broken line ("The\n worker reports…").
- `env={k: v … if k != AUTHORITY_KEY_ENV}` is redundant, because `load_key` already popped the key.
- `Heartbeat` handling stores `_evidence` even when `hang_monitor is None`.
- `receive()` (`authority.py:239-264`) is the single replay boundary and is subtle: the `registers` branch, the `_workers.get(route) is not message` identity trick, and the per-`(rank, evidence)` sequence streams. The tests I could see cover it only partially.

**PR 08 verdict: request changes.** #1 (no opt-out for PMIx fencing) and #2 (worker-controlled fence target) should be settled before merge. #3 is small and should be fixed.

---

## PR 09 — hang evidence, native heartbeat, PMIx fence

**1. [major] `test_hang_evidence.py` is not registered in any test list**
- I grepped `tests/integration/test_lists` for `test_hang_evidence` and found nothing.
- PR 07 and PR 08 each add their files to `l0_cpu.yml`. PR 09 adds none, so CI never runs the reporter, fence, `HeartbeatSender` or `pmix_ranks` tests.
- The file needs a GPU and a built extension (and MPI for `pmix_ranks`), so it belongs in an l0 GPU list, not l0_cpu.
- **Fix:** register it in an appropriate test-db yml. The commit message does not mention this.

**2. [major] Hand-rolled HMAC in C++ with a thin test (`heartbeatSender.cpp:97-122`)**
- The implementation is RFC 2104 and looks right to me, including the `key > block` branch.
- The only test uses `key = b"k"*32`. The `>64` branch (a 65 to 128 byte hex key, which `load_key` permits) is untested.
- **Fix:** add a test-vector case (a key longer than 64 bytes) to `test_the_native_heartbeat…`, or reuse an existing HMAC helper if the repo has one.

**3. [minor] `_kill_rank` can raise `IndexError` in the thread (`hang_evidence.py:301-306`)**
- `self._pmix_ranks[rank]` is indexed with the default `pmix_ranks=()`, or with any `Fence.rank` outside the tuple.
- The `except (PmixUnavailable, RuntimeError)` clause does not cover `IndexError`. The runtime-fence thread dies with an uncaught traceback, and the rank stays hung until the authority's command fence.
- **Fix:** catch `IndexError`, or validate it in `_obey`.

**4. [minor] Single-slot fence thread drops later fences (`hang_evidence.py:676-681`)**
- If one `PMIx_Job_control` call never answers, any later fence for another rank is skipped with a warning. It does not queue.
- That is intentional ("at most one"), but a second hung rank then depends only on the authority's command fence, which has its own single slot (08 #3).

**5. [minor] Fragile libpmix discovery (`pmix_fence.py:314-324`)**
- It parses `/proc/self/maps` and takes `line.split()[-1]`. A mapping line ending in `(deleted)` yields the wrong path.
- The hand-written ctypes struct layout (`_Proc`, `PMIX_BOOL`, `PMIx_Info_load` signature) targets PMIx ≥ 4 only. The functions are macros in v3, so you get `AttributeError` and then `PmixUnavailable`. That is handled correctly.
- Add a comment on the supported PMIx version, plus a skipped-unless-PMIx integration test. At the moment only a fake libpmix is tested.

**6. [minor] `start_native_heartbeat` and `waiting` race (`hang_evidence.py:552-571`, `hang_evidence.py:688-697`)**
- `self._waiting()` and `self._iteration()` are read from the reporter thread while the main thread writes them. Reads of ints are benign under the GIL, but the callbacks' contract does not say "thread-safe".
- `start_native_heartbeat` returns `None` (with a warning) on failure, so a Python-starved rank may be falsely confirmed hung. The warning says so.
- Document the threading contract in `HangEvidenceReporter`.

**7. [minor] C++ details (`heartbeatSender.cpp`)**
- The constructor can throw after the socket is opened but before the thread starts (a failed `std::thread` spawn). The destructor never runs, so the fd leaks. It is harmless in practice.
- On non-Linux the whole class is `#if`ed out but the file still builds, which is fine. `HeartbeatSender` is unbound there, and `start_native_heartbeat` handles the `ImportError`.

**PR 09 verdict: approve with changes.** #1 (tests not wired into CI) should be fixed before merge.

---

## Cross-cutting notes
- The security docstrings are honest ("traffic is authenticated but not encrypted"), but PR 08's #2 and PR 07's #2 are not covered by them.
- Not verified: how the authority is deployed and launched in the intended environments. That is the premise behind the "outlives any worker" claim.
