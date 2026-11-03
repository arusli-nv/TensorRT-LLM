## Scores (1–10)

| File | Read | Name | Comment | Error | Struct | Idiom | **Overall** |
|---|---|---|---|---|---|---|---|
| 1A ipc (ZeroMqQueue) | 5 | 6 | 5 | 5 | 4 | 5 | **5** |
| 1B recovery client | 8 | 8 | 8 | 8 | 8 | 8 | **8** |
| 2A hang-evidence reporter | 7 | 8 | 7 | 7 | 6 | 8 | **7** |
| 2B hang detector/kill | 5 | 7 | 4 | 7 | 5 | 6 | **5.5** |
| 3A MoeExecutionState | 7 | 8 | 7 | 7 | 7 | 7 | **7** |
| 3B EPGroupHealth | 8 | 9 | 8 | 8 | 8 | 8 | **8** |
| 4A virtualMemory.cpp | 6 | 5 | 5 | 6 | 6 | 6 | **6** |
| 4B ghostSlotMemory.cpp | 6 | 7 | 8 | 7 | 5 | 7 | **6.5** |
| 5A health probe | 8 | 8 | 8 | 7 | 8 | 7 | **8** |
| 5B shutdown sender script | 8 | 8 | 8 | 8 | 7 | 8 | **8** |
| 6A recovery authority | 6 | 8 | 7 | 8 | 6 | 8 | **7** |
| 6B all-to-all watchdog | 7 | 8 | 6 | 7 | 6 | 8 | **7** |

All 12 files are in the same fault-tolerance and recovery area. Code quality alone separates the pairs only weakly, so the confidences below are modest.

## Per-file weaknesses

**1A** `pair1_A.py`
1. Send and receive paths are copy-pasted. `put` and `put_nowait` (192–217) are near-identical, as are `_recv_data` and `_recv_data_async` (458–496), and the ROUTER/HMAC branches are repeated.
2. The constructor has a dead branch and a garbled message. Line 54 already rejects `use_hmac_encryption=False`, so lines 91–94 can never run. The text at 56 reads "use_hmac_encryption HMAC encryption…".
3. Error handling is inconsistent.
   - `put_noblock` (219–248) recurses to retry, swallows the failure, and logs the whole object.
   - `put_async_noblock` (271–284) ignores ROUTER and re-implements signing.
   - `Empty` is imported inline at 301.
   - `notify_with_retry` (498–531) has no backoff and catches a bare `Exception`.
   - The `assert` at 234 is used for argument validation.

**1B** `pair1_B.py`
1. The module-level mutable `_KEY` with `global` (68, 80) is hidden state. `load_key` also mutates the environment as a side effect.
2. `_recv` re-checks `is_message` (253) even though `receive_message` already did (157). `discard_pending` (263–268) loops on `RecoveryError` with no bound.
3. Several lines exceed 110 columns (e.g. 57–59, 114–116). `__all__` exports the connection-option constants. `_parse_data` slices before it checks the length (135–136).

**2A** `pair2_A.py`
1. `poll` (322–347) is one long state machine mixing heartbeat, row staleness, and report/clear logic. `_obey` (250–286) mixes receiving, killing, and fence scheduling.
2. `start_native_heartbeat` (150) has no return annotation. Its "Returns" text describes the handle but not the `None` case.
3. Blanket `except Exception` at 312 and 319. `os._exit` and `os.kill(SIGKILL)` sit inside the message handler (257, 271), which makes the code hard to test.

**2B** `pair2_B.py`
1. Comments are over-long essays, some narrating history rather than the current code (138–152, 283–292, 303–314, 574–582). Phrases like "as before" and "old name is load-bearing" describe the past, not the code as it stands.
2. Three near-identical `_best_effort_*` helpers (28–50). `hard_kill_on_rank_crash` (239–320) and `all_ranks_crashed` (177–236) are long and have many exits.
3. Public `HangDetector` methods lack annotations (445, 560, 565, 585, 593). `timeout: Optional[int]` defaults to a magic 300 (430). Detector state is a mix of public attributes and locked fields (434–443).

**3A** `pair3_A.py`
1. Mask bit-twiddling is duplicated (119, 186) and repeats logic that belongs in `EPGroupHealth`. `bool(self.abort.any())` at 182 duplicates `aborted()`.
2. `progress_comm` is a public attribute assigned after construction (51, 77). `for_model` does a local import, and `id()`-keyed dicts are used for de-duplication (61–70).
3. `staging_workspace` calls `max()` on a possibly empty dict, which raises an opaque `ValueError` (102). Long docstring lines (88, 109, 111). The comment at 76 is cryptic.

**3B** `pair3_B.py`
1. `__len__` returns the world size rather than the active count (249–256). Its own docstring has to apologise for this, and it is misleading.
2. `mark_failed` and `mark_active` are mirror duplicates (117–168). `_active_count` redundantly mirrors the mask popcount.
3. The docstrings are heavily boilerplate and verbose (3–15, 56–64, 99–110). `get_mask_words` (188–217) has a `num_words` parameter that no caller needs.

**4A** `pair4_A.cpp`
1. Naming and layout are inconsistent.
   - snake_case helpers (`safe_invoke_helper` at 48, `deviceptr_cast` at 310) sit among camelCase code.
   - The file has two separate `namespace tensorrt_llm::runtime` blocks (8 and 370).
   - There are file-level `static` templates next to an anonymous namespace.
2. The second `add()` (159–190) does manual unlock/relock around `materialize` with a "Hopefully…" comment, and uses two `ScopeGuard`s that hold a `bool const&` (14–27). `addBadHandle` swallows all exceptions (211–220). `retrieveBadHandles` reads `mBadHandles` without the lock (222).
3. `releaseWithTag` counts a chunk even when its release failed (248). The rollback in `materializeWithTag` (281–304) is intricate.

**4B** `pair4_B.cpp`
1. `allocate()` (421–530) is about 110 lines. It mixes setup, slot classification, and mapping, and encodes state in magic ints (-1 and `-2 - chunk`) at 467–479.
2. `hostAccessibleSupported` (288–351) is a long lambda that includes a hardware probe. It uses raw `cuMemAddressFree` / `cuMemUnmap` with unchecked results (44, 193, 323).
3. `std::map`, `std::shared_ptr`, and `std::mutex` are used without the corresponding includes (10–16). The reinterpret_casts between `CUdeviceptr` and pointers are repeated.

**5A** `pair5_A.py`
1. The drain loop uses an exception for control flow (139–143). `_receive` is annotated `-> object` but returns `None`. `status_body` has an implicit contract that a non-"unprotected" state means a non-`None` status (69–79).
2. Typing style is mixed: `Optional[...]` at 104 and 122 versus `"X | None"` at 116. Imports are not sorted (22–28).
3. `close()` has no docstring. `_cached_status` caches a `None` result for `max_age_s`. `type(message) is Status` at 154 is exact-type matching, a deliberate choice that is not explained.

**5B** `pair5_B.py`
1. `main` (70–118) nests a `with`, a `for`, a `with`, a `try`, and a `while`. That is deep, and it returns from inside the context managers.
2. The libc `getenv` / `memset` environment scrubbing is Linux-specific and hacky (57–63). The code duplicates logic from `executor/utils.py`; the comment admits this (52–54).
3. `STOP_MAX_ATTEMPTS = 300` is a magic number. The `continue` on `ZMQError` is silent, with no logging of the failure reason.

**6A** `pair6_A.py`
1. This is a 900-line module. `_on_stopped` (457–497) is long and branchy. `handle`, `receive`, and `_dispatch` are intertwined (190–244, 385–404). `serve` (624–687) nests deeply.
2. Comments are inconsistent or broken. Lines 349–350 contradict each other ("lowest other one" versus "two live"). The `command_fence` docstring has a stray line break (790–792).
3. `hang_monitor is not None` checks are scattered through the file (466, 542, 585, 293). `RecoveryAuthority.__init__` has many parameters. Mutable public attributes are described as "read-only for callers" (112–113).

**6B** `pair6_B.py`
1. `AlltoAllWatchdogCoordinator` (232–467) is a grab-bag of shared state, mask capture, and registration. The double-checked shared-state code is duplicated (256–275). Mask-word logic is re-implemented (308–312, 365–368) rather than reusing `get_mask_words`.
2. `_run` (709–769) is long. The three-line reset of `last_observed_flags` and `poll_timed_out` is repeated three times (745–746, 764–765, 730–731).
3. Many public methods have no docstring (277, 365, 373, 430, 444). The `EPGroupHealthLike` and snapshot Protocols mirror the real class (62–84). `assert` is used for runtime invariants (189, 358, 412). `metainfo` is accessed by string keys (278, 531).

## Pair guesses (newly contributed file)

| Pair | New | Confidence | Quality signals |
|---|---|---|---|
| 1 | **B** | 85% | B is a modern, tightly specified module: complete `X \| None` typing, docstrings with Raises sections, a precise exception hierarchy, a subclass that narrows behaviour. A shows the marks of years of patching: duplicated sync and async blocks, an unreachable validation branch, an inline import, an `assert` for argument validation, and inconsistent quoting and docstring style (`'''` versus `"""`). |
| 2 | **A** | 65% | A is compact and consistent. It has a single purpose, a uniform comment voice, and a dataclass plus Callable-injection design. B reads as heavily iterated: patch-history comments ("compat alias", "as before"), repeated helpers, and untyped public methods next to carefully typed new ones. |
| 3 | **A** | 55% | Weak signal. A is denser and uses the same terse, reference-heavy comment voice as 1B, 2A, and 5A. B uses a different, boilerplate-style docstring approach and a foundational design (Snapshot, `__deepcopy__`, `__len__`). Both quality levels are similar. |
| 4 | **B** | 70% | B has detailed, current-state comments and a structured design with RAII and logged cleanup. It also has a very long `allocate()` and missing includes. A has older-style signs: snake_case helpers in a camelCase code base, two namespace blocks, hedging comments ("Hopefully…"), and swallowed exceptions. |
| 5 | **A** | 60% | A depends on the recovery client from pair 1, and its idiom matches 1B (typing, parenthesised walrus loops, the `__all__` list). B is a self-contained stand-alone script with `nosec` markers and duplicated utility logic, which is more typical of older hardening work. |
| 6 | **A** | 55% | Close to a coin flip. A shares 1B's idiom (the `_Violation` hierarchy, keyword-only parameters, `Outbound` aliases, rate-limited logging) and cites its own protocol throughout. B has a different author style: Protocol "Like" types, an explicit coordinator, and `tllm_logger %`-style formatting. |

**Summary:** the recovery code with the clearest "new author" style is 1B, 2A, 3A, 5A, and 6A, and the guesses above rest on that stylistic clustering. The signals for pairs 3, 5, and 6 are weak. Pairs 1 and 4 are the most separable on quality grounds.
