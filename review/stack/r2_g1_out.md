# Round 2 review of PRs 01–06

No blockers. M4's constant stays as it is, M6's verification is still unfilled, and M3's test gap is still open. I read the six diffs and the tree at `/tmp/claude-161238/wt-stack`. I ran nothing. The description file was truncated at line 706, but M1–M6 sit above that, so I saw all six.

## Status of round-1 findings

| # | Finding | Status |
|---|---|---|
| M1 | Fakes and real schema can drift apart (minor) | JUSTIFIED. The description says the PRs that add `abort_state` and `expert_to_slot` extend the fakes. |
| M2 | `__deepcopy__` returns `self` for every deepcopy (minor) | JUSTIFIED. The description states the aliasing risk and the docstring covers it. |
| M3-1 | `str(error)` sent to every client | JUSTIFIED, with reservations. The description says "The completions stream already sent that text, and OpenAI clients need the cause to act on it." That is sound for completions. For chat and Anthropic it is a new exposure of CUDA, NCCL or ZMQ text. Accepted as a documented trade-off, not re-raised as a blocker. |
| M3-2 | Metrics failure after `[DONE]` is now only logged (minor) | FIXED. The commit message and the PR text both say "A failure after [DONE], such as in metrics extraction, is logged only." |
| M3-3 | `_stream_error_events(e)[0]` builds a list to index it (minor) | STILL A PROBLEM, minor. `openai_server.py:3079` is unchanged, and `:3078` still couples the two places by comment. A `_stream_error_event()` helper that returns one string would remove it. |
| M3-5 | No multi-prompt failure test | STILL A PROBLEM, major. The description claims "one error event per failed prompt while the other prompts keep streaming, then a single [DONE]". Its coverage list (`test_stream_error_events.py`) has no test for `merge_generators` with a failing producer. The decision said "add if cheap" and it was not added. Fix: one test with two `completion_generator`s, one failing, through `generator_wrapper`. |
| M4-1 | Hard-coded heuristic, no data, no gate | JUSTIFIED, with a residual. The commit now has numbers (DEP32 GB200, DeepSeek-R1 NVFP4). Those are the "Measured … closed-loop" figures: releases per wave drop and throughput is "unchanged within noise (-0.18% / -0.82%, 4 pairs)". The description also gives the reason for a class constant. The residual is that only closed-loop load on one hardware and model was measured. My open-loop trickle scenario (a few arrivals, then 4 quiet decisions, with more arriving soon) is untested. The rule never holds longer than before, so I am not blocking, but note it in the PR. |
| M4-2, M4-3 | Rank-identical inputs: `all_new_flat` and `waiting_queue` | FIXED. The message and PR text now state it. Verified in the tree: `py_executor.py:7591` says "all ranks see identical new_requests after _pop_from_waiting_queue", and the queue is filled by `add_requests` after the broadcast (`:7317`). |
| M4-4 | State hygiene (minor) | JUSTIFIED. The state is bounded and only used inside the balance branch. |
| M5 | Sibling `ld.relaxed` asm sites without a clobber | JUSTIFIED. The description scopes the PR to the release/acquire fences ("Other ld.relaxed sites in the file are not touched"). `:274`, `:1146`, `:1234` and `:2092` are flag polls, not fences. Only `:301` carries a clobber. I did not re-audit their downstream ordering. Optional: one sentence on why they need none. |
| M5 | No repro or SASS evidence (minor) | JUSTIFIED. A compiler barrier has no unit test, and the description says the existing multi-GPU tests run these kernels. |
| M6-1 | Patch unverified (hunk anchor, `os` import) | STILL A PROBLEM, major. The PR text still has `Verification of the script in a built container: <placeholder: result>`. The "decision: FIX, verify in container" is not evidenced. `git apply` fails the build on a mismatch, which is the right failure mode. But `os` being in scope in `_core.py` is only checked at runtime, under `TRTLLM_USE_MPI_KVCACHE=1`. The hunk has 3 lines of leading and 2 of trailing context, so git requires it to match at end of file, and I could not verify that against 4.1.2. No mpi4py source is available here. Fill in the build log and a KV-cache-mode smoke run before merge. |
| M6-2 | 3.x→4.x regression evidence | STILL A PROBLEM, major. The description has `Regression evidence … : <placeholder>`. It says "No unit test: the change is an image build step. Evidence is the image build and the existing MPI tests", but gives no results. The bump reaches every user. Do not merge before the MPI test lists have run. |
| M6-3 | Version pins disagree | STILL A PROBLEM, minor. `ATTRIBUTIONS-Python.md:16915` still says 4.1.1, and `requirements.txt:18` is unpinned while `security_scanning/pyproject.toml:21` says `>=4.1.2`. The description does explain that the script alone fixes 4.1.2, so the requirements line is justified. Update ATTRIBUTIONS. |
| M6-4 | Debug `print` | STILL A PROBLEM, minor (`install_mpi4py.sh:67`). Drop it, or say it is intentional. |
| M6-5 | aarch64 `Cython==0.29.37` and setuptools pin removal unverified | STILL A PROBLEM, minor. The commit message mentions only the setuptools pin. The Cython removal is silent and needs an aarch64 build log. |
| M6-6 | No checksum on the tarball | STILL A PROBLEM, minor. `curl -L … \| tar` at `install_mpi4py.sh:19` is unchanged. This is a version bump on a supply-chain path, so add a sha256. |

## New significant issues

None at blocker or major level. Two minor observations:
- **M3, `openai_server.py:3076-3079`:** a failure of `_extract_metrics` in `completion_generator` still emits an error event after a fully delivered completion. The chat paths now log it only, so the two routes disagree. The commit message says "A failure after [DONE] … is logged only" without saying that completions differ.
- **M3, `openai_server.py:326-342`:** the wrapper now sits between Starlette and the inner generator. A client disconnect closes only the wrapper, and the inner chat generator is finalized later by the event loop. The old code had Starlette close the generator directly. This is probably harmless, but nothing tests that disconnect still cancels the request.

## Verdicts

| PR | Verdict |
|---|---|
| M1 | approve |
| M2 | approve |
| M3 | approve with changes (add the multi-prompt failure test) |
| M4 | approve (add an open-loop result if you have one) |
| M5 | approve |
| M6 | request changes (container verification and MPI regression results are still placeholders) |
