# FT-COST-BISECT: which part of FT's mechanism costs the +2.09% steady step at 128/GPU

Status: DRAFT, not submitted. Files: `job.sh.new` (case label FT-COST-BISECT in the FT-COST-E2E|...|PROF block),
`testhooks.py.new` (TLLM_TEST_FT_OFF). The same changes as unified diffs against job.sh (mtime 08:29:36) and
testhooks.py (03:44:16): `FT-COST-BISECT-job.patch`, `FT-COST-BISECT-testhooks.patch` in this directory, in case
job.sh.new is replaced by another edit before it is moved in.

## Question and evidence so far
- CLAIMS #104 (FT-COST-MECH-8063159, n=4, pinned tactics, ft vs noft-ulfm, same ULFM launch): steady step at
  128/GPU 25.45 -> 25.98 ms, +2.09% (CI 1.72..2.45), ~+530 us/step. Off-arm CV 0.13%.
- FT-COST-PROF-8065299 (nsys, round 1 usable; round 2 not): GPU step delta only +130 us (+0.45%); on rank 0,
  `_enqueue_responses` +524 us/step while the other ranks wait for rank 0 in `_fetch_new_requests` (+174 us mean).
  Rank 0 spends ~6 ms/step in `_enqueue_responses` and ~5 ms in `_handle_responses` in both arms, so at 128/GPU
  rank 0's serving thread is on the step's critical path: any FT host work there adds ~1:1 to the step.
  The profiler distorts host time (step 28.8 vs 25.5 ms), so it can only point, not decide.
- Code (fcc86b10b2): FT_STEP_COST_INVENTORY.md and FT_DISPATCH_COST.md list the candidates; the ones reachable by a
  test hook are below.

## Arms (3 rounds, rotating; 128/GPU only)
| Arm | Launch | FT | Hooks |
|---|---|---|---|
| noft-ulfm | ULFM ob1/TCP + detector | off | none |
| ft | same | on | none |
| ft-noH (default 3rd arm) | same | on | TLLM_TEST_FT_OFF=coll,resp,step: group H, all of FT's serving-thread host work |

Shared: wheel build-next-fcc86b10b2, no overlay (preflight enforced); dep32-final.yaml; forced MTP acceptance;
1k/1k, `--loads 128 --iterations 2 --warm-each-load`; tactics pinned from FT_COST_E2E_SEED (default
FT-COST-E2E-8059717/autotuner/pinned.json, the seed of #104 and of FT-COST-PROF); TACTIC_CHECK per case as before.
Hang detection stays off (FT default), so the hang-evidence reporter and the native heartbeat do not run in any arm;
every hook arm's proof line asserts `hang_detection=False`.

Other hook arms (selectable with FT_COST_E2E_ARMS, e.g. `noft-ulfm ft ft-noT` or a follow-up split):
`ft-noT` (watch), `ft-noresp` (resp), `ft-nostep` (step), `ft-nocoll` (coll).

### What each piece removes (testhooks.py.new, monkeypatched at import, all ranks)
| Piece | Inventory items | Patch |
|---|---|---|
| resp | H5 | fetching_requests, broadcast_requests, requests_routed, gathering_responses, responses_gathered -> RecoveryHook's no-ops (no per-rank response history deque, no rank-0 owner map / `_forget_finished` over every response, no control-item filter). The broadcast itself still runs (the base calls it). |
| step | H1, H2, H3 | step_prepared, step_released, commit_step -> no-ops (no context-capacity snapshot, no step-change records, no commit sync / peer_failed check) and `PeerLossRecovery.enabled = False`, which only gates py_executor's three FT orderings: V2 context blocks committed and recompute victims paused in their own step, as without FT (no early input-copy wait). |
| coll | H4 | every `@_interruptible` method of MPIDist (12 in fcc86b10b2) replaced by its `__wrapped__`: no Get_failed + group free after each host collective, no in_host_collective flag (read only by hang detection). |
| watch | T1 | `_watch_peer_failures` installs a stub: no PeerFailureWatch native thread, no MPI_Test polling every 5 ms, no extra communicator Dup. |

Not removable here and the same in ft and every hook arm: G (bound dispatch: compact CTAs, sys fences, valid counts,
MTP-layer compact copy), the idle recovery-client zmq thread, the authority process on node 0 (0.1 s poll), the
mpi_session futures poll (Get_failed every 0.5 s), EPLB host copies (online EPLB: present in both arms anyway).
Healthy runs only: with resp/step/watch removed a peer loss would not recover correctly.

### Proof that the hooks took effect (per case, every rank)
- `WIDEEP_FT_OFF rank R ... module=communicator unwrapped=12 ...` and `... module=peer_loss_recovery taps=... enabled=...
  watch=...` at import.
- `WIDEEP_FT_OFF_ACTIVE rank R ... serving=1 verified=<pieces> hang_detection=False` when the serving loop starts,
  computed from the live executor (class attributes differ from the originals, `enabled is False`,
  `type(dist).tp_gather` is not an _interruptible wrapper, the watch method is the stub). serving=0 is the KV
  estimation loop.
- `WIDEEP_FT_OFF_CALL rank R ... name=<tap>` at the first call of each no-op tap (then the tap is the plain base
  method: no counting overhead afterwards).
- job.sh: `hook_ok` (node 0) after every case writes a HOOK_CHECK line to hook_check.txt: case DONE; hook arm: 32 rank
  logs, 32 `serving=1 verified=<exact pieces> hang_detection=False`, 0 WIDEEP_FT_OFF_ERROR, rank 0 logged the first
  call of every resp tap (fetching_requests, broadcast_requests, requests_routed, gathering_responses,
  responses_gathered) and of step_prepared/commit_step, `unwrapped=12` on 32 ranks for coll; non-hook arms: no
  WIDEEP_FT_OFF line at all. A failed check in round 1 ends the job (SMOKE_FAIL hook check, no idle GPUs); later
  failures print HOOK_CHECK_FAIL (exclude that case).

## Time (8 nodes, <= 2 h)
128-only case ~10 min: FT-COST-MECH startup to server_ready 436-527 s (one 867 s outlier), 128/GPU warm + measured
~105 s, teardown ~10 s. 3 arms x 3 rounds = 9 cases ~95 min + ~2 min start. A round starts only with
FT_COST_E2E_ROUND_MIN = 11 min x arms + 2 (35 min) left, so rounds stay complete; round 4 (FT_COST_E2E_ROUNDS=4)
runs only if the job is fast. Four arms (adding ft-noT) need ~2 h 10 min for 3 rounds: either --time 2:15 or a
second job.

Submit (when approved): `MVP_BUILD=build-next-fcc86b10b2` and `run_job.sh FT-COST-BISECT` on 8 nodes, --time 2:00.

## Outputs
ab_stats-noft-ulfm-vs-ft.txt (F = FT's cost), ab_stats-ft-noH-vs-ft.txt (S = what the arm saves; "cost %" = ft slower
than ft-noH), ab_stats-noft-ulfm-vs-ft-noH.txt (R = what remains), bisect.txt (one BISECT line per hook arm:
steady_step 128/GPU F, S, R with 95% CIs, recovered = S/F), hook_check.txt, tactic_check.txt.

## Decision rules (steady_step_ms at 128/GPU, paired by round, n >= 3)
0. Valid only if every case passes HOOK_CHECK and TACTIC_CHECK (autotune=0, priming_failed=0, cache_miss=0, pin OK),
   and F reproduces #104: F's CI excludes 0 (expected ~2%). Otherwise no attribution from this job.
1. **Source identified (H):** S >= 0.5 x F and S's CI lower bound > 0. Next: split H with ft-noresp / ft-nostep /
   ft-nocoll (one seeded job; resp first: rank 0's response bookkeeping is the nsys suspect) and fix at the source.
2. **H ruled out as the main source:** S's CI upper bound < 0.5 x F. The cost is in G or T: next job ft-noT (watch)
   plus a G probe (dispatch-kernel clock64 tail split, or an unbound-dispatch arm, FT_DISPATCH_COST.md section 5).
3. **Partial:** S's CI excludes 0 but its point estimate < 0.5 x F: H explains part; record S, and bisect the rest
   as in rule 2.
4. **Inconclusive:** S's CI contains 0.5 x F (or 0 and 0.5 x F): pool a second seeded job (same seed, rounds add up).
Sanity: S + R should be ~F (independent paired estimates; a large mismatch means drift between rounds - report it).
Expected resolution: per-round paired SD ~0.2% (off-arm CV 0.13%, #104) -> 95% half-width ~0.5% at n=3, enough to
tell S >= ~1.05% from S ~0.

## Offline validation done (no GPU)
- `bash -n job.sh.new`: OK. testhooks.py.new compiles (Python 3.12).
- Stub modules (the real `_interruptible` and `RecoveryHook` source of fcc86b10b2, stub PeerLossRecovery whose taps
  record FT calls, stub MPIDist with two wrapped collectives) loaded through the hooks' real meta-path finder, for
  FT_OFF = coll,resp,step / watch / resp / step / coll / coll,watch / none / coll,bogus: in each, exactly the
  requested pieces stop calling FT's code (Get_failed calls 0 with coll, 3 without), `enabled` flips only with step,
  the watch stub is installed only with watch, the base broadcast still runs the broadcast, first-call lines appear
  once per tap, the proof line lists exactly the requested pieces, an unknown piece logs WIDEEP_FT_OFF_ERROR, and
  chaining with TLLM_TEST_WATCH_POLL_S on the same module applies both patches.
- `hook_ok` against synthetic case dirs: passes the correct hook arm, ft and noft-ulfm; fails a missing rank, missing
  first-call lines, hooks leaking into ft, wrong pieces, and a case not DONE.
- Not validated: the real modules' import (torch/MPI), that 12 wrappers are found in the real communicator (the
  check requires it), and the effect on a live serving loop - the round-1 HOOK_CHECK covers these on GPU.
