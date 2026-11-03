"""Benchmark-only hooks, loaded through a .pth file into every rank's Python (never part of TRT-LLM).

Always on (counters, one stdout line each, so recovery can be shown to rebuild nothing):
- WIDEEP_CAPTURE rank t=...: a CUDA graph capture.
- WIDEEP_AUTOTUNE rank t=... op=...: an autotuner profiling run (a real tuning, not a cache hit).
- WIDEEP_COMPILE rank t=...: a CuteDSL kernel compilation.
- WIDEEP_MEM rank ...: the all-to-all workspace bytes per rank, and the KV cache quota (startup memory).

Exact-token reference (TLLM_TEST_ORACLE_EXCLUDE_RANK=r): every rank except r routes around r's expert
slots from startup, so a healthy run routes exactly as the survivors do after r is lost; its greedy tokens
are the reference the after-loss tokens must equal.

Kill at an exact moment (TLLM_TEST_KILL_RANK=r and a trigger file the driver creates):
- TLLM_TEST_KILL_IN_DECODE=path: once path exists, rank r SIGKILLs itself right before a decode CUDA graph
  replay, after the step's host collectives, so the survivors wait for it inside that step's MoE all-to-all.
  With TLLM_TEST_KILL_WHEN_BUSY=1 (overlap scheduler): at the first replay after the trigger (step k) rank r's
  GPU is held ~2 s before the step's kernels, and rank r SIGKILLs itself at the next replay (step k+1), after
  step k+1's host collectives: the survivors have prepared step k+1 while step k's all-to-all still waits for
  rank r, so the loss lands in two uncommitted steps.
- TLLM_TEST_KILL_IN_PREFILL=path: the same before a forward that has context requests (eager prefill).
- TLLM_TEST_KILL_IN_EPLB_UPDATE=path: the same right after starting an iteration that updates EPLB weights (online
  EPLB), so the update is in flight when the rank dies.
- TLLM_TEST_STALL_IN_DECODE=path (hang tests): once path exists, rank r's GPU sleeps TLLM_TEST_STALL_S seconds
  (default 30) right before one decode replay, so its peers wait for it inside the step's all-to-all while its
  process stays alive.
- TLLM_TEST_STICKY_IN_DECODE=path (device-failure tests): once path exists, rank r launches an out-of-range
  GPU gather before one decode replay: a device-side assert, a sticky error of its CUDA context.
- TLLM_TEST_HOST_STALL_IN_DECODE=path: the same as the stall, but the main thread sleeps TLLM_TEST_STALL_S seconds (a stuck
  serving loop; the process's other threads keep running).
- TLLM_TEST_GIL_HOG_IN_DECODE=path (hang negative control): once path exists, a thread on rank r runs pure Python
  for TLLM_TEST_STALL_S seconds (GIL switch interval TLLM_TEST_GIL_SWITCH_S, default Python's), so the serving
  loop and the heartbeat thread run only in slices; the rank is slow, not hung.
- TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1: the decode triggers above fire only in a step where rank r serves a real
  (not attention-DP dummy) request, so a loss under load always takes requests with it.
- TLLM_TEST_KILL_TRIGGER_FOR=r=path[,r=path...] (repeated failures): rank r kills itself at a decode replay once its
  own path exists (as TLLM_TEST_KILL_IN_DECODE), so a second loss can follow the first one's recovery; with
  TLLM_TEST_KILL_TRIGGER_ANY_STEP=1 also before a forward with context requests (its next step of either kind).
- TLLM_TEST_KILL_RERUN_OWNER=prefix (rerun-once quarantine): at each recovery rank 0 writes the ids of the requests it
  is about to run again to prefix.ids (and logs WIDEEP_RERUN / WIDEEP_QUARANTINE); the first non-0 rank that
  prefills one of them claims prefix.claim and SIGKILLs itself in that forward, before the request delivers anything.
- TLLM_TEST_LATE_WRITE_IN_DECODE=path (DR-H2, late writes of a lost rank): once path exists, rank r's GPU sleeps
  TLLM_TEST_LATE_WRITE_DELAY_S (default 8) before one decode replay, then the process SIGSTOPs itself with that
  replay queued: Open MPI's detector declares it lost while its CUDA context lives, and its GPU runs the replay's
  all-to-all writes into the peers after the delay. A detached child SIGKILLs it TLLM_TEST_LATE_WRITE_KILL_S
  (default 30) after the stop.
- TLLM_TEST_DELAY_ABORT_S=s (DR-H2, survivors touching freed victim memory): every rank delays its reaction to a
  peer failure (the MoE abort request, the watch's revoke and the recovery) by s seconds after it first notices
  it, so kernels queued at the abort run after the victim's memory was freed. TLLM_TEST_DELAY_ABORT_RANK=r: only
  rank r delays (C10 deadline row: one slow survivor holds the others' shrink past the recovery deadline).
- TLLM_TEST_LOG_DEADLINE=1 (C10): every rank logs WIDEEP_DEADLINE when a RecoveryTimeout is raised and WIDEEP_MPI_CALL
  when it starts revoke, shrink or watch_peer_failures, so the verdict can check that none starts after its deadline.
Healthy-cost attribution: TLLM_TEST_EMPTY_EXTRA_SLOTS=1 empties the extra EPLB slots with FT off;
TLLM_TEST_POPULATE=off|timed skips or times the background page-table build of the EPLB host copies.
Diagnosis only: TLLM_TEST_WATCH_POLL_S=s sets the peer-failure watcher's poll interval;
TLLM_TEST_ALLOC_STATS_S=s logs, every s seconds, one WIDEEP_ALLOC line with the CUDA caching allocator's
retries and device allocations/frees, the process RSS and its page faults (host-side reads only).
TLLM_TEST_EXPERT_STATISTIC_ITER_RANGE=a-b and TLLM_TEST_EXPERT_STATISTIC_PATH=dir set main's EXPERT_STATISTIC_*
(the launcher forwards only TLLM_* variables): every rank dumps per-slot routed-token counts of iterations [a, b),
which main runs without CUDA graphs.
FT step-cost bisection (FT-COST-BISECT; healthy cases only, a peer loss may no longer recover):
TLLM_TEST_FT_OFF=piece[,piece...] runs FT on without pieces of its healthy-path host work:
- resp: the request/response taps (fetching_requests, broadcast_requests, requests_routed, gathering_responses,
  responses_gathered) are RecoveryHook's no-ops: no response history, no owner bookkeeping on rank 0.
- step: the step hooks (step_prepared, step_released, commit_step) are RecoveryHook's no-ops and
  PeerLossRecovery.enabled is False, so the executor commits V2 context blocks and pauses recompute victims in
  their own step, as without FT.
- coll: MPIDist's host collectives run without _interruptible (no Get_failed after each, no in_host_collective).
- watch: no PeerFailureWatch thread (no MPI_Test polling on a private communicator).
Each rank logs WIDEEP_FT_OFF per patched module, WIDEEP_FT_OFF_ACTIVE serving=0|1 verified=<pieces> from the live
executor when a serving loop starts, and WIDEEP_FT_OFF_CALL name=<tap> at the first call of each no-op tap.
FT all-to-all bisection (healthy cases only; a peer loss is out of scope):
- TLLM_TEST_FT_UNBOUND=1: FT stays on (recovery, watch, MoeExecutionState with its abort record, rank mask,
  progress_comm, retained workspaces), but no NVLinkOneSided is bound to it, and the comms are built without
  EPGroupHealth: main's unbound all-to-all (no fixed slots, no compact routing, the sanitize kernel, no rank mask,
  no host watchdog), routed by route() instead of the expert-to-slot table. Logs WIDEEP_FT_UNBOUND applied per
  module, WIDEEP_FT_UNBOUND_ACTIVE layers comms bound rank_mask health slot_table workspaces progress_comm when the
  state is built, and WIDEEP_FT_UNBOUND_CALL at the first dispatch.
- TLLM_TEST_FT_NO_INPLACE_SLOTS=1: the bound path, but CuteDSL answers supports_dispatch_receive_slots() False, so
  the scheduler compacts the receive slots for FC1 as for other backends. Logs WIDEEP_FT_NO_INPLACE applied,
  WIDEEP_FT_NO_INPLACE_ACTIVE layers bound inplace backends after binding, and WIDEEP_FT_NO_INPLACE_CALL at the
  first compaction that copies.
- TLLM_TEST_FT_NO_GHOST=1: the bound path on 256 slots: the load balancer plans for no loss (largest failure domain 0),
  so no ghost slots are appended, and covers_loss_of answers True so startup admits it (a loss cannot recover). The
  weights keep the ghost run's storage (GhostSlotMemory, no slot empty) and GPU copy mode; ring host copies stay. Logs
  WIDEEP_FT_NO_GHOST applied, WIDEEP_FT_NO_GHOST_ACTIVE layers bound num_slots slots_per_rank ghost ghost_storage
  copy_nodes largest_loss real_largest_loss after binding, and WIDEEP_FT_NO_GHOST_CALL at the first dispatch.
- TLLM_TEST_FT_NO_SLOT_TABLE=1: the bound path, but the scheduler has no expert-to-slot table: route() writes the slot
  ids and the dispatch kernel reads them (no in-kernel lookup). Logs WIDEEP_FT_NO_SLOT_TABLE applied,
  WIDEEP_FT_NO_SLOT_TABLE_ACTIVE layers bound recovery slot_table after binding, and WIDEEP_FT_NO_SLOT_TABLE_CALL at the
  first dispatch.
Each logs <EVENT>_REFUSED module=... moved=... and leaves a module unpatched if the code it patches moved.
Slot-table proof: TLLM_TEST_LOG_SLOT_TABLE=1 logs, on every rank, WIDEEP_SLOT_TABLE layer=<idx> used=0|1 once per layer
at the scheduler's first expert-to-slot table query (used=1 table=<entries>; used=0 reason=<the None condition: comm=<class>,
expert_statistic, calibrator=<mode>, updates_enabled, query_empty, or test_hook under TLLM_TEST_FT_NO_SLOT_TABLE>), and
WIDEEP_ROUTE_CALLS layer=<idx> n=<calls> lb=0|1 recovery=0|1 at a layer's 1st and 100th route() call (MoE._load_balancer_route;
eager and capture calls only, CUDA graph replays run no Python). No ROUTE_CALLS line: route() never ran. Each patched
module logs WIDEEP_SLOT_TABLE applied/WIDEEP_ROUTE_CALLS applied, or <EVENT>_REFUSED.
Serving load: TLLM_TEST_SERVE_BACKLOG=n sets the default listen backlog of uvicorn.Config (2048) to n in every
process. At 512/GPU the benchmark client opens ~16k fresh connections at once at each wave boundary, and a 2048
backlog resets some of them before the server sees them; the frontend logs WIDEEP_BACKLOG once.
Page migration test: TLLM_TEST_PIN_HOST_COPIES=1 registers (cudaHostRegister, a long-term pin) every EPLB host
copy segment a rank maps in /dev/shm once its sharing is final, so compaction cannot migrate those pages under the
EPLB weight copies; one WIDEEP_PIN_HOST line per layer with the bytes, seconds and any CUDA error.
Host collectives: TLLM_TEST_COLL_TIMES=n times MPIDist's host collectives on rank 0 (both arms: the same class) and logs,
every n calls of a method, WIDEEP_COLL method=<name> calls=n mean_us p50_us p99_us max_us; TLLM_TEST_COLL_RANKS=all
does it on every rank (a rank that enters a collective early waits in it longer, so the spread of these times across
ranks shows how far apart the ranks enter and leave the step's collectives).
Launch skew: TLLM_TEST_LAUNCH_SKEW=n on every rank logs, every n CUDA graph replays, WIDEEP_SKEW rank=<r> with the
p50/p90/p99/max time from the end of the rank's last host collective to its replay call (us). The step's collectives
end within microseconds of each other on every rank, so the spread of this host gap across ranks is how far apart the
ranks launch the step's graph (clock-free), i.e. how long the first rank's dispatch waits for the last.
Host phases: TLLM_TEST_PHASE_TIMES=n on rank 0 times the executor's host phases (PyExecutor schedule, fetch, forward,
update, response handling, the input-copy wait; the recovery hook's step and response taps) and logs, every n calls of
a phase, WIDEEP_PHASE name=<phase> calls mean_us p50_us p90_us total_ms; plus name=period, the time between successive
schedule calls (the loop's step period).
Admissions: TLLM_TEST_LOG_ADMIT=1 on every rank logs WIDEEP_ADMIT call=<balance call> ctx=<context requests the ADP
balance gate released> held=<context requests it held> each time it releases or newly holds context (a few lines per
wave), so another rank's late refill and its 60-iteration timeout release show directly, and WIDEEP_FETCH call=<fetch
call> new=<requests routed to this rank> whenever a fetch hands the rank new requests (when they arrived).
Eager steps: TLLM_TEST_LOG_EAGER=1 logs WIDEEP_EAGER_STEP for every generation-only forward that replayed no CUDA graph.
Request broadcasts: TLLM_TEST_BCAST_LOG=<ranks> (e.g. 0,31) logs on those ranks WIDEEP_BCAST n=<requests> ms=<wall ms>
bytes=<pickled bytes> for every broadcast of new requests (RequestBroadcaster._broadcast_requests: the payload after
the count probe), and WIDEEP_CTX call=<schedule call> reqs=<context requests> tokens=<context tokens> sched_ms=<wall ms
of _prepare_and_schedule_batch, which holds the fetch and the broadcast> for every step that schedules context. Every
rank logs WIDEEP_BCAST applied module=<communicator|request_utils|py_executor> log=<0|1>; only logging ranks are patched.
Modules are patched when TRT-LLM imports them.
"""

import functools
import importlib.abc
import importlib.util
import os
import signal
import sys
import threading
import time

RANK = os.environ.get("OMPI_COMM_WORLD_RANK")
KILL_DECODE = os.environ.get("TLLM_TEST_KILL_IN_DECODE")
KILL_PREFILL = os.environ.get("TLLM_TEST_KILL_IN_PREFILL")
KILL_EPLB_UPDATE = os.environ.get("TLLM_TEST_KILL_IN_EPLB_UPDATE")
KILL_WHEN_BUSY = os.environ.get("TLLM_TEST_KILL_WHEN_BUSY") == "1"
# TLLM_TEST_KILL_RANK: one rank, or a comma list (a whole node).
KILL_HERE = RANK is not None and RANK in os.environ.get("TLLM_TEST_KILL_RANK", "").split(",")
EXCLUDE = os.environ.get("TLLM_TEST_ORACLE_EXCLUDE_RANK")
WATCH_POLL_S = os.environ.get("TLLM_TEST_WATCH_POLL_S")
ALLOC_STATS_S = os.environ.get("TLLM_TEST_ALLOC_STATS_S")
STALL_DECODE = os.environ.get("TLLM_TEST_STALL_IN_DECODE")
STALL_S = float(os.environ.get("TLLM_TEST_STALL_S", "30"))
HOST_STALL_DECODE = os.environ.get("TLLM_TEST_HOST_STALL_IN_DECODE")
GIL_HOG_DECODE = os.environ.get("TLLM_TEST_GIL_HOG_IN_DECODE")
STICKY_DECODE = os.environ.get("TLLM_TEST_STICKY_IN_DECODE")
# TLLM_TEST_KILL_TRIGGER_FOR: per-rank decode triggers; such a rank is a kill target with its own trigger.
_TRIGGER_FOR = dict(item.split("=", 1) for item in os.environ.get("TLLM_TEST_KILL_TRIGGER_FOR", "").split(",") if item)
if RANK in _TRIGGER_FOR:
    # TLLM_TEST_KILL_TRIGGER_ANY_STEP=1 (stress tests): also before a forward with context requests, so the rank dies in
    # its next step of either kind.
    KILL_HERE, KILL_DECODE = True, _TRIGGER_FOR[RANK]
    KILL_PREFILL = KILL_DECODE if os.environ.get("TLLM_TEST_KILL_TRIGGER_ANY_STEP") == "1" else None
RERUN_OWNER = os.environ.get("TLLM_TEST_KILL_RERUN_OWNER")
LATE_WRITE_DECODE = os.environ.get("TLLM_TEST_LATE_WRITE_IN_DECODE")
DELAY_ABORT_S = (float(os.environ.get("TLLM_TEST_DELAY_ABORT_S", "0"))
                 if os.environ.get("TLLM_TEST_DELAY_ABORT_RANK", RANK) == RANK else 0.0)
SERVE_BACKLOG = os.environ.get("TLLM_TEST_SERVE_BACKLOG")
FT_OFF = frozenset(piece for piece in os.environ.get("TLLM_TEST_FT_OFF", "").split(",") if piece)
for _name in ("ITER_RANGE", "PATH"):
    if os.environ.get(f"TLLM_TEST_EXPERT_STATISTIC_{_name}"):
        os.environ[f"EXPERT_STATISTIC_{_name}"] = os.environ[f"TLLM_TEST_EXPERT_STATISTIC_{_name}"]


def _say(event: str, extra: str = "") -> None:
    print(f"{event} rank {RANK} t={time.time():.6f} {extra}".rstrip(), flush=True)


def _kill_if_triggered(trigger: str) -> None:
    if os.path.exists(trigger):
        _say("WIDEEP_SELF_KILL")
        os.kill(os.getpid(), signal.SIGKILL)


def _patch_graphs(module):
    capture = module.CUDAGraphRunner.capture

    def counted_capture(self, key, *args, **kwargs):
        _say("WIDEEP_CAPTURE", f"key={tuple(key)}")
        return capture(self, key, *args, **kwargs)

    module.CUDAGraphRunner.capture = counted_capture
    if KILL_HERE and KILL_DECODE:
        replay = module.CUDAGraphRunner.replay

        delayed = []

        def killing_replay(self, *args, **kwargs):
            if not _holds_requests():
                pass
            elif not KILL_WHEN_BUSY:
                _kill_if_triggered(KILL_DECODE)
            elif os.path.exists(KILL_DECODE):
                if delayed:
                    # Step k+1's host collectives are done and step k has not started on this GPU.
                    _say("WIDEEP_SELF_KILL", "previous_step_held=1")
                    os.kill(os.getpid(), signal.SIGKILL)
                import torch

                delayed.append(True)
                _say("WIDEEP_HOLD_STEP", "s=2")
                torch.cuda._sleep(int(2 * 2e9))  # holds step k's kernels ~2 s on this GPU
            return replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = killing_replay
    if KILL_HERE and STALL_DECODE:
        replay = module.CUDAGraphRunner.replay
        stalled = []

        def stalling_replay(self, *args, **kwargs):
            if not stalled and _holds_requests() and os.path.exists(STALL_DECODE):
                import torch

                stalled.append(True)
                _say("WIDEEP_SELF_STALL", f"s={STALL_S}")
                # ~2 GHz SM clock: the GPU spins this long before the step's kernels run.
                torch.cuda._sleep(int(STALL_S * 2e9))
            return replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = stalling_replay
    if KILL_HERE and STICKY_DECODE:
        replay = module.CUDAGraphRunner.replay
        faulted = []

        def faulting_replay(self, *args, **kwargs):
            if not faulted and _holds_requests() and os.path.exists(STICKY_DECODE):
                import torch

                faulted.append(True)
                _say("WIDEEP_SELF_FAULT", "device-side assert")
                # An out-of-range gather on the GPU: a device-side assert, a sticky error of this context.
                torch.zeros(1, device="cuda")[torch.full((1,), 1 << 20, device="cuda", dtype=torch.long)]
            return replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = faulting_replay
    if KILL_HERE and LATE_WRITE_DECODE:
        replay = module.CUDAGraphRunner.replay
        stopped = []

        def late_writing_replay(self, *args, **kwargs):
            if stopped or not _holds_requests() or not os.path.exists(LATE_WRITE_DECODE):
                return replay(self, *args, **kwargs)
            import subprocess

            import torch

            stopped.append(True)
            delay = float(os.environ.get("TLLM_TEST_LATE_WRITE_DELAY_S", "8"))
            kill_s = float(os.environ.get("TLLM_TEST_LATE_WRITE_KILL_S", "30"))
            # WIDEEP_SELF_STALL: the drivers' marker that the victim acted on the trigger.
            _say("WIDEEP_SELF_STALL", f"late_write delay_s={delay} kill_s={kill_s}")
            torch.cuda._sleep(int(delay * 2e9))  # ~2 GHz SM clock, as the stall hook
            output = replay(self, *args, **kwargs)  # queued behind the sleep; runs while the process is stopped
            subprocess.Popen(["sh", "-c", f"sleep {kill_s}; kill -KILL {os.getpid()}"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _say("WIDEEP_LATE_WRITE_STOP")
            os.kill(os.getpid(), signal.SIGSTOP)
            return output

        module.CUDAGraphRunner.replay = late_writing_replay
    if KILL_HERE and HOST_STALL_DECODE:
        replay = module.CUDAGraphRunner.replay
        host_stalled = []

        def host_stalling_replay(self, *args, **kwargs):
            if not host_stalled and _holds_requests() and os.path.exists(HOST_STALL_DECODE):
                host_stalled.append(True)
                _say("WIDEEP_SELF_STALL", f"host s={STALL_S}")
                time.sleep(STALL_S)  # the serving loop stops; the process and its other threads run
            return replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = host_stalling_replay
    if KILL_HERE and GIL_HOG_DECODE:
        replay = module.CUDAGraphRunner.replay
        hogged = []

        def hog():
            interval = sys.getswitchinterval()
            # TLLM_TEST_GIL_SWITCH_S: the GIL switch interval during the hog (Python's default 0.005 s unless set).
            sys.setswitchinterval(float(os.environ.get("TLLM_TEST_GIL_SWITCH_S", interval)))
            end = time.monotonic() + STALL_S
            while time.monotonic() < end:
                sum(range(10000))
            sys.setswitchinterval(interval)
            _say("WIDEEP_GIL_HOG_DONE", f"s={STALL_S}")

        def gil_hogging_replay(self, *args, **kwargs):
            if not hogged and _holds_requests() and os.path.exists(GIL_HOG_DECODE):
                hogged.append(True)
                _say("WIDEEP_GIL_HOG", f"s={STALL_S}")
                threading.Thread(target=hog, daemon=True).start()
            return replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = gil_hogging_replay


def _patch_autotuner(module):
    profile = module.AutoTuner._profile_runners

    def counted_profile(self, custom_op, *args, **kwargs):
        _say("WIDEEP_AUTOTUNE", f"op={custom_op}")
        return profile(self, custom_op, *args, **kwargs)

    module.AutoTuner._profile_runners = counted_profile


def _patch_cute(module):
    compile_fn = module.compile

    def counted_compile(*args, **kwargs):
        _say("WIDEEP_COMPILE")
        return compile_fn(*args, **kwargs)

    module.compile = counted_compile


def _patch_eplb(module):
    finalize = module.MoeLoadBalancer.finalize_model

    def excluding_finalize(self):
        finalize(self)
        if self.ep_rank != int(EXCLUDE):
            mode = self.extra_slots_mode() if hasattr(self, "extra_slots_mode") else module.extra_slots_mode()
            if mode in ("empty", "ghost"):
                # The excluded rank's experts have no other replica: restore them as the survivors will.
                self.install_recovery_placement([int(EXCLUDE)], self.recovery_placements([int(EXCLUDE)]))
            else:
                self.load_balancer_impl.reconfigure_mask_only([int(EXCLUDE)])
            _say("WIDEEP_ORACLE", f"excluded={EXCLUDE}")

    covers = module.MoeLoadBalancer.covers_loss_of

    def covers_loss_of(self, dead):
        # Recoverability as if EXCLUDE were already lost, so every rank reports the same set.
        dead = {dead} if isinstance(dead, int) else set(dead)
        return covers(self, dead | {int(EXCLUDE)})

    module.MoeLoadBalancer.finalize_model = excluding_finalize
    module.MoeLoadBalancer.covers_loss_of = covers_loss_of


def _patch_a2a(module):
    init = module.NVLinkOneSided.__init__

    def reporting_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _say("WIDEEP_MEM", f"a2a_workspace_bytes={self.workspace_size_per_rank}")

    module.NVLinkOneSided.__init__ = reporting_init


def _patch_kv(module):
    logger = module.logger

    class Reporting:
        def __getattr__(self, name):
            return getattr(logger, name)

        def info(self, message, *args, **kwargs):
            if str(message).startswith("KV cache manager v2 device quota"):
                _say("WIDEEP_MEM", f"kv_quota_gib={str(message).split()[-1]}")
            return logger.info(message, *args, **kwargs)

    module.logger = Reporting()


# Real (not attention-DP dummy) generation requests in this rank's current step; None until the first step.
_REAL_GENERATION = [None]


# TLLM_TEST_TRIGGER_NEEDS_REQUESTS=1: decode triggers wait for a step that serves a real request (the oracle
# cases exclude the victim from routing, so they leave it unset).
_NEEDS_REQUESTS = os.environ.get("TLLM_TEST_TRIGGER_NEEDS_REQUESTS") == "1"


def _holds_requests() -> bool:
    """Whether a decode trigger may fire in this rank's current step."""
    return not _NEEDS_REQUESTS or _REAL_GENERATION[0] is None or _REAL_GENERATION[0] > 0


def _engine_class(module):
    """The model engine class: PyTorchModelEngine on older builds, ModelEngine on newer main."""
    return getattr(module, "PyTorchModelEngine", None) or module.ModelEngine


def _patch_engine(module):
    forward = _engine_class(module).forward

    def killing_forward(self, batch, *args, **kwargs):
        _REAL_GENERATION[0] = sum(
            1 for r in getattr(batch, "generation_requests", ()) if not getattr(r, "is_attention_dp_dummy", False)
        )
        # Every rank runs this wrapper when TLLM_TEST_KILL_RERUN_OWNER is set; only the kill target acts on the trigger.
        if KILL_HERE and KILL_PREFILL and getattr(batch, "num_context_requests", 0) > 0:
            _kill_if_triggered(KILL_PREFILL)
        if RERUN_OWNER and RANK != "0" and getattr(batch, "num_context_requests", 0) > 0:
            _kill_if_rerun_owner(batch)
        return forward(self, batch, *args, **kwargs)

    _engine_class(module).forward = killing_forward


_RERUN_IDS = [None, frozenset()]  # (mtime of prefix.ids, ids)


def _kill_if_rerun_owner(batch) -> None:
    """SIGKILL this rank if it prefills a request rank 0 runs again, unless another rank claimed the kill."""
    path = RERUN_OWNER + ".ids"
    try:
        mtime = os.stat(path).st_mtime_ns
    except FileNotFoundError:
        return
    if mtime != _RERUN_IDS[0]:
        with open(path) as ids:
            _RERUN_IDS[:] = [mtime, frozenset(int(i) for i in ids.read().split())]
    held = [r.py_request_id for r in batch.context_requests if r.py_request_id in _RERUN_IDS[1]]
    if not held:
        return
    try:
        claim = os.open(RERUN_OWNER + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    os.write(claim, f"{RANK} {held[0]}\n".encode())
    os.close(claim)
    _say("WIDEEP_SELF_KILL", f"rerun_owner request={held[0]}")
    os.kill(os.getpid(), signal.SIGKILL)


def _patch_rerun_owner(module):
    """Rank 0: publish the ids it runs again at each recovery (before readmitting them) and log quarantines."""
    deliver = module.PeerLossRecovery._deliver_responses

    def publishing_deliver(self, failed, buffered):
        owners = self._request_owners or {}
        lost = [rid for rid, (owner, _) in owners.items() if owner in failed and rid not in self._delivered]
        rerun = sorted(rid for rid in lost if rid not in self._rerun)
        quarantined = sorted(rid for rid in lost if rid in self._rerun)
        _say("WIDEEP_RERUN", f"failed={sorted(failed)} count={len(rerun)} ids={rerun[:20]}")
        _say("WIDEEP_QUARANTINE", f"failed={sorted(failed)} count={len(quarantined)} ids={quarantined[:20]}")
        if rerun:
            tmp = f"{RERUN_OWNER}.ids.tmp"
            with open(tmp, "w") as ids:
                ids.write(" ".join(map(str, rerun)) + "\n")
            os.replace(tmp, RERUN_OWNER + ".ids")
        return deliver(self, failed, buffered)

    module.PeerLossRecovery._deliver_responses = publishing_deliver


def _patch_delay_abort(module):
    """Delay this rank's reaction to a peer failure by DELAY_ABORT_S after its first notice (watch or recovery)."""
    cls = module.PeerLossRecovery
    notice = []

    def wait_out_delay(source: str) -> None:
        if not notice:
            notice.append(time.monotonic())
            _say("WIDEEP_ABORT_DELAY", f"source={source} s={DELAY_ABORT_S}")
        remaining = notice[0] + DELAY_ABORT_S - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    on_peer_failure = cls._on_peer_failure

    def delayed_watch(self):
        def on_failure():
            # As the native watch does without a delay: the kernels' abort request first, then the host side.
            wait_out_delay("watch")
            self._executor.model_engine.moe_execution_state.request_abort()
            _say("WIDEEP_ABORT_SET", "source=watch")
            on_peer_failure(self)

        # No abort word: the watch must not release the kernels before the delay.
        self._watch = self._executor.dist.watch_peer_failures(on_failure, 0)

    recover = cls._recover

    def delayed_recover(self, error):
        wait_out_delay("recover")
        _say("WIDEEP_ABORT_SET", "source=recover")
        try:
            return recover(self, error)
        finally:
            notice.clear()

    cls._watch_peer_failures = delayed_watch
    cls._recover = delayed_recover


def _patch_watch(module):
    module._WATCH_POLL_S = float(WATCH_POLL_S)


def _alloc_stats_loop(interval: float) -> None:
    import torch

    while True:
        time.sleep(interval)
        if not torch.cuda.is_initialized():
            continue
        stats = torch.cuda.memory_stats()
        with open("/proc/self/stat") as stat:
            fields = stat.read().rsplit(")", 1)[1].split()
        with open("/proc/self/status") as status:
            rss = next(line.split()[1] for line in status if line.startswith("VmRSS"))
        _say(
            "WIDEEP_ALLOC",
            f"retries={stats.get('num_alloc_retries', 0)} "
            f"device_allocs={stats.get('num_device_alloc', 0)} "
            f"device_frees={stats.get('num_device_free', 0)} "
            f"reserved_mib={stats.get('reserved_bytes.all.current', 0) >> 20} "
            f"rss_kib={rss} minflt={fields[7]} majflt={fields[9]}",
        )


def _patch_client(module):
    # TLLM_TEST_KILL_AFTER_STOPPED=r: rank r SIGKILLs itself right after sending its first Stopped, so it fails
    # while the survivors agree on a recovery.
    send = module.RecoveryClient.send

    def killing_send(self, message):
        send(self, message)
        if type(message).__name__ == "Stopped":
            _say("WIDEEP_SELF_KILL", "after Stopped")
            os.kill(os.getpid(), signal.SIGKILL)

    module.RecoveryClient.send = killing_send


def _patch_recovery(module):
    # TLLM_TEST_BORROW_FROM_KV=1: the device reports no free memory, so borrowed slots always take theirs from the
    # KV cache (the path a full device takes).
    if os.environ.get("TLLM_TEST_BORROW_FROM_KV") == "1":
        module._free_device_bytes = lambda: 0
    if RERUN_OWNER and RANK == "0":
        _patch_rerun_owner(module)
    if DELAY_ABORT_S > 0:
        _patch_delay_abort(module)


def _patch_alloc_stats(module):
    import threading

    threading.Thread(target=_alloc_stats_loop, args=(float(ALLOC_STATS_S),), daemon=True).start()


def _patch_empty_extra_slots(module):
    """FT off (TLLM_TEST_EMPTY_EXTRA_SLOTS=1): the EPLB layer as under FT's empty mode, so a 288-slot layout can be
    measured against eplb0 without the FT executor (healthy-cost attribution). The load balancer believes recovery
    is on: it keeps host copies (needed to change slots) and empties the extra slots before warmup."""
    module.peer_loss_recovery_enabled = lambda: True
    # The C++ load balancer accepts empty (-1) slots, and finalizes a layer without host weights, only under
    # TLLM_FAULT_TOLERANCE_MODE=1 (read at each call). Set it for those calls only: the rest of the process stays FT off.
    def with_ft_mode(call):
        def wrapped(*args, **kwargs):
            previous = os.environ.get("TLLM_FAULT_TOLERANCE_MODE")
            os.environ["TLLM_FAULT_TOLERANCE_MODE"] = "1"
            try:
                return call(*args, **kwargs)
            finally:
                if previous is None:
                    del os.environ["TLLM_FAULT_TOLERANCE_MODE"]
                else:
                    os.environ["TLLM_FAULT_TOLERANCE_MODE"] = previous

        return wrapped

    layer = module.SingleLayerMoeLoadBalancer
    layer.set_initial_weight_assignments = with_ft_mode(layer.set_initial_weight_assignments)
    finalize = with_ft_mode(module.MoeLoadBalancer.finalize_model)

    def logging_finalize(self):
        finalize(self)
        _say("WIDEEP_EMPTY_EXTRA_SLOTS", f"mode={self.extra_slots_mode()}")

    module.MoeLoadBalancer.finalize_model = logging_finalize


def _patch_populate(module):
    """TLLM_TEST_POPULATE (EV6): how the EPLB host copies' page tables get built after startup. ``off``: not at all;
    ``timed``: as the stack does (one background madvise per segment), logging start, end and GiB."""
    mode = os.environ.get("TLLM_TEST_POPULATE")
    name = "_start_populate" if hasattr(module, "_start_populate") else "_populate_in_background"
    if mode == "off":
        setattr(module, name, lambda segments: _say("WIDEEP_POPULATE", "off"))
    elif mode == "timed":
        populate = getattr(module, name)

        def timed(segments):
            gib = sum(segment.size for segment in segments) / 2**30
            _say("WIDEEP_POPULATE", f"start segments={len(segments)} GiB={gib:.1f}")
            start = time.monotonic()
            started = populate(segments)
            for worker in threading.enumerate():
                if worker.name == "eplb_host_copy_populate":
                    threading.Thread(target=lambda t=worker: (t.join(), _say(
                        "WIDEEP_POPULATE", f"done s={time.monotonic() - start:.1f}")), daemon=True).start()
            return started

        setattr(module, name, timed)


def _patch_kill_in_eplb_update(module):
    start_iter = module.MoeLoadBalancer.start_iter

    def killing_start_iter(self):
        start_iter(self)
        if self.enable_update_weights and os.path.exists(KILL_EPLB_UPDATE):
            _say("WIDEEP_KILL_PHASE", f"eplb_update iter={self.iter_id}")
            _kill_if_triggered(KILL_EPLB_UPDATE)

    module.MoeLoadBalancer.start_iter = killing_start_iter


def _patch_deadline_comm(module):
    """C10: log each start of an MPI call that a recovery past its deadline must not make."""
    for name in ("revoke", "shrink", "watch_peer_failures"):
        def logged(*args, _call=getattr(module.MPIDist, name), _name=name, **kwargs):
            _say("WIDEEP_MPI_CALL", f"name={_name}")
            return _call(*args, **kwargs)
        setattr(module.MPIDist, name, logged)


def _patch_deadline_client(module):
    """C10: log every RecoveryTimeout (a recovery deadline passed) when it is raised."""
    init = module.RecoveryTimeout.__init__

    def logged_init(self, *args, **kwargs):
        _say("WIDEEP_DEADLINE", repr(str(args[0]) if args else "")[:160])
        init(self, *args, **kwargs)

    module.RecoveryTimeout.__init__ = logged_init


def _patch_coll_times(module):
    every = int(os.environ["TLLM_TEST_COLL_TIMES"])
    names = ("broadcast", "allgather", "allreduce", "tp_allreduce", "tp_broadcast", "tp_allgather", "tp_allgather_int64",
             "tp_gather")

    def timed(name, call):
        samples = []

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return call(*args, **kwargs)
            finally:
                samples.append((time.perf_counter() - start) * 1e6)
                if len(samples) >= every:
                    ordered = sorted(samples)
                    _say("WIDEEP_COLL", f"method={name} calls={len(ordered)} mean_us={sum(ordered) / len(ordered):.1f} "
                         f"p50_us={ordered[len(ordered) // 2]:.1f} p99_us={ordered[int(len(ordered) * 0.99)]:.1f} "
                         f"max_us={ordered[-1]:.1f}")
                    samples.clear()

        return wrapper

    for name in names:
        if name in module.MPIDist.__dict__:
            setattr(module.MPIDist, name, timed(name, module.MPIDist.__dict__[name]))


_LAST_COLLECTIVE = [None, ""]


def _patch_skew_collectives(module):
    def noted(name, call):
        def wrapper(*args, **kwargs):
            try:
                return call(*args, **kwargs)
            finally:
                _LAST_COLLECTIVE[0], _LAST_COLLECTIVE[1] = time.perf_counter(), name

        return wrapper

    for name in ("broadcast", "allgather", "allreduce", "tp_allreduce", "tp_broadcast", "tp_allgather",
                 "tp_allgather_int64", "tp_gather"):
        if name in module.MPIDist.__dict__:
            setattr(module.MPIDist, name, noted(name, module.MPIDist.__dict__[name]))


def _patch_skew_replay(module):
    every = int(os.environ["TLLM_TEST_LAUNCH_SKEW"])
    replay = module.CUDAGraphRunner.replay
    gaps, after = [], {}

    def timed_replay(self, *args, **kwargs):
        if _LAST_COLLECTIVE[0] is not None:
            gaps.append((time.perf_counter() - _LAST_COLLECTIVE[0]) * 1e6)
            after[_LAST_COLLECTIVE[1]] = after.get(_LAST_COLLECTIVE[1], 0) + 1
            if len(gaps) >= every:
                ordered = sorted(gaps)
                pick = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]  # noqa: E731
                _say("WIDEEP_SKEW", f"replays={len(ordered)} p50_us={pick(0.5):.1f} p90_us={pick(0.9):.1f} "
                     f"p99_us={pick(0.99):.1f} max_us={ordered[-1]:.1f} after={max(after, key=after.get)}")
                gaps.clear()
                after.clear()
        return replay(self, *args, **kwargs)

    module.CUDAGraphRunner.replay = timed_replay


def _patch_admit_log(module):
    balance = module.PyExecutor._balance_adp_requests
    calls, holding = [0], [0]

    def logged_balance(self, context_requests, generation_requests):
        released = balance(self, context_requests, generation_requests)
        calls[0] += 1
        held = len(context_requests) - len(released)
        if released or (held and not holding[0]):
            _say("WIDEEP_ADMIT", f"call={calls[0]} ctx={len(released)} held={held} gen={len(generation_requests)}")
        holding[0] = held
        return released

    module.PyExecutor._balance_adp_requests = logged_balance
    fetch = module.PyExecutor._fetch_new_requests
    fetches = [0]

    def logged_fetch(self, *args, **kwargs):
        new = fetch(self, *args, **kwargs)
        fetches[0] += 1
        if new:
            _say("WIDEEP_FETCH", f"call={fetches[0]} new={len(new)}")
        return new

    module.PyExecutor._fetch_new_requests = logged_fetch


_PHASE_EXECUTOR = ("_prepare_and_schedule_batch", "_fetch_new_requests", "_forward_step", "_sample_async",
                   "_update_requests", "_handle_responses", "_enqueue_responses", "_wait_for_model_engine_input_copy",
                   "_process_previous_batch")
_PHASE_RECOVERY = ("step_prepared", "commit_step", "gathering_responses", "responses_gathered", "fetching_requests",
                   "requests_routed")


def _phase_logger(every):
    samples = {}

    def note(name, us):
        values = samples.setdefault(name, [])
        values.append(us)
        if len(values) >= every:
            ordered = sorted(values)
            _say("WIDEEP_PHASE", f"name={name} calls={len(ordered)} mean_us={sum(ordered) / len(ordered):.1f} "
                 f"p50_us={ordered[len(ordered) // 2]:.1f} p90_us={ordered[int(len(ordered) * 0.9)]:.1f} "
                 f"total_ms={sum(ordered) / 1e3:.1f}")
            values.clear()

    return note


_PHASE_NOTE = [None]


def _timed_methods(cls, names):
    note = _PHASE_NOTE[0]
    for name in names:
        call = cls.__dict__.get(name)
        if call is None:
            continue

        def make(name, call):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return call(*args, **kwargs)
                finally:
                    note(name, (time.perf_counter() - start) * 1e6)

            return wrapper

        setattr(cls, name, make(name, call))


def _patch_phase_executor(module):
    _timed_methods(module.PyExecutor, _PHASE_EXECUTOR)
    schedule = module.PyExecutor._prepare_and_schedule_batch
    last = [None]

    def periodic(self, *args, **kwargs):
        now = time.perf_counter()
        if last[0] is not None:
            _PHASE_NOTE[0]("period", (now - last[0]) * 1e6)
        last[0] = now
        return schedule(self, *args, **kwargs)

    module.PyExecutor._prepare_and_schedule_batch = periodic


def _patch_phase_recovery(module):
    for cls in vars(module).values():
        if isinstance(cls, type) and any(n in cls.__dict__ for n in _PHASE_RECOVERY):
            _timed_methods(cls, _PHASE_RECOVERY)


def _patch_load_balancer(module):
    if os.environ.get("TLLM_TEST_EMPTY_EXTRA_SLOTS") == "1":
        _patch_empty_extra_slots(module)
    _patch_populate(module)


_FT_OFF_TAPS = {
    "resp": ("fetching_requests", "broadcast_requests", "requests_routed", "gathering_responses", "responses_gathered"),
    "step": ("step_prepared", "step_released", "commit_step"),
}
_COMMUNICATOR = "tensorrt_llm._torch.distributed.communicator"
_RECOVERY = "tensorrt_llm._torch.pyexecutor.peer_loss_recovery"


def _interruptible_code():
    """The code object every _interruptible wrapper shares, or None before the communicator is imported."""
    module = sys.modules.get(_COMMUNICATOR)
    return None if module is None else module._interruptible(lambda self: None).__code__


def _patch_ft_off_comm(module):
    """TLLM_TEST_FT_OFF=coll: every host collective _interruptible wrapped runs bare."""
    code = module._interruptible(lambda self: None).__code__
    unwrapped = []
    for cls in [value for value in vars(module).values() if isinstance(value, type)]:
        for name, attr in list(vars(cls).items()):
            if getattr(attr, "__code__", None) is code:
                setattr(cls, name, attr.__wrapped__)
                unwrapped.append(f"{cls.__name__}.{name}")
    _say("WIDEEP_FT_OFF", f"module=communicator unwrapped={len(unwrapped)} {','.join(unwrapped)}")


class _NoWatch:
    """Stands in for PeerFailureWatch (TLLM_TEST_FT_OFF=watch): never fails, nothing to stop."""

    failed = False

    def stop(self) -> None:
        pass


def _no_watch(self) -> None:
    self._watch = _NoWatch()


def _first_call_logged(cls, name, replacement):
    """``replacement`` as cls.name; the first call logs WIDEEP_FT_OFF_CALL and installs ``replacement`` itself."""

    def first_call(self, *args, **kwargs):
        setattr(cls, name, replacement)
        _say("WIDEEP_FT_OFF_CALL", f"name={name}")
        return replacement(self, *args, **kwargs)

    return first_call


def _patch_ft_off_recovery(module):
    """TLLM_TEST_FT_OFF=resp,step,watch on PeerLossRecovery; for every piece, the serving loop's proof line."""
    cls, base = module.PeerLossRecovery, module.RecoveryHook
    originals = {}
    for piece, taps in _FT_OFF_TAPS.items():
        if piece in FT_OFF:
            for name in taps:
                originals[name] = getattr(cls, name)
                setattr(cls, name, _first_call_logged(cls, name, getattr(base, name)))
    if "step" in FT_OFF:
        cls.enabled = False
    if "watch" in FT_OFF:
        cls._watch_peer_failures = _no_watch
    _say("WIDEEP_FT_OFF", f"module=peer_loss_recovery taps={','.join(originals) or '-'} "
         f"enabled={cls.enabled} watch={'stub' if 'watch' in FT_OFF else 'native'}")
    run = cls.run

    def proving_run(self, event_loop):
        # What the live executor runs, piece by piece.
        verified = []
        for piece, taps in _FT_OFF_TAPS.items():
            if all(name in originals and getattr(type(self), name) is not originals[name] for name in taps) and (
                    piece != "step" or self.enabled is False):
                verified.append(piece)
        code = _interruptible_code()
        if code is not None and getattr(type(self._executor.dist).tp_gather, "__code__", code) is not code:
            verified.append("coll")
        if type(self)._watch_peer_failures is _no_watch:
            verified.append("watch")
        try:
            hang = module.fault_tolerance_config().hang_detection
        except Exception as error:  # noqa: BLE001 - reported, not needed for the run
            hang = f"unknown({type(error).__name__})"
        _say("WIDEEP_FT_OFF_ACTIVE", f"serving={int(not self._executor.gather_all_responses)} "
             f"verified={','.join(sorted(verified)) or '-'} hang_detection={hang}")
        return run(self, event_loop)

    cls.run = proving_run


def _patch_uvicorn(module):
    init = module.Config.__init__

    def init_with_backlog(self, *args, **kwargs):
        kwargs.setdefault("backlog", int(SERVE_BACKLOG))
        init(self, *args, **kwargs)
        _say("WIDEEP_BACKLOG", f"backlog={self.backlog}")

    module.Config.__init__ = init_with_backlog


def _patch_pin_host_copies(module):
    finalize = module.HostMoeTensorSharer.finalize_host_tensor_sharing

    def pinning_finalize(self, *args, **kwargs):
        result = finalize(self, *args, **kwargs)
        import ctypes

        import torch

        start, pinned, errors = time.time(), 0, []
        for shm in [self.own_shm, *(self.imported_shms or [])]:
            if shm is None:
                continue
            address = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
            status = torch.cuda.cudart().cudaHostRegister(address, shm.size, 0)
            if int(status) == 0:
                pinned += shm.size
            else:
                errors.append(int(status))
        _say("WIDEEP_PIN_HOST", f"layer={getattr(self, 'layer_id', '?')} bytes={pinned} s={time.time() - start:.2f} "
             f"errors={errors or '-'}")
        return result

    module.HostMoeTensorSharer.finalize_host_tensor_sharing = pinning_finalize


_REPLAYED = [False]


def _patch_replay_flag(module):
    replay = module.CUDAGraphRunner.replay

    def flagged_replay(self, *args, **kwargs):
        _REPLAYED[0] = True
        return replay(self, *args, **kwargs)

    module.CUDAGraphRunner.replay = flagged_replay


def _patch_eager_log(module):
    forward = _engine_class(module).forward
    calls = [0]

    def logged_forward(self, batch, *args, **kwargs):
        _REPLAYED[0] = False
        result = forward(self, batch, *args, **kwargs)
        calls[0] += 1
        if not _REPLAYED[0] and getattr(batch, "num_context_requests", 0) == 0:
            _say("WIDEEP_EAGER_STEP", f"forward={calls[0]} generation={len(getattr(batch, 'generation_requests', ()))}")
        return result

    _engine_class(module).forward = logged_forward


_FT_UNBOUND = os.environ.get("TLLM_TEST_FT_UNBOUND") == "1"
_FT_NO_INPLACE = os.environ.get("TLLM_TEST_FT_NO_INPLACE_SLOTS") == "1"
_FT_NO_GHOST = os.environ.get("TLLM_TEST_FT_NO_GHOST") == "1"
_FT_NO_SLOT_TABLE = os.environ.get("TLLM_TEST_FT_NO_SLOT_TABLE") == "1"
# The EPGroupHealth objects the communication factory created while FT_UNBOUND kept them from the comms.
_UNBOUND_HEALTH = {}


def _missing_anchors(anchors):
    """The labels of (label, object, text) anchors whose object's source lacks text (the patched code moved)."""
    import inspect

    missing = []
    for label, obj, text in anchors:
        try:
            found = obj is not None and text in inspect.getsource(obj)
        except (OSError, TypeError):
            found = False
        if not found:
            missing.append(label)
    return missing


def _refused(event, module, missing) -> bool:
    if missing:
        _say(f"{event}_REFUSED", f"module={module.__name__.rsplit('.', 1)[-1]} moved={','.join(missing)}")
    return bool(missing)


def _moe_comm_modules(model, nvlink_one_sided):
    """The MoE modules of ``model`` on NVLinkOneSided, as MoeExecutionState.for_model finds them."""
    return [m for m in model.modules() if isinstance(getattr(m, "comm", None), nvlink_one_sided)
            and hasattr(m, "backend")]


def _patch_unbound_factory(module):
    """FT_UNBOUND: NVLinkOneSided is built as with FT off (no EPGroupHealth: no rank mask, no host watchdog);
    the factory still creates FT's EPGroupHealth, which the execution state keeps."""
    factory = getattr(module, "CommunicationFactory", None)
    source_calls = 0
    if factory is not None and hasattr(module, "get_wide_ep_ft_options"):
        import inspect

        source_calls = inspect.getsource(factory).count("get_wide_ep_ft_options(")
    if _refused("WIDEEP_FT_UNBOUND", module, [] if source_calls == 2 else [f"get_wide_ep_ft_options_calls={source_calls}"]):
        return
    options = module.get_wide_ep_ft_options

    def unbound_options(model_config):
        health, timeout_s, poll_interval_s = options(model_config)
        if health is None:
            return health, timeout_s, poll_interval_s
        _UNBOUND_HEALTH[id(health)] = health
        return None, None, poll_interval_s

    module.get_wide_ep_ft_options = unbound_options
    _say("WIDEEP_FT_UNBOUND", "applied module=communication_factory health=withheld")


def _patch_unbound_state(module):
    """FT_UNBOUND: MoeExecutionState.for_model builds the state (abort record, rank mask, progress_comm, retained
    workspaces, tracked layers) as bind_execution_state would, but binds no comm: dispatch/combine see state None."""
    cls, comm_cls = module.MoeExecutionState, module.NVLinkOneSided
    if _refused("WIDEEP_FT_UNBOUND", module, _missing_anchors([
            ("for_model.bind_layers", cls.for_model, "NVLinkOneSided.bind_layers(state"),
            ("for_model.progress_comm", cls.for_model, "state.progress_comm = comm"),
            ("bind_execution_state.retain", getattr(comm_cls, "bind_execution_state", None), "state.retain("),
            ("bind_execution_state.track_layer", getattr(comm_cls, "bind_execution_state", None), "state.track_layer("),
            ("dispatch.state", getattr(comm_cls, "dispatch", None), "state = self._execution_state"),
    ])):
        return

    def unbound_for_model(state_cls, model):
        moes = _moe_comm_modules(model, comm_cls)
        if not moes:
            raise RuntimeError("TLLM_TEST_FT_UNBOUND: the model has no NVLinkOneSided MoE layer")
        if len(_UNBOUND_HEALTH) != 1:
            raise RuntimeError(f"TLLM_TEST_FT_UNBOUND: {len(_UNBOUND_HEALTH)} EPGroupHealth objects withheld, need 1")
        comms = list({id(m.comm): m.comm for m in moes}.values())
        state = state_cls(next(iter(_UNBOUND_HEALTH.values())), comms[0].workspace.device)
        state.progress_comm = comms[0]
        for comm in comms:  # bind_execution_state without the binding
            state.retain(comm.mnnvl_mem, comm._reset_local_workspace,
                         lambda source, comm=comm: comm.source_write_ranges(comm.moe_a2a_metainfo, source))
            state.track_layer(comm.ep_rank, comm.reset_state)
        schedulers = [getattr(m, "scheduler", None) for m in moes]
        _say("WIDEEP_FT_UNBOUND_ACTIVE", f"layers={len(moes)} comms={len(comms)} "
             f"bound={sum(c._execution_state is not None for c in comms)} "
             f"rank_mask={sum(bool(c._rank_mask_enabled) for c in comms)} "
             f"health={sum(c.ep_group_health is not None for c in comms)} "
             f"slot_table={sum(bool(getattr(s, '_peer_loss_recovery', True)) for s in schedulers)} "
             f"workspaces={len(state._workspaces)} progress_comm={int(state.progress_comm is not None)}")
        return state

    cls.for_model = classmethod(unbound_for_model)
    _say("WIDEEP_FT_UNBOUND", "applied module=execution_state for_model=unbound")


def _patch_unbound_scheduler(module):
    """FT_UNBOUND: the MoE scheduler routes as with FT off: route() instead of the expert-to-slot table, and no
    receive-slot compaction or source slots in the comm plan."""
    cls = module.ExternalCommMoEScheduler
    if _refused("WIDEEP_FT_UNBOUND", module, _missing_anchors([
            ("init.peer_loss_recovery", cls.__init__, "self._peer_loss_recovery = peer_loss_recovery_enabled()"),
    ]) + ([] if hasattr(module, "peer_loss_recovery_enabled") else ["peer_loss_recovery_enabled"])):
        return
    module.peer_loss_recovery_enabled = lambda: False
    _say("WIDEEP_FT_UNBOUND", "applied module=moe_scheduler slot_table=off")


def _first_call_proof(cls, name, event, describe, done=lambda *a: True):
    """Wrap cls.name: log ``event`` with describe(self, args, kwargs, result) on the first call, and restore the
    original once done(self, args, kwargs, result) holds."""
    original = cls.__dict__[name]
    logged = []

    @functools.wraps(original)  # __wrapped__: source anchors still read the original
    def proving(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not logged:
            logged.append(True)
            _say(event, describe(self, args, kwargs, result))
        if done(self, args, kwargs, result):
            setattr(cls, name, original)
        return result

    setattr(cls, name, proving)


def _patch_unbound_dispatch(module):
    """FT_UNBOUND: the first NVLinkOneSided dispatch says how it ran."""
    _first_call_proof(module.NVLinkOneSided, "dispatch", "WIDEEP_FT_UNBOUND_CALL", lambda self, args, kwargs, _: (
        f"name=dispatch state={'None' if self._execution_state is None else 'bound'} "
        f"rank_mask={int(bool(self._rank_mask_enabled))} expert_to_slot={int(kwargs.get('expert_to_slot') is not None)}"))


def _patch_no_inplace_backend(module):
    """FT_NO_INPLACE_SLOTS: CuteDSL reports that it cannot read the receive slots in place, so the scheduler hands
    it a compact copy of the hidden states and scales, as every other backend gets."""
    cls = module.CuteDslFusedMoE
    if _refused("WIDEEP_FT_NO_INPLACE", module, _missing_anchors([
            ("supports_dispatch_receive_slots", cls.__dict__.get("supports_dispatch_receive_slots"), "has_nvfp4"),
    ])):
        return
    cls.supports_dispatch_receive_slots = lambda self: False
    _say("WIDEEP_FT_NO_INPLACE", f"applied module=fused_moe_cute_dsl class={cls.__name__} receive_slots=compact")


def _patch_no_inplace_scheduler(module):
    """FT_NO_INPLACE_SLOTS: the first receive-slot compaction that copies says so."""
    cls = module.ExternalCommMoEScheduler
    if _refused("WIDEEP_FT_NO_INPLACE", module, _missing_anchors([
            ("_compact_receive_slots", cls.__dict__.get("_compact_receive_slots"), "supports_dispatch_receive_slots()"),
            ("_build_comm_plan", cls.__dict__.get("_build_comm_plan"), "supports_dispatch_receive_slots()"),
    ])):
        return
    _first_call_proof(
        cls, "_compact_receive_slots", "WIDEEP_FT_NO_INPLACE_CALL",
        lambda self, args, kwargs, result: (f"name=_compact_receive_slots compacted={int(result[0] is not args[0])} "
                                            f"rows_in={args[0].shape[0]} rows_out={result[0].shape[0]}"),
        done=lambda self, args, kwargs, result: result[0] is not args[0])
    _say("WIDEEP_FT_NO_INPLACE", "applied module=moe_scheduler proof=_compact_receive_slots")


def _prove_after_binding(module, event, describe):
    """Once MoeExecutionState.for_model binds the layers, log ``event`` with layers=<n> bound=<n bound to the state>
    and describe(moes)."""
    cls, comm_cls = module.MoeExecutionState, module.NVLinkOneSided
    for_model = cls.for_model.__func__

    def proving_for_model(state_cls, model):
        state = for_model(state_cls, model)
        moes = _moe_comm_modules(model, comm_cls)
        _say(event, f"layers={len(moes)} bound={sum(m.comm._execution_state is state for m in moes)} {describe(moes)}")
        return state

    cls.for_model = classmethod(proving_for_model)


def _patch_no_inplace_state(module):
    """FT_NO_INPLACE_SLOTS: once for_model binds the layers, prove they are bound and none reads the slots in place."""
    _prove_after_binding(module, "WIDEEP_FT_NO_INPLACE_ACTIVE", lambda moes: (
        f"inplace={sum(bool(m.backend.supports_dispatch_receive_slots()) for m in moes)} "
        f"backends={','.join(sorted({type(m.backend).__name__ for m in moes}))}"))


def _values(items):
    """The distinct values of ``items``, sorted and comma separated (- if none)."""
    return ",".join(str(v) for v in sorted(set(items), key=str)) or "-"


# FT_NO_GHOST: [the MoeLoadBalancer, its real largest failure domain] once its __init__ asks for the latter.
_NO_GHOST_BALANCER = []


def _patch_no_ghost_balancer(module):
    """FT_NO_GHOST: the load balancer plans for no loss, so add_layer appends no ghost slot (256 slots, not 320), and
    every loss counts as covered, so the executor's startup check admits a layout no survivors could serve. The
    weights keep the ghost run's storage (GhostSlotMemory on the device, with no empty slot) and GPU copy mode,
    which register_weight_slots_after_to_cuda sets only for a ghost layout. Ring host copies stay (copy_nodes: one
    slot per expert, static). Healthy cases only: a peer loss cannot recover."""
    cls = module.MoeLoadBalancer
    if _refused("WIDEEP_FT_NO_GHOST", module, _missing_anchors([
            ("init.largest_loss", cls.__init__, "self._largest_loss = self._largest_failure_domain(hosts)"),
            ("add_layer.ghost_slot_count", cls.add_layer, "survivors = self.ep_size - self._largest_loss"),
            ("add_layer.copy_nodes", cls.add_layer, "self.copy_nodes = self._ring_copy_nodes"),
            ("register.ghost_mode", cls.register_weight_slots_after_to_cuda, 'if mode in ("ghost", "replicas+ghost"):'),
            ("register.gpu_memcpy", cls.register_weight_slots_after_to_cuda, "self.set_use_gpu_memcpy(True)"),
            ("make_tensor_host_accessible.ghost_slots",
             getattr(module, "SingleLayerMoeLoadBalancer", None) and module.SingleLayerMoeLoadBalancer.__dict__.get(
                 "make_tensor_host_accessible"), "if self.ghost_slots is None:"),
            ("covers_loss_of", cls.__dict__.get("covers_loss_of"), "self.recovery_placements(dead_ranks)"),
    ])):
        return
    largest = cls._largest_failure_domain

    def no_loss(self, hosts):
        _NO_GHOST_BALANCER[:] = [self, largest(self, hosts)]
        return 0

    register = cls.register_weight_slots_after_to_cuda

    def ghost_storage_register(self):
        if self._largest_loss == 0 and self.copy_nodes is not None:
            self.set_use_gpu_memcpy(True)
            for layer in self.single_layer_load_balancers:
                layer.ghost_slots = []  # moe_ghost_slot_storage, no slot empty
        return register(self)

    cls._largest_failure_domain = no_loss
    cls.register_weight_slots_after_to_cuda = ghost_storage_register
    cls.covers_loss_of = lambda self, dead_ranks: True
    _say("WIDEEP_FT_NO_GHOST", "applied module=moe_load_balancer largest_loss=0 covers_loss_of=always storage=ghost_memory")


def _patch_no_ghost_state(module):
    """FT_NO_GHOST: once for_model binds the layers, prove they are bound on the 256-slot layout, ghost storage."""
    def describe(moes):
        balancers = [getattr(m, "layer_load_balancer", None) for m in moes]
        balancer, real = _NO_GHOST_BALANCER or (None, "?")
        return (f"num_slots={_values(getattr(m, 'num_slots', '?') for m in moes)} "
                f"slots_per_rank={_values(getattr(b, 'slot_count_per_rank', '?') for b in balancers)} "
                f"ghost={_values(getattr(b, 'ghost_slot_count', '?') for b in balancers)} "
                f"ghost_storage={sum(getattr(b, 'ghost_slots', None) == [] for b in balancers)} "
                f"copy_nodes={'ring' if getattr(balancer, 'copy_nodes', None) is not None else 'none'} "
                f"largest_loss={getattr(balancer, '_largest_loss', '?')} real_largest_loss={real}")

    _prove_after_binding(module, "WIDEEP_FT_NO_GHOST_ACTIVE", describe)


def _patch_no_ghost_dispatch(module):
    """FT_NO_GHOST: the first NVLinkOneSided dispatch says how many slots it routes to."""
    _first_call_proof(module.NVLinkOneSided, "dispatch", "WIDEEP_FT_NO_GHOST_CALL", lambda self, args, kwargs, _: (
        f"name=dispatch state={'None' if self._execution_state is None else 'bound'} num_slots={self.num_experts} "
        f"expert_to_slot={int(kwargs.get('expert_to_slot') is not None)} "
        f"table={0 if kwargs.get('expert_to_slot') is None else kwargs['expert_to_slot'].numel()}"))


def _patch_no_slot_table_scheduler(module):
    """FT_NO_SLOT_TABLE: the bound path, but the scheduler has no expert-to-slot table, so it routes with route()
    (the load balancer's routing kernel) and the dispatch kernel reads the slot ids instead of looking them up."""
    cls = module.ExternalCommMoEScheduler
    if _refused("WIDEEP_FT_NO_SLOT_TABLE", module, _missing_anchors([
            ("_forward_chunk_impl.slot_table", cls.__dict__.get("_forward_chunk_impl"),
             "expert_to_slot = self._dispatch_slot_table()"),
            ("_forward_chunk_impl.route", cls.__dict__.get("_forward_chunk_impl"), "if expert_to_slot is None:"),
            ("_dispatch_slot_table", cls.__dict__.get("_dispatch_slot_table"), "one_slot_table()"),
    ])):
        return
    cls._dispatch_slot_table = lambda self: None
    _say("WIDEEP_FT_NO_SLOT_TABLE", "applied module=moe_scheduler slot_table=route")


def _patch_no_slot_table_state(module):
    """FT_NO_SLOT_TABLE: once for_model binds the layers, prove they are bound, FT-routed, without a table."""
    _prove_after_binding(module, "WIDEEP_FT_NO_SLOT_TABLE_ACTIVE", lambda moes: (
        f"recovery={sum(bool(getattr(m.scheduler, '_peer_loss_recovery', False)) for m in moes)} "
        f"slot_table={sum(m.scheduler._dispatch_slot_table() is not None for m in moes)}"))


def _patch_no_slot_table_dispatch(module):
    """FT_NO_SLOT_TABLE: the first NVLinkOneSided dispatch says how it ran."""
    _first_call_proof(module.NVLinkOneSided, "dispatch", "WIDEEP_FT_NO_SLOT_TABLE_CALL", lambda self, args, kwargs, _: (
        f"name=dispatch state={'None' if self._execution_state is None else 'bound'} "
        f"rank_mask={int(bool(self._rank_mask_enabled))} expert_to_slot={int(kwargs.get('expert_to_slot') is not None)}"))


def _slot_table_none_reason(module, scheduler, replaced) -> str:
    """Which of _dispatch_slot_table's None conditions holds for ``scheduler`` (read once per layer)."""
    if replaced:
        return "test_hook"
    if not scheduler._is_using_nvlink_one_sided():
        return f"comm={type(scheduler.moe.comm).__name__}"
    if module.ExpertStatistic.expert_statistic_obj is not None:
        return "expert_statistic"
    mode = module.get_calibrator().mode
    if mode != module.CalibratorMode.NONE:
        return f"calibrator={getattr(mode, 'name', mode)}"
    if scheduler.moe.layer_load_balancer.updates_enabled:
        return "updates_enabled"
    return "query_empty"


def _patch_slot_table_log(module):
    """LOG_SLOT_TABLE: each layer's first expert-to-slot table query says whether the dispatch uses the table."""
    cls = module.ExternalCommMoEScheduler
    # FT_NO_SLOT_TABLE (patched first) checked _dispatch_slot_table and replaced it: this wraps its replacement.
    if _refused("WIDEEP_SLOT_TABLE", module, _missing_anchors([
            ("_forward_chunk_impl.slot_table", cls.__dict__.get("_forward_chunk_impl"),
             "expert_to_slot = self._dispatch_slot_table()"),
    ] + ([] if _FT_NO_SLOT_TABLE else [
            ("_dispatch_slot_table", cls.__dict__.get("_dispatch_slot_table"), "one_slot_table()"),
    ])) + [name for name in ("ExpertStatistic", "get_calibrator", "CalibratorMode") if not hasattr(module, name)]):
        return
    original = cls.__dict__["_dispatch_slot_table"]
    replaced = original.__name__ == "<lambda>"  # FT_NO_SLOT_TABLE's
    seen = set()

    @functools.wraps(original)
    def logged(self):
        table = original(self)
        layer = self.moe.layer_idx
        if layer not in seen:
            seen.add(layer)
            _say("WIDEEP_SLOT_TABLE", f"layer={layer} used=1 table={table.numel()}" if table is not None else
                 f"layer={layer} used=0 reason={_slot_table_none_reason(module, self, replaced)}")
        return table

    cls._dispatch_slot_table = logged
    _say("WIDEEP_SLOT_TABLE", "applied module=moe_scheduler")


def _patch_route_calls(module):
    """LOG_SLOT_TABLE: count each layer's route() calls; log the 1st and the 100th."""
    cls = module.MoE
    if _refused("WIDEEP_ROUTE_CALLS", module, _missing_anchors([
            ("_load_balancer_route", cls.__dict__.get("_load_balancer_route"), ".route("),
    ])):
        return
    original = cls.__dict__["_load_balancer_route"]
    calls = {}

    @functools.wraps(original)
    def counted(self, *args, **kwargs):
        layer = self.layer_idx
        n = calls[layer] = calls.get(layer, 0) + 1
        if n == 1 or n == 100:
            recovery = getattr(getattr(self, "scheduler", None), "_peer_loss_recovery", None)
            _say("WIDEEP_ROUTE_CALLS", f"layer={layer} n={n} lb={int(self.layer_load_balancer is not None)} "
                 f"recovery={'?' if recovery is None else int(bool(recovery))}")
        return original(self, *args, **kwargs)

    cls._load_balancer_route = counted
    _say("WIDEEP_ROUTE_CALLS", "applied module=interface")


_BCAST_LOG = os.environ.get("TLLM_TEST_BCAST_LOG")
_BCAST_HERE = _BCAST_LOG is not None and RANK is not None and RANK in _BCAST_LOG.split(",")
# The byte count of the request broadcast running on this thread (armed by its wrapper).
_BCAST_SIZE = threading.local()


def _patch_bcast_pickle(module):
    """BCAST_LOG: the communicator's pickle records the size of a request broadcast's payload (dumps on the root,
    loads elsewhere), so no rank pickles it a second time."""
    if _refused("WIDEEP_BCAST", module, _missing_anchors([
            ("safe_broadcast.dumps", getattr(module, "safe_broadcast", None), "pickle.dumps(obj"),
            ("safe_broadcast.loads", getattr(module, "safe_broadcast", None), "pickle.loads(dst_buf)")])):
        return
    if _BCAST_HERE:
        import types

        real = module.pickle

        class SizedPickle(types.ModuleType):
            def __getattr__(self, name):
                return getattr(real, name)

            def dumps(self, obj, *args, **kwargs):
                data = real.dumps(obj, *args, **kwargs)
                if getattr(_BCAST_SIZE, "armed", False):
                    _BCAST_SIZE.bytes = len(data)
                return data

            def loads(self, data, *args, **kwargs):
                if getattr(_BCAST_SIZE, "armed", False):
                    _BCAST_SIZE.bytes = len(data)
                return real.loads(data, *args, **kwargs)

        module.pickle = SizedPickle("pickle")
    _say("WIDEEP_BCAST", f"applied module=communicator log={int(_BCAST_HERE)}")


def _patch_bcast_requests(module):
    """BCAST_LOG: time each broadcast of new requests and log its size."""
    cls = getattr(module, "RequestBroadcaster", None)
    if _refused("WIDEEP_BCAST", module, _missing_anchors([
            ("RequestBroadcaster._broadcast_requests", cls, "self.dist.broadcast(payloads, root=0)")])):
        return
    if _BCAST_HERE:
        original = cls.__dict__["_broadcast_requests"]

        @functools.wraps(original)
        def timed(self, new_requests, py_request_objects):
            _BCAST_SIZE.armed, _BCAST_SIZE.bytes = True, None
            start = time.perf_counter()
            try:
                result = original(self, new_requests, py_request_objects)
            finally:
                _BCAST_SIZE.armed = False
            ms = (time.perf_counter() - start) * 1e3
            requests = result[0] if isinstance(result, tuple) and result else None
            _say("WIDEEP_BCAST", f"n={len(requests) if requests is not None else '?'} ms={ms:.3f} bytes={_BCAST_SIZE.bytes}")
            return result

        cls._broadcast_requests = timed
    _say("WIDEEP_BCAST", f"applied module=request_utils log={int(_BCAST_HERE)}")


def _patch_bcast_ctx(module):
    """BCAST_LOG: each step that schedules context: its context requests and tokens, and the schedule's wall time."""
    cls = module.PyExecutor
    if _refused("WIDEEP_BCAST", module, _missing_anchors([
            ("_prepare_and_schedule_batch.return", cls.__dict__.get("_prepare_and_schedule_batch"),
             "return scheduled_batch, iter_stats")])):
        return
    if _BCAST_HERE:
        original = cls.__dict__["_prepare_and_schedule_batch"]
        calls = [0]

        @functools.wraps(original)
        def logged(self):
            start = time.perf_counter()
            result = original(self)
            calls[0] += 1
            batch = result[0] if isinstance(result, tuple) else None
            requests = tokens = 0
            for request in getattr(batch, "context_requests", None) or ():
                if getattr(request, "is_attention_dp_dummy", False) or getattr(request, "is_dummy", False):
                    continue
                requests += 1
                try:
                    tokens += request.context_chunk_size
                except (AttributeError, RuntimeError):
                    pass
            if requests:
                _say("WIDEEP_CTX", f"call={calls[0]} reqs={requests} tokens={tokens} "
                                   f"sched_ms={(time.perf_counter() - start) * 1e3:.3f}")
            return result

        cls._prepare_and_schedule_batch = logged
    _say("WIDEEP_BCAST", f"applied module=py_executor log={int(_BCAST_HERE)}")


def _chain(name, patch):
    """Add ``patch`` for module ``name`` after any patch it already has."""
    previous = PATCHES.get(name)
    PATCHES[name] = patch if previous is None else (lambda module: (previous(module), patch(module)))


PATCHES = {
    "tensorrt_llm._torch.pyexecutor.cuda_graph_runner": _patch_graphs,
    "tensorrt_llm._torch.autotuner": _patch_autotuner,
    "cutlass.cute": _patch_cute,
    "tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided": _patch_a2a,
    "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2": _patch_kv,
}
if EXCLUDE is not None:
    PATCHES["tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"] = _patch_eplb
if os.environ.get("TLLM_TEST_EMPTY_EXTRA_SLOTS") == "1" or os.environ.get("TLLM_TEST_POPULATE"):
    PATCHES["tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"] = _patch_load_balancer
if WATCH_POLL_S is not None:
    PATCHES["tensorrt_llm._torch.distributed.communicator"] = _patch_watch
if RANK is not None and RANK == os.environ.get("TLLM_TEST_KILL_AFTER_STOPPED"):
    PATCHES["tensorrt_llm.executor.recovery.client"] = _patch_client
if (os.environ.get("TLLM_TEST_BORROW_FROM_KV") == "1" or (RERUN_OWNER and RANK == "0")
        or DELAY_ABORT_S > 0):
    PATCHES["tensorrt_llm._torch.pyexecutor.peer_loss_recovery"] = _patch_recovery
if ALLOC_STATS_S is not None:
    PATCHES["tensorrt_llm._torch.pyexecutor.py_executor"] = _patch_alloc_stats
if (KILL_HERE and (KILL_PREFILL or KILL_DECODE or STALL_DECODE or STICKY_DECODE or HOST_STALL_DECODE
                   or GIL_HOG_DECODE or LATE_WRITE_DECODE)) or RERUN_OWNER:
    PATCHES["tensorrt_llm._torch.pyexecutor.model_engine"] = _patch_engine
if SERVE_BACKLOG:
    PATCHES["uvicorn.config"] = _patch_uvicorn
if os.environ.get("TLLM_TEST_LOG_EAGER") == "1":
    _chain("tensorrt_llm._torch.pyexecutor.cuda_graph_runner", _patch_replay_flag)
    _chain("tensorrt_llm._torch.pyexecutor.model_engine", _patch_eager_log)
if RANK == "0" and os.environ.get("TLLM_TEST_PHASE_TIMES"):
    _PHASE_NOTE[0] = _phase_logger(int(os.environ["TLLM_TEST_PHASE_TIMES"]))
    _chain("tensorrt_llm._torch.pyexecutor.py_executor", _patch_phase_executor)
    _chain(_RECOVERY, _patch_phase_recovery)
if RANK is not None and os.environ.get("TLLM_TEST_LOG_ADMIT") == "1":
    _chain("tensorrt_llm._torch.pyexecutor.py_executor", _patch_admit_log)
if RANK is not None and os.environ.get("TLLM_TEST_LAUNCH_SKEW"):
    _chain(_COMMUNICATOR, _patch_skew_collectives)
    _chain("tensorrt_llm._torch.pyexecutor.cuda_graph_runner", _patch_skew_replay)
if (RANK == "0" or (RANK is not None and os.environ.get("TLLM_TEST_COLL_RANKS") == "all")) and os.environ.get(
        "TLLM_TEST_COLL_TIMES"):
    _chain(_COMMUNICATOR, _patch_coll_times)
if RANK is not None and os.environ.get("TLLM_TEST_LOG_DEADLINE") == "1":
    _chain(_COMMUNICATOR, _patch_deadline_comm)
    _chain("tensorrt_llm.executor.recovery.client", _patch_deadline_client)
if KILL_HERE and KILL_EPLB_UPDATE:
    _chain("tensorrt_llm._torch.moe.fused_moe.moe_load_balancer", _patch_kill_in_eplb_update)
if os.environ.get("TLLM_TEST_PIN_HOST_COPIES") == "1":
    _chain("tensorrt_llm._torch.moe.fused_moe.moe_load_balancer", _patch_pin_host_copies)
if FT_OFF and RANK is not None:
    if FT_OFF - {"resp", "step", "coll", "watch"}:
        _say("WIDEEP_FT_OFF_ERROR", f"unknown pieces {sorted(FT_OFF - {'resp', 'step', 'coll', 'watch'})}")
    _chain(_RECOVERY, _patch_ft_off_recovery)
    if "coll" in FT_OFF:
        _chain(_COMMUNICATOR, _patch_ft_off_comm)
if _BCAST_LOG is not None and RANK is not None:
    _chain(_COMMUNICATOR, _patch_bcast_pickle)
    _chain("tensorrt_llm._torch.pyexecutor.request_utils", _patch_bcast_requests)
    _chain("tensorrt_llm._torch.pyexecutor.py_executor", _patch_bcast_ctx)
_MOE = "tensorrt_llm._torch.moe.fused_moe."
if _FT_UNBOUND and RANK is not None:
    _chain(_MOE + "communication.communication_factory", _patch_unbound_factory)
    _chain(_MOE + "communication.execution_state", _patch_unbound_state)
    _chain(_MOE + "moe_scheduler", _patch_unbound_scheduler)
    _chain(_MOE + "communication.nvlink_one_sided", _patch_unbound_dispatch)
if _FT_NO_INPLACE and RANK is not None:
    _chain(_MOE + "fused_moe_cute_dsl", _patch_no_inplace_backend)
    _chain(_MOE + "moe_scheduler", _patch_no_inplace_scheduler)
    _chain(_MOE + "communication.execution_state", _patch_no_inplace_state)
if _FT_NO_GHOST and RANK is not None:
    _chain(_MOE + "moe_load_balancer", _patch_no_ghost_balancer)
    _chain(_MOE + "communication.execution_state", _patch_no_ghost_state)
    _chain(_MOE + "communication.nvlink_one_sided", _patch_no_ghost_dispatch)
if _FT_NO_SLOT_TABLE and RANK is not None:
    _chain(_MOE + "moe_scheduler", _patch_no_slot_table_scheduler)
    _chain(_MOE + "communication.execution_state", _patch_no_slot_table_state)
    _chain(_MOE + "communication.nvlink_one_sided", _patch_no_slot_table_dispatch)
if os.environ.get("TLLM_TEST_LOG_SLOT_TABLE") == "1" and RANK is not None:
    _chain(_MOE + "moe_scheduler", _patch_slot_table_log)
    _chain(_MOE + "interface", _patch_route_calls)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = PATCHES.pop(name, None)
        if patch is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            if PATCHES:
                sys.meta_path.insert(0, self)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = patched_exec
        return spec


if RANK is not None or SERVE_BACKLOG:
    sys.meta_path.insert(0, _Finder())
