"""Diagnostic hooks for the EP32 experiments, loaded via a .pth file in MPI worker processes.

- TLLM_TEST_ORACLE_EXCLUDE_RANK=r: every rank except r excludes r's EPLB slots from routing at
  startup. A healthy all-alive run then routes exactly as after rank r is lost, so its greedy
  tokens are the exact oracle for post-recovery tokens.
- TLLM_TEST_TRACE_CAPTURE=1: print a line for every CUDA graph capture, to show that recovery
  captures nothing.
- TLLM_TEST_KILL_IN_REPLAY=path (with TLLM_TEST_KILL_RANK=r): once ``path`` exists, rank r
  SIGKILLs itself at a decode graph replay, after the step's host collectives and before
  launching its GPU work, so the survivors' MoE collectives of that step wait for it.

- TLLM_TEST_NO_COMMIT_SYNC=1: the executor's commit gate checks the peer-failure flag without
  waiting for the step's device work (performance attribution only; unsafe).
- TLLM_TEST_NO_PEER_WATCH=1: do not start the peer-failure watcher thread (performance
  attribution only; in-flight losses are then detected only by host collectives).
- TLLM_TEST_DUMP_STACKS=dir: every 60 s after the kill trigger exists, every process appends
  all thread stacks to dir/stacks-<rank>-<pid>.txt (to diagnose hangs).

Modules are patched when TRT-LLM imports them, never at interpreter startup.
"""
import importlib.abc
import importlib.util
import os
import sys
import time

EPLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"
GRAPHS = "tensorrt_llm._torch.pyexecutor.cuda_graph_runner"
EXECUTOR = "tensorrt_llm._torch.pyexecutor.py_executor"
COMMUNICATOR = "tensorrt_llm._torch.distributed.communicator"
exclude_rank = os.environ.get("TLLM_TEST_ORACLE_EXCLUDE_RANK")
trace_capture = os.environ.get("TLLM_TEST_TRACE_CAPTURE") == "1"
mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
kill_trigger = os.environ.get("TLLM_TEST_KILL_IN_REPLAY")
kill_here = kill_trigger is not None and os.environ.get("TLLM_TEST_KILL_RANK") == mpi_rank


def _patch_eplb(module):
    original = module.MoeLoadBalancer.finalize_model

    def finalize_model(self):
        original(self)
        if self.ep_rank != int(exclude_rank):
            self.load_balancer_impl.reconfigure_mask_only([int(exclude_rank)])
            print(f"WIDEEP_ORACLE rank {self.ep_rank} excluded rank {exclude_rank}", flush=True)

    module.MoeLoadBalancer.finalize_model = finalize_model


def _patch_graphs(module):
    if trace_capture:
        original_capture = module.CUDAGraphRunner.capture

        def capture(self, key, *args, **kwargs):
            print(f"WIDEEP_CAPTURE rank {mpi_rank} t={time.time():.3f} key={tuple(key)}",
                  flush=True)
            return original_capture(self, key, *args, **kwargs)

        module.CUDAGraphRunner.capture = capture
    if kill_here:
        import signal

        original_replay = module.CUDAGraphRunner.replay

        def replay(self, *args, **kwargs):
            if os.path.exists(kill_trigger):
                print(f"WIDEEP_SELF_KILL rank {mpi_rank} t={time.time():.3f}", flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
            return original_replay(self, *args, **kwargs)

        module.CUDAGraphRunner.replay = replay


def _patch_executor(module):
    def commit_step(self, sample_state):
        if self._peer_failed.is_set():
            raise module.HostCollectiveInterrupted("MoE all-to-all", frozenset())
        self._uncommitted_step = None

    module.PyExecutor._commit_step = commit_step


def _patch_communicator(module):
    module.MPIDist.watch_peer_failures = lambda self, on_failure, stop, poll_interval=0.01: None


PATCHES = {}
if os.environ.get("TLLM_TEST_NO_PEER_WATCH") == "1":
    PATCHES[COMMUNICATOR] = _patch_communicator
if os.environ.get("TLLM_TEST_NO_COMMIT_SYNC") == "1":
    PATCHES[EXECUTOR] = _patch_executor
if exclude_rank is not None:
    PATCHES[EPLB] = _patch_eplb
if trace_capture or kill_here:
    PATCHES[GRAPHS] = _patch_graphs


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


if PATCHES and mpi_rank is not None:
    sys.meta_path.insert(0, _Finder())

dump_dir = os.environ.get("TLLM_TEST_DUMP_STACKS")
if dump_dir and kill_trigger and mpi_rank is not None:
    import faulthandler
    import threading

    def _dump_after_kill():
        while not os.path.exists(kill_trigger):
            time.sleep(1)
        path = os.path.join(dump_dir, f"stacks-{mpi_rank}-{os.getpid()}.txt")
        while True:
            time.sleep(60)
            with open(path, "a") as out:
                out.write(f"==== {time.strftime('%H:%M:%S')}\n")
                faulthandler.dump_traceback(file=out, all_threads=True)

    threading.Thread(target=_dump_after_kill, daemon=True).start()
