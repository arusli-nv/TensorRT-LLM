# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only fault hook for the WideEP peer-loss recovery tests, loaded through ``PYTHONPATH``.

``WIDEEP_FT_TEST_KILL_AFTER_STOPPED=r``: the worker of MPI rank r SIGKILLs itself right after it
sends its first ``Stopped`` to the recovery authority, so it fails while the survivors agree on a
recovery. The patch is applied when TRT-LLM imports the recovery client; other processes are
untouched. A ``sitecustomize`` this module shadows still runs.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import signal
import sys

_CLIENT = "tensorrt_llm.executor.recovery.client"
SELF_KILL_MARKER = "WIDEEP_FT_TEST_SELF_KILL"


def _run_shadowed_sitecustomize() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [entry for entry in sys.path if os.path.abspath(entry or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec(__name__, rest)
    if spec is not None and spec.loader is not None:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _patch_client(module) -> None:
    send = module.RecoveryClient.send

    def killing_send(self, message):
        send(self, message)
        if type(message).__name__ == "Stopped":
            print(f"{SELF_KILL_MARKER} rank={os.environ['OMPI_COMM_WORLD_RANK']}", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)

    module.RecoveryClient.send = killing_send


class _ClientFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _CLIENT:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _patch_client(module)

        spec.loader.exec_module = patched_exec
        return spec


_run_shadowed_sitecustomize()
_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
if _rank is not None and _rank == os.environ.get("WIDEEP_FT_TEST_KILL_AFTER_STOPPED"):
    sys.meta_path.insert(0, _ClientFinder())
