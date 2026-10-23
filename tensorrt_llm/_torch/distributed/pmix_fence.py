# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End one process of an MPI job through its runtime (PMIx job control), for fencing a hung rank.

The runtime (PRRTE under ``mpirun``) kills the target's process group; the other processes keep running and
ULFM survivors see the process fail like any other. Only ``PMIX_JOB_CTRL_KILL`` aimed at one process is used:
``PMIX_JOB_CTRL_SIGNAL`` signals the whole job. The calls go through the libpmix that libmpi already loaded
and initialized in this process.
"""

import ctypes
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI

__all__ = ["PmixUnavailable", "kill_rank", "pmix_ranks"]

_PMIX_MAX_NSLEN = 255
_PMIX_BOOL = 1
_PMIX_JOB_CTRL_KILL = b"pmix.jctrl.kill"


class PmixUnavailable(RuntimeError):
    """This process was not launched by a PMIx runtime, or its libpmix lacks job control."""


class _Proc(ctypes.Structure):
    _fields_ = [("nspace", ctypes.c_char * (_PMIX_MAX_NSLEN + 1)), ("rank", ctypes.c_uint32)]


def _loaded_libpmix() -> ctypes.CDLL:
    path = next(
        (
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libpmix.so" in line
        ),
        None,
    )
    if path is None:
        raise PmixUnavailable("libpmix is not loaded (MPI not initialized by a PMIx runtime)")
    lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
    lib.PMIx_Info_create.restype = ctypes.c_void_p
    lib.PMIx_Info_create.argtypes = [ctypes.c_size_t]
    lib.PMIx_Info_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_uint16,
    ]
    lib.PMIx_Info_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.PMIx_Job_control.argtypes = [
        ctypes.POINTER(_Proc),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.PMIx_Error_string.restype = ctypes.c_char_p
    return lib


def pmix_ranks(comm: "MPI.Comm") -> tuple[int, ...]:
    """The rank in this job's PMIx namespace (its MPI_COMM_WORLD rank) of each rank of ``comm``."""
    from mpi4py import MPI

    group, world = comm.Get_group(), MPI.COMM_WORLD.Get_group()
    try:
        return tuple(MPI.Group.Translate_ranks(group, list(range(group.Get_size())), world))
    finally:
        group.Free()
        world.Free()


def kill_rank(rank: int) -> None:
    """Have the runtime kill the process of ``rank`` (its rank in this job's PMIx namespace, i.e. MPI_COMM_WORLD).

    Blocks until the runtime answers, which an unresponsive runtime never does.

    Raises:
        PmixUnavailable: No PMIx runtime launched this process.
        RuntimeError: The runtime refused the request.
    """
    nspace = os.environ.get("PMIX_NAMESPACE")
    if not nspace:
        raise PmixUnavailable("PMIX_NAMESPACE is not set")
    try:
        lib = _loaded_libpmix()
    except (OSError, AttributeError) as error:
        raise PmixUnavailable(f"libpmix job control is not usable: {error}") from error
    directive = lib.PMIx_Info_create(1)
    try:
        flag = ctypes.c_bool(True)
        status = lib.PMIx_Info_load(directive, _PMIX_JOB_CTRL_KILL, ctypes.byref(flag), _PMIX_BOOL)
        if status == 0:
            target = _Proc(nspace.encode(), rank)
            results, count = ctypes.c_void_p(), ctypes.c_size_t()
            status = lib.PMIx_Job_control(
                ctypes.byref(target), 1, directive, 1, ctypes.byref(results), ctypes.byref(count)
            )
            if results.value:
                lib.PMIx_Info_free(results, count.value)
    finally:
        lib.PMIx_Info_free(directive, 1)
    if status != 0:
        raise RuntimeError(
            f"PMIx refused to kill rank {rank}: {lib.PMIx_Error_string(status).decode()} ({status})"
        )
