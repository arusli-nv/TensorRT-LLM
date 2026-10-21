# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a process of a fault-tolerant MPI world ends: a bounded MPI_Finalize, or leaving a world that lost a rank."""

import ctypes
import faulthandler
import logging
import os
import sys
import threading
import time
from concurrent.futures import BrokenExecutor
from typing import NoReturn

from tensorrt_llm._utils import (
    global_mpi_rank,
    is_mpi_peer_failure,
    mpi_failed_ranks,
    peer_loss_recovery_enabled,
)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.logger import logger

if ENABLE_MULTI_DEVICE:
    import mpi4py


def num_failed_ranks() -> int:
    """Number of processes ULFM reports failed in the world.

    Zero when recovery is disabled or the MPI library cannot report failures.
    """
    if not (ENABLE_MULTI_DEVICE and peer_loss_recovery_enabled()):
        return 0
    try:
        return len(mpi_failed_ranks(mpi4py.MPI.COMM_WORLD))
    except NotImplementedError:
        # A startup that refused such a library is torn down here; its error must surface.
        return 0


# How long a BrokenExecutor waits for ULFM to report the failed rank that
# broke the executor.
_FAILED_RANK_EVIDENCE_S = 1.0


def is_departed_peer_failure(exc: BaseException) -> bool:
    """Whether *exc* only reports a departed MPI peer while recovery is on.

    Such failures are left to the surviving ranks' recovery path instead of
    tearing down the session: a ULFM peer-failure error, or a
    ``BrokenExecutor`` from the mpi4py futures manager once ULFM reports a
    failed rank.
    """
    if not peer_loss_recovery_enabled():
        return False
    if isinstance(exc, BrokenExecutor):
        deadline = time.monotonic() + _FAILED_RANK_EVIDENCE_S
        while not num_failed_ranks():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True
    return is_mpi_peer_failure(exc)


# How long a process of a fault-tolerant MPI world waits in MPI_Finalize before
# it exits without it.
_FINALIZE_GRACE_S = 30.0


def _mpi_finalize() -> int:
    """MPI_Finalize through ctypes; returns the MPI error code.

    ctypes releases the GIL while it runs (mpi4py's call holds it), so that the timer thread of
    ``_finalize_and_exit`` can run.
    """
    try:
        finalize = ctypes.CDLL(mpi4py.MPI.__file__).MPI_Finalize
    except (
        AttributeError
    ):  # an mpi4py that loads MPI at run time: the GIL-free backstop still bounds it
        mpi4py.MPI.Finalize()
        return 0
    return finalize()


def _finalize_and_exit(status: int, where: str) -> NoReturn:
    """Finalize MPI (*where* names the exit path in the logs), then end this process with *status*.

    ULFM's MPI_Finalize can block for good, and it holds every process that
    reached it; after ``_FINALIZE_GRACE_S`` this process exits with *status*
    without it. A native timer, which needs no GIL, exits with status 1 if
    that cannot run. Exit handlers do not run, so the peer-failure watch they
    would stop is stopped first.
    """
    recovery = sys.modules.get("tensorrt_llm._torch.pyexecutor.peer_loss_recovery")
    if recovery is not None:
        recovery.stop_startup_watch()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    # Read before MPI_Finalize starts: no MPI call may run while it does.
    rank = global_mpi_rank()

    def give_up() -> None:
        print(
            f"Rank {rank}: MPI_Finalize ({where}) did not end within "
            f"{_FINALIZE_GRACE_S:.0f} s; exiting with status {status} without it",
            file=sys.stderr,
            flush=True,
        )
        os._exit(status)

    watchdog = threading.Timer(_FINALIZE_GRACE_S, give_up)
    watchdog.daemon = True
    watchdog.start()
    # A backstop that needs no GIL, should the Timer still not run.
    faulthandler.dump_traceback_later(2 * _FINALIZE_GRACE_S, exit=True)
    error = _mpi_finalize()
    if error:
        print(f"MPI_Finalize failed: MPI error {error}", file=sys.stderr, flush=True)
        status = status or 1
    faulthandler.cancel_dump_traceback_later()
    watchdog.cancel()
    watchdog.join()
    os._exit(status)


def exit_fault_tolerant_process(status: int = 0) -> None:
    """End this process through a bounded MPI_Finalize under peer-loss recovery.

    With recovery enabled, exits with *status* (see ``_finalize_and_exit``);
    otherwise returns.
    """
    if ENABLE_MULTI_DEVICE and peer_loss_recovery_enabled():
        _finalize_and_exit(status, "process end")


def leave_degraded_world(clean: bool) -> NoReturn:
    """End this process of an MPI world that lost a rank, once its task ended.

    mpi4py.futures cannot release such a world: task results and its stop
    message travel on communicators that include the lost rank, so each
    process leaves on its own. After a clean task end it finalizes MPI with
    the other survivors (ULFM's MPI_Finalize first agrees on the lost ranks)
    and exits with status 0 (see ``_finalize_and_exit``). After an error it
    ends the world with MPI_Abort.
    """
    if clean:
        logger.warning(f"Rank {global_mpi_rank()}: finalizing the MPI world, which lost a rank")
        _finalize_and_exit(0, "degraded world")
    logger.error(
        f"Rank {global_mpi_rank()}: ending the MPI world, which lost a rank, after its task failed"
    )
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    mpi4py.MPI.COMM_WORLD.Abort(1)
