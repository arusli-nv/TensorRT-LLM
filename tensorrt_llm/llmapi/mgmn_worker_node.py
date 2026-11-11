#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys

# With WideEP fault tolerance, the node supervisor starts every rank but 0 as
# its own process, outside the launcher's MPI job, with its rank in this
# variable (FAULT_TOLERANCE_RANK_ENV in tensorrt_llm/_utils.py): the rank takes
# its worker arguments from rank 0 through the rendezvous store.
if "TLLM_FAULT_TOLERANCE_RANK" in os.environ:
    from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import \
        worker_node_main

    worker_node_main()
    sys.exit(0)

from mpi4py.futures import MPICommExecutor  # noqa: E402
from mpi4py.MPI import COMM_WORLD  # noqa: E402

# For multi-node MPI, the worker nodes should launch MPICommExecutor to accept tasks sent from rank0
with MPICommExecutor(COMM_WORLD) as executor:
    if executor is not None:
        raise RuntimeError(f"rank{COMM_WORLD.rank} should not have executor")

logging.warning(f"worker rank{COMM_WORLD.rank} quited")
