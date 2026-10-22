#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os

# trtllm-llmapi-launch passes these when it execs this process (see run_worker_rank).
parser = argparse.ArgumentParser()
parser.add_argument("--flashinfer-temporary-workspace")
parser.add_argument("--flashinfer-lock-fd", type=int)
args = parser.parse_args()
# The launcher's FlashInfer slot lock: children must not inherit it, so set it before any import can fork.
if args.flashinfer_lock_fd is not None:
    os.set_inheritable(args.flashinfer_lock_fd, False)

import logging  # noqa: E402

from mpi4py.futures import MPICommExecutor  # noqa: E402
from mpi4py.MPI import COMM_WORLD  # noqa: E402

from tensorrt_llm.llmapi.mpi_session import run_worker_rank  # noqa: E402


def serve() -> None:
    # For multi-node MPI, the worker nodes should launch MPICommExecutor to accept tasks sent from rank0
    with MPICommExecutor(COMM_WORLD) as executor:
        if executor is not None:
            raise RuntimeError(
                f"rank{COMM_WORLD.rank} should not have executor")

    logging.warning(f"worker rank{COMM_WORLD.rank} quited")


run_worker_rank(serve, args.flashinfer_temporary_workspace,
                args.flashinfer_lock_fd)
