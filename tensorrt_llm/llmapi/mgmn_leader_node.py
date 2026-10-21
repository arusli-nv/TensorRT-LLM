# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
'''
This script is used to start the MPICommSession in the rank0 and wait for the
MPI Proxy process to connect and get the MPI task to run.
'''
from typing import Literal

import click
import zmq

from tensorrt_llm._utils import global_mpi_rank, mpi_world_size
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.utils import (
    get_spawn_proxy_process_ipc_addr_env,
    get_spawn_proxy_process_ipc_hmac_key_env)
from tensorrt_llm.llmapi.mpi_session import (RemoteMpiCommSessionServer,
                                             exit_fault_tolerant_process)
from tensorrt_llm.llmapi.utils import logger_debug


def launch_server_main(sub_comm=None):
    num_ranks = sub_comm.Get_size() if sub_comm is not None else mpi_world_size(
    )
    logger_debug(f"Starting MPI Comm Server with {num_ranks} workers\n",
                 "yellow")
    server = RemoteMpiCommSessionServer(
        comm=sub_comm,
        n_workers=num_ranks,
        addr=get_spawn_proxy_process_ipc_addr_env(),
        hmac_key=get_spawn_proxy_process_ipc_hmac_key_env(),
        is_comm=True)
    logger_debug(
        f"MPI Comm Server started at {get_spawn_proxy_process_ipc_addr_env()}")
    server.serve()

    logger_debug("RemoteMpiCommSessionServer stopped\n", "yellow")


def stop_server_main():
    hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
    queue = ZeroMqQueue((get_spawn_proxy_process_ipc_addr_env(), hmac_key),
                        use_hmac_encryption=True,
                        is_server=False,
                        socket_type=zmq.PAIR)

    try:
        logger_debug(
            f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] send shutdown signal to server\n",
            "green")
        queue.put(None)  # ask RemoteMpiCommSessionServer to shutdown
    except zmq.error.ZMQError as e:
        logger_debug(f"Error during RemoteMpiCommSessionClient shutdown: {e}\n",
                     "red")


@click.command()
@click.option("--action", type=click.Choice(["start", "stop"]), default="start")
def main(action: Literal["start", "stop"] = "start"):
    '''
    Arguments:
        action: The action to perform.
            start: Start the MPI Comm Server.
            stop: Stop the MPI Comm Server.
    '''
    if action == "start":
        launch_server_main()
        exit_fault_tolerant_process()
    elif action == "stop":
        stop_server_main()
    else:
        raise ValueError(f"Invalid action: {action}")


if __name__ == '__main__':
    main()
