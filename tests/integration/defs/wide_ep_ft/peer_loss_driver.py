# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM API driver for the WideEP peer-loss recovery test.

Runs on MPI rank 0 under ``trtllm-llmapi-launch``: serves DeepSeek-V3-Lite on EP4 with
attention DP, SIGKILLs (or SIGSTOPs) the worker processes of the victim ranks, keeps serving
and writes what every request returned to ``--output``.

- ``idle``: fault between two rounds of requests.
- ``inflight``: fault while one request per attention-DP rank is decoding.

Only worker processes that carry this run's ``RUN_TOKEN_ENV`` value are faulted.
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path

KILL_MARKER = "PEER_LOSS_DRIVER_KILL"
# Environment variable whose value marks the processes of one test run.
RUN_TOKEN_ENV = "WIDEEP_FT_TEST_RUN_TOKEN"
WORLD_SIZE = 4
NUM_EXPERTS = 72
FIRST_MOE_LAYER = 1
NUM_LAYERS = 30
SHORT_MAX_TOKENS = 16
# Long enough that every in-flight request is still decoding when the victim dies.
LONG_MAX_TOKENS = 384
TOKENS_BEFORE_KILL = 16
RESULT_TIMEOUT_S = 300

# Attention-DP rank -> (prompt, substring of a correct greedy answer).
PROMPTS = {
    0: ("The capital of France is", "Paris"),
    1: ("The capital of Italy is", "Rome"),
    2: ("The capital of Spain is", "Madrid"),
    3: ("The capital of Japan is", "Tokyo"),
}


def placement(replica_offset: int = 1) -> list[int]:
    """Two replicas of every expert: rank r holds its own experts and those of rank r + offset.

    Offset 1 survives the loss of any one rank; offset 2 also that of ranks 2 and 3 together.
    """
    per_rank = NUM_EXPERTS // WORLD_SIZE
    slots = []
    for rank in range(WORLD_SIZE):
        for owner in (rank, (rank + replica_offset) % WORLD_SIZE):
            slots += range(owner * per_rank, (owner + 1) * per_rank)
    return slots


def llm_kwargs(overlap: bool = False, replica_offset: int = 1, online_slots: int = 0) -> dict:
    """With ``online_slots``, online EPLB over that many slots from its default placement."""
    slots = placement(replica_offset)
    if online_slots:
        load_balancer = dict(num_slots=online_slots, layer_updates_per_iter=1)
    else:
        load_balancer = dict(
            num_slots=len(slots),
            layer_updates_per_iter=0,
            initial_global_assignments={
                layer: slots for layer in range(FIRST_MOE_LAYER, NUM_LAYERS)
            },
        )
    return dict(
        tensor_parallel_size=WORLD_SIZE,
        moe_expert_parallel_size=WORLD_SIZE,
        moe_tensor_parallel_size=1,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=False,
        disable_overlap_scheduler=not overlap,
        enable_autotuner=False,
        max_batch_size=8,
        max_num_tokens=256,
        max_seq_len=512,
        cuda_graph_config=dict(batch_sizes=[1, 2, 4, 8], enable_padding=True),
        kv_cache_config=dict(enable_block_reuse=False, free_gpu_memory_fraction=0.2),
        moe_config=dict(
            # CuteDSL NVFP4 reads the dispatch receive slots in place (other backends get a copy).
            backend="CUTEDSL",
            disable_finalize_fusion=True,
            load_balancer=load_balancer,
        ),
    )


def find_worker(rank: int) -> int:
    """Return the pid of this run's local ``mgmn_worker_node`` process of MPI ``rank``."""
    run_token = f"{RUN_TOKEN_ENV}={os.environ[RUN_TOKEN_ENV]}".encode()
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
            environ = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            b"mgmn_worker_node" in cmdline
            and run_token in environ
            and f"OMPI_COMM_WORLD_RANK={rank}".encode() in environ
        ):
            pids.append(int(entry.name))
    if len(pids) != 1:
        raise LookupError(f"Expected one local worker process of rank {rank}, found {pids}")
    return pids[0]


def submit(llm, ranks, max_tokens: int, stream_rank: int | None = None) -> list:
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.scheduling_params import SchedulingParams

    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    return [
        (
            rank,
            llm.generate_async(
                PROMPTS[rank][0],
                params,
                streaming=rank == stream_rank,
                scheduling_params=SchedulingParams(
                    attention_dp_rank=rank, attention_dp_relax=False
                ),
            ),
        )
        for rank in ranks
    ]


def collect(pending: list) -> list[dict]:
    from tensorrt_llm.llmapi import RequestError

    rows = []
    for rank, request in pending:
        prompt, expected = PROMPTS[rank]
        row = dict(rank=rank, prompt=prompt, expected=expected)
        try:
            row["text"] = request.result(timeout=RESULT_TIMEOUT_S).outputs[0].text
        except RequestError as error:
            row["error"] = str(error)
        rows.append(row)
    return rows


def fault(ranks: list[int], signum: signal.Signals) -> None:
    pids = [find_worker(rank) for rank in ranks]
    for rank, pid in zip(ranks, pids):
        print(f"{KILL_MARKER} rank={rank} pid={pid} signal={signum.name}", flush=True)
        os.kill(pid, signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("idle", "inflight"), required=True)
    parser.add_argument("--victims", type=int, nargs="+", required=True)
    parser.add_argument(
        "--also-lost",
        type=int,
        nargs="*",
        default=[],
        help="ranks that end themselves during the recovery (not served afterwards)",
    )
    parser.add_argument(
        "--fault", choices=("kill", "stop"), default="kill", help="SIGKILL or SIGSTOP the victims"
    )
    parser.add_argument("--replica-offset", type=int, default=1, help="see placement()")
    parser.add_argument("--online-slots", type=int, default=0, help="see llm_kwargs()")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap", action="store_true", help="run the overlap scheduler")
    args = parser.parse_args()

    from tensorrt_llm import LLM

    all_ranks = list(range(WORLD_SIZE))
    lost = set(args.victims) | set(args.also_lost)
    survivors = [rank for rank in all_ranks if rank not in lost]
    signum = signal.SIGKILL if args.fault == "kill" else signal.SIGSTOP
    result = {}
    llm = LLM(model=args.model, **llm_kwargs(args.overlap, args.replica_offset, args.online_slots))
    try:
        result["healthy"] = collect(submit(llm, all_ranks, SHORT_MAX_TOKENS))
        if args.mode == "idle":
            time.sleep(2)
            fault(args.victims, signum)
        else:
            probe_rank = survivors[0]
            pending = submit(llm, all_ranks, LONG_MAX_TOKENS, stream_rank=probe_rank)
            probe = dict(pending)[probe_rank]
            while len(next(probe).outputs[0].token_ids) < TOKENS_BEFORE_KILL:
                pass
            fault(args.victims, signum)
            result["during_loss"] = collect(pending)
        result["after_loss"] = collect(submit(llm, survivors, SHORT_MAX_TOKENS))
    finally:
        # Written before shutdown: ending a world that lost a rank calls MPI_Abort.
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        llm.shutdown()


if __name__ == "__main__":
    main()
