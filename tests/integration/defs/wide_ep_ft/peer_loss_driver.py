# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM API driver for the WideEP peer-loss recovery test.

Runs on MPI rank 0 under ``trtllm-llmapi-launch``: serves DeepSeek-V3-Lite on EP4 with
attention DP, SIGKILLs the worker process of one rank, keeps serving and writes what every
request returned to ``--output``.

- ``idle``: kill between two rounds of requests.
- ``inflight``: kill while one request per attention-DP rank is decoding.
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path

KILL_MARKER = "PEER_LOSS_DRIVER_KILL"
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


def placement() -> list[int]:
    """Two replicas of every expert: rank r holds its own experts and those of rank r + 1."""
    per_rank = NUM_EXPERTS // WORLD_SIZE
    slots = []
    for rank in range(WORLD_SIZE):
        for owner in (rank, (rank + 1) % WORLD_SIZE):
            slots += range(owner * per_rank, (owner + 1) * per_rank)
    return slots


def llm_kwargs() -> dict:
    slots = placement()
    return dict(
        tensor_parallel_size=WORLD_SIZE,
        moe_expert_parallel_size=WORLD_SIZE,
        moe_tensor_parallel_size=1,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=False,
        disable_overlap_scheduler=True,
        enable_autotuner=False,
        max_batch_size=8,
        max_num_tokens=256,
        max_seq_len=512,
        cuda_graph_config=dict(batch_sizes=[1, 2, 4, 8], enable_padding=True),
        kv_cache_config=dict(enable_block_reuse=False, free_gpu_memory_fraction=0.2),
        moe_config=dict(
            backend="CUTLASS",
            disable_finalize_fusion=True,
            load_balancer=dict(
                num_slots=len(slots),
                layer_updates_per_iter=0,
                initial_global_assignments={
                    layer: slots for layer in range(FIRST_MOE_LAYER, NUM_LAYERS)
                },
            ),
        ),
    )


def find_worker(rank: int) -> int:
    """Return the pid of the local ``mgmn_worker_node`` process of MPI ``rank``."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
            environ = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"mgmn_worker_node" in cmdline and f"OMPI_COMM_WORLD_RANK={rank}".encode() in environ:
            return int(entry.name)
    raise LookupError(f"No local worker process for rank {rank}")


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


def kill(rank: int) -> None:
    pid = find_worker(rank)
    print(f"{KILL_MARKER} rank={rank} pid={pid}", flush=True)
    os.kill(pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("idle", "inflight"), required=True)
    parser.add_argument("--victim", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tensorrt_llm import LLM

    all_ranks = list(range(WORLD_SIZE))
    survivors = [rank for rank in all_ranks if rank != args.victim]
    result = {}
    llm = LLM(model=args.model, **llm_kwargs())
    try:
        result["healthy"] = collect(submit(llm, all_ranks, SHORT_MAX_TOKENS))
        if args.mode == "idle":
            time.sleep(2)
            kill(args.victim)
        else:
            probe_rank = survivors[0]
            pending = submit(llm, all_ranks, LONG_MAX_TOKENS, stream_rank=probe_rank)
            probe = dict(pending)[probe_rank]
            while len(next(probe).outputs[0].token_ids) < TOKENS_BEFORE_KILL:
                pass
            kill(args.victim)
            result["during_loss"] = collect(pending)
        result["after_loss"] = collect(submit(llm, survivors, SHORT_MAX_TOKENS))
    finally:
        # Written before shutdown: ending a world that lost a rank calls MPI_Abort.
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        llm.shutdown()


if __name__ == "__main__":
    main()
