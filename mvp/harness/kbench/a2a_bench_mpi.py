"""NVLinkOneSided dispatch -> routing sort -> combine at full EP scale under mpirun, FT off vs on, driven like the model.

Every rank runs T tokens (T = requests per GPU in decode) routed to a uniform random top-8 of 256 experts, the
DeepSeek-R1 payloads (NVFP4 hidden states and scales, expert ids, router weights) and an FP8 combine of the
7168-wide expert output. Between them the CuteDSL MoE's routing sort (moe_sort) reads the received expert ids and
weights, as in the model, so a layout that makes those reads slower or faster is counted where it is paid. One CUDA
graph per token count is replayed; rank 0 prints µs per MoE layer, and the
rank wrapped by nsys (rank.sh, TLLM_TEST_NSYS_RANK) gives per-kernel timings under NVTX ranges "tokens=T".

FT mode is set by the environment the launcher forwards (TLLM_FAULT_TOLERANCE_*).
Usage (one process per GPU): python3 a2a_bench_mpi.py MODE   MODE = off | on
"""

import os
import sys

import torch
from mpi4py import MPI

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.mapping import Mapping

# Model shape; the defaults are DeepSeek-R1. KBENCH_DISPATCH=bf16 sends unquantized hidden states.
TOP_K = int(os.environ.get("KBENCH_TOPK", 8))
EXPERTS = int(os.environ.get("KBENCH_EXPERTS", 256))
HIDDEN = int(os.environ.get("KBENCH_HIDDEN", 7168))
NVFP4 = os.environ.get("KBENCH_DISPATCH", "nvfp4") == "nvfp4"
TOKENS = [int(t) for t in os.environ.get("KBENCH_TOKENS", "8,16,32,64,128,192,256").split(",")]
WARMUP, ITERS = 20, 200


def main(mode: str) -> None:
    comm = MPI.COMM_WORLD
    rank, ep = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())))
    mapping = Mapping(ep, rank, ep, tp_size=ep, moe_ep_size=ep, enable_attention_dp=True)
    health = None
    if mode == "on":
        from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth

        health = EPGroupHealth(ep)
    a2a = NVLinkOneSided(mapping, EXPERTS, TOP_K, max(TOKENS), hidden_size=HIDDEN, dtype=torch.bfloat16,
                         num_experts=EXPERTS, use_low_precision_combine=True, ep_group_health=health)
    if mode == "on":
        from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState

        a2a.bind_execution_state(MoeExecutionState(health, torch.device("cuda")))
    comm.Barrier()

    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    graphs, inputs = {}, {}
    for tokens in TOKENS:
        slots = torch.rand(tokens, EXPERTS, device="cuda", generator=gen).argsort(dim=1)[:, :TOP_K].to(torch.int32)
        if NVFP4:
            hidden = torch.randint(0, 256, (tokens, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=gen)
            sf = torch.randint(0, 256, (tokens, HIDDEN // 16), dtype=torch.uint8, device="cuda", generator=gen)
        else:
            hidden, sf = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen), None
        inputs[tokens] = (
            hidden,
            sf,
            slots,
            torch.rand(tokens, TOP_K, device="cuda", generator=gen),
            torch.randn(ep * tokens, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen),
        )

        def step(tokens=tokens):
            hidden, sf, slots, scales, expert_out = inputs[tokens]
            _, _, ids, weights = a2a.dispatch(hidden, sf, slots, scales, [tokens] * ep)
            sort_args = dict(num_experts=EXPERTS, top_k=TOP_K, local_expert_offset=rank * (EXPERTS // ep),
                             local_num_experts=EXPERTS // ep, tile_tokens_dim=128)
            # main's NVLinkOneSided has no receive slots.
            source_slots = a2a.dispatch_source_slots() if hasattr(a2a, "dispatch_source_slots") else None
            # FT on, wheels with the standalone compaction op: the routing of the receive slots, compacted for
            # the sort. KBENCH_FT_COMPACT=0 skips it (measurement only: the sort then reads unsanitized rows) to
            # price it. Wheels whose dispatch writes the compact lists itself return them as ids / weights.
            separate_compaction = hasattr(torch.ops.trtllm, "moe_a2a_compact_source_slots")
            if source_slots is not None and separate_compaction and os.environ.get("KBENCH_FT_COMPACT", "1") == "1":
                ids, weights = torch.ops.trtllm.moe_a2a_compact_source_slots(
                    ids, weights, source_slots.valid_counts, source_slots.source_tokens)
            torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=weights, **sort_args)
            return a2a.combine(expert_out)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        graphs[tokens] = graph
        comm.Barrier()

    for tokens in TOKENS:
        for _ in range(WARMUP):
            graphs[tokens].replay()
        torch.cuda.synchronize()
        comm.Barrier()
        torch.cuda.nvtx.range_push(f"tokens={tokens}")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            graphs[tokens].replay()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        us = comm.gather(start.elapsed_time(end) * 1e3 / ITERS, root=0)
        if rank == 0:
            print(f"BENCH mode={mode} ep={ep} hidden={HIDDEN} topk={TOP_K} experts={EXPERTS} nvfp4={int(NVFP4)} tokens={tokens} us_per_layer={max(us):.2f} "
                  f"median_rank={sorted(us)[len(us) // 2]:.2f}", flush=True)
        comm.Barrier()


if __name__ == "__main__":
    main(sys.argv[1])
