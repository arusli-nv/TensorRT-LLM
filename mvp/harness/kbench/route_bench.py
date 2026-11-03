"""EPLB routing (expert -> slot) alone on one GPU, as rank 0 of the DEP32 DeepSeek-R1 placement.

256 experts, top-8, 32 ranks; slots per rank 8 (no replicas: moeComputeRouteNoRedundantKernel) or 9 (8 home
experts + 1 replica of rank (r+16)%32's first expert: moeComputeRouteKernel). Each call is captured in a CUDA
graph of 50 calls and replayed; prints the median microseconds per call. The placement lives where the load
balancer puts it: on GB200 host-registered NUMA memory, or plain device memory with TRTLLM_EPLB_FORCE_GDRCOPY=1.
Usage: python3 route_bench.py LABEL
"""

import statistics
import sys

import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer

EXPERTS, TOP_K, EP = 256, 8, 32
CALLS, REPLAYS = 50, 40


def time_us(fn) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(CALLS):
                fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / CALLS)
    return statistics.median(samples)


def main(label: str) -> None:
    per_rank = EXPERTS // EP
    balancer = MoeLoadBalancer(0, EP, 0)
    layers = {}
    for slots in (per_rank, per_rank + 1):
        layer = balancer.add_layer(EXPERTS, TOP_K, slots)
        assignment = []
        for rank in range(EP):
            assignment += list(range(rank * per_rank, (rank + 1) * per_rank))
            if slots > per_rank:
                assignment.append(((rank + EP // 2) % EP) * per_rank)
        layer.set_initial_weight_assignments(assignment)
        layers[slots] = layer
    balancer.finalize_model()
    gen = torch.Generator(device="cuda").manual_seed(0)
    for tokens in (1, 8, 64, 256):
        ids = torch.rand((tokens, EXPERTS), device="cuda", generator=gen).argsort(dim=1)[:, :TOP_K]
        ids = ids.to(torch.int32).contiguous()
        line = f"ROUTE label={label} tokens={tokens}"
        for slots, layer in layers.items():
            us = time_us(lambda: torch.ops.trtllm.moe_load_balance_routing(
                ids, False, layer.single_layer_load_balancer_ptr))
            line += f" slots{slots * EP}_us={us:.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
