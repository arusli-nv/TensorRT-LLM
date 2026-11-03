"""Routing sort alone on one GPU: moe_sort on the compact list vs moe_sort_source_slots on the slots.

Same shapes as the EP32 DeepSeek-R1 decode (32 sources, top-8 of 256 experts, 8 local experts per rank).
Each op is captured in a CUDA graph of 50 calls and replayed; prints the median microseconds per call.
Usage: python3 sort_bench.py LABEL
"""

import statistics
import sys

import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

SOURCES, EXPERTS, TOP_K, LOCAL, TILE = 32, 256, 8, 8, 128
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
    has_slots = hasattr(torch.ops.trtllm, "moe_sort_source_slots")
    gen = torch.Generator(device="cuda").manual_seed(0)
    for sent in (8, 64, 256):
        slot = 256
        num_tokens = SOURCES * sent
        slot_experts = torch.rand((SOURCES * slot, EXPERTS), device="cuda", generator=gen).argsort(dim=1)[:, :TOP_K]
        slot_experts = slot_experts.to(torch.int32).contiguous()
        slot_weights = torch.rand((SOURCES * slot, TOP_K), device="cuda", generator=gen)
        counts = torch.full((SOURCES,), sent, dtype=torch.int32, device="cuda")
        compact_experts = slot_experts.view(SOURCES, slot, TOP_K)[:, :sent].reshape(num_tokens, TOP_K).contiguous()
        compact_weights = slot_weights.view(SOURCES, slot, TOP_K)[:, :sent].reshape(num_tokens, TOP_K).contiguous()
        compact = time_us(lambda: torch.ops.trtllm.moe_sort(
            compact_experts, compact_weights, EXPERTS, TOP_K, 0, LOCAL, TILE))
        line = f"SORT label={label} tokens={sent} moe_sort_us={compact:.2f}"
        if has_slots:
            slots = time_us(lambda: torch.ops.trtllm.moe_sort_source_slots(
                slot_experts, slot_weights, counts, sent, EXPERTS, TOP_K, 0, LOCAL, TILE))
            line += f" source_slots_us={slots:.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
