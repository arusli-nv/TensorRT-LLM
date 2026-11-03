# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost of FT's compact copy for MoE backends that cannot read NVLinkOneSided's receive slots (e.g. DEEPGEMM).

DispatchSourceSlots.compact copies each source's valid rows out of its fixed receive slot, once per MoE layer.
Timed with CUDA events at GLM-5.3 shapes (32 sources, hidden 6144), per tokens per source, for a BF16 payload and
for an FP8 payload with its 128-element block scales (float32). Usage: python3 compact_bench.py [SLOT_TOKENS]
"""

import sys

import torch

from tensorrt_llm._torch.moe.fused_moe.impl_contract import DispatchSourceSlots

SOURCES, HIDDEN = 32, 6144
slot_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def time_us(fn, reps: int = 200) -> float:
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


for tokens in (1, 4, 8, 16, 32, 64, 128, 256):
    if tokens > slot_tokens:
        break
    slots = DispatchSourceSlots(torch.full((SOURCES,), tokens, dtype=torch.int32, device="cuda"),
                                source_tokens=tokens, slot_tokens=slot_tokens)
    bf16 = torch.randn(SOURCES * slot_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    fp8 = bf16.to(torch.float8_e4m3fn)
    scales = torch.rand(SOURCES * slot_tokens, HIDDEN // 128, device="cuda", dtype=torch.float32)
    bf16_us = time_us(lambda: slots.compact(bf16))
    fp8_us = time_us(lambda: (slots.compact(fp8), slots.compact(scales)))
    print(f"RESULT tokens_per_source={tokens} rows={SOURCES * tokens} bf16_us={bf16_us:.1f} fp8_scales_us={fp8_us:.1f}",
          flush=True)
print("COMPACT_BENCH_DONE")
