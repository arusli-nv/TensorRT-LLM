# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MASK-SPILL resource check: STACK of the fence-path a2a kernels by rank-mask flag, from `cuobjdump -res-usage`.

Usage: cuobjdump -res-usage libth_common.so | python3 maskspill_res.py [EXPECT_STACK]
Prints one MASKSPILL_RES line per kernel (dispatch, combine) with the STACK values seen for MASK=0 and MASK=1
(value x count), then MASKSPILL_RES_OK when every MASK=1 instantiation has STACK <= EXPECT_STACK (default 24, the
MASK=0 level), else MASKSPILL_RES_SPILL. Exit status 0 only for MASKSPILL_RES_OK.
"""

import re
import sys
from collections import Counter

# moeA2ADispatchKernel<TOP_K, EPLB, MASK, COMPACT, SLOT>; moeA2ACombineKernel<T, TOP_K, LOW_PRECISION, MASK>.
KERNELS = {
    "dispatch": re.compile(r"moe_comm20moeA2ADispatchKernelILi\d+ELb[01]ELb([01])ELb[01]ELb[01]E"),
    "combine": re.compile(r"moe_comm19moeA2ACombineKernelI.+?Li\d+ELb[01]ELb([01])EE"),
}
expect = int(sys.argv[1]) if len(sys.argv) > 1 else 24
stacks = {(k, m): Counter() for k in KERNELS for m in "01"}
kernel = mask = None
for line in sys.stdin:
    if line.lstrip().startswith("Function "):
        kernel = mask = None
        for name, pattern in KERNELS.items():
            match = pattern.search(line)
            if match:
                kernel, mask = name, match.group(1)
    elif kernel and "STACK:" in line:
        stacks[(kernel, mask)][int(re.search(r"STACK:(\d+)", line).group(1))] += 1
        kernel = None

ok = True
for name in KERNELS:
    parts = []
    for m in "01":
        seen = stacks[(name, m)]
        parts.append(f"mask{m}=" + (",".join(f"{s}x{n}" for s, n in sorted(seen.items())) or "none"))
    print(f"MASKSPILL_RES {name} " + " ".join(parts))
    mask_on = stacks[(name, "1")]
    ok = ok and bool(mask_on) and max(mask_on) <= expect
print("MASKSPILL_RES_OK" if ok else f"MASKSPILL_RES_SPILL (a MASK=1 instantiation above STACK {expect}, or none found)")
sys.exit(0 if ok else 1)
