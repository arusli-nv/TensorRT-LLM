# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A2A FT ablation resource gate: `cuobjdump -res-usage` of the a2a-ablate libth_common.so, per (kernel, FT_ABL).

Usage: cuobjdump -res-usage libth_common.so | python3 ablate_res.py [REF_RES_USAGE_FILE]
Kernels (mangled): moeA2ADispatchKernel<TOP_K, EPLB, MASK, COMPACT, SLOT, FT_ABL>,
moeA2ACombineKernel<T, TOP_K, LOW_PRECISION, MASK, FT_ABL>. Prints one ABLATE_RES line per (kernel, FT_ABL, MASK)
with the instantiation count and the STACK values (value x count), then ABLATE_RES_OK when
  - every variant (S S1 F F8 L A SO ALL) is instantiated for both kernels, only with MASK=1 and TOP_K=8, as many times
    as the FT_ABL=0, MASK=1, TOP_K=8 control (same EPLB/compact/slot or dtype/low-precision set), and
  - every ALL instantiation has STACK 24 (no rank-mask array left, so no spill);
else ABLATE_RES_FAIL. Exit status 0 only for ABLATE_RES_OK.
REF_RES_USAGE_FILE (optional, informational): the v5-final freeze lib's -res-usage; every FT_ABL=0 instantiation's
resource line is compared with the reference kernel of the same template arguments (ABLATE_RES_CTRL_SAME / _DIFF).
"""

import re
import sys
from collections import Counter, defaultdict

VARIANTS = {0: "0", 1: "S", 2: "S1", 4: "F", 64: "F8", 8: "L", 16: "A", 32: "SO", 61: "ALL"}
DISPATCH = re.compile(r"moe_comm20moeA2ADispatchKernelILi(\d+)ELb([01])ELb([01])ELb([01])ELb([01])ELi(\d+)EE")
COMBINE = re.compile(r"moe_comm19moeA2ACombineKernelI(.+?)Li(\d+)ELb([01])ELb([01])ELi(\d+)EE")
# The same kernels without FT_ABL (v5-final), for the reference comparison.
REF_DISPATCH = re.compile(r"moe_comm20moeA2ADispatchKernelILi(\d+)ELb([01])ELb([01])ELb([01])ELb([01])EE")
REF_COMBINE = re.compile(r"moe_comm19moeA2ACombineKernelI(.+?)Li(\d+)ELb([01])ELb([01])EE")
RESOURCES = re.compile(r"(REG:\d+.*)$")


def parse(lines, dispatch, combine, with_abl):
    """{(arch, kernel, key-without-FT_ABL, FT_ABL): resource line}."""
    found = {}
    arch = key = None
    for line in lines:
        if line.strip().startswith("arch = "):
            arch = line.split("=", 1)[1].strip()
        elif line.lstrip().startswith("Function "):
            key = None
            match = dispatch.search(line)
            if match:
                g = match.groups()
                # (TOP_K, MASK, rest), FT_ABL
                key = ("dispatch", (int(g[0]), g[2], g[1] + g[3] + g[4]), int(g[5]) if with_abl else 0)
            match = None if key else combine.search(line)
            if match:
                g = match.groups()
                key = ("combine", (int(g[1]), g[3], g[0] + g[2]), int(g[4]) if with_abl else 0)
        elif key and "STACK:" in line:
            found[(arch,) + key] = RESOURCES.search(line.strip()).group(1)
            key = None
    return found


def stack(resources):
    return int(re.search(r"STACK:(\d+)", resources).group(1))


def main():
    found = parse(sys.stdin, DISPATCH, COMBINE, True)
    ok = bool(found)
    groups = defaultdict(Counter)  # (kernel, FT_ABL, MASK) -> STACK counter
    for (arch, kernel, (top_k, mask, rest), abl), res in found.items():
        groups[(kernel, abl, mask)][stack(res)] += 1
        if abl and (mask != "1" or top_k != 8):
            print(f"ABLATE_RES_BAD {kernel} FT_ABL={abl} instantiated for TOP_K={top_k} MASK={mask} ({arch})")
            ok = False
    for kernel in ("dispatch", "combine"):
        # The control's MASK=1, TOP_K=8 instantiations, per arch: each variant must cover the same set.
        control = {(a, r) for (a, k, (t, m, r), b) in found if k == kernel and b == 0 and m == "1" and t == 8}
        for abl, name in VARIANTS.items():
            for mask in ("0", "1") if abl == 0 else ("1",):
                seen = groups.get((kernel, abl, mask), Counter())
                n = sum(seen.values())
                stacks = ",".join(f"{s}x{c}" for s, c in sorted(seen.items())) or "none"
                print(f"ABLATE_RES {kernel} abl={name} (FT_ABL={abl}) mask{mask} n={n} STACK={stacks}")
            if abl == 0:
                continue
            have = {(a, r) for (a, k, (t, m, r), b) in found if k == kernel and b == abl and m == "1" and t == 8}
            if not control or have != control:
                print(f"ABLATE_RES_MISSING {kernel} {name}: {len(have)} of {len(control)} control instantiations")
                ok = False
        all_stacks = groups.get((kernel, 61, "1"), Counter())
        if not all_stacks or set(all_stacks) != {24}:
            print(f"ABLATE_RES_ALL_SPILL {kernel}: ALL STACK {dict(all_stacks)} (want only 24)")
            ok = False
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            ref = parse(f, REF_DISPATCH, REF_COMBINE, False)
        same = diff = unmatched = 0
        for key, res in found.items():
            if key[-1] != 0:
                continue
            if key not in ref:
                unmatched += 1
            elif ref[key] == res:
                same += 1
            else:
                diff += 1
                if diff <= 5:
                    print(f"ABLATE_RES_CTRL_DIFF {key}: new {res} | ref {ref[key]}")
        verdict = "ABLATE_RES_CTRL_SAME" if diff == 0 and unmatched == 0 and same else "ABLATE_RES_CTRL_DIFF"
        print(f"{verdict} FT_ABL=0 vs reference: same={same} diff={diff} unmatched={unmatched} (informational)")
    print("ABLATE_RES_OK" if ok else "ABLATE_RES_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
