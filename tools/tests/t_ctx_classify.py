# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: python3 -I t_ctx_classify.py tools/ctx_classify.py mvp/runs/STD-8147471   (reproduces CLAIMS 170)
import subprocess, sys
d = sys.argv[2]
out = subprocess.run([sys.executable, "-I", sys.argv[1], "--off", f"{d}/std-online-1", f"{d}/std-online-2",
                      "--on", f"{d}/std-online-ft-1", f"{d}/std-online-ft-2"], capture_output=True, text=True, check=True).stdout
dev = out.split("### dev step ms")[1]
mixed = [c.split(" / ")[0] for c in [l for l in dev.splitlines() if l.startswith("| mixed |")][0].split(" | ")[1:]]
assert mixed == ["513.4", "426.5", "506.0", "521.5"], mixed                 # mixed ctx median off 513/427 vs ft 506/522
totals = [l.split(" | ")[3] for l in out.split("### Totals")[1].splitlines() if l.startswith("| o")]
assert totals == ["67.8", "62.7", "66.0", "66.3"], totals                   # total ctx s off 67.8/62.7 vs ft 66.0/66.3
assert "refill -1.1 (n=24), backlog +1.6 (n=36)" in out and "refill -12.8 (n=34), backlog +14.3 (n=25)" in out, out
assert "-0.31*fetched" in out, out                                          # ~0 excess per new request
print("T_CTX_CLASSIFY PASS")
