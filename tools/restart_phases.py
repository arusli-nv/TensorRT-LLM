"""TRT-LLM's own rank-0 phase times for FT-off warm restarts (slide 1 notes, CLAIMS 224).

usage: python3 tools/restart_phases.py <COLD-RESTART run dir>...   (dirs under mvp/runs)
For each COLD_RESTART line, reads the relaunched server's rank-0.log (the line's relaunch= dir) and prints model load
(executor_creation/model), attention JIT and autotuner (memory-profiling pass 1), and CUDA graph capture for the
memory dry run and the final pass.
"""
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "mvp" / "runs"
PATTERNS = {"model_load": r"executor_creation/model: done in ([\d.]+)s",
            "attention_jit": r"\[purpose=memory_profiling\]\[pass=1\] attention_jit: done in ([\d.]+)s",
            "autotuner": r"\[purpose=memory_profiling\]\[pass=1\] autotuner: done in ([\d.]+)s",
            "graphs_dry": r"\[purpose=memory_profiling\]\[pass=1\] cuda_graph_capture: done in ([\d.]+)s",
            "graphs_final": r"\[purpose=(?!memory_profiling)[^\]]*\]\[pass=\d+\] cuda_graph_capture: done in ([\d.]+)s"}


def phases(log):
    text = log.read_text(errors="replace")
    return {k: (float(m[1]) if (m := re.search(p, text)) else None) for k, p in PATTERNS.items()}


print("dir run down_s " + " ".join(PATTERNS))
for d in sys.argv[1:]:
    for line in open(RUNS / d / "cold_restart.txt"):
        if not line.startswith("COLD_RESTART "):
            continue
        run = re.search(r"\brun=(\d+)", line)[1]
        down = re.search(r"\bdown_s=([\d.]+)", line)[1]
        rel = re.search(r"\brelaunch=(\S+)", line)[1]
        ph = phases(RUNS / d / rel / "rank-0.log")
        print(d, run, down, " ".join("-" if v is None else f"{v:.1f}" for v in ph.values()))
