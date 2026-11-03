"""Per-case recovery sub-step durations from the server logs of KILL-FINAL runs (CPU only).

usage: python3 tools/recovery_substeps.py > review/diag/recovery_substeps.txt
For every KILL-FINAL-*/<case>/ with a first recovery, one line:
  SUBSTEP <run>/<case> mapped_ms=.. fence_ms=.. remap_ms=.. staged_ms=.. restored_ms=..
Each value is the largest over the case's rank logs (authority.log for fences) of the first recovery's line:
  mapped   "Mapped N bytes behind ghost expert slots in X ms"   (spare-slot memory, Plan)
  fence    "departed rank R fenced X ms into the recovery"      (authority, Plan)
  remap    "Remapped N bytes of all-to-all workspace ... in X ms" (Ready)
  staged   "Staged N expert weight pieces in X ms"               (ring copies, no EPLB only; Ready)
  restored "Restored N expert slots ... in X ms"                 (Install)
0 = no such line (e.g. staged with EPLB). Survivors run each step in parallel; the slowest gates the phase.
"""
import glob
import os
import re

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mvp", "runs")
PATTERNS = {
    "mapped_ms": r"Mapped \d+ bytes behind ghost expert slots in ([\d.]+) ms",
    "remap_ms": r"Remapped \d+ bytes of all-to-all workspace .*? in ([\d.]+) ms",
    "staged_ms": r"Staged \d+ expert weight pieces in ([\d.]+) ms",
    "restored_ms": r"Restored \d+ expert slots .*? in ([\d.]+) ms",
}
FIRST = re.compile(r"WideEP recovery gen=1 phase=first_step")


def first_value(text, pattern):
    """The pattern's value in the first recovery (before any gen=2 line), or None."""
    head = text.split("WideEP recovery gen=2 ", 1)[0]
    m = re.search(pattern, head)
    return float(m.group(1)) if m else None


def main():
    for case in sorted(glob.glob(f"{RUNS}/KILL-FINAL-*/*/")):
        r0 = os.path.join(case, "rank-0.log")
        if not os.path.exists(r0) or not FIRST.search(open(r0, errors="ignore").read()):
            continue
        logs = [open(f, errors="ignore").read() for f in sorted(glob.glob(f"{case}/rank-*.log"))]
        vals = {k: max([v for v in (first_value(t, p) for t in logs) if v is not None] or [0.0])
                for k, p in PATTERNS.items()}
        auth = os.path.join(case, "authority.log")
        head = open(auth, errors="ignore").read().split("gen=2", 1)[0] if os.path.exists(auth) else ""
        vals["fence_ms"] = max([float(x) for x in re.findall(r"fenced ([\d.]+) ms into the recovery", head)] or [0.0])
        name = os.path.relpath(case, RUNS).rstrip("/")
        print("SUBSTEP", name, *(f"{k}={vals[k]:.1f}" for k in ("mapped_ms", "fence_ms", "remap_ms", "staged_ms",
                                                                   "restored_ms")))


if __name__ == "__main__":
    main()
