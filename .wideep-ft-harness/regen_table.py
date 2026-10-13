"""Refresh the commit headings in DESIGN.md from the commits of wideep-ft-folded (run in the fold worktree)."""
import os, re, subprocess
R = "https://github.com/arusli-nv/TensorRT-LLM"; base = "ca3a6c4c62"
tip = subprocess.check_output(["git", "rev-parse", "--short=10", "HEAD"], text=True).strip()
work = {
 "keep EPGroupHealth identity across ConfigurableMoE deepcopy": "prerequisite (1a.8)",
 "upgrade mpi4py to 4.1.2 for ULFM fault tolerance": "1c.3",
 "add an external recovery authority for WideEP peer loss": "1c.4b, 1c.3a, 1b.2a",
 "ULFM survivor communicators for host collectives": "1c.3, 1c.3a, 1c.4a",
 "add a run-time rank mask and abort to the NVLinkOneSided all-to-all": "1a.8, 1a.8a, 1a.11",
 "bind MoE layers to the all-to-all execution descriptor for recovery": "1a.8a, 1a.8b",
 "route a lost rank's experts to their replicas after a peer loss": "1b.2a, 1b.4a",
 "let attention-DP routing skip absent ranks": "1b.4a, 1c.4a",
 "keep the MPI session serving after a peer loss": "1d.1",
 "roll back KV Cache Manager V2 allocations of a discarded step": "1c.4c",
 "recover the executor from a WideEP peer loss": "1c.4, 1c.4c, 1d.1, 1d.4a",
}
p = "../DESIGN.md"; s = open(p).read()
log = subprocess.check_output(["git", "log", "--reverse", "--format=%h|%s", f"{base}..HEAD"], text=True)
for i, line in enumerate(filter(None, log.split("\n")), 1):
    sha, subj = line.split("|", 1)
    kind, title = re.match(r"\[None\]\[(\w+)\] (.*)", subj).groups()
    heading = f"#### {i}. {work[title]}: [{kind}] {title} ([{sha}]({R}/commit/{sha}))"
    s, n = re.subn(rf"^#### {i}\. .*$", lambda _: heading, s, flags=re.M)
    assert n == 1, f"no heading for commit {i}"
open(p, "w").write(s); print("headings at", tip, i, "commits")
