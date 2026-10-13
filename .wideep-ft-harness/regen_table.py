"""Regenerate the PR table in DESIGN.md from the commits of wideep-ft-folded (run in the fold worktree)."""
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
rows = []
log = subprocess.check_output(["git", "log", "--reverse", "--format=%h|%s", f"{base}..HEAD"], text=True)
for i, line in enumerate(filter(None, log.split("\n")), 1):
    sha, subj = line.split("|", 1)
    kind, title = re.match(r"\[None\]\[(\w+)\] (.*)", subj).groups()
    files, docs = [], []
    for st in filter(None, subprocess.check_output(["git", "show", "--numstat", "--format=", sha], text=True).split("\n")):
        a, d, f = st.split("\t")
        if f.startswith("docs/source/features/"):
            docs.append(f)
        if f.startswith(("tests/", "docs/", "jenkins/", "security_scanning/")) or f.endswith((".md", ".yml", ".txt", ".lock", ".toml")):
            continue
        files.append((int(a) if a != "-" else 0, f))
    files.sort(reverse=True)
    shown = [f for _, f in files[:3]] + docs
    code = ", ".join(f"[{os.path.basename(f)}]({R}/blob/{tip}/{f})" for f in shown)
    rows.append(f"| {i} | [{sha}]({R}/commit/{sha}) | `[{kind}]` {title} | {work[title]} | {code} |")
p = "../DESIGN.md"; s = open(p).read()
i = s.index("| # | Commit | PR | Work items | Main code |"); j = s.index("\n\n", i)
s = s[:i] + "| # | Commit | PR | Work items | Main code |\n|---|---|---|---|---|\n" + "\n".join(rows) + s[j:]
open(p, "w").write(s); print("table at", tip, len(rows), "rows")
