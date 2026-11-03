"""Compare every .py file in a wheel with the same path in a git commit; print mismatches, extras and stale files.

Extras under the trees the build generates or vendors are expected; any other extra is a stale file a shared build root
left in the wheel (setuptools never deletes from its build/lib).
"""
import subprocess, sys, zipfile
GENERATED = ("tensorrt_llm/include/", "tensorrt_llm/_torch/cute_dsl_kernels/", "tensorrt_llm/_torch/cuda_tile_kernels/",
             "tensorrt_llm/deep_gemm/", "tensorrt_llm/deep_ep/", "tensorrt_llm/flash_mla/")
wheel, repo, commit = sys.argv[1:4]
z = zipfile.ZipFile(wheel)
tree = set(subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "ls-tree", "-r", "--name-only", commit, "tensorrt_llm"],
                          capture_output=True, text=True, check=True).stdout.split())
bad, extra, n = [], [], 0
for name in z.namelist():
    if not (name.startswith("tensorrt_llm/") and name.endswith(".py")):
        continue
    n += 1
    if name not in tree:
        extra.append(name); continue
    want = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "show", f"{commit}:{name}"], capture_output=True, check=True).stdout
    if z.read(name) != want:
        bad.append(name)
stale = [e for e in extra if not e.startswith(GENERATED)]
print(f"{wheel.split('/')[-3]}: {n} py, {len(bad)} differ, {len(extra)} not in {commit}, {len(stale)} stale")
for b in bad[:15]: print("  DIFF", b)
for e in stale: print("  STALE", e)
