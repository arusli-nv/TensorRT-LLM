import os, sys, types, importlib.util, io, contextlib, time
os.environ["TLLM_TEST_LAUNCH_SKEW"] = "3"
spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
class MPIDist:
    def tp_allgather(self, obj): return [obj]
class CUDAGraphRunner:
    def replay(self, key, inputs): return key
comm = types.SimpleNamespace(MPIDist=MPIDist); runner = types.SimpleNamespace(CUDAGraphRunner=CUDAGraphRunner)
th._patch_skew_collectives(comm); th._patch_skew_replay(runner)
d, r, out = MPIDist(), CUDAGraphRunner(), io.StringIO()
with contextlib.redirect_stdout(out):
    assert r.replay("k", {}) == "k"  # no collective yet: not counted
    for _ in range(6):
        d.tp_allgather(1); time.sleep(0.001); assert r.replay("k", {}) == "k"
lines = [l for l in out.getvalue().splitlines() if "WIDEEP_SKEW" in l]
assert len(lines) == 2 and "after=tp_allgather" in lines[0], lines
p50 = float(lines[0].split("p50_us=")[1].split()[0]); assert 900 < p50 < 5000, p50
print("PASS", lines[0][:140])
