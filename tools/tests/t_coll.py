import os, sys, types, importlib.util, io, contextlib
os.environ["TLLM_TEST_COLL_TIMES"] = "3"
spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
class MPIDist:
    def allgather(self, obj, *, departed=None): return [obj]
    def tp_gather(self, obj): return obj
m = types.SimpleNamespace(MPIDist=MPIDist)
th._patch_coll_times(m)
d = MPIDist(); out = io.StringIO()
with contextlib.redirect_stdout(out):
    for i in range(7): assert d.allgather(i, departed=1) == [i]
    assert d.tp_gather(5) == 5
lines = [l for l in out.getvalue().splitlines() if "WIDEEP_COLL" in l]
assert len(lines) == 2 and all("method=allgather calls=3" in l for l in lines), lines
print("PASS", lines[0][:120])
