"""testhooks TLLM_TEST_BCAST_LOG against a fake TRT-LLM tree imported through the hook.

Run: python3 -I t_bcast.py <testhooks.py>  (each mode runs in its own child process: the hooks read env at import)."""
import contextlib, importlib, importlib.util, io, os, subprocess, sys, tempfile, textwrap

FILES = {
    "tensorrt_llm/_torch/distributed/communicator.py": """
        import pickle
        class Comm:
            def __init__(self, rank): self.rank = rank
        def safe_broadcast(comm, obj, root=0):
            if comm.rank == root:
                serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                return obj
            dst_buf = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))  # what the root sent
            @LOADS@
        class Dist:
            def __init__(self, rank): self.rank, self.world_size, self.has_pp = rank, 32, False
            def broadcast(self, obj, root=0): return safe_broadcast(Comm(self.rank), obj, root=root)
        """,
    "tensorrt_llm/_torch/pyexecutor/request_utils.py": """
        class RequestBroadcaster:
            def __init__(self, dist): self.dist = dist
            def broadcast(self, new_requests):
                return self._broadcast_requests(new_requests, None)
            def _broadcast_requests(self, new_requests, py_request_objects):
                payloads = (new_requests, py_request_objects)
                return self.dist.broadcast(payloads, root=0)
        """,
    "tensorrt_llm/_torch/pyexecutor/py_executor.py": """
        class Req:
            def __init__(self, chunk, dummy=False): self.context_chunk_size, self.is_attention_dp_dummy = chunk, dummy
        class Batch:
            def __init__(self, ctx): self.context_requests = ctx
        class PyExecutor:
            def __init__(self, batches): self.batches = list(batches)
            def _prepare_and_schedule_batch(self):
                scheduled_batch, iter_stats = self.batches.pop(0), None
                return scheduled_batch, iter_stats
        """,
}
LOADS = {"normal": "return pickle.loads(dst_buf)", "refused": "return pickle.loads(bytes(dst_buf))"}


def write_tree(mode):
    root = tempfile.mkdtemp(prefix="t_bcast.", dir=os.environ.get("TMPDIR"))
    for path, body in FILES.items():
        parts = path.split("/")
        for i in range(1, len(parts)):
            init = os.path.join(root, *parts[:i], "__init__.py")
            os.makedirs(os.path.dirname(init), exist_ok=True)
            open(init, "a").close()
        with open(os.path.join(root, path), "w") as f:
            f.write(textwrap.dedent(body).replace("@LOADS@", LOADS["refused" if mode == "refused" else "normal"]))
    return root


def child(hooks, mode):
    rank = {"root": "0", "peer": "31", "quiet": "5", "refused": "31"}[mode]
    os.environ["OMPI_COMM_WORLD_RANK"] = rank
    os.environ["TLLM_TEST_BCAST_LOG"] = "0,31"
    sys.path.insert(0, write_tree(mode))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        spec = importlib.util.spec_from_file_location("th", hooks); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
        comm = importlib.import_module("tensorrt_llm._torch.distributed.communicator")
        ru = importlib.import_module("tensorrt_llm._torch.pyexecutor.request_utils")
        pe = importlib.import_module("tensorrt_llm._torch.pyexecutor.py_executor")
        requests = [{"id": i, "tokens": list(range(1024))} for i in range(3)]
        got = ru.RequestBroadcaster(comm.Dist(int(rank))).broadcast(requests)
        executor = pe.PyExecutor([pe.Batch([]), pe.Batch([pe.Req(512), pe.Req(1024), pe.Req(8, dummy=True)]), None])
        steps = [executor._prepare_and_schedule_batch() for _ in range(3)]
        other = comm.pickle.dumps({"x": 1})  # outside a request broadcast: not recorded
    text = out.getvalue()
    print(text, end="")
    assert got[0] == requests and [s[0] is None for s in steps] == [False, False, True], (got, steps)
    assert comm.pickle.loads(other) == {"x": 1} and comm.pickle.HIGHEST_PROTOCOL >= 5
    size = len(__import__("pickle").dumps((requests, None), protocol=5))
    if mode == "refused":
        assert "WIDEEP_BCAST_REFUSED rank 31 t=" in text and "module=communicator moved=safe_broadcast.loads" in text, text
        assert "applied module=communicator" not in text, text
        assert "applied module=request_utils log=1" in text and f"bytes=None" in text, text  # timed, size unknown
        return
    applied = [l for l in text.splitlines() if " applied module=" in l]
    log = int(mode != "quiet")
    assert len(applied) == 3 and all(l.endswith(f"log={log}") for l in applied), text
    if mode == "quiet":
        assert "WIDEEP_BCAST rank 5 t=" in text and " n=" not in text and "WIDEEP_CTX" not in text, text
        assert type(comm.pickle).__name__ == "module" and "_broadcast_requests" in vars(ru.RequestBroadcaster)
        return
    lines = [l for l in text.splitlines() if l.startswith("WIDEEP_BCAST") and " n=" in l]
    assert len(lines) == 1 and f" n=3 " in lines[0] and lines[0].endswith(f"bytes={size}"), (size, text)
    ctx = [l for l in text.splitlines() if l.startswith("WIDEEP_CTX")]
    assert len(ctx) == 1 and " call=2 reqs=2 tokens=1536 sched_ms=" in ctx[0], text
    assert "REFUSED" not in text, text


if len(sys.argv) > 2:
    child(sys.argv[1], sys.argv[2])
else:
    bad = 0
    for mode in ("root", "peer", "quiet", "refused"):
        run = subprocess.run([sys.executable, "-I", __file__, sys.argv[1], mode], capture_output=True, text=True)
        ok = run.returncode == 0
        bad += not ok
        first = next((l for l in run.stdout.splitlines() if " n=" in l or "REFUSED" in l), run.stdout.splitlines()[:1])
        print(("PASS" if ok else "FAIL"), mode, "|", first if ok else (run.stdout + run.stderr)[-1500:])
    sys.exit(bad)
