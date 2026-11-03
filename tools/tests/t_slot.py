"""testhooks TLLM_TEST_LOG_SLOT_TABLE (alone and with TLLM_TEST_FT_NO_SLOT_TABLE) against a fake TRT-LLM tree imported
through the hook, plus the hooks' source anchors in a real checkout.

Run: python3 -I t_slot.py <testhooks.py> [<checkout>]  (each mode runs in its own child process: the hooks read env at import)."""
import ast, os, sys, subprocess, tempfile, textwrap, importlib, importlib.util, io, contextlib

FILES = {
    "tensorrt_llm/_torch/moe/fused_moe/interface.py": """
        class MoE:
            def __init__(self, layer_idx, balancer):
                self.layer_idx, self.layer_load_balancer = layer_idx, balancer
            def _load_balancer_route(self, token_selected_experts, use_dp):
                if self.layer_load_balancer:
                    return self.layer_load_balancer.@ROUTE@(token_selected_experts, use_dp)
                else:
                    return token_selected_experts
        """,
    "tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py": """
        import types
        from .interface import MoE
        class ExpertStatistic:
            expert_statistic_obj = None
        class CalibratorMode:
            NONE, COLLECT = "NONE", "COLLECT"
        CALIBRATOR = types.SimpleNamespace(mode=CalibratorMode.NONE)
        def get_calibrator(): return CALIBRATOR
        class NVLinkOneSided: pass
        def peer_loss_recovery_enabled(): return True
        class ExternalCommMoEScheduler:
            def __init__(self, moe):
                self.moe = moe
                self._peer_loss_recovery = peer_loss_recovery_enabled()
            def _is_using_nvlink_one_sided(self):
                return isinstance(self.moe.comm, NVLinkOneSided)
            def _dispatch_slot_table(self):
                if not (
                    self._is_using_nvlink_one_sided()
                    and ExpertStatistic.expert_statistic_obj is None
                    and get_calibrator().mode == CalibratorMode.NONE
                ):
                    return None
                return self.moe.layer_load_balancer.one_slot_table()
            def _forward_chunk_impl(self, x):
                moe, expert_to_slot = self.moe, None
                if self._peer_loss_recovery:
                    expert_to_slot = @SLOT_TABLE@
                if expert_to_slot is None:
                    return moe._load_balancer_route(x, False)
                return x
        """,
}
TREE = {"@SLOT_TABLE@": "self._dispatch_slot_table()", "@ROUTE@": "route"}
MOVED = {"refused": {"@SLOT_TABLE@": "self._dispatch_slot_table(*())"}, "route_refused": {"@ROUTE@": "route_moved"}}
# mode: the env it sets, then per layer (comm, updates_enabled, table) and the global None conditions it sets.
ENV = {"mixed": ["TLLM_TEST_LOG_SLOT_TABLE"], "statistic": ["TLLM_TEST_LOG_SLOT_TABLE"],
       "calibrator": ["TLLM_TEST_LOG_SLOT_TABLE"], "noslottable": ["TLLM_TEST_LOG_SLOT_TABLE", "TLLM_TEST_FT_NO_SLOT_TABLE"],
       "refused": ["TLLM_TEST_LOG_SLOT_TABLE"], "route_refused": ["TLLM_TEST_LOG_SLOT_TABLE"], "off": []}
# layer -> the line its first query must log (mixed: 4 layers covering every per-layer reason; else layers 0 and 1 as mixed's 0).
WANT = {"mixed": {0: "used=1 table=256", 1: "used=0 reason=updates_enabled", 2: "used=0 reason=comm=Other",
                  3: "used=0 reason=query_empty"},
        "statistic": {0: "used=0 reason=expert_statistic", 1: "used=0 reason=expert_statistic"},
        "calibrator": {0: "used=0 reason=calibrator=COLLECT", 1: "used=0 reason=calibrator=COLLECT"},
        "noslottable": {0: "used=0 reason=test_hook", 1: "used=0 reason=test_hook"}}
STEPS = 150


def write_tree(moved):
    root = tempfile.mkdtemp(prefix="t_slot.", dir=os.environ.get("TMPDIR"))
    for path, body in FILES.items():
        parts = path.split("/")
        for i in range(1, len(parts)):
            init = os.path.join(root, *parts[:i], "__init__.py")
            os.makedirs(os.path.dirname(init), exist_ok=True)
            open(init, "a").close()
        with open(os.path.join(root, path), "w") as f:
            body = textwrap.dedent(body)
            for key, value in {**TREE, **moved}.items():
                body = body.replace(key, value)
            f.write(body)
    return root


def child(hooks, mode):
    os.environ["OMPI_COMM_WORLD_RANK"] = "3"
    for var in ENV[mode]:
        os.environ[var] = "1"
    sys.path.insert(0, write_tree(MOVED.get(mode, {})))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        spec = importlib.util.spec_from_file_location("th", hooks); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
        P = "tensorrt_llm._torch.moe.fused_moe."
        iface = importlib.import_module(P + "interface")
        sch = importlib.import_module(P + "moe_scheduler")
        original_route = iface.MoE.__dict__["_load_balancer_route"]

        class Table:
            def numel(self): return 256

        class Balancer:
            def __init__(self, updates, table): self.updates_enabled, self.table, self.routed = updates, table, 0
            def one_slot_table(self): return None if self.updates_enabled else self.table
            def route(self, x, use_dp): self.routed += 1; return x

        layers = [("nvl", False, Table()), ("nvl", True, Table()), ("other", False, Table()), ("nvl", False, None)]
        if mode != "mixed":
            layers = layers[:1] * 2
        if mode == "statistic":
            sch.ExpertStatistic.expert_statistic_obj = object()
        if mode == "calibrator":
            sch.CALIBRATOR.mode = sch.CalibratorMode.COLLECT
        moes = []
        for idx, (comm, updates, table) in enumerate(layers):
            moe = iface.MoE(idx, Balancer(updates, table))
            moe.comm = sch.NVLinkOneSided() if comm == "nvl" else type("Other", (), {})()
            moe.scheduler = sch.ExternalCommMoEScheduler(moe)
            moes.append(moe)
        for _ in range(STEPS):
            for moe in moes:
                moe.scheduler._forward_chunk_impl("x")
    text = out.getvalue()
    print(text, end="")
    lines = text.splitlines()
    slot = [l for l in lines if l.startswith("WIDEEP_SLOT_TABLE rank 3 t=") and " layer=" in l]
    route = [l for l in lines if l.startswith("WIDEEP_ROUTE_CALLS rank 3 t=") and " layer=" in l]
    # The fake's routing ran exactly as without the hooks: a layer without a table routes every step.
    routed = {i: m.layer_load_balancer.routed for i, m in enumerate(moes)}
    if mode in ("off", "refused", "route_refused"):
        if mode == "off":
            assert not text, text
        else:
            event = "WIDEEP_SLOT_TABLE" if mode == "refused" else "WIDEEP_ROUTE_CALLS"
            moved = "_forward_chunk_impl.slot_table" if mode == "refused" else "_load_balancer_route"
            assert f"{event}_REFUSED rank 3 t=" in text and f"moved={moved}" in text, text
            assert f"{event} rank 3" not in text and not (slot if mode == "refused" else route), text
            other = "WIDEEP_ROUTE_CALLS" if mode == "refused" else "WIDEEP_SLOT_TABLE"
            assert sum(l.startswith(f"{other} rank 3 t=") and l.endswith(" applied module=" + ("interface" if mode == "refused" else "moe_scheduler"))
                       for l in lines) == 1, text  # the other hook still applies
        if mode == "route_refused":
            assert iface.MoE.__dict__["_load_balancer_route"] is original_route
        return
    assert "REFUSED" not in text, text
    assert text.count("WIDEEP_SLOT_TABLE rank 3 t=") - len(slot) == 1 and "applied module=moe_scheduler" in text, text
    assert text.count("WIDEEP_ROUTE_CALLS rank 3 t=") - len(route) == 1 and "applied module=interface" in text, text
    want = WANT[mode]
    assert len(slot) == len(want), slot  # once per layer, not per step
    for layer, tail in want.items():
        assert any(l.endswith(f" layer={layer} {tail}") for l in slot), (layer, tail, slot)
    for layer, tail in want.items():
        n = routed[layer]
        assert n == (0 if "used=1" in tail else STEPS), (layer, n)
        got = sorted(l.split(" layer=")[1] for l in route if f" layer={layer} " in l + " ")
        assert got == ([] if n == 0 else [f"{layer} n=1 lb=1 recovery=1", f"{layer} n=100 lb=1 recovery=1"]), (layer, got)
    if mode == "noslottable":
        assert "WIDEEP_FT_NO_SLOT_TABLE rank 3" in text and "slot_table=route" in text, text


def real_anchors(checkout):
    """Both slot-table hooks' anchors in the real sources: [(file, class, method, text)] each found."""
    base = os.path.join(checkout, "tensorrt_llm/_torch/moe/fused_moe")
    want = [("moe_scheduler.py", "ExternalCommMoEScheduler", "_forward_chunk_impl", "expert_to_slot = self._dispatch_slot_table()"),
            ("moe_scheduler.py", "ExternalCommMoEScheduler", "_forward_chunk_impl", "if expert_to_slot is None:"),
            ("moe_scheduler.py", "ExternalCommMoEScheduler", "_dispatch_slot_table", "one_slot_table()"),
            ("moe_scheduler.py", "ExternalCommMoEScheduler", "_dispatch_slot_table", "ExpertStatistic.expert_statistic_obj is None"),
            ("moe_scheduler.py", "ExternalCommMoEScheduler", "_dispatch_slot_table", "get_calibrator().mode == CalibratorMode.NONE"),
            ("interface.py", "MoE", "_load_balancer_route", ".route(")]
    missing = []
    for name, cls, method, text in want:
        source = open(os.path.join(base, name)).read()
        found = False
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method:
                        found = text in ast.get_source_segment(source, item)
        if not found:
            missing.append(f"{cls}.{method}:{text}")
    sched = open(os.path.join(base, "moe_scheduler.py")).read()
    missing += [n for n in ("ExpertStatistic", "get_calibrator", "CalibratorMode") if f"import {n}" not in sched
                and f"as {n}" not in sched]
    return missing


if len(sys.argv) > 2 and sys.argv[2] in ENV:
    child(sys.argv[1], sys.argv[2])
else:
    for mode in ENV:
        run = subprocess.run([sys.executable, "-I", __file__, sys.argv[1], mode], capture_output=True, text=True)
        assert run.returncode == 0, (mode, run.stdout, run.stderr)
        print("PASS", mode, "|", " / ".join(l.split(" t=")[0] + " " + l.split(" ", 4)[-1] for l in run.stdout.splitlines()
                                             if " layer=0 " in l + " " or "REFUSED" in l))
    if len(sys.argv) > 2:
        missing = real_anchors(sys.argv[2])
        assert not missing, missing
        print("PASS real_anchors", sys.argv[2])
