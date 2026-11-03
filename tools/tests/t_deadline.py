# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# testhooks C10: TLLM_TEST_LOG_DEADLINE lines and TLLM_TEST_DELAY_ABORT_RANK. Usage: python3 -I t_deadline.py testhooks.py
import os, sys, types, importlib.util, io, contextlib
def load(**env):
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("th", sys.argv[1]); th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
    return th
th = load(OMPI_COMM_WORLD_RANK="9", TLLM_TEST_DELAY_ABORT_S="45", TLLM_TEST_DELAY_ABORT_RANK="9", TLLM_TEST_LOG_DEADLINE="1")
assert th.DELAY_ABORT_S == 45.0
assert th.PATCHES.get(th._COMMUNICATOR) and th.PATCHES.get("tensorrt_llm.executor.recovery.client")
class MPIDist:
    def revoke(self): return "r"
    def shrink(self): return frozenset({5})
    def watch_peer_failures(self, cb, word): return (cb, word)
class RecoveryTimeout(RuntimeError): pass
comm, client = types.SimpleNamespace(MPIDist=MPIDist), types.SimpleNamespace(RecoveryTimeout=RecoveryTimeout)
th._patch_deadline_comm(comm); th._patch_deadline_client(client)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    d = MPIDist()
    assert d.revoke() == "r" and d.shrink() == frozenset({5}) and d.watch_peer_failures(print, 0) == (print, 0)
    e = RecoveryTimeout("The recovery deadline passed before x")
assert str(e) == "The recovery deadline passed before x"
lines = out.getvalue().splitlines()
assert [l.split()[0] for l in lines] == ["WIDEEP_MPI_CALL"] * 3 + ["WIDEEP_DEADLINE"], lines
assert [l.split("name=")[1] for l in lines[:3]] == ["revoke", "shrink", "watch_peer_failures"], lines
assert lines[0].startswith("WIDEEP_MPI_CALL rank 9 t="), lines
assert load(OMPI_COMM_WORLD_RANK="3").DELAY_ABORT_S == 0.0  # another rank: no delay
assert load(OMPI_COMM_WORLD_RANK="3", TLLM_TEST_DELAY_ABORT_RANK="").DELAY_ABORT_S == 0.0
os.environ.pop("TLLM_TEST_DELAY_ABORT_RANK")
assert load(OMPI_COMM_WORLD_RANK="3").DELAY_ABORT_S == 45.0  # unset: every rank, as before
print("PASS", lines[-1][:100])
