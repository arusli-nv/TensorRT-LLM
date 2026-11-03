# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: python3 -I t_h23.py mvp/harness/draft_cases/val_verdicts.py   (H23 kill thresholds and the restored-load readout)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("vv", sys.argv[1]); vv = importlib.util.module_from_spec(spec); spec.loader.exec_module(vv)

# Resume <= 10 s (every loss), after/before >= 0.8 x live/total, >= 20 requests in full after the kill.
c, f = vv.h23_checks([0.7], 329664, 314123, 31, 32, 12160); assert all(c.values()), (c, f)
assert f["h23_tput_ratio"] == 0.953 and f["h23_tput_floor"] == 0.775 and f["h23_live"] == "31/32", f
assert vv.h23_checks([10.0], 100, 78, 31, 32, 20)[0] == dict(h23_resume=True, h23_tput=True, h23_completed=True)
c, f = vv.h23_checks([10.01], 100, 77, 31, 32, 19)                                    # each threshold just missed
assert c == dict(h23_resume=False, h23_tput=False, h23_completed=False), c
c, f = vv.h23_checks([1.2], 100, 70, 28, 32, 500); assert c["h23_tput"] and f["h23_tput_floor"] == 0.7, f  # node of 4
c, f = vv.h23_checks([1.0, 11.0], 100, 90, 30, 32, 500); assert not c["h23_resume"] and f["h23_resume_s"] == "1.0,11.0"
c, f = vv.h23_checks([1.0, None], 100, 90, 30, 32, 500); assert not c["h23_resume"]   # second loss never resumed
c, f = vv.h23_checks([None], None, None, 31, 32, None)                                # nothing measured: FAIL, no crash
assert c == dict(h23_resume=False, h23_tput=True, h23_completed=False) and f["h23_tput_ratio"] == "N/A", (c, f)
# An empty after-window (no full decode steps after the resume) is N/A, not a throughput loss; the other checks hold.
c, f = vv.h23_checks([3.43], 240384, None, 31, 32, 1984, confirm_s=33.65); assert all(c.values()), (c, f)
assert f["h23_tput_ratio"] == "N/A", f
c, f = vv.h23_checks([10.5], 240384, None, 31, 32, 1984); assert c == dict(h23_resume=False, h23_tput=True, h23_completed=True)
c, f = vv.h23_checks([3.0], 240384, None, 31, 32, 19); assert not c["h23_completed"] and c["h23_tput"]
c, f = vv.h23_checks([3.0], None, 200000, 31, 32, 500); assert not c["h23_tput"] and f["h23_tput_ratio"] is None  # no before rate

# A frozen rank (row stop): resume timed from the hang confirmation, the confirm time reported apart and not checked.
c, f = vv.h23_checks([9.5], 100, 90, 31, 32, 500, confirm_s=36.43); assert all(c.values()), (c, f)
assert list(f)[:2] == ["h23_resume_s", "h23_confirm_s"] and f["h23_confirm_s"] == 36.43, f
assert not vv.h23_checks([12.2], 100, 90, 31, 32, 500, confirm_s=30.5)[0]["h23_resume"]
assert "h23_confirm_s" not in vv.h23_checks([1.0], 100, 90, 31, 32, 500)[1]                # kill rows: no field
# hang_confirmed_wall_s: the authority's "is hung ... fencing it" (whole-second local stamp) or the detect phase,
# whichever is earlier.
import time
hung = ("[10/22/2026-15:00:40] [TRT-LLM] [I] Recovery authority: rank 2 suspect\n"
        "[10/22/2026-15:00:41] [TRT-LLM] [E] Recovery authority: rank 2 (pid 1 on n) is hung (every peer waits on it "
        "and it made no progress for 30.0s); fencing it\n")
t41 = time.mktime(time.strptime("10/22/2026-15:00:41", "%m/%d/%Y-%H:%M:%S"))
assert vv.hang_confirmed_wall_s(hung, t41 + 2.2) == t41
assert vv.hang_confirmed_wall_s(hung, t41 - 3.0) == t41 - 3.0                           # detector detected first
assert vv.hang_confirmed_wall_s("", t41) == t41 and vv.hang_confirmed_wall_s(None, None) is None
assert vv.hang_confirmed_wall_s(hung.replace("fencing it", "not fencing"), None) is None

# completed_after: the first max_concurrency start at the anchor, each later one when the earliest slot frees.
res = lambda e2els, conc, errors=None, lens=None: dict(e2els=e2els, max_concurrency=conc, errors=errors or [""] * len(e2els),
                                                       output_lens=lens or [8] * len(e2els))
# conc 2: finishes 10, 12, then 10+5=15, 12+5=17 (anchor 100, so wall 110, 112, 115, 117).
r = res([10, 12, 5, 5], 2)
assert vv.completed_after([r], 100.0, 111.0, 8) == 3 and vv.completed_after([r], 100.0, 120.0, 8) == 0
assert vv.completed_after([r], 100.0, 99.0, 8) == 4
# An errored or short (cut) request does not count, but still holds its slot until it ended.
assert vv.completed_after([res([10, 12, 5, 5], 2, errors=["", "x", "", ""])], 100.0, 111.0, 8) == 2
assert vv.completed_after([res([10, 12, 5, 5], 2, lens=[8, 8, 3, 8])], 100.0, 111.0, 8) == 2
# Client parts replay separately and add up.
assert vv.completed_after([r, res([20, 20], 1)], 100.0, 111.0, 8) == 3 + 2
assert vv.completed_after([], 100.0, 111.0, 8) is None and vv.completed_after([r], None, 111.0, 8) is None
assert vv.completed_after([{}], 100.0, 111.0, 8) is None and vv.completed_after([r], 100.0, None, 8) is None
assert vv.completed_after([res([1, 2], 2, lens=[8])], 100.0, 0.0, 8) is None             # mismatched lists

# restored_load: memory each rank mapped behind ghost slots, summed over its recoveries; the max names its ranks.
mapped = lambda r, n: f"[x] [TRT-LLM] [I] [_torch][RANK {r}] Mapped {n} bytes behind ghost expert slots in 1 ms\n"
mb = 2**20
logs = {0: mapped(0, 444 * mb), 1: mapped(1, 444 * mb), 8: mapped(8, 360 * mb), 9: mapped(9, 360 * mb), 10: mapped(10, 360 * mb)}
assert vv.restored_load(logs) == "max444@0,1/med360/min360/x1.23", vv.restored_load(logs)
assert vv.restored_load({**logs, 9: mapped(9, 100 * mb) + mapped(9, 400 * mb)}).startswith("max500@9/"), "summed"
assert vv.restored_load({0: "no mapping", 1: ""}) is None
assert vv.restored_load({0: mapped(0, 0), 1: mapped(1, 0)}) == "max0@0,1"
print("PASS t_h23")
