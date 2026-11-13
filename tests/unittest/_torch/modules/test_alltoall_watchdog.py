# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for AlltoAllWatchdog (WideEP fault tolerance, PR 1a.4)."""

import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.alltoall_watchdog import (
    DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
    DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
    AlltoAllWatchdog,
    AlltoAllWatchdogTimeout,
    reject_rank_mask_cuda_graph_capture,
)
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.wide_ep_ft import get_wide_ep_ft_options


class FakeWaitedFlags:
    """Stand-in for the waits that the kernels publish in the abort state."""

    def __init__(self) -> None:
        self.current: dict[int, int] = {}

    def __call__(self) -> dict[int, int]:
        return dict(self.current)


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def _watchdog(waits: FakeWaitedFlags, events: list, **kwargs) -> AlltoAllWatchdog:
    options = {"timeout_s": 0.02, "poll_interval_s": 0.005, "on_timeout": events.append}
    options.update(kwargs)
    return AlltoAllWatchdog(ep_rank=0, waited_flags=waits, **options)


def test_watchdog_reports_nothing_for_waits_that_end_before_timeout() -> None:
    waits = FakeWaitedFlags()
    events: list[AlltoAllWatchdogTimeout] = []

    with _watchdog(waits, events, timeout_s=0.2):
        waits.current = {1: 1, 2: 1}
        time.sleep(0.05)
        waits.current = {}
        time.sleep(0.25)

    assert events == []


def test_watchdog_reports_peers_waited_on_past_timeout_once() -> None:
    """One event per waited round flag, naming every peer waited on for it, and only once."""
    health = EPGroupHealth(4)
    committed_before = health.snapshot()
    waits = FakeWaitedFlags()
    events: list[AlltoAllWatchdogTimeout] = []

    with _watchdog(waits, events):
        waits.current = {3: 1, 1: 1}
        _wait_for(lambda: len(events) == 1)
        time.sleep(0.05)
        assert len(events) == 1
        # The combine of the same round waits on peer 2: a new wait, so a new event.
        waits.current = {2: 2}
        _wait_for(lambda: len(events) == 2)
        time.sleep(0.05)

    assert [(e.phase, e.expected_flag, e.missing_ranks) for e in events] == [
        ("dispatch", 1, (1, 3)),
        ("combine", 2, (2,)),
    ]
    assert events[0].elapsed_s >= 0.02
    assert not hasattr(events[0], "marked_failed_ranks")
    assert health.snapshot() == committed_before


def test_watchdog_times_each_round_flag_separately() -> None:
    """A peer that delivers new rounds is not reported, even if the kernels keep waiting on it."""
    waits = FakeWaitedFlags()
    events: list[AlltoAllWatchdogTimeout] = []

    with _watchdog(waits, events, timeout_s=0.1):
        for flag in range(1, 9):
            waits.current = {1: flag}
            time.sleep(0.03)
        assert events == []
        # Flag values above the int32 range keep their uint32 value.
        waits.current = {1: (1 << 32) - 1}
        _wait_for(lambda: len(events) == 1)

    assert events[0].expected_flag == (1 << 32) - 1
    assert events[0].phase == "dispatch"


def test_watchdog_defaults_match_design_doc() -> None:
    watchdog = AlltoAllWatchdog(ep_rank=0, waited_flags=FakeWaitedFlags())

    assert watchdog._timeout_s == DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
    assert watchdog._poll_interval_s == DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S


def test_watchdog_stop_is_terminal() -> None:
    watchdog = _watchdog(FakeWaitedFlags(), [])
    watchdog.start()
    watchdog.stop(timeout_s=1.0)

    with pytest.raises(RuntimeError, match="stopped AlltoAllWatchdog"):
        watchdog.start()


def test_watchdog_callback_error_stops_it() -> None:
    waits = FakeWaitedFlags()
    waits.current = {1: 1}

    def raise_from_callback(event: AlltoAllWatchdogTimeout) -> None:
        raise RuntimeError(f"callback failed for {event.phase}")

    with _watchdog(waits, [], on_timeout=raise_from_callback) as watchdog:
        _wait_for(lambda: watchdog.last_error is not None)
        assert isinstance(watchdog.last_error, RuntimeError)
        with pytest.raises(RuntimeError, match="stopped AlltoAllWatchdog"):
            watchdog.start()


def test_wide_ep_ft_options_create_shared_health_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    health, timeout_s, poll_interval_s = get_wide_ep_ft_options(model_config)
    health_again, timeout_again_s, poll_again_s = get_wide_ep_ft_options(model_config)

    assert isinstance(health, EPGroupHealth)
    assert health_again is health
    assert timeout_s == DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S
    assert timeout_again_s == timeout_s
    assert poll_interval_s == DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S
    assert poll_again_s == poll_interval_s


def test_wide_ep_ft_options_reject_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=4),
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="does not support CUDA graphs"):
        get_wide_ep_ft_options(model_config)
    assert model_config.extra_attrs == {}


def test_rank_mask_mode_rejects_direct_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    reject_rank_mask_cuda_graph_capture(False)
    with pytest.raises(RuntimeError, match="does not support CUDA graphs"):
        reject_rank_mask_cuda_graph_capture(True)


def test_wide_ep_ft_options_ignore_legacy_enable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    monkeypatch.setenv("TRTLLM_ENABLE_WIDE_EP_FT", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    health, timeout_s, _ = get_wide_ep_ft_options(model_config)

    assert health is None
    assert timeout_s is None
