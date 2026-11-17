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
"""Hold a CUDA stream on a host flag, so that a test controls when the work queued on it runs."""

import threading
from contextlib import contextmanager

import pytest
import torch


class _Gate:
    def __init__(self, flag: torch.Tensor):
        self._flag = flag
        self.opened = False
        self.tripped_by_watchdog = False

    def open(self) -> None:
        self._flag.numpy()[0] = 1
        self.opened = True


def _wait_on_flag(stream: torch.cuda.Stream) -> _Gate:
    from cuda.bindings import driver

    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    (err,) = driver.cuStreamWaitValue32(
        driver.CUstream(stream.cuda_stream),
        driver.CUdeviceptr(flag.data_ptr()),
        1,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
    )
    if err == driver.CUresult.CUDA_ERROR_NOT_SUPPORTED:
        pytest.skip("this device cannot hold a stream on a host flag")
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return _Gate(flag)


@contextmanager
def held_stream(stream: torch.cuda.Stream, strict: bool = True, watchdog_seconds: float = 120.0):
    """Hold ``stream`` on a host flag until ``gate.open()``: work queued meanwhile does not start.

    ``strict`` makes a torch host sync raise; a watchdog opens the gate if the test never does, and
    the block then fails, so a host wait on the stream cannot pass unnoticed.
    """
    gate = _wait_on_flag(stream)

    def trip():
        gate.tripped_by_watchdog = True
        gate.open()

    watchdog = threading.Timer(watchdog_seconds, trip)
    watchdog.start()
    previous = torch.cuda.get_sync_debug_mode()
    if strict:
        torch.cuda.set_sync_debug_mode("error")
    try:
        yield gate
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        gate.open()
        watchdog.cancel()
        watchdog.join()
        stream.synchronize()
    assert not gate.tripped_by_watchdog, "the stream stayed held: something waited on it"


@contextmanager
def gated_stream(stream: torch.cuda.Stream, open_after: float):
    """Hold ``stream`` until a timer opens it after ``open_after`` seconds.

    A host wait on the stream then simply lasts that long. ``gate.opened`` tells whether it has.
    """
    gate = _wait_on_flag(stream)
    timer = threading.Timer(open_after, gate.open)
    timer.start()
    try:
        yield gate
    finally:
        timer.cancel()
        gate.open()
        timer.join()
        stream.synchronize()
