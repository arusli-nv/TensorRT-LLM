# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore copy rate: experts from a host shared-memory segment (as online EPLB keeps them) into GPU slots.

One rank lost at 288 slots moves ~58 experts (~23.4 MB each, R1 NVFP4) into each survivor GPU (RESTORE.md).
Paths timed, each over the same expert set:
  pageable   cudaMemcpyAsync from the shm segment as allocated (main's segments are not registered)
  registered the same segment after cudaHostRegister (what restore could do once at startup)
  cpu-N      N CPU threads memcpy into host-visible device memory (main's default copyWeightsCpu path),
             approximated by writing a device tensor through its C2C/ATS mapping when the platform allows it
Usage: python3 restore_copy.py [EXPERTS] [EXPERT_MB] [REPEATS]
"""

import concurrent.futures
import ctypes
import statistics
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import torch

experts = int(sys.argv[1]) if len(sys.argv) > 1 else 58
expert_bytes = int(float(sys.argv[2] if len(sys.argv) > 2 else 23.4) * 2**20)
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 5
total = experts * expert_bytes

shm = shared_memory.SharedMemory(create=True, size=total)
try:
    host = np.ndarray((experts, expert_bytes), dtype=np.uint8, buffer=shm.buf)
    host[:] = np.arange(expert_bytes, dtype=np.uint64).astype(np.uint8)[None, :]
    host_t = torch.from_numpy(host)
    slots = torch.empty((experts, expert_bytes), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()

    def gpu_copy() -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for e in range(experts):
                slots[e].copy_(host_t[e], non_blocking=True)
        stream.synchronize()
        return time.perf_counter() - start

    def report(name: str, times: list) -> None:
        best, med = min(times), statistics.median(times)
        print(f"RESTORE_COPY {name} experts={experts} MB={total / 2**20:.0f} median_ms={med * 1e3:.2f} "
              f"best_ms={best * 1e3:.2f} GBps={total / med / 1e9:.1f}", flush=True)

    gpu_copy()
    report("pageable", [gpu_copy() for _ in range(repeats)])

    cudart = None
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    assert cudart is not None, "libcudart not found"
    ptr = ctypes.c_void_p(host.ctypes.data)
    rc = cudart.cudaHostRegister(ptr, ctypes.c_size_t(total), 0)
    if rc == 0:
        gpu_copy()
        report("registered", [gpu_copy() for _ in range(repeats)])
        cudart.cudaHostUnregister(ptr)
    else:
        print(f"RESTORE_COPY registered skipped cudaHostRegister rc={rc}", flush=True)

    # CPU threads writing device memory directly: GB200 maps GPU memory into the CPU's address space (C2C + ATS),
    # so a cudaMallocManaged buffer preferred on the GPU stands in for main's host-accessible slot allocation.
    managed = ctypes.c_void_p()
    if cudart.cudaMallocManaged(ctypes.byref(managed), ctypes.c_size_t(total), 1) == 0:
        device = torch.cuda.current_device()
        cudart.cudaMemAdvise(managed, ctypes.c_size_t(total), 3, device)  # SetPreferredLocation
        cudart.cudaMemAdvise(managed, ctypes.c_size_t(total), 5, -1)  # SetAccessedBy CPU
        cudart.cudaMemPrefetchAsync(managed, ctypes.c_size_t(total), device, None)
        torch.cuda.synchronize()
        dst = np.ctypeslib.as_array((ctypes.c_uint8 * total).from_address(managed.value)).reshape(experts, -1)
        for threads in (4, 8, 16):
            def cpu_copy() -> float:
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(threads) as pool:
                    list(pool.map(lambda e: np.copyto(dst[e], host[e]), range(experts)))
                return time.perf_counter() - start

            cpu_copy()
            report(f"cpu-{threads}", [cpu_copy() for _ in range(repeats)])
        cudart.cudaFree(managed)
    else:
        print("RESTORE_COPY cpu skipped: cudaMallocManaged failed", flush=True)
    del host_t, host
finally:
    shm.close()
    shm.unlink()
print("RESTORE_COPY_DONE", flush=True)
