#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: can a VMM reservation be backed by HOST_NUMA memory on the GPU's own memory NUMA node (GB200 HBM),
CPU-written through the VMM VA, and GPU-read at HBM speed? Compares against cudaMalloc and main's EPLB pool
(mmap + mbind(GPU memory node) + MADV_HUGEPAGE + cudaHostRegister).

Usage: python3 host_numa_vmm_probe.py [--device 0] [--gib 1] [--iters 20]
Each step prints PASS / FAIL / INFO. The CPU dereference of the VA is first tried in a forked child, so a
segfault there is reported instead of killing the probe.
"""
import argparse
import ctypes
import os
import platform
import subprocess
import sys
import threading
import time

import torch

try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu

RESULTS = []


def report(step, ok, msg):
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    RESULTS.append((step, tag, msg))
    print(f"[{tag}] {step}: {msg}", flush=True)


def ename(err):
    try:
        e, name = cu.cuGetErrorName(err)
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return str(err)


def ok(err):
    return int(err) == 0


def attr(dev, name):
    a = getattr(cu.CUdevice_attribute, name, None)
    if a is None:
        return None
    err, v = cu.cuDeviceGetAttribute(a, dev)
    return v if ok(err) else f"err {ename(err)}"


# ---------------------------------------------------------------- (a) NUMA ids
def find_numa(device):
    err, dev = cu.cuDeviceGet(device)
    numa_config = attr(dev, "CU_DEVICE_ATTRIBUTE_NUMA_CONFIG")
    numa_id = attr(dev, "CU_DEVICE_ATTRIBUTE_NUMA_ID")
    host_numa_id = attr(dev, "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID")
    ats = attr(dev, "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES")
    pageable = attr(dev, "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS")
    vmm = attr(dev, "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED")
    props = torch.cuda.get_device_properties(device)
    pci_numa = None
    try:
        bus = torch.cuda.get_device_properties(device).pci_bus_id
        dom = getattr(props, "pci_domain_id", 0)
        devid = getattr(props, "pci_device_id", 0)
        path = f"/sys/bus/pci/devices/{dom:04x}:{bus:02x}:{devid:02x}.0/numa_node"
        pci_numa = int(open(path).read().strip())
    except Exception as e:
        pci_numa = f"unreadable ({e})"
    report("a.numa", None, f"NUMA_CONFIG={numa_config} (1 = GPU memory is a NUMA node), NUMA_ID={numa_id}, "
           f"HOST_NUMA_ID={host_numa_id}, pci numa_node={pci_numa}, ATS(host page tables)={ats}, "
           f"pageable access={pageable}, VMM={vmm}, kernel page size={os.sysconf('SC_PAGE_SIZE')}")
    try:
        out = subprocess.run(["numactl", "-H"], capture_output=True, text=True, timeout=10).stdout
        print("\n".join(l for l in out.splitlines() if "size" in l or "available" in l), flush=True)
    except Exception as e:
        print(f"numactl unavailable: {e}")
    mem_numa = numa_id if numa_config == 1 and isinstance(numa_id, int) and numa_id >= 0 else -1
    report("a.gpu_memory_numa_id", mem_numa >= 0,
           f"GPU memory NUMA id = {mem_numa} (TopologyDetector::getCurrentGpuMemoryNumaId equivalent)")
    cpu_numa = host_numa_id if isinstance(host_numa_id, int) and host_numa_id >= 0 else (
        pci_numa if isinstance(pci_numa, int) and pci_numa >= 0 else 0)
    return dev, mem_numa, cpu_numa


# ---------------------------------------------------------------- (b) VMM with HOST_NUMA
def location(kind, ident):
    loc = cu.CUmemLocation()
    loc.type = getattr(cu.CUmemLocationType, kind)
    loc.id = ident
    return loc


def access_desc(kind, ident):
    d = cu.CUmemAccessDesc()
    d.location = location(kind, ident)
    d.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return d


def alloc_prop(kind, ident):
    p = cu.CUmemAllocationProp()
    p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location = location(kind, ident)
    return p


class VmmRange:
    """One reservation, mapped with chunks of a given location; access set for the device (+ host)."""

    def __init__(self, label, kind, ident, nbytes, device, want_host, chunk_bytes=None):
        self.label, self.ptr, self.size, self.handles, self.mapped = label, 0, 0, [], []
        prop = alloc_prop(kind, ident)
        gmin = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        grec = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        e1, g_min = cu.cuMemGetAllocationGranularity(prop, gmin)
        e2, g_rec = cu.cuMemGetAllocationGranularity(prop, grec)
        report(f"b.{label}.granularity", ok(e1) and ok(e2) and g_min <= (2 << 20),
               f"{kind} id={ident}: minimum={g_min} recommended={g_rec} ({ename(e1)}/{ename(e2)}); "
               f"PASS needs minimum <= 2 MiB (ghost chunks are 2 MiB today)")
        if not ok(e1):
            raise RuntimeError("granularity query failed")
        chunk = chunk_bytes or max(g_min, 2 << 20)
        chunk = (chunk + g_min - 1) // g_min * g_min
        self.size = (nbytes + chunk - 1) // chunk * chunk
        err, ptr = cu.cuMemAddressReserve(self.size, chunk, 0, 0)
        report(f"b.{label}.reserve", ok(err), f"cuMemAddressReserve({self.size}) -> {ename(err)}")
        if not ok(err):
            raise RuntimeError("reserve failed")
        self.ptr = int(ptr)
        # Map in chunks, as ghostSlotMemory maps resident runs and pieces separately.
        n_chunks = self.size // chunk
        first_err = None
        for i in range(n_chunks):
            err, h = cu.cuMemCreate(chunk, prop, 0)
            if not ok(err):
                first_err = f"cuMemCreate(chunk {i}, {chunk} B) -> {ename(err)}"
                break
            self.handles.append(h)
            (err,) = cu.cuMemMap(self.ptr + i * chunk, chunk, 0, h, 0)
            if not ok(err):
                first_err = f"cuMemMap(chunk {i}) -> {ename(err)}"
                break
            self.mapped.append((self.ptr + i * chunk, chunk))
        report(f"b.{label}.create_map", first_err is None,
               first_err or f"{n_chunks} x cuMemCreate({chunk} B, {kind} id={ident}) + cuMemMap OK")
        if first_err:
            raise RuntimeError(first_err)
        (err,) = cu.cuMemSetAccess(self.ptr, self.size, [access_desc("CU_MEM_LOCATION_TYPE_DEVICE", device)], 1)
        report(f"b.{label}.access_device", ok(err), f"cuMemSetAccess(DEVICE {device}) -> {ename(err)}")
        self.host_access = False
        if want_host:
            tries = [("CU_MEM_LOCATION_TYPE_HOST", 0)]
            if kind == "CU_MEM_LOCATION_TYPE_HOST_NUMA":
                tries.append(("CU_MEM_LOCATION_TYPE_HOST_NUMA", ident))
            for hk, hid in tries:
                if getattr(cu.CUmemLocationType, hk, None) is None:
                    report(f"b.{label}.access_{hk}", False, "enum not in these bindings")
                    continue
                (err,) = cu.cuMemSetAccess(self.ptr, self.size, [access_desc(hk, hid)], 1)
                gerr, flags = cu.cuMemGetAccess(location(hk, hid), self.ptr)
                report(f"b.{label}.access_{hk.split('_')[-1].lower()}", ok(err),
                       f"cuMemSetAccess({hk} id={hid}) -> {ename(err)}; cuMemGetAccess -> {ename(gerr)} flags={flags}")
                self.host_access |= ok(err)
            # Both at once, as the allocator would pass them.
            (err,) = cu.cuMemSetAccess(self.ptr, self.size, [
                access_desc("CU_MEM_LOCATION_TYPE_DEVICE", device),
                access_desc("CU_MEM_LOCATION_TYPE_HOST", 0)], 2)
            report(f"b.{label}.access_device+host", ok(err), f"cuMemSetAccess([DEVICE, HOST]) -> {ename(err)}")

    def free(self):
        for p, s in self.mapped:
            cu.cuMemUnmap(p, s)
        for h in self.handles:
            cu.cuMemRelease(h)
        if self.ptr:
            cu.cuMemAddressFree(self.ptr, self.size)


# ---------------------------------------------------------------- main's pool
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.syscall.restype = ctypes.c_long
PROT_NONE, PROT_RW = 0, 3
MAP_PRIVATE, MAP_ANON, MAP_FIXED = 0x02, 0x20, 0x10
MADV_HUGEPAGE, MPOL_BIND = 14, 2
SYS_MBIND = {"aarch64": 235, "x86_64": 237}[platform.machine()]


class NumaPool:
    """main's NumaHugePagePoolAllocator: 512 MiB-aligned mmap, mbind, THP, touch, cudaHostRegister."""
    HUGE = 512 << 20

    def __init__(self, nbytes, numa):
        self.size = (nbytes + self.HUGE - 1) // self.HUGE * self.HUGE
        base = libc.mmap(None, self.size + self.HUGE, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)
        aligned = (base + self.HUGE - 1) // self.HUGE * self.HUGE
        addr = libc.mmap(aligned, self.size, PROT_RW, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0)
        assert addr == aligned, "mmap fixed failed"
        mask = (ctypes.c_ulong * 16)()
        mask[numa // 64] = 1 << (numa % 64)
        r = libc.syscall(SYS_MBIND, ctypes.c_void_p(addr), ctypes.c_size_t(self.size), MPOL_BIND, mask,
                         ctypes.c_ulong(16 * 64 + 1), 0)
        report("pool.mbind", r == 0, f"mbind(node {numa}) -> {r} errno {ctypes.get_errno()}")
        libc.madvise(addr, self.size, MADV_HUGEPAGE)
        ctypes.memset(addr, 0, self.size)
        err = torch.cuda.cudart().cudaHostRegister(addr, self.size, 0)
        report("pool.cudaHostRegister", int(err) == 0, f"-> {err}")
        e, dptr = cu.cuMemHostGetDevicePointer(addr, 0)
        self.ptr, self.dptr = addr, int(dptr) if ok(e) else addr
        report("pool.device_ptr", ok(e), f"host {addr:#x} device {self.dptr:#x} (equal: {addr == self.dptr})")


# ---------------------------------------------------------------- helpers
class CAI:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def wrap(ptr, n):
    try:
        t = torch.as_tensor(CAI(ptr, n), device="cuda")
        return t if t.data_ptr() == ptr else None
    except Exception as e:
        print(f"torch cannot wrap {ptr:#x}: {e}")
        return None


def cpu_touch_in_child(ptr):
    """Writes one page through ptr in a forked child: 0 = ok, else the signal/exit code."""
    pid = os.fork()
    if pid == 0:
        try:
            buf = (ctypes.c_ubyte * 4096)(*([0x5A] * 4096))
            ctypes.memmove(ptr, buf, 4096)
            os._exit(0 if ctypes.string_at(ptr, 1) == b"\x5a" else 3)
        except BaseException:
            os._exit(4)
    _, status = os.waitpid(pid, 0)
    return status


def maps_entry(ptr):
    for line in open("/proc/self/maps"):
        lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
        if lo <= ptr < hi:
            return line.strip()
    return "not in /proc/self/maps"


def gpu_read_bw(label, src_ptr, n, iters):
    """Copy-engine (cuMemcpyDtoDAsync) and SM (elementwise kernel) read bandwidth from src into cudaMalloc."""
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = {}
    for _ in range(2):
        cu.cuMemcpyDtoDAsync(dst.data_ptr(), src_ptr, n, stream.cuda_stream)
    s.record()
    for _ in range(iters):
        (err,) = cu.cuMemcpyDtoDAsync(dst.data_ptr(), src_ptr, n, stream.cuda_stream)
    e.record()
    torch.cuda.synchronize()
    res["CE"] = n * iters / (s.elapsed_time(e) / 1e3) / 1e9 if ok(err) else None
    src = wrap(src_ptr, n)
    if src is not None:
        a, b = src.view(torch.int64), dst.view(torch.int64)
        torch.bitwise_not(a, out=b)
        s.record()
        for _ in range(iters):
            torch.bitwise_not(a, out=b)
        e.record()
        torch.cuda.synchronize()
        res["SM"] = n * iters / (s.elapsed_time(e) / 1e3) / 1e9
    print(f"    {label:28s} GPU read GB/s: CE={res.get('CE')} SM={res.get('SM')}", flush=True)
    return res


def cpu_write_bw(label, dst_ptr, n, threads):
    src = (ctypes.c_ubyte * (64 << 20))()
    ctypes.memset(src, 0x33, len(src))
    per = n // threads

    def work(t):
        off = t * per
        while off < (t + 1) * per:
            k = min(len(src), (t + 1) * per - off)
            ctypes.memmove(dst_ptr + off, src, k)
            off += k

    work(0) if threads == 1 else None  # warm
    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    bw = n / (time.perf_counter() - t0) / 1e9
    print(f"    {label:28s} CPU write GB/s ({threads} thr): {bw:.1f}", flush=True)
    return bw


# ---------------------------------------------------------------- (f) memory accounting
def node_free_gib(numa):
    try:
        with open(f"/sys/devices/system/node/node{numa}/meminfo") as f:
            for line in f:
                if "MemFree" in line:
                    return int(line.split()[-2]) / (1 << 20)
    except OSError:
        return float("nan")
    return float("nan")


def accounting(device, numa, mode):
    """Run in a fresh process: does HOST_NUMA on the GPU node count in cudaMemGetInfo, can it reuse released
    DEVICE memory, and what happens when the node is short (clean error or OOM kill)?"""
    gib = 1 << 30
    free0, total = torch.cuda.mem_get_info()
    print(f"    start: cudaMemGetInfo free {free0 / gib:.2f} / {total / gib:.2f} GiB, node{numa} MemFree "
          f"{node_free_gib(numa):.2f} GiB", flush=True)
    if mode == "count":
        r = VmmRange("acct_1g", "CU_MEM_LOCATION_TYPE_HOST_NUMA", numa, 4 * gib, device, True, chunk_bytes=gib)
        free1, _ = torch.cuda.mem_get_info()
        drop = (free0 - free1) / gib
        report("f.host_numa_counts_in_meminfo", drop > 3.5, f"4 GiB HOST_NUMA: cudaMemGetInfo free dropped "
               f"{drop:.2f} GiB, node{numa} MemFree {node_free_gib(numa):.2f} GiB")
        r.free()
        free2, _ = torch.cuda.mem_get_info()
        report("f.host_numa_release_returns", (free2 - free1) / gib > 3.5, f"after release free {free2 / gib:.2f} GiB")
    elif mode == "reuse":
        size = (free0 - 4 * gib) // gib * gib
        d = VmmRange("acct_device_big", "CU_MEM_LOCATION_TYPE_DEVICE", device, size, device, False, chunk_bytes=gib)
        print(f"    DEVICE {size / gib:.0f} GiB held: free {torch.cuda.mem_get_info()[0] / gib:.2f} GiB, node{numa} "
              f"MemFree {node_free_gib(numa):.2f} GiB", flush=True)
        d.free()
        try:
            h = VmmRange("acct_reuse", "CU_MEM_LOCATION_TYPE_HOST_NUMA", numa, size, device, True, chunk_bytes=gib)
            report("f.released_device_reusable_as_host_numa", True, f"{size / gib:.0f} GiB HOST_NUMA after DEVICE release; "
                   f"free {torch.cuda.mem_get_info()[0] / gib:.2f} GiB")
            h.free()
        except RuntimeError as e:
            report("f.released_device_reusable_as_host_numa", False, str(e))
    elif mode == "short":
        size = (free0 + 8 * gib) // gib * gib
        try:
            h = VmmRange("acct_short", "CU_MEM_LOCATION_TYPE_HOST_NUMA", numa, size, device, True, chunk_bytes=gib)
            report("f.short_node_clean_error", False, f"{size / gib:.0f} GiB > free succeeded (spilled?) node{numa} "
                   f"MemFree {node_free_gib(numa):.2f} GiB")
            h.free()
        except RuntimeError as e:
            report("f.short_node_clean_error", True, f"clean failure: {e}")


def accounting_in_child(device, numa, mode):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--device", str(device), "--accounting", mode,
                        "--numa", str(numa)], capture_output=True, text=True, timeout=900)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        report(f"f.{mode}.child", False, f"child exit {r.returncode} (negative = signal; -9 = OOM kill?) "
               f"{r.stderr[-400:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-fork", action="store_true", help="skip the child-process dereference check")
    ap.add_argument("--accounting", choices=("count", "reuse", "short"), help=argparse.SUPPRESS)
    ap.add_argument("--numa", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    n = int(args.gib * (1 << 30))
    torch.cuda.set_device(args.device)
    torch.zeros(1, device="cuda")  # primary context current on this thread
    (err,) = cu.cuInit(0)
    if args.accounting:
        accounting(args.device, args.numa, args.accounting)
        for s, t, m in RESULTS:
            print(f"  {t} {s}")
        return
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, driver {cu.cuDriverGetVersion()[1]}, "
          f"host {platform.node()} {platform.machine()}", flush=True)

    dev, mem_numa, cpu_numa = find_numa(args.device)
    numa = mem_numa if mem_numa >= 0 else cpu_numa

    # (b) 2 MiB chunk first, then the 1 GiB range used for (c)-(e).
    try:
        VmmRange("2mib", "CU_MEM_LOCATION_TYPE_HOST_NUMA", numa, 2 << 20, args.device, True).free()
    except RuntimeError as e:
        report("b.2mib", False, str(e))
    try:
        vmm = VmmRange("hbm_host_numa", "CU_MEM_LOCATION_TYPE_HOST_NUMA", numa, n, args.device, True)
    except RuntimeError as e:
        report("b", False, f"HOST_NUMA VMM range unusable: {e}")
        vmm = None

    # (c) CPU write through the VA, GPU readback.
    if vmm is not None:
        print(f"    /proc/self/maps for VA: {maps_entry(vmm.ptr)}", flush=True)
        status = 0 if args.no_fork else cpu_touch_in_child(vmm.ptr)
        child_ok = status == 0
        report("c.cpu_deref_child", child_ok if not args.no_fork else None,
               f"forked child wrote through VA {vmm.ptr:#x}: wait status {status}"
               + ("" if child_ok else " (signal 11 = not CPU-mapped; DONTFORK mappings also fail here: try --no-fork)"))
        if child_ok:
            pat = torch.arange(n, dtype=torch.int64).remainder_(251).to(torch.uint8)  # host tensor
            ctypes.memmove(vmm.ptr, pat.data_ptr(), n)
            back = torch.empty(n, dtype=torch.uint8, device="cuda")
            (err,) = cu.cuMemcpyDtoD(back.data_ptr(), vmm.ptr, n)
            torch.cuda.synchronize()
            same = ok(err) and torch.equal(back.cpu(), pat)
            report("c.cpu_write_gpu_readback", same, f"CPU memmove {n} B, GPU copy back -> equal={same}")
            # And GPU write, CPU read.
            back.fill_(7)
            cu.cuMemcpyDtoD(vmm.ptr, back.data_ptr(), 4096)
            torch.cuda.synchronize()
            report("c.gpu_write_cpu_read", ctypes.string_at(vmm.ptr, 4096) == b"\x07" * 4096, "4 KiB")
            report("c.host_ptr", None, "host pointer == VMM VA (no separate mapping needed)")

    # (d)/(e) bandwidth.
    print("(d) GPU read bandwidth from 1 GiB sources into cudaMalloc", flush=True)
    base = torch.empty(n, dtype=torch.uint8, device="cuda")
    bw = {"cudaMalloc": gpu_read_bw("cudaMalloc", base.data_ptr(), n, args.iters)}
    pool = None
    if mem_numa >= 0:
        pool = NumaPool(n, mem_numa)
        bw["pool"] = gpu_read_bw(f"main pool (node {mem_numa})", pool.dptr, n, args.iters)
    if vmm is not None:
        bw["vmm"] = gpu_read_bw(f"VMM HOST_NUMA (node {numa})", vmm.ptr, n, args.iters)
    try:
        lp = VmmRange("lpddr_host_numa_ref", "CU_MEM_LOCATION_TYPE_HOST_NUMA", cpu_numa, n, args.device, False)
        gpu_read_bw(f"ref: VMM HOST_NUMA LPDDR ({cpu_numa})", lp.ptr, n, args.iters)
        lp.free()
    except RuntimeError as e:
        print(f"    LPDDR reference skipped: {e}")
    if "vmm" in bw:
        for k in ("CE", "SM"):
            ref, got = bw["cudaMalloc"].get(k), bw["vmm"].get(k)
            if ref and got:
                report(f"d.gpu_read_{k}", got >= 0.9 * ref, f"VMM HOST_NUMA {got:.0f} vs cudaMalloc {ref:.0f} GB/s "
                       f"({100 * got / ref:.0f}%; PASS >= 90%)")
    print("(e) CPU write bandwidth", flush=True)
    for thr in (1, 8):
        p = cpu_write_bw("main pool", pool.ptr, n, thr) if pool else None
        v = cpu_write_bw("VMM HOST_NUMA", vmm.ptr, n, thr) if vmm is not None and RESULTS and any(
            s == "c.cpu_write_gpu_readback" and t == "PASS" for s, t, _ in RESULTS) else None
        if p and v:
            report(f"e.cpu_write_{thr}thr", v >= 0.8 * p, f"VMM {v:.1f} vs pool {p:.1f} GB/s (PASS >= 80%)")
    if vmm is not None:
        vmm.free()
    del base
    torch.cuda.empty_cache()
    print("(f) memory accounting, each in a fresh process", flush=True)
    for mode in ("count", "reuse", "short"):
        accounting_in_child(args.device, numa, mode)
    print("\nSUMMARY")
    for s, t, m in RESULTS:
        if t != "INFO":
            print(f"  {t} {s}")


if __name__ == "__main__":
    main()
