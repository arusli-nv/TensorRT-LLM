#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: GDRCopy pins of single 2 MiB cuMem chunks (gpuDirectRDMACapable=1) inside one VA reservation, and
whether those per-chunk pins survive remapping a chunk to fresh physical memory.

Usage: python3 gdr_remap_probe.py [--device 0] [--remap-iters 20] [--coherence-iters 200] [--n-pins 300]
Steps (each runs in its own child process so a crash in one step does not hide the others):
  1 per-chunk pin+map of 4 chunks: copy_to_mapping + GPU readback, gdr_get_info va == pinned address
  2 remap chunk 1 N times (unpin, cuMemUnmap/Release, cuMemCreate/Map/SetAccess, repin), neighbours checked
  3 remap chunk 1 to a shared scratch handle (also mapped at a second VA) without pinning, then back to a fresh
    allocation and pin
  4 coherence: GPU kernel / cuMemcpyDtoD reads immediately after copy_to_mapping, no extra fence
  5 N concurrent pins of N separate 2 MiB allocations, per-allocation pin/map time
ctypes gdrapi bindings and cuMem helpers are copied from gdr_vmm_probe.py.
"""
import argparse
import ctypes
import os
import statistics
import subprocess
import sys
import time

import torch

try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu

MiB = 1 << 20
CHUNK = 2 * MiB
RESULTS = []


def report(step, ok, msg):
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    RESULTS.append((step, tag, msg))
    print(f"[{tag}] {step}: {msg}", flush=True)


def print_summary(title="SUMMARY"):
    print(f"\n{title}", flush=True)
    for step, tag, msg in RESULTS:
        print(f"  {tag} {step}: {msg}", flush=True)


# ---------------------------------------------------------------- CUDA driver helpers
def ename(err):
    try:
        _, name = cu.cuGetErrorName(err)
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return str(err)


class CuErr(RuntimeError):
    pass


def chk(res, what):
    if isinstance(res, tuple):
        err, rest = res[0], res[1:]
    else:
        err, rest = res, ()
    if int(err) != 0:
        raise CuErr(f"{what} -> {ename(err)}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def setup_cuda(device):
    torch.cuda.set_device(device)
    torch.zeros(1, device="cuda")
    chk(cu.cuInit(0), "cuInit")
    dev = chk(cu.cuDeviceGet(device), "cuDeviceGet")
    ctx = chk(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
    chk(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return dev


def make_prop(device, rdma=True):
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    if rdma:
        prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def set_access(ptr, size, device):
    desc = cu.CUmemAccessDesc()
    desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    chk(cu.cuMemSetAccess(ptr, size, [desc], 1), "cuMemSetAccess")


def check_granularity(device):
    gran = int(chk(cu.cuMemGetAllocationGranularity(
        make_prop(device), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity"))
    if CHUNK % gran:
        raise CuErr(f"granularity {gran} does not divide the 2 MiB chunk")
    return gran


class Chunks:
    """n chunks of CHUNK bytes in one VA reservation, each its own cuMemCreate allocation."""

    def __init__(self, device, n):
        self.device, self.n = device, n
        self.prop = make_prop(device)
        self.handles = [None] * n
        self.base = int(chk(cu.cuMemAddressReserve(n * CHUNK, CHUNK, 0, 0), "cuMemAddressReserve"))
        for i in range(n):
            self.map_new(i)

    def va(self, i):
        return self.base + i * CHUNK

    def map_new(self, i):
        h = chk(cu.cuMemCreate(CHUNK, self.prop, 0), f"cuMemCreate chunk {i}")
        try:
            chk(cu.cuMemMap(self.va(i), CHUNK, 0, h, 0), f"cuMemMap chunk {i}")
        except CuErr:
            cu.cuMemRelease(h)
            raise
        self.handles[i] = h
        set_access(self.va(i), CHUNK, self.device)

    def unmap_release(self, i):
        chk(cu.cuMemUnmap(self.va(i), CHUNK), f"cuMemUnmap chunk {i}")
        chk(cu.cuMemRelease(self.handles[i]), f"cuMemRelease chunk {i}")
        self.handles[i] = None

    def free(self):
        for i, h in enumerate(self.handles):
            if h is not None:
                cu.cuMemUnmap(self.va(i), CHUNK)
                cu.cuMemRelease(h)
                self.handles[i] = None
        cu.cuMemAddressFree(self.base, self.n * CHUNK)


def gpu_copy_read(ptr, n):
    """cuMemcpyDtoD (copy engine) of n bytes at device VA ptr into a fresh torch CUDA tensor."""
    out = torch.empty(n, dtype=torch.uint8, device="cuda")
    chk(cu.cuMemcpyDtoD(out.data_ptr(), ptr, n), "cuMemcpyDtoD readback")
    chk(cu.cuCtxSynchronize(), "cuCtxSynchronize")
    return out


class _CudaArray:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3,
                                         "strides": None}


def gpu_view(ptr, n):
    """Zero-copy torch uint8 view of device VA ptr; ops on it run as GPU kernels reading the memory directly."""
    return torch.as_tensor(_CudaArray(ptr, n), device="cuda")


def gpu_kernel_count_eq(ptr, n, value):
    """Number of bytes equal to value, computed by a GPU kernel reading ptr directly."""
    return int((gpu_view(ptr, n) == value).sum().item())


# ---------------------------------------------------------------- GDRCopy via ctypes
class gdr_mh_t(ctypes.Structure):
    _fields_ = [("h", ctypes.c_ulong)]


class gdr_info_t(ctypes.Structure):
    # Layout from gdrwrap.h; mapping_type (gdrcopy >= 2.4) plus padding so a newer lib cannot overrun.
    _fields_ = [
        ("va", ctypes.c_uint64),
        ("mapped_size", ctypes.c_uint64),
        ("page_size", ctypes.c_uint32),
        ("tm_cycles", ctypes.c_uint64),
        ("cycles_per_ms", ctypes.c_uint32),
        ("mapped", ctypes.c_uint, 1),
        ("wc_mapping", ctypes.c_uint, 1),
        ("mapping_type", ctypes.c_int),
        ("_pad", ctypes.c_uint8 * 64),
    ]


class Pin:
    """One gdr pin + map of [addr, addr+size). cpu is the CPU pointer for addr."""

    def __init__(self, mh, mva, cpu, info, addr, size, pin_ms, map_ms):
        self.mh, self.mva, self.cpu, self.info = mh, mva, cpu, info
        self.addr, self.size, self.pin_ms, self.map_ms = addr, size, pin_ms, map_ms


class Gdr:
    def __init__(self):
        last = None
        self.lib = None
        for name in ("libgdrapi.so", "libgdrapi.so.2"):
            try:
                self.lib = ctypes.CDLL(name, use_errno=True)
                self.libname = name
                break
            except OSError as e:
                last = e
        if self.lib is None:
            raise OSError(f"dlopen libgdrapi failed: {last}")
        L = self.lib
        vp, sz, u64, u32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32
        L.gdr_open.restype = vp
        L.gdr_open.argtypes = []
        L.gdr_close.argtypes = [vp]
        L.gdr_pin_buffer.argtypes = [vp, ctypes.c_ulong, sz, u64, u32, ctypes.POINTER(gdr_mh_t)]
        L.gdr_unpin_buffer.argtypes = [vp, gdr_mh_t]
        L.gdr_get_info.argtypes = [vp, gdr_mh_t, ctypes.POINTER(gdr_info_t)]
        L.gdr_map.argtypes = [vp, gdr_mh_t, ctypes.POINTER(vp), sz]
        L.gdr_unmap.argtypes = [vp, gdr_mh_t, vp, sz]
        L.gdr_runtime_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
        L.gdr_runtime_get_version.restype = None
        L.gdr_driver_get_version.argtypes = [vp] + [ctypes.POINTER(ctypes.c_int)] * 2
        L.gdr_driver_get_version.restype = None
        L.gdr_copy_to_mapping.argtypes = [gdr_mh_t, vp, vp, sz]
        L.gdr_copy_from_mapping.argtypes = [gdr_mh_t, vp, vp, sz]
        for f in ("gdr_close", "gdr_pin_buffer", "gdr_unpin_buffer", "gdr_get_info", "gdr_map", "gdr_unmap",
                  "gdr_copy_to_mapping", "gdr_copy_from_mapping"):
            getattr(L, f).restype = ctypes.c_int
        self.g = L.gdr_open()
        if not self.g:
            raise OSError(f"gdr_open returned NULL (errno {ctypes.get_errno()})")

    def versions(self):
        a, b, c, d = (ctypes.c_int() for _ in range(4))
        self.lib.gdr_runtime_get_version(ctypes.byref(a), ctypes.byref(b))
        self.lib.gdr_driver_get_version(self.g, ctypes.byref(c), ctypes.byref(d))
        return f"lib {a.value}.{b.value}, driver(api) {c.value}.{d.value}"

    def pin_map(self, addr, size):
        """Return a Pin or raise OSError with the return code and errno."""
        mh = gdr_mh_t()
        ctypes.set_errno(0)
        t0 = time.perf_counter()
        rc = self.lib.gdr_pin_buffer(self.g, addr, size, 0, 0, ctypes.byref(mh))
        t1 = time.perf_counter()
        if rc != 0:
            e = ctypes.get_errno()
            raise OSError(f"gdr_pin_buffer(0x{addr:x}, {size}) rc={rc} errno={e} ({os.strerror(e)})")
        mva = ctypes.c_void_p()
        rc = self.lib.gdr_map(self.g, mh, ctypes.byref(mva), size)
        t2 = time.perf_counter()
        if rc != 0:
            e = ctypes.get_errno()
            self.lib.gdr_unpin_buffer(self.g, mh)
            raise OSError(f"gdr_map(0x{addr:x}, {size}) rc={rc} errno={e} ({os.strerror(e)})")
        info = self.info(mh)
        cpu = mva.value + (info.va - addr)
        return Pin(mh, mva.value, cpu, info, addr, size, (t1 - t0) * 1e3, (t2 - t1) * 1e3)

    def info(self, mh):
        info = gdr_info_t()
        rc = self.lib.gdr_get_info(self.g, mh, ctypes.byref(info))
        if rc != 0:
            raise OSError(f"gdr_get_info rc={rc}")
        return info

    def unmap_unpin(self, p):
        r1 = self.lib.gdr_unmap(self.g, p.mh, p.mva, p.size)
        r2 = self.lib.gdr_unpin_buffer(self.g, p.mh)
        return r1, r2

    def copy_to(self, p, host, off=0, n=None):
        n = p.size - off if n is None else n
        return self.lib.gdr_copy_to_mapping(p.mh, p.cpu + off, host.data_ptr(), n)

    def close(self):
        self.lib.gdr_close(self.g)


def info_str(info):
    return (f"va=0x{info.va:x} mapped_size={info.mapped_size} page_size={info.page_size} mapped={info.mapped} "
            f"wc_mapping={info.wc_mapping}")


def host_pattern(value):
    return torch.full((CHUNK,), value & 0xFF, dtype=torch.uint8)


def write_verify(gdr, p, value):
    """copy_to_mapping a constant byte pattern into the whole pin, then GPU readback. Returns (ok, detail)."""
    rc = gdr.copy_to(p, host_pattern(value))
    if rc != 0:
        return False, f"copy_to_mapping rc={rc}"
    got = gpu_copy_read(p.addr, p.size)
    nbad = int((got != (value & 0xFF)).sum().item())
    return nbad == 0, f"rc=0, {nbad} bytes differ from 0x{value & 0xFF:02x}"


def stats_ms(xs):
    return (f"n={len(xs)} min={min(xs):.3f} median={statistics.median(xs):.3f} mean={statistics.mean(xs):.3f} "
            f"max={max(xs):.3f} ms")


# ---------------------------------------------------------------- shared setup: 4 chunks, each pinned separately
def pin_all(gdr, ch, step):
    pins = []
    for i in range(ch.n):
        p = gdr.pin_map(ch.va(i), CHUNK)
        pins.append(p)
        report(f"{step}.pin_chunk{i}", True, f"addr=0x{p.addr:x} pin {p.pin_ms:.3f} ms map {p.map_ms:.3f} ms, "
               f"{info_str(p.info)}")
    return pins


def check_neighbours(gdr, pins, skip, value, step):
    bad = []
    for i, p in enumerate(pins):
        if i == skip or p is None:
            continue
        ok, detail = write_verify(gdr, p, value + i)
        if not ok:
            bad.append(f"chunk{i}: {detail}")
    if bad:
        report(step, False, "; ".join(bad))
    return not bad


def release_all(gdr, pins, ch):
    for p in pins:
        if p is not None:
            gdr.unmap_unpin(p)
    ch.free()


# ---------------------------------------------------------------- steps
def step1(gdr, device, args):
    ch = Chunks(device, 4)
    report("1.alloc", True, f"4 x 2 MiB cuMemCreate (gpuDirectRDMACapable=1) in one reservation at 0x{ch.base:x}")
    pins = []
    try:
        pins = pin_all(gdr, ch, "1")
        for i, p in enumerate(pins):
            ok, detail = write_verify(gdr, p, 0x10 + i)
            report(f"1.copy_readback_chunk{i}", ok, detail)
        for i, p in enumerate(pins):
            off = p.info.va - p.addr
            report(f"1.info_va_offset0_chunk{i}", off == 0,
                   f"info.va=0x{p.info.va:x} pinned=0x{p.addr:x} offset={off} mapped_size={p.info.mapped_size}")
        # Cross-check: each chunk still holds its own pattern (no aliasing between per-chunk mappings).
        for i in range(4):
            n = gpu_kernel_count_eq(ch.va(i), CHUNK, 0x10 + i)
            report(f"1.no_alias_chunk{i}", n == CHUNK, f"{n}/{CHUNK} bytes still 0x{0x10 + i:02x}")
        r = [gdr.unmap_unpin(p) for p in pins]
        pins = []
        report("1.unmap_unpin", all(x == (0, 0) for x in r), f"rc={r}")
    finally:
        release_all(gdr, pins, ch)


def step2(gdr, device, args):
    ch = Chunks(device, 4)
    pins = []
    k = 1
    first_fail = None
    pin_ms, total_ms = [], []
    try:
        pins = pin_all(gdr, ch, "2.setup")
        for it in range(args.remap_iters):
            stage = "start"
            try:
                t0 = time.perf_counter()
                stage = "gdr_unmap_unpin"
                r = gdr.unmap_unpin(pins[k])
                pins[k] = None
                if r != (0, 0):
                    raise OSError(f"gdr unmap/unpin rc={r}")
                stage = "cuMemUnmap+Release"
                ch.unmap_release(k)
                stage = "cuMemCreate+Map+SetAccess"
                ch.map_new(k)
                stage = "pin_map"
                pins[k] = gdr.pin_map(ch.va(k), CHUNK)
                total_ms.append((time.perf_counter() - t0) * 1e3)
                pin_ms.append(pins[k].pin_ms)
                stage = "write_verify"
                off = pins[k].info.va - pins[k].addr
                if off != 0:
                    raise OSError(f"info.va offset {off} != 0")
                ok, detail = write_verify(gdr, pins[k], 0x40 + it)
                if not ok:
                    raise OSError(detail)
                stage = "neighbours"
                if not check_neighbours(gdr, pins, k, 0x80 + 4 * it, f"2.iter{it}.neighbours"):
                    raise OSError("neighbour chunk mapping failed")
            except (OSError, CuErr) as e:
                if first_fail is None:
                    first_fail = it
                report(f"2.iter{it}", False, f"failed at stage {stage}: {e}")
                if stage not in ("write_verify", "neighbours"):
                    break  # chunk k is not pinned and mapped, so the cycle cannot continue
        n_ok = args.remap_iters if first_fail is None else first_fail
        report("2.remap_cycles", first_fail is None,
               f"{args.remap_iters} iterations requested, first failure at iteration "
               f"{first_fail if first_fail is not None else 'none'} (0-based), clean iterations before it: {n_ok}")
        if pin_ms:
            report("2.repin_ms", None, f"gdr_pin_buffer after remap: {stats_ms(pin_ms)}")
            report("2.cycle_ms", None, f"full unpin+remap+repin cycle: {stats_ms(total_ms)}")
    finally:
        release_all(gdr, pins, ch)


def step3(gdr, device, args):
    ch = Chunks(device, 4)
    pins = []
    k = 1
    scratch_h = None
    scratch_va = None
    try:
        pins = pin_all(gdr, ch, "3.setup")
        scratch_h = chk(cu.cuMemCreate(CHUNK, ch.prop, 0), "cuMemCreate scratch")
        scratch_va = int(chk(cu.cuMemAddressReserve(CHUNK, CHUNK, 0, 0), "reserve scratch VA"))
        chk(cu.cuMemMap(scratch_va, CHUNK, 0, scratch_h, 0), "map scratch at its own VA")
        set_access(scratch_va, CHUNK, device)
        chk(cu.cuMemsetD8(scratch_va, 0x5A, CHUNK), "memset scratch")
        chk(cu.cuCtxSynchronize(), "sync")
        report("3.scratch", True, f"shared scratch handle mapped at 0x{scratch_va:x}, filled 0x5A")
        for cyc in range(args.shared_cycles):
            tag = f"3.cycle{cyc}"
            stage = "start"
            try:
                stage = "gdr_unmap_unpin"
                r = gdr.unmap_unpin(pins[k])
                pins[k] = None
                if r != (0, 0):
                    raise OSError(f"gdr unmap/unpin rc={r}")
                stage = "cuMemUnmap+Release"
                ch.unmap_release(k)
                stage = "map scratch at chunk VA"
                chk(cu.cuMemMap(ch.va(k), CHUNK, 0, scratch_h, 0), "cuMemMap scratch at chunk VA")
                set_access(ch.va(k), CHUNK, device)
                stage = "alias check"
                n = gpu_kernel_count_eq(ch.va(k), CHUNK, 0x5A)
                if n != CHUNK:
                    raise OSError(f"chunk VA reads {n}/{CHUNK} bytes of scratch 0x5A")
                stage = "neighbours while shared"
                if not check_neighbours(gdr, pins, k, 0x20 + cyc, f"{tag}.neighbours_shared"):
                    raise OSError("neighbour chunk mapping failed while chunk is shared")
                stage = "unmap scratch at chunk VA"
                chk(cu.cuMemUnmap(ch.va(k), CHUNK), "cuMemUnmap scratch at chunk VA")
                stage = "cuMemCreate+Map+SetAccess fresh"
                ch.map_new(k)
                stage = "pin_map fresh"
                pins[k] = gdr.pin_map(ch.va(k), CHUNK)
                stage = "write_verify fresh"
                ok, detail = write_verify(gdr, pins[k], 0xA0 + cyc)
                if not ok:
                    raise OSError(detail)
                stage = "scratch untouched"
                n = gpu_kernel_count_eq(scratch_va, CHUNK, 0x5A)
                if n != CHUNK:
                    raise OSError(f"scratch reads {n}/{CHUNK} bytes of 0x5A after the fresh chunk was written")
                stage = "neighbours after"
                if not check_neighbours(gdr, pins, k, 0x60 + cyc, f"{tag}.neighbours_after"):
                    raise OSError("neighbour chunk mapping failed after re-pin")
                report(tag, True, f"shared-scratch map -> fresh alloc -> pin ok (pin {pins[k].pin_ms:.3f} ms, "
                       f"info.va offset {pins[k].info.va - pins[k].addr})")
            except (OSError, CuErr) as e:
                report(tag, False, f"failed at stage {stage}: {e}")
                break
    finally:
        release_all(gdr, pins, ch)
        if scratch_va is not None:
            cu.cuMemUnmap(scratch_va, CHUNK)
            cu.cuMemAddressFree(scratch_va, CHUNK)
        if scratch_h is not None:
            cu.cuMemRelease(scratch_h)


def step4(gdr, device, args):
    ch = Chunks(device, 4)
    pins = []
    try:
        pins = pin_all(gdr, ch, "4.setup")
        p = pins[0]
        g = torch.Generator().manual_seed(1234)
        for size in (64, 4096, CHUNK):
            bad_kernel = bad_copy = bad_tail = 0
            for it in range(args.coherence_iters):
                host = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=g)
                # Last 8 bytes carry the iteration number so a stale tail is caught too.
                host[-8:] = torch.tensor(list((it + 1).to_bytes(8, "little")), dtype=torch.uint8)
                ref = host.to("cuda")
                torch.cuda.synchronize()
                rc = gdr.copy_to(p, host, 0, size)
                if rc != 0:
                    report(f"4.size{size}", False, f"copy_to_mapping rc={rc} at iteration {it}")
                    break
                # GPU kernel reading the pinned memory directly, launched right after the CPU store.
                same_k = bool(torch.equal(gpu_view(p.addr, size), ref))
                tail_k = bool(torch.equal(gpu_view(p.addr + size - 8, 8), ref[-8:]))
                # Copy-engine read right after.
                same_c = bool(torch.equal(gpu_copy_read(p.addr, size), ref))
                bad_kernel += not same_k
                bad_tail += not tail_k
                bad_copy += not same_c
            ok = bad_kernel == 0 and bad_copy == 0 and bad_tail == 0
            report(f"4.size{size}", ok, f"{args.coherence_iters} iters: kernel-read mismatches {bad_kernel}, "
                   f"tail-word mismatches {bad_tail}, cuMemcpyDtoD mismatches {bad_copy}")
        # Same check on a cuMemAlloc buffer (main's GDR path) for comparison.
        ptr = int(chk(cu.cuMemAlloc(CHUNK), "cuMemAlloc"))
        try:
            q = gdr.pin_map(ptr, CHUNK)
            bad = 0
            for it in range(args.coherence_iters):
                host = torch.randint(0, 256, (CHUNK,), dtype=torch.uint8, generator=g)
                ref = host.to("cuda")
                torch.cuda.synchronize()
                gdr.copy_to(q, host)
                bad += not bool(torch.equal(gpu_view(ptr, CHUNK), ref))
            gdr.unmap_unpin(q)
            report("4.cumemalloc_kernel_read", bad == 0, f"{args.coherence_iters} iters 2 MiB: mismatches {bad}")
        finally:
            cu.cuMemFree(ptr)
    finally:
        release_all(gdr, pins, ch)


def step5(gdr, device, args):
    n = args.n_pins
    ch = Chunks(device, n)
    report("5.alloc", True, f"{n} x 2 MiB cuMemCreate (gpuDirectRDMACapable=1) in one reservation at 0x{ch.base:x}")
    pins = []
    try:
        for i in range(n):
            try:
                pins.append(gdr.pin_map(ch.va(i), CHUNK))
            except OSError as e:
                report("5.pin_all", False, f"pin {i} of {n} failed: {e}")
                break
        else:
            report("5.pin_all", True, f"{n} concurrent pins + maps held")
        if pins:
            report("5.pin_ms", None, f"gdr_pin_buffer per 2 MiB allocation: {stats_ms([p.pin_ms for p in pins])}")
            report("5.map_ms", None, f"gdr_map per 2 MiB allocation: {stats_ms([p.map_ms for p in pins])}")
            report("5.pin_ms_first_last10", None,
                   f"first 10 mean {statistics.mean(p.pin_ms for p in pins[:10]):.3f} ms, last 10 mean "
                   f"{statistics.mean(p.pin_ms for p in pins[-10:]):.3f} ms")
            offs = sum(p.info.va != p.addr for p in pins)
            report("5.info_va_offset0", offs == 0, f"{offs}/{len(pins)} pins with info.va != pinned address")
            bad = []
            for i, p in enumerate(pins):
                ok, detail = write_verify(gdr, p, i)
                if not ok:
                    bad.append(f"{i}: {detail}")
            report("5.copy_readback_all", not bad, f"{len(pins) - len(bad)}/{len(pins)} ok" +
                   (f"; first bad {bad[:3]}" if bad else ""))
            t0 = time.perf_counter()
            r = [gdr.unmap_unpin(p) for p in pins]
            dt = (time.perf_counter() - t0) * 1e3
            pins = []
            report("5.unmap_unpin_all", all(x == (0, 0) for x in r),
                   f"{len(r)} unmap+unpin in {dt:.1f} ms ({dt / len(r):.3f} ms each)")
    finally:
        release_all(gdr, pins, ch)


STEPS = {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5}


def run_child(args):
    step = args.only
    setup_cuda(args.device)
    gdr = Gdr()
    try:
        check_granularity(args.device)
        STEPS[step](gdr, args.device, args)
    except (OSError, CuErr) as e:
        report(f"{step}.error", False, f"{type(e).__name__}: {e}")
        return 1
    finally:
        gdr.close()
        print_summary(f"STEP {step} SUMMARY")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--remap-iters", type=int, default=20)
    ap.add_argument("--shared-cycles", type=int, default=5)
    ap.add_argument("--coherence-iters", type=int, default=200)
    ap.add_argument("--n-pins", type=int, default=300)
    ap.add_argument("--steps", default="1,2,3,4,5")
    ap.add_argument("--only", choices=sorted(STEPS), default=None, help="internal: run one step in this process")
    args = ap.parse_args()

    if args.only:
        sys.exit(run_child(args))

    print(f"host={os.uname().nodename} torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={torch.cuda.get_device_name(args.device)}", flush=True)
    try:
        g = Gdr()
        report("0.gdr_open", True, f"{g.libname}: {g.versions()}")
        g.close()
    except OSError as e:
        report("0.gdr_open", False, str(e))
        print_summary()
        sys.exit(1)
    for step in args.steps.split(","):
        cmd = [sys.executable, os.path.abspath(__file__), "--device", str(args.device), "--only", step,
               "--remap-iters", str(args.remap_iters), "--shared-cycles", str(args.shared_cycles),
               "--coherence-iters", str(args.coherence_iters), "--n-pins", str(args.n_pins)]
        print(f"\n===== step {step}: {' '.join(cmd)}", flush=True)
        try:
            rc = subprocess.run(cmd, timeout=900).returncode
            if rc < 0:
                report(f"{step}.child", False, f"killed by signal {-rc} (7=SIGBUS, 11=SIGSEGV); see output above")
            else:
                report(f"{step}.child", rc == 0, f"exit code {rc}")
        except subprocess.TimeoutExpired:
            report(f"{step}.child", False, "timed out after 900 s")
    print_summary("PARENT SUMMARY (per-step PASS/FAIL detail is in each STEP SUMMARY above)")


if __name__ == "__main__":
    main()
