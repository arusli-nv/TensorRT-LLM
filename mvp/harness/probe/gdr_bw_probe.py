#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: CPU write bandwidth into GDRCopy-mapped memory at EPLB update sizes, cuMemAlloc vs cuMem VMM.

Cases (each target pinned + mapped through GDRCopy; source is a pageable host buffer):
  A  cuMemAlloc of --mib MiB + gdr_pin_buffer + gdr_map (main's HostAccessibleDeviceAllocator way).
  B  one cuMemCreate of --mib MiB (DEVICE, gpuDirectRDMACapable=1), one gdr pin/map.
  C  --mib/2 separate 2 MiB cuMemCreate allocations (gpuDirectRDMACapable=1), each its own gdr pin/map.
Per case: gdr_copy_to_mapping and plain memcpy into the mapped pointer, with each thread count splitting the
buffer (A/B: contiguous slices; C: whole pieces per thread). One untimed warm-up iteration ("first"), then
--iters timed iterations (best, median). GPU readback verifies each method once per case.

Usage: python3 gdr_bw_probe.py [--device 0] [--mib 38] [--iters 5] [--threads 1,4,8,16]
Bindings and gdr_info_t layout copied from gdr_vmm_probe.py (gdrwrap.{h,cpp}).
"""
import argparse
import ctypes
import os
import statistics
import threading
import time

import torch

try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu

MiB = 1 << 20
GPU_PAGE = 64 * 1024
PIECE = 2 * MiB


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


def make_prop(device, rdma):
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


class VmmRange:
    """size bytes reserved, backed by chunk-sized cuMemCreate handles mapped back to back."""

    def __init__(self, device, size, chunk, rdma):
        self.device, self.size, self.chunk = device, size, chunk
        self.prop = make_prop(device, rdma)
        self.base = None
        self.handles = []
        self.base = int(chk(cu.cuMemAddressReserve(size, chunk, 0, 0), "cuMemAddressReserve"))
        for i in range(size // chunk):
            h = chk(cu.cuMemCreate(chunk, self.prop, 0), f"cuMemCreate chunk {i}")
            self.handles.append(h)
            chk(cu.cuMemMap(self.base + i * chunk, chunk, 0, h, 0), f"cuMemMap chunk {i}")
        set_access(self.base, size, device)

    def free(self):
        if self.base is None:
            return
        for i, h in enumerate(self.handles):
            if h is not None:
                cu.cuMemUnmap(self.base + i * self.chunk, self.chunk)
                cu.cuMemRelease(h)
        cu.cuMemAddressFree(self.base, self.size)
        self.base = None


def gpu_read(ptr, n):
    """Copy n bytes at device VA ptr into a fresh torch CUDA tensor (GPU-side readback)."""
    out = torch.empty(n, dtype=torch.uint8, device="cuda")
    chk(cu.cuMemcpyDtoD(out.data_ptr(), ptr, n), "cuMemcpyDtoD readback")
    chk(cu.cuCtxSynchronize(), "cuCtxSynchronize")
    return out


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
        """Return (mh, map_va, cpu_ptr_for_addr, info) or raise with the return code and errno."""
        mh = gdr_mh_t()
        ctypes.set_errno(0)
        rc = self.lib.gdr_pin_buffer(self.g, addr, size, 0, 0, ctypes.byref(mh))
        if rc != 0:
            raise OSError(f"gdr_pin_buffer rc={rc} errno={ctypes.get_errno()} ({os.strerror(ctypes.get_errno())})")
        mva = ctypes.c_void_p()
        rc = self.lib.gdr_map(self.g, mh, ctypes.byref(mva), size)
        if rc != 0:
            e = ctypes.get_errno()
            self.lib.gdr_unpin_buffer(self.g, mh)
            raise OSError(f"gdr_map rc={rc} errno={e} ({os.strerror(e)})")
        info = self.info(mh)
        off = info.va - addr
        return mh, mva.value, mva.value + off, info

    def info(self, mh):
        info = gdr_info_t()
        rc = self.lib.gdr_get_info(self.g, mh, ctypes.byref(info))
        if rc != 0:
            raise OSError(f"gdr_get_info rc={rc}")
        return info

    def unmap_unpin(self, mh, mva, size):
        r1 = self.lib.gdr_unmap(self.g, mh, mva, size)
        r2 = self.lib.gdr_unpin_buffer(self.g, mh)
        return r1, r2

    def copy_to(self, mh, cpu_ptr, host_ptr, size):
        return self.lib.gdr_copy_to_mapping(mh, cpu_ptr, host_ptr, size)


LIBC = ctypes.CDLL(None)
LIBC.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
LIBC.memcpy.restype = ctypes.c_void_p


# ---------------------------------------------------------------- benchmark
class Mapping:
    """One gdr pin+map: device VA dev_addr, CPU pointer cpu_ptr, size bytes; offset is its place in the buffer."""

    def __init__(self, gdr, dev_addr, size, offset):
        self.gdr, self.dev_addr, self.size, self.offset = gdr, dev_addr, size, offset
        t0 = time.perf_counter()
        self.mh, self.mva, self.cpu_ptr, self.info = gdr.pin_map(dev_addr, size)
        self.pin_map_ms = (time.perf_counter() - t0) * 1e3

    def close(self):
        return self.gdr.unmap_unpin(self.mh, self.mva, self.size)


def work_items(maps, total, threads, piecewise):
    """Per-thread lists of (mapping, offset_in_mapping, offset_in_buffer, nbytes)."""
    if piecewise:
        n = len(maps)
        return [[(m, 0, m.offset, m.size) for m in maps[i * n // threads:(i + 1) * n // threads]]
                for i in range(threads)]
    m = maps[0]
    per = (total // threads) // GPU_PAGE * GPU_PAGE
    out = []
    for i in range(threads):
        off = i * per
        n = per if i < threads - 1 else total - off
        out.append([(m, off, off, n)])
    return out


def run_once(items, host_ptr, method):
    """Run one timed pass: threads wait on a barrier, time from release to the last join."""
    rcs = []
    bar = threading.Barrier(len(items) + 1)

    def work(lst):
        bar.wait()
        for m, moff, boff, n in lst:
            if n == 0:
                continue
            if method == "gdr_copy":
                rcs.append(m.gdr.copy_to(m.mh, m.cpu_ptr + moff, host_ptr + boff, n))
            else:
                LIBC.memcpy(m.cpu_ptr + moff, host_ptr + boff, n)

    ts = [threading.Thread(target=work, args=(lst,)) for lst in items]
    for t in ts:
        t.start()
    bar.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    if any(rc != 0 for rc in rcs):
        raise OSError(f"gdr_copy_to_mapping rc={sorted(set(rcs))}")
    return dt


def verify(maps, total, host, method):
    """Write a fresh pattern with one thread via method; compare a GPU readback of every mapping."""
    host.copy_(torch.randint(0, 256, (total,), dtype=torch.uint8))
    run_once(work_items(maps, total, 1, len(maps) > 1), host.data_ptr(), method)
    torch.cuda.synchronize()
    ref = host.to("cuda")
    bad = 0
    for m in maps:
        got = gpu_read(m.dev_addr, m.size)
        bad += int((got != ref[m.offset:m.offset + m.size]).sum().item())
    return bad


def bench_case(name, gdr, maps, total, host, thread_counts, iters, rows):
    for i, m in enumerate(maps):
        inf = m.info
        print(f"  {name} map[{i:2d}] dev=0x{m.dev_addr:x} size={m.size // MiB} MiB pin+map={m.pin_map_ms:.2f} ms "
              f"mapping_type={inf.mapping_type} wc_mapping={inf.wc_mapping} mapped={inf.mapped} "
              f"page_size={inf.page_size}", flush=True)
    hp = host.data_ptr()
    for method in ("gdr_copy", "memcpy"):
        for th in thread_counts:
            items = work_items(maps, total, th, len(maps) > 1)
            first = run_once(items, hp, method)
            times = [run_once(items, hp, method) for _ in range(iters)]
            rows.append((name, method, th, total / first / 1e9, total / min(times) / 1e9,
                         total / statistics.median(times) / 1e9))
            print(f"  {name} {method:8s} {th:2d}t: first {rows[-1][3]:.2f} best {rows[-1][4]:.2f} "
                  f"median {rows[-1][5]:.2f} GB/s", flush=True)
        bad = verify(maps, total, host, method)
        rows.append((name, f"verify:{method}", 0, None, None, bad))
        print(f"  {name} verify {method}: {'OK' if bad == 0 else f'{bad} bytes differ'}", flush=True)


def case_a(gdr, device, total, host, args, rows):
    ptr = int(chk(cu.cuMemAlloc(total + GPU_PAGE), "cuMemAlloc"))
    maps = []
    try:
        addr = (ptr + GPU_PAGE - 1) & ~(GPU_PAGE - 1)
        maps.append(Mapping(gdr, addr, total, 0))
        bench_case("A_cuMemAlloc", gdr, maps, total, host, args.threads, args.iters, rows)
    finally:
        for m in maps:
            m.close()
        cu.cuMemFree(ptr)


def case_vmm(name, gdr, device, total, chunk, host, args, rows):
    vr = VmmRange(device, total, chunk, True)
    maps = []
    try:
        print(f"  {name}: reserved 0x{vr.base:x}, {len(vr.handles)} cuMemCreate x {chunk // MiB} MiB, "
              f"gpuDirectRDMACapable=1", flush=True)
        for i in range(total // chunk):
            maps.append(Mapping(gdr, vr.base + i * chunk, chunk, i * chunk))
        bench_case(name, gdr, maps, total, host, args.threads, args.iters, rows)
    finally:
        for m in maps:
            m.close()
        vr.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mib", type=int, default=38, help="buffer size in MiB (multiple of 2)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--threads", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8, 16])
    ap.add_argument("--cases", default="A,B,C")
    args = ap.parse_args()
    total = args.mib * MiB
    if total % PIECE:
        raise SystemExit("--mib must be a multiple of 2")

    print(f"host={os.uname().nodename} machine={os.uname().machine} cpus_allowed={len(os.sched_getaffinity(0))} "
          f"torch={torch.__version__} gpu={torch.cuda.get_device_name(args.device)}", flush=True)
    setup_cuda(args.device)
    gdr = Gdr()
    print(f"gdrcopy {gdr.libname}: {gdr.versions()}", flush=True)
    try:
        print("gdrdrv module " + open("/sys/module/gdrdrv/version").read().strip(), flush=True)
    except OSError as e:
        print(f"gdrdrv version unreadable: {e}", flush=True)
    gran = int(chk(cu.cuMemGetAllocationGranularity(
        make_prop(args.device, True), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "granularity"))
    print(f"VMM granularity (rdma) = {gran} B; buffer {args.mib} MiB = {total // PIECE} x 2 MiB", flush=True)
    if PIECE % gran:
        raise SystemExit(f"2 MiB is not a multiple of the granularity {gran}")

    # Pageable source (plain malloc'd, not cudaHostRegister'd), pre-touched.
    host = torch.randint(0, 256, (total,), dtype=torch.uint8)
    rows = []
    cases = {
        "A": lambda: case_a(gdr, args.device, total, host, args, rows),
        "B": lambda: case_vmm("B_vmm_1x", gdr, args.device, total, total, host, args, rows),
        "C": lambda: case_vmm(f"C_vmm_{total // PIECE}x2MiB", gdr, args.device, total, PIECE, host, args, rows),
    }
    for c in args.cases.split(","):
        print(f"\n===== case {c}", flush=True)
        try:
            cases[c]()
        except (OSError, CuErr) as e:
            print(f"  case {c} FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append((c, "FAILED", 0, None, None, str(e)))
        torch.cuda.empty_cache()
    gdr.lib.gdr_close(gdr.g)

    print(f"\nTABLE  ({args.mib} MiB, GB/s = 1e9 B/s, {args.iters} iters after 1 warm-up)")
    print(f"{'case':<18}{'method':<10}{'thr':>4}{'first':>9}{'best':>9}{'median':>9}")
    for name, method, th, first, best, med in rows:
        if first is None:
            print(f"{name:<18}{method:<26}{'OK' if med == 0 else med}")
        else:
            print(f"{name:<18}{method:<10}{th:>4}{first:>9.2f}{best:>9.2f}{med:>9.2f}")


if __name__ == "__main__":
    main()
