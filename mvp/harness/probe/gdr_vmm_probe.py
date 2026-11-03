#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: can GDRCopy pin and map a cuMem VMM DEVICE allocation made with gpuDirectRDMACapable=1, and what CPU
write bandwidth does the mapping give compared with GDRCopy on a cuMemAlloc (cudaMalloc-equivalent) buffer?

Usage: python3 gdr_vmm_probe.py [--device 0] [--gib 1] [--iters 3]
Steps: 1 env, 2 cudaMalloc baseline, 3 VMM with RDMA flag, 4 VMM without it, 5 remap under an active pin.
Step 5 runs in child processes (the old mapping may SIGBUS after invalidation); the parent reports the exit.
Library and gdr_info_t layout follow cpp/tensorrt_llm/runtime/moeLoadBalancer/gdrwrap.{h,cpp}.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import threading
import time

import torch

try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu

MiB = 1 << 20
GPU_PAGE = 64 * 1024
RESULTS = []


def report(step, ok, msg):
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    RESULTS.append((step, tag, msg))
    print(f"[{tag}] {step}: {msg}", flush=True)


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


def attr(dev, name):
    a = getattr(cu.CUdevice_attribute, name, None)
    if a is None:
        return "n/a"
    err, v = cu.cuDeviceGetAttribute(a, dev)
    return v if int(err) == 0 else f"err {ename(err)}"


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


def info_str(info):
    return (f"va=0x{info.va:x} mapped_size={info.mapped_size} page_size={info.page_size} mapped={info.mapped} "
            f"wc_mapping={info.wc_mapping} mapping_type={info.mapping_type}(lib>=2.4 only)")


def timed_copy(gdr, mh, cpu_ptr, host, size, threads, iters):
    """gdr_copy_to_mapping of host[0:size] into the mapping, split across threads. Returns best GB/s and mean."""
    hp = host.data_ptr()
    per = (size // threads) // GPU_PAGE * GPU_PAGE
    ranges = [(i * per, per if i < threads - 1 else size - i * per) for i in range(threads)]
    times, rcs = [], []

    def work(off, n):
        rcs.append(gdr.copy_to(mh, cpu_ptr + off, hp + off, n))

    for _ in range(iters):
        ts = [threading.Thread(target=work, args=r) for r in ranges]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        times.append(time.perf_counter() - t0)
    if any(rc != 0 for rc in rcs):
        raise OSError(f"gdr_copy_to_mapping rc={sorted(set(rcs))}")
    best = size / min(times) / 1e9
    mean = size / (sum(times) / len(times)) / 1e9
    return best, mean


def bench_and_verify(step, gdr, addr, size, host, iters):
    """Pin+map [addr, addr+size), copy with 1 and 8 threads, verify by GPU readback. Returns True on pin."""
    try:
        mh, mva, cpu_ptr, info = gdr.pin_map(addr, size)
    except OSError as e:
        report(f"{step}.pin_map", False, str(e))
        return False
    report(f"{step}.pin_map", True, f"size={size // MiB} MiB, {info_str(info)}")
    try:
        for th in (1, 8):
            host.add_(1)
            best, mean = timed_copy(gdr, mh, cpu_ptr, host, size, th, iters)
            report(f"{step}.bw_{th}t", None, f"{size // MiB} MiB, {th} thread(s): best {best:.2f} GB/s, "
                   f"mean {mean:.2f} GB/s over {iters} iters")
        torch.cuda.synchronize()
        got = gpu_read(addr, size)
        ref = host[:size].to("cuda")
        same = torch.equal(got, ref)
        nbad = 0 if same else int((got != ref).sum().item())
        report(f"{step}.verify", same, "GPU readback matches host buffer" if same else f"{nbad} bytes differ")
        del got, ref
    except (OSError, CuErr) as e:
        report(f"{step}.copy", False, str(e))
    finally:
        r = gdr.unmap_unpin(mh, mva, size)
        report(f"{step}.unmap_unpin", r == (0, 0), f"rc={r}")
    return True


# ---------------------------------------------------------------- steps
def step1_env(gdr_holder, dev):
    report("1.dev_gdrdrv", os.path.exists("/dev/gdrdrv"), "/dev/gdrdrv exists" if os.path.exists("/dev/gdrdrv")
           else "/dev/gdrdrv missing")
    try:
        v = open("/sys/module/gdrdrv/version").read().strip()
        report("1.gdrdrv_version", None, v)
    except OSError as e:
        report("1.gdrdrv_version", None, f"unreadable: {e}")
    try:
        g = Gdr()
        gdr_holder.append(g)
        report("1.gdr_open", True, f"{g.libname}: {g.versions()}")
    except (OSError, AttributeError) as e:
        report("1.gdr_open", False, str(e))
    for a in ("CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED",
              "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED",
              "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS",
              "CU_DEVICE_ATTRIBUTE_NUMA_CONFIG",
              "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"):
        report(f"1.{a.replace('CU_DEVICE_ATTRIBUTE_', '').lower()}", None, str(attr(dev, a)))
    try:
        out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60).stdout.splitlines()
        hits = []
        for i, line in enumerate(out):
            if "addressing" in line.lower():
                hits.extend(out[i:i + 3])
        report("1.nvidia_smi_addressing", None, " | ".join(s.strip() for s in hits) or "no 'addressing' lines")
    except (OSError, subprocess.SubprocessError) as e:
        report("1.nvidia_smi_addressing", None, f"nvidia-smi unavailable: {e}")


def step2_cudamalloc(gdr, size, host, iters):
    ptr = None
    try:
        ptr = int(chk(cu.cuMemAlloc(size + GPU_PAGE), "cuMemAlloc"))
        addr = (ptr + GPU_PAGE - 1) & ~(GPU_PAGE - 1)
        report("2.alloc", True, f"cuMemAlloc (cudaMalloc-equivalent, non-VMM) {size // MiB} MiB, "
               f"ptr=0x{ptr:x} aligned=0x{addr:x}")
        bench_and_verify("2.cudamalloc", gdr, addr, size, host, iters)
    except CuErr as e:
        report("2.alloc", False, str(e))
    finally:
        if ptr is not None:
            cu.cuMemFree(ptr)


def step_vmm(tag, gdr, device, size, host, iters, rdma):
    prop = make_prop(device, rdma)
    try:
        gran = chk(cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity")
        rec = chk(cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "granularity rec")
        report(f"{tag}.granularity", None, f"minimum={gran} recommended={rec}")
    except CuErr as e:
        report(f"{tag}.granularity", False, str(e))
        gran = 2 * MiB
    chunk = max(2 * MiB, int(gran))
    vr = None
    try:
        vr = VmmRange(device, size, chunk, rdma)
        report(f"{tag}.alloc", True, f"{size // MiB} MiB reserved at 0x{vr.base:x}, {len(vr.handles)} chunks of "
               f"{chunk // MiB} MiB, gpuDirectRDMACapable={int(rdma)}")
    except CuErr as e:
        report(f"{tag}.alloc", False, str(e))
        if vr is not None:
            vr.free()
        return
    try:
        bench_and_verify(f"{tag}.a_1chunk", gdr, vr.base, chunk, host, iters)
        n = min(32, len(vr.handles))
        bench_and_verify(f"{tag}.b_{n}chunks", gdr, vr.base, n * chunk, host, iters)
        bench_and_verify(f"{tag}.c_whole", gdr, vr.base, size, host, iters)
    finally:
        vr.free()


# ---------------------------------------------------------------- step 5 (child process)
def step5_child(device, release_old, iters):
    """Pin+map a 4-chunk RDMA range, remap chunk 1, then write through the old mapping."""
    tag = "5.release" if release_old else "5.keep"
    dev = setup_cuda(device)
    del dev
    gdr = Gdr()
    chunk = 2 * MiB
    gran = chk(cu.cuMemGetAllocationGranularity(
        make_prop(device, True), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), "gran")
    chunk = max(chunk, int(gran))
    size = 4 * chunk
    vr = VmmRange(device, size, chunk, True)
    k = 1
    kva = vr.base + k * chunk
    mh, mva, cpu_ptr, info = gdr.pin_map(vr.base, size)
    report(f"{tag}.pin_map", True, info_str(info))
    host = torch.full((chunk,), 0xA1, dtype=torch.uint8)
    rc = gdr.copy_to(mh, cpu_ptr + k * chunk, host.data_ptr(), chunk)
    ok = rc == 0 and bool((gpu_read(kva, chunk) == 0xA1).all().item())
    report(f"{tag}.write_before_remap", ok, f"rc={rc}, chunk {k} reads 0xA1 on GPU: {ok}")

    old = vr.handles[k]
    chk(cu.cuMemUnmap(kva, chunk), "cuMemUnmap old chunk")
    report(f"{tag}.unmap_old", True, "cuMemUnmap of chunk under an active pin returned success")
    if release_old:
        chk(cu.cuMemRelease(old), "cuMemRelease old")
        old = None
        report(f"{tag}.release_old", True, "old physical handle released")
    new = chk(cu.cuMemCreate(chunk, vr.prop, 0), "cuMemCreate new")
    chk(cu.cuMemMap(kva, chunk, 0, new, 0), "cuMemMap new at same VA")
    set_access(kva, chunk, device)
    vr.handles[k] = new
    chk(cu.cuMemsetD8(kva, 0x11, chunk), "memset new")
    chk(cu.cuCtxSynchronize(), "sync")
    alt = None
    if old is not None:
        alt = int(chk(cu.cuMemAddressReserve(chunk, chunk, 0, 0), "reserve alt VA"))
        chk(cu.cuMemMap(alt, chunk, 0, old, 0), "map old at alt VA")
        set_access(alt, chunk, device)
        report(f"{tag}.old_still_a1", bool((gpu_read(alt, chunk) == 0xA1).all().item()),
               "old physical chunk (via alt VA) still holds 0xA1")
    try:
        report(f"{tag}.info_after_remap", None, info_str(gdr.info(mh)))
    except OSError as e:
        report(f"{tag}.info_after_remap", None, f"gdr_get_info failed: {e}")

    print(f"[MARK] {tag}: writing 0xB2 through the OLD gdr mapping (SIGBUS here = invalidated)", flush=True)
    host.fill_(0xB2)
    rc = gdr.copy_to(mh, cpu_ptr + k * chunk, host.data_ptr(), chunk)
    print(f"[MARK] {tag}: old-mapping write returned rc={rc}", flush=True)
    new_val = gpu_read(kva, chunk)
    nb2 = int((new_val == 0xB2).sum().item())
    report(f"{tag}.old_mapping_hits_new", None, f"new chunk bytes == 0xB2: {nb2}/{chunk} (0x11 kept: "
           f"{int((new_val == 0x11).sum().item())})")
    if alt is not None:
        ov = gpu_read(alt, chunk)
        report(f"{tag}.old_mapping_hits_old", None, f"old chunk bytes == 0xB2: {int((ov == 0xB2).sum().item())}"
               f"/{chunk} (0xA1 kept: {int((ov == 0xA1).sum().item())})")
    # Other chunks still pinned: does chunk 0 still take writes?
    host.fill_(0xC3)
    rc0 = gdr.copy_to(mh, cpu_ptr, host.data_ptr(), chunk)
    c0 = bool((gpu_read(vr.base, chunk) == 0xC3).all().item())
    report(f"{tag}.untouched_chunk0_write", None, f"rc={rc0}, chunk 0 reads 0xC3: {c0}")

    r = gdr.unmap_unpin(mh, mva, size)
    report(f"{tag}.unmap_unpin_after_remap", None, f"rc={r}")
    try:
        mh2, mva2, cpu2, info2 = gdr.pin_map(vr.base, size)
        report(f"{tag}.repin", True, info_str(info2))
        host.fill_(0xD4)
        rc = gdr.copy_to(mh2, cpu2 + k * chunk, host.data_ptr(), chunk)
        good = rc == 0 and bool((gpu_read(kva, chunk) == 0xD4).all().item())
        report(f"{tag}.repin_write_new", good, f"rc={rc}, new chunk reads 0xD4: {good}")
        r = gdr.unmap_unpin(mh2, mva2, size)
        report(f"{tag}.repin_unmap_unpin", r == (0, 0), f"rc={r}")
    except OSError as e:
        report(f"{tag}.repin", False, str(e))
    if alt is not None:
        cu.cuMemUnmap(alt, chunk)
        cu.cuMemAddressFree(alt, chunk)
        cu.cuMemRelease(old)
    vr.free()
    print_summary()


def step5_parent(args):
    for mode in ("keep", "release"):
        cmd = [sys.executable, os.path.abspath(__file__), "--device", str(args.device), "--step5", mode]
        print(f"\n===== step 5 ({mode} old handle) in child: {' '.join(cmd)}", flush=True)
        try:
            p = subprocess.run(cmd, timeout=600)
            rc = p.returncode
            if rc < 0:
                report(f"5.{mode}.child", False, f"child killed by signal {-rc} (e.g. 7=SIGBUS, 11=SIGSEGV); see "
                       f"[MARK] lines above for where")
            else:
                report(f"5.{mode}.child", rc == 0, f"child exit code {rc}")
        except subprocess.TimeoutExpired:
            report(f"5.{mode}.child", False, "child timed out after 600 s")


def print_summary():
    print("\nSUMMARY", flush=True)
    for step, tag, msg in RESULTS:
        print(f"  {tag} {step}: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gib", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--step5", choices=("keep", "release"), default=None, help="internal: run one step-5 child")
    args = ap.parse_args()

    if args.step5:
        try:
            step5_child(args.device, args.step5 == "release", args.iters)
        except (OSError, CuErr) as e:
            report(f"5.{args.step5}.error", False, str(e))
            print_summary()
            sys.exit(1)
        return

    print(f"host={os.uname().nodename} torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={torch.cuda.get_device_name(args.device)} PYTORCH_CUDA_ALLOC_CONF="
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    dev = setup_cuda(args.device)
    holder = []
    step1_env(holder, dev)
    if not holder:
        report("abort", False, "no GDRCopy handle; skipping steps 2-5")
        print_summary()
        return
    gdr = holder[0]
    size = args.gib << 30
    host = torch.randint(0, 256, (size,), dtype=torch.uint8)
    steps = [
        ("2", lambda: step2_cudamalloc(gdr, size, host, args.iters)),
        ("3", lambda: step_vmm("3.vmm_rdma", gdr, args.device, size, host, args.iters, True)),
        ("4", lambda: step_vmm("4.vmm_nordma", gdr, args.device, size, host, args.iters, False)),
    ]
    for name, fn in steps:
        print(f"\n===== step {name}", flush=True)
        try:
            fn()
        except Exception as e:  # keep later steps running whatever fails here
            report(f"{name}.unexpected", False, f"{type(e).__name__}: {e}")
        torch.cuda.empty_cache()
    gdr.lib.gdr_close(gdr.g)
    step5_parent(args)
    print_summary()


if __name__ == "__main__":
    main()
