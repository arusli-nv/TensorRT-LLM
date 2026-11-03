// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Restore copies on the GPU path (borrowed slots): many host -> device copies from pageable shared memory (EPLB's
// host copies), one per (slot, weight tensor). Timed per path over the same copies:
//   memcpy      one cudaMemcpyAsync each (MoeLoadBalancer's GPU path today)
//   batch       one cudaMemcpyBatchAsync (srcAccessOrder stream)
//   batch-call  the same with srcAccessOrder during-API-call (the host copies never change)
//   kernel      one kernel that reads the host pointers directly (pageable memory access, GB200 ATS)
// Usage: restore_batch_bench SLOTS TENSOR_BYTES...   e.g. Lite: 312 2883584 1441792 184320 92160 4 4
//                                                     R1:   58 15728640 7864320 983040 491520 4 4
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <vector>

#define CK(x)                                                                                                        \
    do                                                                                                               \
    {                                                                                                                \
        cudaError_t e_ = (x);                                                                                        \
        if (e_ != cudaSuccess)                                                                                       \
        {                                                                                                            \
            std::printf("FAIL %s: %s\n", #x, cudaGetErrorString(e_));                                                \
            std::exit(1);                                                                                            \
        }                                                                                                            \
    } while (0)

struct Copy
{
    char const* src;
    char* dst;
    size_t bytes;
};

__global__ void gather(Copy const* copies, int count)
{
    for (int c = blockIdx.x; c < count; c += gridDim.x)
    {
        Copy const copy = copies[c];
        size_t const words = copy.bytes / 16;
        auto const* src = reinterpret_cast<uint4 const*>(copy.src);
        auto* dst = reinterpret_cast<uint4*>(copy.dst);
        for (size_t i = threadIdx.x; i < words; i += blockDim.x)
            dst[i] = src[i];
        for (size_t i = words * 16 + threadIdx.x; i < copy.bytes; i += blockDim.x)
            copy.dst[i] = copy.src[i];
    }
}

int main(int argc, char** argv)
{
    if (argc < 3)
        return 1;
    int const slots = std::atoi(argv[1]);
    std::vector<size_t> tensors;
    for (int i = 2; i < argc; ++i)
        tensors.push_back(std::strtoull(argv[i], nullptr, 10));
    size_t perSlot = 0;
    for (size_t t : tensors)
        perSlot += (t + 255) / 256 * 256;
    size_t const total = perSlot * slots;
    // Shared anonymous memory, touched, like the host copies' shm segments.
    auto* host = static_cast<char*>(mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    for (size_t i = 0; i < total; ++i)
        host[i] = static_cast<char>(i * 131);
    char* dev = nullptr;
    CK(cudaMalloc(&dev, total));
    std::vector<Copy> copies;
    for (size_t off = 0, s = 0; s < static_cast<size_t>(slots); ++s)
        for (size_t t : tensors)
        {
            copies.push_back({host + off, dev + off, t});
            off += (t + 255) / 256 * 256;
        }
    int pageable = 0;
    CK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
    std::printf("copies %zu, %.1f MB, pageableMemoryAccess=%d\n", copies.size(), total / 1e6, pageable);
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Copy* devCopies = nullptr;
    CK(cudaMalloc(&devCopies, copies.size() * sizeof(Copy)));
    CK(cudaMemcpy(devCopies, copies.data(), copies.size() * sizeof(Copy), cudaMemcpyHostToDevice));

    std::vector<void*> dsts, srcs;
    std::vector<size_t> sizes;
    for (auto const& c : copies)
    {
        dsts.push_back(c.dst);
        srcs.push_back(const_cast<char*>(c.src));
        sizes.push_back(c.bytes);
    }
    auto run = [&](char const* name, auto&& body)
    {
        for (int rep = 0; rep < 4; ++rep)
        {
            CK(cudaMemset(dev, 0, total));
            CK(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            body();
            CK(cudaStreamSynchronize(stream));
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::vector<char> back(perSlot);
            CK(cudaMemcpy(back.data(), dev + total - perSlot, perSlot, cudaMemcpyDeviceToHost));
            bool ok = std::memcmp(back.data(), host + total - perSlot, perSlot) == 0;
            if (rep > 0)
                std::printf("RESULT path=%s ms=%.2f GBps=%.1f ok=%d\n", name, ms, total / ms / 1e6, ok);
        }
    };
    run("memcpy",
        [&]
        {
            for (auto const& c : copies)
                CK(cudaMemcpyAsync(c.dst, c.src, c.bytes, cudaMemcpyHostToDevice, stream));
        });
    for (auto order : {cudaMemcpySrcAccessOrderStream, cudaMemcpySrcAccessOrderDuringApiCall})
    {
        run(order == cudaMemcpySrcAccessOrderStream ? "batch" : "batch-call",
            [&]
            {
                cudaMemcpyAttributes attrs{};
                attrs.srcAccessOrder = order;
                size_t attrIdx = 0;
                CK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), copies.size(), &attrs, &attrIdx, 1,
                    stream));
            });
    }
    if (pageable)
        run("kernel", [&] { gather<<<264, 512, 0, stream>>>(devCopies, static_cast<int>(copies.size())); });
    std::printf("RESTORE_BATCH_DONE\n");
    return 0;
}
