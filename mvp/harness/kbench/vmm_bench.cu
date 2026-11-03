// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// CUDA VMM costs for KV-borrowed EPLB slots (KV_BORROWED_SLOTS.md, BorrowedSlotMemory), one GB200.
// RANGES empty-slot runs of CHUNKS chunks in total. Startup: each run maps shared scratch allocations of 8/4/2/1
// chunks (the driver maps whole allocations only). Restore, timed: unmap every scratch piece, then per run one
// cuMemCreate of the run, map and set access. Then a write + read check through the new mappings.
// Usage: vmm_bench [CHUNKS=650] [RANGES=116]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <utility>
#include <vector>

#define CK(x)                                                                                                        \
    do                                                                                                               \
    {                                                                                                                \
        CUresult r_ = (x);                                                                                           \
        if (r_ != CUDA_SUCCESS)                                                                                      \
        {                                                                                                            \
            char const* s_;                                                                                          \
            cuGetErrorString(r_, &s_);                                                                               \
            std::printf("FAIL %s: %s\n", #x, s_);                                                                    \
            std::exit(1);                                                                                            \
        }                                                                                                            \
    } while (0)

static double ms(std::chrono::steady_clock::time_point a)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

__global__ void fill(unsigned* p, size_t n, unsigned v)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = v + static_cast<unsigned>(i);
}

__global__ void check(unsigned const* p, size_t n, unsigned v, int* bad)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (p[i] != v + static_cast<unsigned>(i))
            atomicAdd(bad, 1);
}

int main(int argc, char** argv)
{
    int chunks = argc > 1 ? std::atoi(argv[1]) : 650;
    int ranges = argc > 2 ? std::atoi(argv[2]) : 116;
    cudaFree(nullptr);
    CUdevice dev;
    CK(cuCtxGetDevice(&dev));
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gran = 0;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    size_t const bytes = gran * chunks;
    std::printf("granularity %zu bytes, %d chunks (%.2f GB), %d ranges\n", gran, chunks, bytes / 1e9, ranges);

    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, bytes, gran, 0, 0));
    std::map<size_t, CUmemGenericAllocationHandle> scratch;
    for (size_t piece = 1; piece <= 8; piece *= 2)
        CK(cuMemCreate(&scratch[piece], piece * gran, &prop, 0));

    // Runs: [first, count) in chunks.
    std::vector<std::pair<size_t, size_t>> runs;
    for (int r = 0, per = chunks / ranges; r < ranges; ++r)
        runs.emplace_back(static_cast<size_t>(r) * per, r == ranges - 1 ? chunks - r * per : per);

    std::vector<std::pair<size_t, size_t>> pieces;
    auto t = std::chrono::steady_clock::now();
    for (auto [first, count] : runs)
    {
        while (count > 0)
        {
            size_t piece = 8;
            while (piece > count)
                piece /= 2;
            CK(cuMemMap(va + first * gran, piece * gran, 0, scratch[piece], 0));
            CK(cuMemSetAccess(va + first * gran, piece * gran, &access, 1));
            pieces.emplace_back(first, piece);
            first += piece;
            count -= piece;
        }
    }
    std::printf("RESULT alias pieces=%zu ms=%.2f\n", pieces.size(), ms(t));

    t = std::chrono::steady_clock::now();
    for (auto [first, piece] : pieces)
        CK(cuMemUnmap(va + first * gran, piece * gran));
    double const unalias = ms(t);
    t = std::chrono::steady_clock::now();
    std::vector<CUmemGenericAllocationHandle> handles(runs.size());
    for (size_t r = 0; r < runs.size(); ++r)
        CK(cuMemCreate(&handles[r], runs[r].second * gran, &prop, 0));
    double const create = ms(t);
    t = std::chrono::steady_clock::now();
    for (size_t r = 0; r < runs.size(); ++r)
    {
        CK(cuMemMap(va + runs[r].first * gran, runs[r].second * gran, 0, handles[r], 0));
        CK(cuMemSetAccess(va + runs[r].first * gran, runs[r].second * gran, &access, 1));
        CK(cuMemRelease(handles[r]));
    }
    double const map = ms(t);
    std::printf("RESULT restore unmap_ms=%.2f create_ms=%.2f map_ms=%.2f total_ms=%.2f\n", unalias, create, map,
        unalias + create + map);

    int* bad;
    cudaMalloc(&bad, sizeof(int));
    cudaMemset(bad, 0, sizeof(int));
    fill<<<264, 256>>>(reinterpret_cast<unsigned*>(va), bytes / 4, 3);
    check<<<264, 256>>>(reinterpret_cast<unsigned*>(va), bytes / 4, 3, bad);
    int hbad = -1;
    cudaMemcpy(&hbad, bad, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("RESULT backed mismatches=%d\n", hbad);
    std::printf("VMM_BENCH_DONE\n");
    return 0;
}
