/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "borrowedSlotMemory.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/memoryCounters.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <utility>

namespace tensorrt_llm::runtime
{

namespace
{
constexpr size_t kMaxScratchChunks = 8;

// Maps a handle at a fixed address; unlike UnicastConfigurator, a failed access grant does not leave it mapped.
struct MapConfigurator : CUDAVirtualMemoryChunk::Configurator
{
    MapConfigurator(CUdeviceptr address, size_t size, CUmemAccessDesc const& desc)
        : mAddress(address)
        , mSize(size)
        , mDesc(desc)
    {
    }

    void setup(CUmemGenericAllocationHandle handle) override
    {
        TLLM_CU_CHECK(cuMemMap(mAddress, mSize, 0, handle, 0));
        CUresult const result = cuMemSetAccess(mAddress, mSize, &mDesc, 1);
        if (result != CUDA_SUCCESS)
        {
            cuMemUnmap(mAddress, mSize);
            TLLM_CU_CHECK(result);
        }
    }

    void teardown(CUmemGenericAllocationHandle, bool) override
    {
        TLLM_CU_CHECK_FREE_RESOURCE(cuMemUnmap(mAddress, mSize));
    }

    CUdeviceptr mAddress;
    size_t mSize;
    CUmemAccessDesc mDesc;
};

// Device memory created ahead of its mapping, so that a restore allocates everything before it unmaps anything.
class DeviceMemory
{
public:
    DeviceMemory(CUmemAllocationProp const& prop, size_t size)
        : mSize(size)
    {
        TLLM_CU_CHECK(cuMemCreate(&mHandle, size, &prop, 0));
        MemoryCounters::getInstance().allocate(MemoryType::kGPU, size);
    }

    DeviceMemory(DeviceMemory&& other) noexcept
        : mHandle(std::exchange(other.mHandle, 0))
        , mSize(other.mSize)
    {
    }

    DeviceMemory(DeviceMemory const&) = delete;
    DeviceMemory& operator=(DeviceMemory const&) = delete;
    DeviceMemory& operator=(DeviceMemory&&) = delete;

    ~DeviceMemory()
    {
        if (mHandle != 0)
        {
            cuMemRelease(mHandle);
            MemoryCounters::getInstance().deallocate(MemoryType::kGPU, mSize);
        }
    }

    // The creator that owns the memory from now on.
    CUDAVirtualMemoryChunk::CreatorPtr creator() &&
    {
        return std::make_unique<Owner>(std::exchange(mHandle, 0), mSize);
    }

private:
    struct Owner : CUDAVirtualMemoryChunk::Creator
    {
        Owner(CUmemGenericAllocationHandle handle, size_t size)
            : mHandle(handle)
            , mSize(size)
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            TLLM_CHECK_WITH_INFO(mHandle != 0, "Borrowed slot memory is mapped once");
            return mHandle;
        }

        void release(CUmemGenericAllocationHandle handle, bool) override
        {
            mHandle = 0;
            MemoryCounters::getInstance().deallocate(MemoryType::kGPU, mSize);
            TLLM_CU_CHECK_FREE_RESOURCE(cuMemRelease(handle));
        }

        CUmemGenericAllocationHandle mHandle;
        size_t mSize;
    };

    CUmemGenericAllocationHandle mHandle = 0;
    size_t mSize;
};

// Lists [first, first + count) runs of consecutive chunks.
std::vector<std::pair<size_t, size_t>> runsOf(std::vector<size_t> const& chunks)
{
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t chunk : chunks)
    {
        if (!runs.empty() && runs.back().first + runs.back().second == chunk)
        {
            ++runs.back().second;
        }
        else
        {
            runs.emplace_back(chunk, 1);
        }
    }
    return runs;
}
} // namespace

// A scratch allocation, shared by every mapping of it and released with the last one.
struct BorrowedSlotMemory::Scratch
{
    // A mapping's creator: hands out the shared handle and keeps the allocation alive.
    struct Creator : CUDAVirtualMemoryChunk::Creator
    {
        explicit Creator(std::shared_ptr<Scratch> scratch)
            : mScratch(std::move(scratch))
        {
        }

        CUmemGenericAllocationHandle create() override
        {
            return mScratch->handle;
        }

        void release(CUmemGenericAllocationHandle, bool) override {}

        std::shared_ptr<Scratch> mScratch;
    };

    Scratch(CUmemAllocationProp const& prop, size_t size)
    {
        TLLM_CU_CHECK(cuMemCreate(&handle, size, &prop, 0));
    }

    Scratch(Scratch const&) = delete;
    Scratch& operator=(Scratch const&) = delete;

    ~Scratch()
    {
        cuMemRelease(handle);
    }

    CUmemGenericAllocationHandle handle = 0;
};

BorrowedSlotMemory& BorrowedSlotMemory::getInstance()
{
    // Never destroyed: CUDA may be torn down before static destructors run.
    static auto* instance = new BorrowedSlotMemory;
    return *instance;
}

CUmemAllocationProp BorrowedSlotMemory::allocationProp() const
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDevice;
    return prop;
}

BorrowedSlotMemory::Piece BorrowedSlotMemory::mapPiece(CUdeviceptr address, size_t count,
    CUDAVirtualMemoryChunk::CreatorPtr creator, bool scratch, Owner pendingOwner) const
{
    CUmemAccessDesc access{};
    access.location = allocationProp().location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<MapConfigurator>(address, count * mGranularity, access));
    Piece piece{count, scratch, pendingOwner, CUDAVirtualMemoryChunk(std::move(creator), std::move(configurators))};
    try
    {
        piece.mapping.materialize();
    }
    catch (...)
    {
        // Undo what materialize() did; an error of the undo must not hide the original one.
        try
        {
            piece.mapping.release();
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Releasing a borrowed slot mapping after a failed map failed: %s", error.what());
        }
        throw;
    }
    return piece;
}

void BorrowedSlotMemory::mapScratch(CUdeviceptr base, Allocation& allocation, size_t first, size_t count) const
{
    while (count > 0)
    {
        size_t piece = kMaxScratchChunks;
        while (piece > count)
        {
            piece /= 2;
        }
        allocation.pieces.emplace(first,
            mapPiece(base + first * mGranularity, piece, std::make_unique<Scratch::Creator>(mScratch.at(piece)),
                /*scratch=*/true, /*pendingOwner=*/nullptr));
        first += piece;
        count -= piece;
    }
}

void BorrowedSlotMemory::toScratch(
    CUdeviceptr base, Allocation& allocation, std::map<size_t, Piece>::iterator piece) const
{
    size_t const first = piece->first;
    size_t const count = piece->second.count;
    piece->second.mapping.release();
    allocation.pieces.erase(piece);
    mapScratch(base, allocation, first, count);
}

void BorrowedSlotMemory::coverHoles(CUdeviceptr base, Allocation& allocation) const noexcept
{
    size_t const chunks = allocation.size / mGranularity;
    size_t chunk = 0;
    try
    {
        while (chunk < chunks)
        {
            auto const next = allocation.pieces.lower_bound(chunk);
            if (next != allocation.pieces.end() && next->first == chunk)
            {
                chunk += next->second.count;
                continue;
            }
            size_t const end = next == allocation.pieces.end() ? chunks : next->first;
            mapScratch(base, allocation, chunk, end - chunk);
            chunk = end;
        }
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Borrowed slot memory at %p left chunks from %zu unmapped: %s", reinterpret_cast<void*>(base),
            chunk, error.what());
    }
}

void* BorrowedSlotMemory::allocate(size_t slotBytes, int slotCount, std::vector<int> const& emptySlots)
{
    TLLM_CHECK_WITH_INFO(slotBytes > 0 && slotCount > 0, "No slots to allocate");
    std::lock_guard<std::mutex> lock(mMutex);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CHECK_WITH_INFO(mDevice < 0 || mDevice == device, "Borrowed slot memory serves one device per process");
    if (mDevice < 0)
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        size_t granularity = 0;
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        // Every scratch size up front, so that a restore needs no memory beyond what unbackedBytes() reports.
        std::map<size_t, std::shared_ptr<Scratch>> scratch;
        for (size_t chunks = 1; chunks <= kMaxScratchChunks; chunks *= 2)
        {
            scratch.emplace(chunks, std::make_shared<Scratch>(prop, chunks * granularity));
        }
        mScratch = std::move(scratch);
        mGranularity = granularity;
        mDevice = device;
    }
    std::vector<bool> empty(slotCount, false);
    for (int slot : emptySlots)
    {
        TLLM_CHECK_WITH_INFO(slot >= 0 && slot < slotCount, "Empty slot %d is not in [0, %d)", slot, slotCount);
        empty[slot] = true;
    }
    size_t const bytes = slotBytes * slotCount;
    size_t const chunks = common::divUp(bytes, mGranularity);
    std::vector<bool> unbacked(chunks);
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        // Unbacked if every byte of the chunk is in an empty slot or past the last slot.
        size_t pos = chunk * mGranularity;
        size_t const end = pos + mGranularity;
        while (pos < end && pos < bytes && empty[pos / slotBytes])
        {
            pos = (pos / slotBytes + 1) * slotBytes;
        }
        unbacked[chunk] = pos >= end || pos >= bytes;
    }

    CUdeviceptr base = 0;
    TLLM_CU_CHECK(cuMemAddressReserve(&base, chunks * mGranularity, mGranularity, 0, 0));
    Allocation allocation{
        std::shared_ptr<void>(reinterpret_cast<void*>(base),
            [size = chunks * mGranularity](void* ptr) { cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size); }),
        chunks * mGranularity, {}};
    auto const prop = allocationProp();
    try
    {
        for (size_t chunk = 0; chunk < chunks;)
        {
            size_t run = chunk;
            while (run < chunks && unbacked[run] == unbacked[chunk])
            {
                ++run;
            }
            if (unbacked[chunk])
            {
                mapScratch(base, allocation, chunk, run - chunk);
            }
            else
            {
                allocation.pieces.emplace(chunk,
                    mapPiece(base + chunk * mGranularity, run - chunk,
                        DeviceMemory(prop, (run - chunk) * mGranularity).creator(), /*scratch=*/false,
                        /*pendingOwner=*/nullptr));
            }
            chunk = run;
        }
    }
    catch (...)
    {
        // Unmap what was mapped; the reservation is freed with the allocation.
        for (auto& [first, piece] : allocation.pieces)
        {
            try
            {
                piece.mapping.release();
            }
            catch (std::exception const& error)
            {
                TLLM_LOG_ERROR("Unmapping borrowed slot memory after a failed allocation failed: %s", error.what());
            }
        }
        throw;
    }
    mAllocations.emplace(base, std::move(allocation));
    return reinterpret_cast<void*>(base);
}

void BorrowedSlotMemory::free(void* ptr) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mAllocations.find(reinterpret_cast<CUdeviceptr>(ptr));
    if (it == mAllocations.end())
    {
        TLLM_LOG_ERROR("%p is not borrowed slot memory", ptr);
        return;
    }
    for (auto& [first, piece] : it->second.pieces)
    {
        try
        {
            piece.mapping.release();
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Unmapping borrowed slot memory at %p failed: %s", ptr, error.what());
        }
    }
    mAllocations.erase(it);
}

std::map<CUdeviceptr, std::vector<size_t>> BorrowedSlotMemory::unbackedChunks(std::vector<Range> const& ranges) const
{
    std::map<CUdeviceptr, std::vector<size_t>> chunks;
    for (auto const& [ptr, size] : ranges)
    {
        auto const start = reinterpret_cast<CUdeviceptr>(ptr);
        auto it = mAllocations.upper_bound(start);
        if (it == mAllocations.begin() || size == 0)
        {
            continue;
        }
        --it;
        auto const& [base, allocation] = *it;
        if (start >= base + allocation.size)
        {
            continue;
        }
        size_t const lo = (start - base) / mGranularity;
        size_t const hi = (std::min<size_t>(start - base + size, allocation.size) - 1) / mGranularity;
        auto piece = allocation.pieces.upper_bound(lo);
        --piece; // the piece holding chunk lo: the pieces tile the range
        for (; piece != allocation.pieces.end() && piece->first <= hi; ++piece)
        {
            if (piece->second.scratch)
            {
                for (size_t chunk = std::max(lo, piece->first);
                     chunk < std::min(hi + 1, piece->first + piece->second.count); ++chunk)
                {
                    chunks[base].push_back(chunk);
                }
            }
        }
    }
    for (auto& [base, list] : chunks)
    {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    return chunks;
}

size_t BorrowedSlotMemory::unbackedBytes(std::vector<Range> const& ranges)
{
    std::lock_guard<std::mutex> lock(mMutex);
    size_t count = 0;
    for (auto const& [base, list] : unbackedChunks(ranges))
    {
        count += list.size();
    }
    return count * mGranularity;
}

void BorrowedSlotMemory::back(std::vector<Range> const& ranges, Owner owner)
{
    TLLM_CHECK_WITH_INFO(owner != nullptr, "Pending slot backing needs an owner");
    std::lock_guard<std::mutex> lock(mMutex);

    // Device memory for every run first: if that fails, nothing changed.
    struct Run
    {
        CUdeviceptr base;
        size_t first;
        size_t count;
        DeviceMemory memory;
    };

    std::vector<Run> runs;
    auto const prop = allocationProp();
    for (auto const& [base, chunks] : unbackedChunks(ranges))
    {
        for (auto const& [first, count] : runsOf(chunks))
        {
            runs.push_back({base, first, count, DeviceMemory(prop, count * mGranularity)});
        }
    }

    CUdeviceptr base = 0;
    try
    {
        for (auto& run : runs)
        {
            base = run.base;
            auto& allocation = mAllocations.at(run.base);
            // The scratch pieces overlapping the run: unmap them, map the run, map scratch over the rest.
            auto piece = std::prev(allocation.pieces.upper_bound(run.first));
            size_t const lo = piece->first;
            size_t hi = lo;
            while (piece != allocation.pieces.end() && piece->first < run.first + run.count)
            {
                hi = piece->first + piece->second.count;
                piece->second.mapping.release();
                piece = allocation.pieces.erase(piece);
            }
            allocation.pieces.emplace(run.first,
                mapPiece(run.base + run.first * mGranularity, run.count, std::move(run.memory).creator(),
                    /*scratch=*/false, owner));
            mapScratch(run.base, allocation, lo, run.first - lo);
            mapScratch(run.base, allocation, run.first + run.count, hi - run.first - run.count);
        }
    }
    catch (...)
    {
        coverHoles(base, mAllocations.at(base));
        throw;
    }
}

size_t BorrowedSlotMemory::releasePending(Owner owner)
{
    std::lock_guard<std::mutex> lock(mMutex);
    size_t chunks = 0;
    for (auto& [base, allocation] : mAllocations)
    {
        try
        {
            for (auto piece = allocation.pieces.begin(); piece != allocation.pieces.end();)
            {
                if (owner == nullptr || piece->second.pendingOwner != owner)
                {
                    ++piece;
                    continue;
                }
                size_t const next = piece->first + piece->second.count;
                chunks += piece->second.count;
                toScratch(base, allocation, piece);
                piece = allocation.pieces.lower_bound(next);
            }
        }
        catch (...)
        {
            coverHoles(base, allocation);
            throw;
        }
    }
    return chunks * mGranularity;
}

void BorrowedSlotMemory::keepPending(Owner owner)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& [base, allocation] : mAllocations)
    {
        for (auto& [first, piece] : allocation.pieces)
        {
            if (piece.pendingOwner == owner)
            {
                piece.pendingOwner = nullptr;
            }
        }
    }
}

} // namespace tensorrt_llm::runtime
