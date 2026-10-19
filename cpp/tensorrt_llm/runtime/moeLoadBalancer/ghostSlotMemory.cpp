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

#include "ghostSlotMemory.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <exception>
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

// Splits chunks into runs of the scratch sizes, largest first: (first chunk, count) each.
std::vector<std::pair<size_t, size_t>> scratchRuns(size_t chunks)
{
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t first = 0; first < chunks;)
    {
        size_t count = kMaxScratchChunks;
        while (count > chunks - first)
        {
            count /= 2;
        }
        runs.emplace_back(first, count);
        first += count;
    }
    return runs;
}

// Releases a chunk that materialize() left mapped in full or in part; errors are logged.
void releaseLogged(CUDAVirtualMemoryChunk& chunk) noexcept
{
    auto const status = chunk.status();
    if (status == CUDAVirtualMemoryChunk::INVALID || status == CUDAVirtualMemoryChunk::RELEASED)
    {
        return;
    }
    try
    {
        chunk.release();
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Releasing ghost slot memory failed: %s", error.what());
    }
}
} // namespace

// The scratch allocations of 1, 2, 4 and 8 chunks, shared by every piece and released with the last one.
struct GhostSlotMemory::Scratch
{
    Scratch(CUmemAllocationProp const& prop, size_t chunkBytes)
        : granularity(chunkBytes)
    {
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        try
        {
            for (size_t chunks = 1; chunks <= kMaxScratchChunks; chunks *= 2)
            {
                CUmemGenericAllocationHandle handle{};
                TLLM_CU_CHECK(cuMemCreate(&handle, chunks * chunkBytes, &prop, 0));
                handles.emplace(chunks, handle);
            }
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    Scratch(Scratch const&) = delete;
    Scratch& operator=(Scratch const&) = delete;

    ~Scratch()
    {
        release();
    }

    void release() noexcept
    {
        for (auto const& [chunks, handle] : handles)
        {
            cuMemRelease(handle);
        }
        handles.clear();
    }

    // Maps scratch over chunks [0, chunks) at address, every run it can; throws the first error.
    void map(CUdeviceptr address, size_t chunks) const
    {
        CUresult error = CUDA_SUCCESS;
        for (auto const& [first, count] : scratchRuns(chunks))
        {
            CUdeviceptr const at = address + first * granularity;
            size_t const bytes = count * granularity;
            CUresult result = cuMemMap(at, bytes, 0, handles.at(count), 0);
            if (result == CUDA_SUCCESS)
            {
                result = cuMemSetAccess(at, bytes, &access, 1);
                if (result != CUDA_SUCCESS)
                {
                    cuMemUnmap(at, bytes);
                }
            }
            error = error == CUDA_SUCCESS ? result : error;
        }
        TLLM_CU_CHECK_FREE_RESOURCE(error);
    }

    // Unmaps what map(address, chunks) mapped, every run it can; throws the first error.
    void unmap(CUdeviceptr address, size_t chunks) const
    {
        CUresult error = CUDA_SUCCESS;
        for (auto const& [first, count] : scratchRuns(chunks))
        {
            CUresult const result = cuMemUnmap(address + first * granularity, count * granularity);
            error = error == CUDA_SUCCESS ? result : error;
        }
        TLLM_CU_CHECK_FREE_RESOURCE(error);
    }

    size_t granularity;
    CUmemAccessDesc access{};
    std::map<size_t, CUmemGenericAllocationHandle> handles; // chunks -> allocation
};

// The first configurator of a piece's manager entry: setup() unmaps the piece's scratch before the piece's own
// memory is mapped, teardown() maps the scratch back after that memory is unmapped. Every release, rollback and
// eviction of the entry thus ends with the piece on scratch.
class GhostSlotMemory::ScratchSwap : public CUDAVirtualMemoryChunk::Configurator
{
public:
    ScratchSwap(std::shared_ptr<Scratch> scratch, CUdeviceptr address, size_t chunks)
        : mScratch(std::move(scratch))
        , mAddress(address)
        , mChunks(chunks)
    {
    }

    void setup(CUmemGenericAllocationHandle) override
    {
        try
        {
            mScratch->unmap(mAddress, mChunks);
        }
        catch (...)
        {
            // Map back the runs that were unmapped (the runs still mapped fail to map again).
            try
            {
                mScratch->map(mAddress, mChunks);
            }
            catch (std::exception const&)
            {
            }
            throw;
        }
    }

    void teardown(CUmemGenericAllocationHandle, bool destructing) override
    {
        try
        {
            mScratch->map(mAddress, mChunks);
        }
        catch (std::exception const& error)
        {
            if (!destructing)
            {
                throw;
            }
            // A destructor must not throw.
            TLLM_LOG_ERROR(
                "Ghost slot memory at %p left unmapped: %s", reinterpret_cast<void*>(mAddress), error.what());
        }
    }

private:
    std::shared_ptr<Scratch> mScratch;
    CUdeviceptr mAddress;
    size_t mChunks;
};

GhostSlotMemory& GhostSlotMemory::getInstance()
{
    // Never destroyed: CUDA may be torn down before static destructors run.
    static auto* instance = new GhostSlotMemory;
    return *instance;
}

CUmemAllocationProp GhostSlotMemory::allocationProp() const
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = mDevice;
    return prop;
}

CUDAVirtualMemoryChunk GhostSlotMemory::mapResident(CUdeviceptr address, size_t size) const
{
    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<MapConfigurator>(address, size, mScratch->access));
    CUDAVirtualMemoryChunk chunk(std::make_unique<LocalCreator<>>(allocationProp(), size), std::move(configurators));
    try
    {
        chunk.materialize();
    }
    catch (...)
    {
        releaseLogged(chunk);
        throw;
    }
    return chunk;
}

void GhostSlotMemory::addPiece(Piece const& piece) const
{
    size_t const bytes = piece.count * mGranularity;
    CUDAVirtualMemoryChunk::Configurators configurators;
    configurators.push_back(std::make_unique<ScratchSwap>(mScratch, piece.address, piece.count));
    configurators.push_back(std::make_unique<MapConfigurator>(piece.address, bytes, mScratch->access));
    getVirtualMemoryManager().add(piece.address, piece.tag,
        CUDAVirtualMemoryChunk(std::make_unique<LocalCreator<>>(allocationProp(), bytes), std::move(configurators)));
}

void GhostSlotMemory::dropPiece(Piece const& piece) const noexcept
{
    // A backed piece maps its scratch back as it is released.
    auto chunk = getVirtualMemoryManager().remove(piece.address);
    releaseLogged(chunk);
    try
    {
        mScratch->unmap(piece.address, piece.count);
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Unmapping the scratch of ghost slot memory at %p failed: %s",
            reinterpret_cast<void*>(piece.address), error.what());
    }
}

void* GhostSlotMemory::allocate(size_t slotBytes, int slotCount, std::vector<int> const& emptySlots)
{
    TLLM_CHECK_WITH_INFO(slotBytes > 0 && slotCount > 0, "No slots to allocate");
    std::lock_guard<std::mutex> lock(mMutex);
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    TLLM_CHECK_WITH_INFO(mDevice < 0 || mDevice == device, "Ghost slot memory serves one device per process");
    if (mDevice < 0)
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        size_t granularity = 0;
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        mGranularity = granularity;
        mDevice = device;
    }
    if (!mScratch)
    {
        // Allocated up front, so that a restore needs no memory beyond what unbackedBytes() reports.
        mScratch = std::make_shared<Scratch>(allocationProp(), mGranularity);
    }
    size_t const slots = static_cast<size_t>(slotCount);
    std::vector<bool> empty(slots, false);
    for (int slot : emptySlots)
    {
        TLLM_CHECK_WITH_INFO(slot >= 0 && slot < slotCount, "Empty slot %d is not in [0, %d)", slot, slotCount);
        empty[slot] = true;
    }
    size_t const bytes = slotBytes * slots;
    size_t const chunks = common::divUp(bytes, mGranularity);
    // Per chunk: the empty slot holding all its bytes (slots past the last slot), -1 if a byte is in a backed slot,
    // or a value of its own below -1 if it spans several empty slots. A run of one value is a piece.
    std::vector<int64_t> slotOf(chunks);
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        size_t const first = std::min(chunk * mGranularity / slotBytes, slots);
        size_t const last = std::min(((chunk + 1) * mGranularity - 1) / slotBytes, slots);
        bool ghost = true;
        for (size_t slot = first; slot <= last && ghost; ++slot)
        {
            ghost = slot == slots || empty[slot];
        }
        slotOf[chunk] = !ghost ? -1 : first == last ? static_cast<int64_t>(first) : -2 - static_cast<int64_t>(chunk);
    }

    CUdeviceptr base = 0;
    TLLM_CU_CHECK(cuMemAddressReserve(&base, chunks * mGranularity, mGranularity, 0, 0));
    Allocation allocation{
        std::shared_ptr<void>(reinterpret_cast<void*>(base),
            [size = chunks * mGranularity](void* ptr) { cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size); }),
        chunks * mGranularity, {}, {}};
    try
    {
        for (size_t chunk = 0; chunk < chunks;)
        {
            size_t run = chunk;
            while (run < chunks && slotOf[run] == slotOf[chunk])
            {
                ++run;
            }
            CUdeviceptr const address = base + chunk * mGranularity;
            if (slotOf[chunk] == -1)
            {
                allocation.resident.push_back(mapResident(address, (run - chunk) * mGranularity));
            }
            else
            {
                auto const& piece = allocation.pieces.emplace_back(Piece{address, chunk, run - chunk,
                    common::fmtstr("ghost/%llx/%zu", static_cast<unsigned long long>(base), chunk), false});
                mScratch->map(piece.address, piece.count);
                addPiece(piece);
            }
            chunk = run;
        }
    }
    catch (...)
    {
        for (auto const& piece : allocation.pieces)
        {
            dropPiece(piece);
        }
        for (auto& chunk : allocation.resident)
        {
            releaseLogged(chunk);
        }
        if (mAllocations.empty())
        {
            mScratch.reset();
        }
        throw;
    }
    mAllocations.emplace(base, std::move(allocation));
    return reinterpret_cast<void*>(base);
}

void GhostSlotMemory::free(void* ptr) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mAllocations.find(reinterpret_cast<CUdeviceptr>(ptr));
    if (it == mAllocations.end())
    {
        TLLM_LOG_ERROR("%p is not ghost slot memory", ptr);
        return;
    }
    CUdeviceptr const base = it->first;
    auto& allocation = it->second;
    for (auto& [owner, pieces] : mPending)
    {
        pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                         [&](Piece const* piece) { return piece->address - base < allocation.size; }),
            pieces.end());
    }
    for (auto const& piece : allocation.pieces)
    {
        dropPiece(piece);
    }
    for (auto& chunk : allocation.resident)
    {
        releaseLogged(chunk);
    }
    mAllocations.erase(it);
    if (mAllocations.empty())
    {
        // The last mapping of the scratch is gone: release it too.
        mScratch.reset();
    }
}

std::vector<GhostSlotMemory::Piece*> GhostSlotMemory::unbackedPieces(std::vector<Range> const& ranges)
{
    std::vector<Piece*> pieces;
    for (auto const& [ptr, size] : ranges)
    {
        auto const start = reinterpret_cast<CUdeviceptr>(ptr);
        auto it = mAllocations.upper_bound(start);
        if (it == mAllocations.begin() || size == 0)
        {
            continue;
        }
        --it;
        auto& [base, allocation] = *it;
        if (start >= base + allocation.size)
        {
            continue;
        }
        size_t const lo = (start - base) / mGranularity;
        size_t const hi = (std::min<size_t>(start - base + size, allocation.size) - 1) / mGranularity;
        for (auto& piece : allocation.pieces)
        {
            if (!piece.backed && piece.first <= hi && piece.first + piece.count > lo)
            {
                pieces.push_back(&piece);
            }
        }
    }
    std::sort(pieces.begin(), pieces.end(), [](Piece const* a, Piece const* b) { return a->address < b->address; });
    pieces.erase(std::unique(pieces.begin(), pieces.end()), pieces.end());
    return pieces;
}

size_t GhostSlotMemory::unbackedBytes(std::vector<Range> const& ranges)
{
    std::lock_guard<std::mutex> lock(mMutex);
    size_t chunks = 0;
    for (auto const* piece : unbackedPieces(ranges))
    {
        chunks += piece->count;
    }
    return chunks * mGranularity;
}

void GhostSlotMemory::back(std::vector<Range> const& ranges, Owner owner)
{
    TLLM_CHECK_WITH_INFO(owner != nullptr, "Pending slot backing needs an owner");
    std::lock_guard<std::mutex> lock(mMutex);
    auto& manager = getVirtualMemoryManager();
    auto const pieces = unbackedPieces(ranges);
    size_t done = 0;
    try
    {
        for (; done < pieces.size(); ++done)
        {
            TLLM_CHECK_WITH_INFO(manager.materializeWithTag(pieces[done]->tag) == 1,
                "Ghost slot memory %s has no released manager entry", pieces[done]->tag.c_str());
        }
    }
    catch (...)
    {
        // The manager dropped the entry that failed (its piece is on scratch): add it again, released. Then
        // release the pieces materialized before it.
        try
        {
            addPiece(*pieces[done]);
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Ghost slot memory %s cannot be backed again: %s", pieces[done]->tag.c_str(), error.what());
        }
        while (done-- > 0)
        {
            try
            {
                manager.releaseWithTag(pieces[done]->tag);
            }
            catch (std::exception const& error)
            {
                TLLM_LOG_ERROR("Releasing ghost slot memory %s failed: %s", pieces[done]->tag.c_str(), error.what());
            }
        }
        throw;
    }
    auto& pending = mPending[owner];
    for (auto* piece : pieces)
    {
        piece->backed = true;
        pending.push_back(piece);
    }
}

size_t GhostSlotMemory::releasePending(Owner owner)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto const it = mPending.find(owner);
    if (it == mPending.end())
    {
        return 0;
    }
    auto const pieces = std::move(it->second);
    mPending.erase(it);
    auto& manager = getVirtualMemoryManager();
    std::exception_ptr error;
    size_t chunks = 0;
    for (auto* piece : pieces)
    {
        try
        {
            manager.releaseWithTag(piece->tag);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        piece->backed = false;
        chunks += piece->count;
    }
    if (error)
    {
        std::rethrow_exception(error);
    }
    return chunks * mGranularity;
}

void GhostSlotMemory::keepPending(Owner owner)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mPending.erase(owner);
}

} // namespace tensorrt_llm::runtime
