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

#pragma once

#include "tensorrt_llm/runtime/virtualMemory.h"

#include <cstddef>
#include <cuda.h>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

// Device storage for an EPLB slot tensor whose empty slots hold no device memory until a restore fills them.
// The tensor is one virtual address range, tiled by mappings of whole chunks of the VMM granularity (the driver
// maps only whole allocations). Each mapping is a CUDAVirtualMemoryChunk. The chunks that lie wholly inside empty
// slots map scratch memory shared by every range of the process (one allocation each of 1, 2, 4 and 8 chunks, freed
// with the last range), so an access that strays into an empty slot reads or writes scratch instead of faulting;
// back() maps device memory behind them. CudaVirtualMemoryManager is not used: it releases and re-materializes whole
// allocations by tag, which would leave holes under captured CUDA graphs, and it cannot swap the mapping of part of a
// range.
//
// Every operation leaves the range fully mapped, even when a CUDA call fails: on an error, back() and
// releasePending() map scratch over whatever they unmapped before rethrowing. A failed allocate() releases every
// mapping, device memory and the address range it made.
//
// The instance is shared by every MoeLoadBalancer of the process, so the backing that back() maps is pending for
// the owner that asked for it: keepPending() and releasePending() act on that owner's backing only.
class GhostSlotMemory
{
public:
    using Range = std::pair<void const*, size_t>;
    // Whose pending backing it is (the MoeLoadBalancer that restores the slots).
    using Owner = void const*;

    static GhostSlotMemory& getInstance();

    // A range for slotCount slots of slotBytes each on the current device; the slots in emptySlots are not backed.
    void* allocate(size_t slotBytes, int slotCount, std::vector<int> const& emptySlots);
    // Unmaps and releases the range; errors are logged (a tensor deleter must not throw).
    void free(void* ptr) noexcept;

    // Device bytes that back(ranges) would allocate.
    size_t unbackedBytes(std::vector<Range> const& ranges);

    // Maps device memory behind every unbacked chunk that overlaps ranges (ranges outside these allocations are
    // ignored). All memory is allocated first: if that fails, nothing changed. The new backing is pending until
    // keepPending(owner). Only while no kernel uses the slots.
    void back(std::vector<Range> const& ranges, Owner owner);
    // Returns owner's pending backing to scratch (a plan that was abandoned); the device bytes it freed.
    size_t releasePending(Owner owner);
    // Owner's pending backing stays (its plan was installed).
    void keepPending(Owner owner);

private:
    struct Scratch;

    // Chunks [first, first + count) of an allocation, mapped to scratch or to their own device memory.
    struct Piece
    {
        size_t count;
        bool scratch;
        Owner pendingOwner; // nullptr unless back() mapped it and its owner has not kept or released it
        CUDAVirtualMemoryChunk mapping;
    };

    struct Allocation
    {
        // Declared first so that it is freed after the pieces are unmapped.
        std::shared_ptr<void> reservation;
        size_t size;
        std::map<size_t, Piece> pieces; // first chunk -> piece; the pieces tile the range
    };

    GhostSlotMemory() = default;

    // The scratch chunks overlapping ranges, by allocation base, each list sorted and unique.
    std::map<CUdeviceptr, std::vector<size_t>> unbackedChunks(std::vector<Range> const& ranges) const;
    CUmemAllocationProp allocationProp() const;
    Piece mapPiece(CUdeviceptr address, size_t count, CUDAVirtualMemoryChunk::CreatorPtr creator, bool scratch,
        Owner pendingOwner) const;
    // Maps chunks [first, first + count) of the allocation at base to scratch, in pieces of at most 8 chunks.
    void mapScratch(CUdeviceptr base, Allocation& allocation, size_t first, size_t count) const;
    // Unmaps a piece, then maps scratch over its chunks.
    void toScratch(CUdeviceptr base, Allocation& allocation, std::map<size_t, Piece>::iterator piece) const;
    // Best effort after an error: maps scratch over every chunk that no piece covers.
    void coverHoles(CUdeviceptr base, Allocation& allocation) const noexcept;

    std::mutex mMutex;
    std::map<CUdeviceptr, Allocation> mAllocations;
    int mDevice = -1;
    size_t mGranularity = 0;
    std::map<size_t, std::shared_ptr<Scratch>> mScratch; // chunks -> scratch allocation
};

} // namespace tensorrt_llm::runtime
