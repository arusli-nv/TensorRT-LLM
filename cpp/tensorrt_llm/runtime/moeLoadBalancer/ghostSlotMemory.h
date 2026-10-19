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
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

// Device storage for an EPLB slot tensor whose empty (ghost) slots hold no device memory until a restore fills them.
// The tensor is one virtual address range that stays fully mapped for the life of the process (CUDA graphs capture
// its address). Chunks of the VMM granularity that lie wholly inside empty slots form ghost pieces, cut at slot
// boundaries. While released, a piece maps scratch memory shared by every range of the process (one allocation each
// of 1, 2, 4 and 8 chunks), so an access that strays into a ghost slot reads or writes scratch instead of faulting.
// Each piece is an entry of the process's CudaVirtualMemoryManager under its own tag (ghost/<range>/<first chunk>),
// added released; materializing it maps device memory in place of its scratch, releasing it maps scratch back. The
// other chunks of the range are mapped to their own device memory at allocate().
//
// The instance is shared by every MoeLoadBalancer of the process, so the pieces that back() materializes are pending
// for the owner that asked for them: keepPending() and releasePending() act on that owner's pieces only.
class GhostSlotMemory
{
public:
    using Range = std::pair<void const*, size_t>;
    // Whose pending backing it is (the MoeLoadBalancer that restores the slots).
    using Owner = void const*;

    static GhostSlotMemory& getInstance();

    // A range for slotCount slots of slotBytes each on the current device; the slots in emptySlots are not backed.
    void* allocate(size_t slotBytes, int slotCount, std::vector<int> const& emptySlots);
    // Waits for the device's work, then unmaps and releases the range; errors are logged (a tensor deleter must not
    // throw).
    void free(void* ptr) noexcept;

    // Device bytes that back(ranges) would allocate.
    size_t unbackedBytes(std::vector<Range> const& ranges);

    // Materializes every released ghost piece that overlaps ranges (ranges outside these allocations are ignored).
    // If one fails, the pieces materialized before it are released again and the manager's entry of the failed one
    // is added again: every piece is back on scratch. The new backing is pending until keepPending(owner). Only
    // while no kernel uses the slots.
    void back(std::vector<Range> const& ranges, Owner owner);
    // Returns owner's pending backing to scratch (a plan that was abandoned); the device bytes it freed.
    size_t releasePending(Owner owner);
    // Owner's pending backing stays (its plan was installed).
    void keepPending(Owner owner);

private:
    struct Scratch;
    class ScratchSwap;

    // Chunks [first, first + count) of an allocation, at address, all inside the same empty slots.
    struct Piece
    {
        CUdeviceptr address;
        size_t first;
        size_t count;
        std::string tag;
        bool backed;
    };

    struct Allocation
    {
        // Declared first so that it is freed after the chunks are unmapped.
        std::shared_ptr<void> reservation;
        size_t size;
        std::vector<Piece> pieces;                    // the ghost pieces, by first chunk
        std::vector<CUDAVirtualMemoryChunk> resident; // the other chunks, mapped to their own device memory
    };

    GhostSlotMemory() = default;

    CUmemAllocationProp allocationProp() const;
    // The released pieces overlapping ranges, each once.
    std::vector<Piece*> unbackedPieces(std::vector<Range> const& ranges);
    // Device memory mapped at [address, address + size).
    CUDAVirtualMemoryChunk mapResident(CUdeviceptr address, size_t size) const;
    // Adds the released manager entry of a piece whose scratch is mapped.
    void addPiece(Piece const& piece) const;
    // Removes the manager entry of a piece and unmaps its scratch; errors are logged.
    void dropPiece(Piece const& piece) const noexcept;

    std::mutex mMutex;
    std::map<CUdeviceptr, Allocation> mAllocations;
    std::map<Owner, std::vector<Piece*>> mPending; // owner -> pieces back() materialized, not yet kept or released
    int mDevice = -1;
    size_t mGranularity = 0;
    std::shared_ptr<Scratch> mScratch; // while any allocation exists
};

} // namespace tensorrt_llm::runtime
