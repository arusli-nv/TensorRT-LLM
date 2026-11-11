/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/runtime/virtualMemory.h"

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief Device storage for an EPLB slot tensor whose unbacked (spare) slots hold no device memory of their own.
//!
//! The range stays fully mapped at a fixed address, because CUDA graphs capture it. Chunks wholly inside unbacked
//! slots map one shared scratch chunk, so a stray access reads or writes scratch instead of faulting. Backing a slot
//! maps memory of its own at the same address.
class UnbackedSlotMemory
{
public:
    static UnbackedSlotMemory& getInstance();

    //! Return whether allocate() would leave at least one chunk unbacked on the current device.
    bool savesMemory(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    //! Allocate slotCount slots of slotBytes each on the current device, leaving unbackedSlots unbacked.
    void* allocate(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots, bool hostAccessible = false);
    //! Synchronize the range's device, then unmap and release the range. Not valid during CUDA graph capture.
    void free(void* ptr) noexcept;

    struct SlotRange
    {
        void* ptr;
        std::vector<int> slots;
    };

    //! Map memory of their own (contents undefined) or the scratch at the slots and return the bytes changed.
    //! A changed mapping synchronizes the device, so no kernel may use the slots and no CUDA graph capture may run.
    //! On failure every slot keeps its state.
    size_t setBacked(std::vector<SlotRange> const& ranges, bool backed);

    size_t setBacked(void* ptr, std::vector<int> const& slots, bool backed)
    {
        return setBacked({SlotRange{ptr, slots}}, backed);
    }

    //! Return the device bytes that setBacked(ptr, slots, true) would map now.
    size_t bytesToBack(void* ptr, std::vector<int> const& slots);

private:
    struct Scratch;

    // lastSlot may be the slot count, which stands for the range's tail.
    struct Piece
    {
        CUdeviceptr address;
        size_t size;
        int firstSlot;
        int lastSlot;
        bool backed;
        CUDAVirtualMemoryChunk chunk;
    };

    struct Allocation
    {
        // Declared first so that it is freed after the chunks are unmapped.
        std::shared_ptr<void> reservation;
        std::vector<uint8_t> unbacked;
        std::vector<uint8_t> backed;
        std::vector<Piece> pieces;
        std::vector<CUDAVirtualMemoryChunk> resident;
        bool hostAccessible;
    };

    struct Layout
    {
        std::vector<uint8_t> unbacked;
        size_t chunks;
        // Per chunk, the unbacked slot holding all its bytes, -1 if a byte is in a backed slot, or a value of its own
        // below -1 if it spans several unbacked slots. A run of one value is a piece.
        std::vector<int64_t> slotOf;
    };

    UnbackedSlotMemory() = default;

    //! Plan the layout of the slots on the current device, which becomes the device this instance serves.
    Layout plan(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    //! Return the first or last slot holding bytes of chunk, counting the tail as slot number slots.
    size_t firstSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;
    size_t lastSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;

    Allocation& findAllocation(void* ptr, std::vector<int> const& slots);
    static bool needsBacking(Piece const& piece, std::vector<uint8_t> const& backed);

    //! Map [address, address + size) to memory of its own (own) or to the scratch.
    CUDAVirtualMemoryChunk mapChunk(CUdeviceptr address, size_t size, bool own, bool hostAccessible) const;
    //! Map a piece to its other state, or restore its state and throw.
    void remap(Piece& piece, bool hostAccessible) const;
    static void releaseChunks(Allocation& allocation) noexcept;

    std::mutex mMutex;
    std::map<CUdeviceptr, Allocation> mAllocations;
    int mDevice = -1;
    size_t mGranularity = 0;
    // The chunks that map the scratch keep it alive.
    mutable std::weak_ptr<Scratch const> mScratch;
};

} // namespace tensorrt_llm::runtime
