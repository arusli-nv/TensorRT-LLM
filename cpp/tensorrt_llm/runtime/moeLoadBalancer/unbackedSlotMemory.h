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

// Device storage for an EPLB slot tensor whose unbacked (spare) slots hold no device memory of their own.
// The tensor is one virtual address range that stays fully mapped for the life of the tensor (CUDA graphs capture
// its address). Chunks of the VMM granularity that lie wholly inside unbacked slots form pieces, cut at slot
// boundaries. Each piece is a CUDAVirtualMemoryChunk that maps one scratch chunk of the process at each of its chunks
// and grants access to all of them at once, so an access that strays into an unbacked slot reads or writes scratch
// instead of faulting, and unbacked slots cost one chunk of device memory per process.
// Backing a slot releases its pieces and materializes them again at the same addresses with memory of their own.
// The other chunks of the range are mapped to their own device memory, one allocation per run of them.
// A host accessible range (for online EPLB, whose updates write the slots with CPU copies) maps each allocation of its
// own memory for the host with GDRCopy while the allocation lives (HostAccessibleDeviceAllocator::mapDeviceMemory), so
// a slot whose bytes lie in several allocations is written piece by piece.
class UnbackedSlotMemory
{
public:
    static UnbackedSlotMemory& getInstance();

    // Whether allocate(slotBytes, slotCount, unbackedSlots) on the current device would leave at least one chunk
    // unbacked, so that it takes less device memory than a plain allocation of the slots (otherwise it takes up to
    // one chunk more).
    bool savesMemory(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    // A range for slotCount slots of slotBytes each on the current device; the slots in unbackedSlots are not backed.
    // hostAccessible needs host accessible device memory to be GDRCopy mappings
    // (HostAccessibleDeviceAllocator::usesGdrCopy) and a load balancer to have initialized it.
    void* allocate(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots, bool hostAccessible = false);
    // Synchronizes the range's device, then unmaps and releases the range; errors are logged (a tensor deleter must not
    // throw). The synchronization makes it invalid during CUDA graph capture: free the tensor outside capture.
    void free(void* ptr) noexcept;

    // Slots of the range at ptr that allocate() left unbacked.
    struct SlotRange
    {
        void* ptr;
        std::vector<int> slots;
    };

    // Backs the slots of every range with device memory of their own at the same addresses (contents undefined), or
    // with backed=false maps them to the scratch again. If that changes a mapping, synchronizes the current device (the
    // ranges') once first. Returns the device bytes mapped or released. No contents are kept: only for slots that hold
    // nothing the caller needs. On failure (e.g. out of device memory) every slot keeps its state and stays mapped;
    // only if mapping a piece's state again fails too is it left unmapped (an access to it faults; logged): then the
    // slots must not be used again, and free() still releases the ranges. Only while no kernel uses the slots, and not
    // during CUDA graph capture.
    size_t setBacked(std::vector<SlotRange> const& ranges, bool backed);

    size_t setBacked(void* ptr, std::vector<int> const& slots, bool backed)
    {
        return setBacked({SlotRange{ptr, slots}}, backed);
    }

    // The device bytes setBacked(ptr, slots, true) would map now.
    size_t bytesToBack(void* ptr, std::vector<int> const& slots);

private:
    struct Scratch;

    // Chunks at address, all inside slots [firstSlot, lastSlot] (lastSlot may be the slot count: the range's tail).
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
        std::vector<uint8_t> unbacked;                // per slot: unbacked by allocate()
        std::vector<uint8_t> backed;                  // per slot: backed by setBacked()
        std::vector<Piece> pieces;                    // the pieces of unbacked slots, by address
        std::vector<CUDAVirtualMemoryChunk> resident; // the other chunks, mapped to their own device memory
        bool hostAccessible;                          // memory of its own is mapped for the host
    };

    // How allocate() maps a range of slots.
    struct Layout
    {
        std::vector<uint8_t> unbacked; // per slot: unbacked
        size_t chunks;
        // Per chunk: the unbacked slot holding all its bytes, -1 if a byte is in a backed slot, or a value of its own
        // below -1 if it spans several unbacked slots. A run of one value is a piece.
        std::vector<int64_t> slotOf;
    };

    UnbackedSlotMemory() = default;

    // The layout of slotCount slots of slotBytes each on the current device, which becomes the device this instance
    // serves; with mMutex held.
    Layout plan(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    // The first and the last of slots slots of slotBytes each holding bytes of chunk (the tail past the last slot
    // counts as slot slots).
    size_t firstSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;
    size_t lastSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;

    // The allocation at ptr, after checking that slots are among its unbacked slots.
    Allocation& findAllocation(void* ptr, std::vector<int> const& slots);
    // Whether a piece needs memory of its own when backed (per slot) marks the backed slots: while any of its slots
    // is backed (the range's tail never is).
    static bool needsBacking(Piece const& piece, std::vector<uint8_t> const& backed);

    // A materialized chunk mapping [address, address + size) to device memory of its own (own), mapped for the host
    // too with hostAccessible, or to the scratch.
    CUDAVirtualMemoryChunk mapChunk(CUdeviceptr address, size_t size, bool own, bool hostAccessible) const;
    // Maps a piece to its other state (memory of its own or the scratch); if that fails, maps its state again and
    // throws.
    void remap(Piece& piece, bool hostAccessible) const;
    // Unmaps and releases every chunk of an allocation; errors are logged.
    static void releaseChunks(Allocation& allocation) noexcept;

    std::mutex mMutex;
    std::map<CUdeviceptr, Allocation> mAllocations;
    int mDevice = -1;
    size_t mGranularity = 0;
    // The scratch; the chunks mapping it keep it alive, so it is released once no piece maps it.
    mutable std::weak_ptr<Scratch const> mScratch;
};

} // namespace tensorrt_llm::runtime
