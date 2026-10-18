/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// Configuration constants
static constexpr int kMaxTopK = 22;      // Maximum top-k experts per token
static constexpr int kMaxPayloads = 4;   // Maximum number of different payload types
static constexpr int kMaxRanks = 256;    // Maximum supported EP size (covers NVL576 EP256)
static constexpr int kRankMaskWords = 4; // uint64 words to hold the active-rank bitmask
                                         // (kRankMaskWords * 64 must be >= kMaxRanks)
static_assert(kRankMaskWords * 64 >= kMaxRanks, "active_rank_mask too small for kMaxRanks");

// CFT counted write counter stride: 256B per counter to avoid L2 XBAR port camping
// when concurrent counters update. See NCCL CFT perf study (CST tracking structure).
static constexpr size_t kCftCounterStride = 256;
static constexpr size_t kCftCounterStrideU64 = kCftCounterStride / sizeof(uint64_t);

// Smem slot reserved for an mbarrier in the CFT dispatch / push-combine kernels.
// The mbarrier itself is only 8 B (a uint64_t initialized via mbarrier.init.shared.b64);
// the 64 B reservation is padding so the staging buffer that immediately follows is
// 16 B-aligned (cp.async.bulk requires 16 B-aligned source AND destination).
static constexpr int kCftMbarrierSlotBytes = 64;

// Fixed-size peer metadata passed by value to the CFT combine push kernel.
struct CftCombinePeerInfo
{
    uint32_t ids[kMaxRanks];
    uint32_t* completion_flags[kMaxRanks];
    uint64_t const* active_rank_mask; // See DispatchKernelPointers::active_rank_mask
};

// Nominal clock64 rate used to convert timeout seconds to SM cycles.
static constexpr int64_t kAssumedClockHz = 2000ll * 1000ll * 1000ll;
static constexpr int64_t kDefaultTimeoutCycles = 300ll * kAssumedClockHz;

// A wait on a peer reads the host abort record only once it has lasted this long (50 us at an assumed 2 GHz), and
// then once per period: healthy waits are shorter, and concurrent host reads from every waiting thread would slow
// them down.
static constexpr int64_t kAbortPollCycles = 100ll * 1000ll;

// Abort record in pinned host memory. The host sets `requested` with a plain CPU store to make rank-mask kernels
// that wait for peers return; a kernel that observes it records `status = kCancelled`. While the record is nonzero
// the outputs of every collective that uses it are invalid.
struct MoeA2AAbortState
{
    static constexpr int32_t kCancelled = 1;

    int32_t requested; // Set by the host to release kernels waiting on lost peers
    int32_t status;    // Diagnostic: kCancelled once a dispatch or combine returned early
};

// Default per-block dynamic shared-memory cap on sm_90+; larger requests must opt in via
// cudaFuncAttributeMaxDynamicSharedMemorySize.
static constexpr int kDefaultDynamicSmemBytes = 48 * 1024;

// Describes a single payload type to be communicated
struct PayloadDescriptor
{
    void const* src_data;   // Source data pointer [local_num_tokens, elements_per_token]
    int element_size;       // Size of each element in bytes
    int elements_per_token; // Number of elements per token (e.g., hidden_size, top_k)
};

// Kernel pointers packed into a struct for device access
// Dispatch kernel pointers - const source data
struct DispatchKernelPointers
{
    // Payload pointers
    void const* src_data_ptrs[kMaxPayloads];     // Array of source data pointers
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // 2D array of receive buffer pointers
    int payload_bytes_per_token[kMaxPayloads];   // Bytes per token for each payload
    // Completion flags for synchronization (fence-based path)
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // LE dispatch counters: HW-incremented byte counters for dispatch data.
    // le_dispatch_counters[target_rank][source_rank] is incremented by fabric engine
    // by the number of bytes written from source_rank to target_rank.
    uint64_t* le_dispatch_counters[kMaxRanks];

    // Local aux data pointers
    int* send_counters; // [ep_size] How many tokens have been sent to each target rank
    // Each rank owns recv_counters[parity][source_rank]. The two parity banks
    // alternate between A2A rounds.
    int* recv_counters[kMaxRanks];
    int* local_token_counter; // Atomic counter for completed tokens

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int* topk_target_ranks;   // target rank per k, -1 for invalid or duplicate routes
    int* topk_target_indices; // dst index per k, -1 for invalid or duplicate routes

    // Optional: Statistics for EPLB
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // CFT handle-based counted writes (fabric.try_put.counted via Logical Endpoints)
    uint32_t peer_le_ids[kMaxRanks];           // LE ID per target rank (from CftLeManager)
    uint64_t le_payload_offsets[kMaxPayloads]; // Byte offset of each payload's recv region within LE
    uint64_t le_counter_base;                  // Base byte offset for per-source-rank 8B counters within LE

    // Cumulative counter baseline for dispatch data counters (regular device memory, not LE-backed).
    // LE counters grow monotonically; the baseline tracks the previous cumulative value.
    uint64_t* dispatch_counter_baseline; // [ep_size]

    // Optional CFT dispatch-side expert-id sanitization.  Invalid padding tokens in the received
    // expert-id payload are filled after recv_counters are known, avoiding a separate sanitize kernel.
    bool sanitize_expert_ids;
    int expert_id_payload_index;
    int32_t invalid_expert_id;

    // Active-rank bitmask in device memory, read at execution time: bit i set => rank i is alive and participates
    // in this collective. Word w covers ranks 64*w..64*w+63 for w in [0, kRankMaskWords). Tokens routed to a masked
    // rank are dropped (topk_*[k] = -1); flag writes/waits to/from masked peers are skipped. Its address is fixed,
    // so captured CUDA graphs observe a mask rewritten in place while no collective runs. Its writer keeps the
    // local rank's own bit set.
    uint64_t const* active_rank_mask{nullptr};
    // Optional (rank-mask fence path): an abort request releases the completion-flag waits.
    MoeA2AAbortState* abort{nullptr};
    // Optional (rank-mask fence path): with slot_tokens > 0, source rank r writes token t of payload p at row
    // r * slot_tokens + t of recv_buffers[target][p], a fixed slot per source whatever the runtime batch, so a late
    // write from a lost rank reaches only its own slot. source_valid_counts ([ep_size], local) then receives the rows
    // each active source sent this round once its flag arrived, and 0 otherwise. When the round aborts, sources
    // whose flags arrived may still get their counts: the caller discards an aborted step without reading them.
    int slot_tokens{0};
    int* source_valid_counts{nullptr};
    // Optional (fixed slots): one extra CTA per source copies the routed expert ids (payload compact_payload_index,
    // int32 [TOP_K] per token) and router weights (the next payload, float [TOP_K]) of the rows that source sent to
    // compact [ep_size * max_tokens_per_rank, TOP_K] lists once its flag arrives; its other rows get -1 and 0.
    int32_t* compact_expert_ids{nullptr};
    float* compact_weights{nullptr};
    int compact_payload_index{0};
    // Optional (fixed slots with EPLB statistics): [ep_size, eplb_stats_num_experts], local. The same CTAs copy each
    // active source's row of the gathered statistics here once its flag arrives (0 for a masked source), so readers
    // never see a lost rank's late writes into the gathered buffer.
    int* eplb_stats_out{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};
};

// Gather one contribution slice per expert rank.
struct CombineKernelPointers
{
    void* output;
    // Fence: peer input slices. CFT: peer contributions in the local receive inbox.
    uint8_t const* source_buffers[kMaxRanks];

    // Combine readiness flags shared by the fence and CFT paths.
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint32_t* flag_val;                    // The value of the flag for this round (stored on the local rank)

    // Top-K compact routing info per local token (size: [local_num_tokens, top_k])
    int const* topk_target_ranks;   // target rank per k, -1 for invalid or duplicate routes
    int const* topk_target_indices; // dst index per k, -1 for invalid or duplicate routes

    // ---- CFT combine (counted-write) fields. Unused by the fence combine path. ----
    // Local LE combine counters: per receive-slot HW-incremented byte counters.
    uint64_t* combine_counters;
    // Cumulative per-slot counter baselines (regular device memory, single-buffer, advanced in place by reduce).
    uint64_t* combine_counter_baseline; // [ep_size * max_tokens_per_rank]
    int combine_counter_ep_stride = 0;  // STABLE static stride (maxNumTokens) for counter/baseline slot indexing
    // Active-rank bitmask: see DispatchKernelPointers::active_rank_mask. Combine skips
    // completion flag writes/waits to/from inactive peers.
    uint64_t const* active_rank_mask{nullptr};
    // Optional: see DispatchKernelPointers::abort; an aborted combine writes zeros.
    MoeA2AAbortState* abort{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};
};

// Dispatch phase parameters
struct MoeA2ADispatchParams
{
    // EP configuration
    int ep_size;     // Number of EP ranks
    int ep_rank;     // Current EP rank
    int num_experts; // Total number of experts

    // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
    int top_k;               // Number of experts per token

    // Expert routing information
    int32_t const* token_selected_experts; // [local_num_tokens, top_k]

    // Generic payloads
    int num_payloads;                         // Number of different payload types
    PayloadDescriptor payloads[kMaxPayloads]; // Array of payload descriptors

    // Local aux data
    uint32_t* flag_val;       // The value of the flag for this round (stored on the local rank)
    int* local_token_counter; // Atomic counter for completed tokens on this rank
    int* send_counters;       // [ep_size] atomic counters - tracks tokens sent to each target rank
    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
                            // per k, -1 for duplicates
    int* topk_target_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                              // per k, -1 for duplicates

    // Distributed aux data and recv buffers
    // Each rank owns recv_counters[parity][source_rank]. The two parity banks
    // alternate between A2A rounds.
    int* recv_counters[kMaxRanks];
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint64_t* le_dispatch_counters[kMaxRanks];   // HW-incremented byte counters (counted writes path)
    void* recv_buffers[kMaxRanks][kMaxPayloads]; // Per-rank receive buffers for each payload

    // Optional: Statistics for EPLB
    bool enable_eplb;                    // Whether to enable EPLB
    int eplb_stats_num_experts;          // Number of experts for EPLB stats
    int const* eplb_local_stats;         // [eplb_stats_num_experts]
    int* eplb_gathered_stats[kMaxRanks]; // [ep_size, eplb_stats_num_experts] per rank

    // CFT handle-based counted writes: use fabric.try_put.counted via Logical Endpoints
    bool use_cft_counted_writes;
    uint32_t cft_peer_le_ids[kMaxRanks];           // LE ID per target rank
    uint64_t cft_le_payload_offsets[kMaxPayloads]; // Byte offset of each payload's recv region within LE
    uint64_t cft_le_counter_base;                  // Base byte offset for per-source-rank 8B counters within LE

    // Cumulative counter baselines (regular device memory)
    uint64_t* cft_dispatch_counter_baseline; // [ep_size]

    // Optional CFT dispatch-side expert-id sanitization.
    bool sanitize_expert_ids;
    int expert_id_payload_index;
    int32_t invalid_expert_id;

    // Whether to instantiate a kernel with active-rank checks.
    // This is a launch-lifetime mode, independent of future execution-abort handling.
    bool enable_rank_mask{false};

    // Active-rank bitmask in device memory: see DispatchKernelPointers::active_rank_mask. Required when
    // enable_rank_mask is true; kernels read it through this pointer at execution time.
    uint64_t const* active_rank_mask{nullptr};
    // Optional: see DispatchKernelPointers::abort, slot_tokens, source_valid_counts and compact_*.
    MoeA2AAbortState* abort{nullptr};
    int slot_tokens{0};
    int* source_valid_counts{nullptr};
    int32_t* compact_expert_ids{nullptr};
    float* compact_weights{nullptr};
    int compact_payload_index{0};
    int* eplb_stats_out{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};

    // CUDA stream
    cudaStream_t stream;
};

// Dispatch kernels
void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params);
// Prepare for dispatch: zero send_counters, local_token_counter and increment flag_val
void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params);

// Combine phase parameters
struct MoeA2ACombineParams
{
    // EP configuration
    int ep_size; // Number of EP ranks
    int ep_rank; // Current EP rank

    // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
    int top_k;               // Number of experts per token

    // Resolved payload plan. The source always points to the MoE output tensor;
    // the host determines all storage strides before launching any kernel.
    void const* source_payload;
    int source_stride_per_token;
    int workspace_stride_per_token;
    int wire_bytes_per_token;
    int prepare_first_token;
    int prepare_num_tokens;
    void const* cft_push_payload;
    int cft_push_stride_per_token;
    int reduce_stride_per_token;

    // Output tensor
    void* output_data; // Output buffer [local_num_tokens, elements_per_token]
    // Payload information
    int elements_per_token;       // Number of elements per token
    tensorrt_llm::DataType dtype; // Data type of the payload (used for combine kernel dispatch)
    bool
        use_low_precision; // If true, prepare kernel quantizes payload→FP8; combine kernel accumulates FP8→output dtype

    // Local aux data
    uint32_t* flag_val;     // The value of the flag for this round (stored on the local rank)
    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
                            // per k, -1 for duplicates
    int* topk_target_indices; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), dst index
                              // per k, -1 for duplicates
    // Local recv_counters[parity][source_rank]. The two parity banks alternate
    // between A2A rounds.
    int const* recv_counters;

    // Distributed aux data and recv buffers
    uint32_t* completion_flags[kMaxRanks]; // If completion_flags[target_rank][source_rank] == *flag_val, then source
                                           // rank has signaled the target rank
    uint8_t* combine_input_buffers[kMaxRanks]; // Expert-output/staging region on each rank

    // ---- CFT combine (counted-write) path. Gated by use_cft_for_combine. ----
    // When true, moe_a2a_combine_launch takes the CFT push+reduce path and the base fence
    // combine below is bypassed. The base fence combine is unaffected when false.
    bool use_cft_for_combine;
    uint32_t cft_peer_le_ids[kMaxRanks];    // LE ID per target rank
    uint64_t cft_combine_recv_offset;       // LE byte offset of the combine receive inbox
    uint64_t cft_le_combine_counter_base;   // LE byte offset for combine counters
    uint64_t* cft_le_combine_counters;      // Direct pointer to local LE combine counters
    uint8_t* cft_combine_recv_payload;      // Local CFT receive inbox, including the self contribution
    uint64_t* cft_combine_counter_baseline; // [ep_size * max_tokens_per_rank] regular device memory
    int combine_counter_ep_stride = 0;      // STABLE static stride (maxNumTokens) for counter/baseline slot indexing

    // Whether to instantiate a kernel with active-rank checks in peer synchronization.
    // This is a launch-lifetime mode, independent of future execution-abort handling.
    bool enable_rank_mask{false};

    // Active-rank bitmask in device memory: see DispatchKernelPointers::active_rank_mask. Required when
    // enable_rank_mask is true; kernels read it through this pointer at execution time.
    uint64_t const* active_rank_mask{nullptr};
    // Optional: see CombineKernelPointers::abort.
    MoeA2AAbortState* abort{nullptr};

    // Host-selected wait budget in clock64() cycles.
    int64_t timeout_cycles{kDefaultTimeoutCycles};

    // CUDA stream
    cudaStream_t stream;
};

// Combine kernels
void moe_a2a_combine_launch(MoeA2ACombineParams const& params);

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params);

// CFT combine push: processing rank pushes expert output back to originating rank's LE.
void moe_a2a_cft_combine_push_launch(MoeA2ACombineParams const& params);

// Sanitize expert IDs for invalid tokens
// expert_ids: [ep_size, max_tokens_per_rank, top_k] (int32)
// recv_counters: [2, ep_size] (int32), number of valid tokens per source
// invalid_id: value to fill for invalid tokens' expert ids
void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, uint32_t const* flag_val,
    int32_t invalid_id, int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
