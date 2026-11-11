/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/core/Allocator.h>   // for c10::DataPtr
#include <c10/core/StorageImpl.h> // for c10::StorageImpl and use_byte_size_t()
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/accumulate.h>
#include <c10/util/intrusive_ptr.h> // for c10::make_intrusive#include <torch/extension.h>
#include <vector>

#include "tensorrt_llm/kernels/moe/loadBalance/moeLoadBalanceKernels.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/unbackedSlotMemory.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor moeLoadBalanceWaitGpuStage(int64_t singleLayerLoadBalancerPtr)
{
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto enabled = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto signal = loadBalancer->getSignal();
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeWaitSignalForGpuStageDevice(signal, enabled.data_ptr<int>(), stream);

    return enabled;
}

void moeLoadBalanceSetCpuStage(int64_t singleLayerLoadBalancerPtr)
{
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto signal = loadBalancer->getSignal();
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeSetSignalForCpuStageDevice(signal, stream);
}

void moeLoadBalanceStatistic(torch::Tensor gatheredRawExpertIds, torch::Tensor enabled,
    int64_t singleLayerLoadBalancerPtr, int64_t isFirstStage, int64_t isLastStage)
{
    CHECK_INPUT(gatheredRawExpertIds, torch::kInt32);
    CHECK_INPUT(enabled, torch::kInt32);
    TORCH_CHECK(gatheredRawExpertIds.dim() == 2, "gatheredRawExpertIds must be a 2D tensor");
    int topK = gatheredRawExpertIds.size(1);
    TORCH_CHECK(enabled.dim() == 1, "enabled must be a 1D tensor");
    TORCH_CHECK(enabled.size(0) == 1, "enabled must have 1 element");
    TORCH_CHECK(isFirstStage == 0 || isFirstStage == 1, "isFirstStage must be 0 or 1");
    TORCH_CHECK(isLastStage == 0 || isLastStage == 1, "isLastStage must be 0 or 1");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");

    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();

    TORCH_CHECK(topK == metaInfo.topK, "topK must be equal to metaInfo.topK");

    auto statisticInfo = loadBalancer->getStatisticInfo();
    int numTotalTokens = gatheredRawExpertIds.size(0);

    tensorrt_llm::kernels::moeStatisticDevice(metaInfo, *statisticInfo, numTotalTokens, enabled.data_ptr<int>(),
        static_cast<bool>(isFirstStage), static_cast<bool>(isLastStage), gatheredRawExpertIds.data_ptr<int>(), stream);
}

void moeHierarchicalStatisticLocalDevice(torch::Tensor localRawExpertIds, torch::Tensor localExpertTokenCount,
    torch::Tensor enabled, int64_t singleLayerLoadBalancerPtr, int64_t isFirstStage, int64_t isLastStage)
{
    CHECK_INPUT(localRawExpertIds, torch::kInt32);
    CHECK_INPUT(localExpertTokenCount, torch::kInt32);
    CHECK_INPUT(enabled, torch::kInt32);
    TORCH_CHECK(localRawExpertIds.dim() == 2, "localRawExpertIds must be a 2D tensor");
    TORCH_CHECK(localExpertTokenCount.dim() == 1, "localExpertTokenCount must be a 1D tensor");
    int topK = localRawExpertIds.size(1);
    TORCH_CHECK(enabled.dim() == 1, "enabled must be a 1D tensor");
    TORCH_CHECK(enabled.size(0) == 1, "enabled must have 1 element");
    TORCH_CHECK(isFirstStage == 0 || isFirstStage == 1, "isFirstStage must be 0 or 1");
    TORCH_CHECK(isLastStage == 0 || isLastStage == 1, "isLastStage must be 0 or 1");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");

    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();

    TORCH_CHECK(localExpertTokenCount.size(0) == metaInfo.expertCount, "localExpertTokenCount should have shape (%d,)",
        metaInfo.expertCount);
    TORCH_CHECK(topK == metaInfo.topK, "topK must be equal to metaInfo.topK");

    int numTotalTokens = localRawExpertIds.size(0);

    tensorrt_llm::kernels::moeHierarchicalStatisticLocalDevice(metaInfo, numTotalTokens,
        localExpertTokenCount.data_ptr<int>(), enabled.data_ptr<int>(), static_cast<bool>(isFirstStage),
        static_cast<bool>(isLastStage), localRawExpertIds.data_ptr<int>(), stream);
}

void moeHierarchicalStatisticUpdate(
    torch::Tensor globalExpertTokenCount, torch::Tensor enabled, int64_t singleLayerLoadBalancerPtr)
{
    CHECK_INPUT(globalExpertTokenCount, torch::kInt32);
    CHECK_INPUT(enabled, torch::kInt32);
    TORCH_CHECK(globalExpertTokenCount.dim() == 1, "globalExpertTokenCount must be a 1D tensor");
    TORCH_CHECK(enabled.dim() == 1, "enabled must be a 1D tensor");
    TORCH_CHECK(enabled.size(0) == 1, "enabled must have 1 element");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();
    auto statisticInfo = loadBalancer->getStatisticInfo();

    TORCH_CHECK(globalExpertTokenCount.size(0) == metaInfo.expertCount,
        "globalExpertTokenCount should have shape (%d,)", metaInfo.expertCount);
    tensorrt_llm::kernels::moeHierarchicalStatisticUpdate(
        metaInfo, *statisticInfo, globalExpertTokenCount.data_ptr<int>(), enabled.data_ptr<int>(), stream);
}

torch::Tensor moeLoadBalanceRouting(
    torch::Tensor tokenSelectedExperts, bool offsetByEpRank, int64_t singleLayerLoadBalancerPtr)
{
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    int topK = tokenSelectedExperts.size(1);
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    TORCH_CHECK(singleLayerLoadBalancerPtr != 0, "singleLayerLoadBalancerPtr must be non-null");
    auto* loadBalancer
        = reinterpret_cast<tensorrt_llm::runtime::SingleLayerMoeLoadBalancer*>(singleLayerLoadBalancerPtr);
    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo = loadBalancer->getMetaInfo();

    TORCH_CHECK(topK == metaInfo.topK, "topK must be equal to metaInfo.topK");

    int tokenCount = tokenSelectedExperts.size(0);

    auto tokenRoutedSlotIds = torch::empty_like(tokenSelectedExperts);

    tensorrt_llm::kernels::moeComputeRouteDevice(metaInfo, loadBalancer->getGpuPlacementInfo(),
        tokenSelectedExperts.data_ptr<int>(), tokenRoutedSlotIds.data_ptr<int>(), tokenCount, offsetByEpRank, stream,
        loadBalancer->routesEachExpertToOneSlot());

    return tokenRoutedSlotIds;
}

void migrateToHostAccessible(at::Tensor& tensor)
{
    TORCH_CHECK(tensor.device().is_cuda(), "only support CUDA Tensor");

    TLLM_CHECK_WITH_INFO(tensorrt_llm::runtime::HostAccessibleDeviceAllocator::getInstance().isSupported(),
        "host accessible allocator is not supported on system, please install GDRCopy.");

    // 1) compute total bytes
    size_t byte_size = tensor.numel() * tensor.element_size();

    // 2) allocate host accessible memory
    void* devPtr = tensorrt_llm::runtime::HostAccessibleDeviceAllocator::getInstance().allocate(byte_size);

    // 3) copy old data to new memory
    TLLM_CUDA_CHECK(cudaMemcpy(devPtr, tensor.data_ptr(), byte_size, cudaMemcpyDeviceToDevice));

    // 4) use new DataPtr/StorageImpl to construct storage
    //    here managed_ptr is data，and also context，use cudaFree as deleter
    c10::DataPtr dp(
        devPtr, devPtr,
        [](void* ptr) { tensorrt_llm::runtime::HostAccessibleDeviceAllocator::getInstance().free(ptr); },
        tensor.device());
    auto allocator = c10::GetAllocator(tensor.device().type());
    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), byte_size,
        std::move(dp), allocator,
        /*resizable=*/false);
    at::Storage new_storage(storage_impl);

    // Finally replace tensor's storage，offset = 0，shape and stride kept unchanged
    tensor.set_(new_storage,
        /*storage_offset=*/0, tensor.sizes().vec(), tensor.strides().vec());
}

// Whether slotCount slots of slotBytes each take less device memory with the unbackedSlots unbacked than plainly
// allocated (UnbackedSlotMemory::savesMemory, on the current device).
bool moeUnbackedSlotsSaveMemory(int64_t slotBytes, int64_t slotCount, at::IntArrayRef unbackedSlots)
{
    TORCH_CHECK(slotBytes > 0 && slotCount > 0, "expects slots of at least one byte");
    return tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().savesMemory(static_cast<size_t>(slotBytes),
        static_cast<int>(slotCount), std::vector<int>(unbackedSlots.begin(), unbackedSlots.end()));
}

// A new contiguous tensor of slots (dimension 0 of sizes) on the current CUDA device whose unbackedSlots map no device
// memory of their own (UnbackedSlotMemory); for slots that moeUnbackedSlotsSaveMemory says save memory. Its contents
// are undefined, and an access to an unbacked slot reads or writes the scratch. Releasing its storage synchronizes its
// device, so the tensor must not be freed during CUDA graph capture. With hostAccessible, online EPLB's updates write
// its slots with CPU copies, as the slots migrateToHostAccessible makes.
at::Tensor moeUnbackedSlotTensor(
    at::IntArrayRef sizes, at::ScalarType dtype, at::IntArrayRef unbackedSlots, bool hostAccessible)
{
    TORCH_CHECK(!sizes.empty() && sizes[0] > 0, "expects the sizes of a tensor of slots");
    size_t const byteSize = c10::multiply_integers(sizes) * c10::elementSize(dtype);
    TORCH_CHECK(byteSize > 0, "expects slots of at least one byte");
    int const slotCount = static_cast<int>(sizes[0]);
    at::Device const device(at::kCUDA, c10::cuda::current_device());
    void* devPtr = tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().allocate(
        byteSize / slotCount, slotCount, std::vector<int>(unbackedSlots.begin(), unbackedSlots.end()), hostAccessible);
    c10::DataPtr dp(
        devPtr, devPtr, [](void* ptr) { tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().free(ptr); }, device);
    auto storageImpl = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), byteSize,
        std::move(dp), c10::GetAllocator(device.type()), /*resizable=*/false);
    return at::empty({0}, at::TensorOptions().dtype(dtype).device(device))
        .set_(at::Storage(storageImpl), /*storage_offset=*/0, sizes);
}

// Backs unbacked slots of tensors that moeUnbackedSlotTensor created with device memory of their own at the same
// addresses (contents undefined: the caller loads them), or with backed=false maps them to the shared scratch again:
// of tensors[i], the next slotCounts[i] of slots. Returns the device bytes mapped or released. Synchronizes the device
// once if a mapping changes: only while no kernel uses the slots, and not during CUDA graph capture.
int64_t moeUnbackedSlotsSetBacked(
    at::TensorList tensors, at::IntArrayRef slots, at::IntArrayRef slotCounts, bool backed)
{
    TORCH_CHECK(!tensors.empty() && tensors.size() == slotCounts.size(), "expects a slot count per tensor of slots");
    std::vector<tensorrt_llm::runtime::UnbackedSlotMemory::SlotRange> ranges;
    ranges.reserve(tensors.size());
    auto next = slots.begin();
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        TORCH_CHECK(tensors[i].device() == tensors.front().device() && tensors[i].device().is_cuda(),
            "expects CUDA tensors of slots on one device");
        TORCH_CHECK(
            slotCounts[i] >= 0 && slotCounts[i] <= slots.end() - next, "expects slotCounts to sum to the slots");
        ranges.push_back({tensors[i].storage().data_ptr().get(), std::vector<int>(next, next + slotCounts[i])});
        next += slotCounts[i];
    }
    TORCH_CHECK(next == slots.end(), "expects slotCounts to sum to the slots");
    c10::cuda::CUDAGuard const deviceGuard(tensors.front().device());
    return static_cast<int64_t>(tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().setBacked(ranges, backed));
}

// The device bytes moeUnbackedSlotsSetBacked({tensor}, slots, {slots.size()}, true) would map now.
int64_t moeUnbackedSlotsBytesToBack(at::Tensor const& tensor, at::IntArrayRef slots)
{
    TORCH_CHECK(tensor.device().is_cuda(), "expects a CUDA tensor of slots");
    return static_cast<int64_t>(tensorrt_llm::runtime::UnbackedSlotMemory::getInstance().bytesToBack(
        tensor.storage().data_ptr().get(), std::vector<int>(slots.begin(), slots.end())));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_load_balance_wait_gpu_stage(int single_layer_load_balancer_ptr) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_load_balance_wait_gpu_stage", &tensorrt_llm::torch_ext::moeLoadBalanceWaitGpuStage);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_load_balance_set_cpu_stage(int single_layer_load_balancer_ptr) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_load_balance_set_cpu_stage", &tensorrt_llm::torch_ext::moeLoadBalanceSetCpuStage);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_load_balance_statistic(Tensor gathered_raw_expert_ids, Tensor enabled, int "
        "single_layer_load_balancer_ptr, int is_first_stage, int is_last_stage) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_load_balance_statistic", &tensorrt_llm::torch_ext::moeLoadBalanceStatistic);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_hierarchical_statistic_local_device(Tensor local_raw_expert_ids, Tensor local_expert_token_count, Tensor "
        "enabled, int "
        "single_layer_load_balancer_ptr, int is_first_stage, int is_last_stage) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_hierarchical_statistic_local_device", &tensorrt_llm::torch_ext::moeHierarchicalStatisticLocalDevice);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_hierarchical_statistic_update(Tensor global_expert_token_count, Tensor enabled, int "
        "single_layer_load_balancer_ptr) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_hierarchical_statistic_update", &tensorrt_llm::torch_ext::moeHierarchicalStatisticUpdate);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_load_balance_routing(Tensor token_selected_experts, bool offset_by_ep_rank, "
        "int single_layer_load_balancer_ptr) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_load_balance_routing", &tensorrt_llm::torch_ext::moeLoadBalanceRouting);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("migrate_to_host_accessible(Tensor tensor) -> ()");
    m.def("moe_unbacked_slots_save_memory(int slot_bytes, int slot_count, int[] unbacked_slots) -> bool");
    m.def(
        "moe_unbacked_slot_tensor(int[] sizes, ScalarType dtype, int[] unbacked_slots, bool host_accessible=False) -> "
        "Tensor");
    m.def("moe_unbacked_slots_set_backed(Tensor(a!)[] tensors, int[] slots, int[] slot_counts, bool backed) -> int");
    m.def("moe_unbacked_slots_bytes_to_back(Tensor tensor, int[] slots) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("migrate_to_host_accessible", &tensorrt_llm::torch_ext::migrateToHostAccessible);
    m.impl("moe_unbacked_slots_set_backed", &tensorrt_llm::torch_ext::moeUnbackedSlotsSetBacked);
    m.impl("moe_unbacked_slots_bytes_to_back", &tensorrt_llm::torch_ext::moeUnbackedSlotsBytesToBack);
}

// No tensor argument picks the backend: the slot tensor is created on the current CUDA device.
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_unbacked_slots_save_memory", &tensorrt_llm::torch_ext::moeUnbackedSlotsSaveMemory);
    m.impl("moe_unbacked_slot_tensor", &tensorrt_llm::torch_ext::moeUnbackedSlotTensor);
}
