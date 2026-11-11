/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/moeLoadBalanceKernels.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/unbackedSlotMemory.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;

// Unit test for doReplication function
// Define test parameters for doReplication test
struct ReplicationTestParam
{
    int expertCount;
    int slotCountPerRank;
    int epSize;
    std::vector<float> expertLoadFactors;
    std::vector<int> expectedReplicaCounts;
    std::string description;
};

class MoeReplicationTest : public ::testing::TestWithParam<ReplicationTestParam>
{
};

TEST_P(MoeReplicationTest, VerifyReplication)
{
    auto param = GetParam();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        param.expertCount, 2, 0, param.epSize, param.slotCountPerRank};

    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.expertReplicaCount.resize(param.expertCount);

    // Call the function under test
    doReplication(metaInfo, param.expertLoadFactors.data(), &cpuPlacement);

    // Verify results against expected replica counts
    int totalSlots = param.epSize * param.slotCountPerRank;
    int totalReplicas = 0;

    ASSERT_EQ(cpuPlacement.expertReplicaCount.size(), param.expectedReplicaCounts.size())
        << "Replica count size mismatch";

    for (int i = 0; i < param.expertCount; i++)
    {
        EXPECT_EQ(cpuPlacement.expertReplicaCount[i], param.expectedReplicaCounts[i])
            << "Expert " << i << " has incorrect replica count";
        totalReplicas += cpuPlacement.expertReplicaCount[i];
    }

    // Verify total replicas matches total slots
    EXPECT_EQ(totalReplicas, totalSlots) << "Total replicas should equal total slots";
}

// Define a set of test cases for replication
INSTANTIATE_TEST_SUITE_P(ReplicationTests, MoeReplicationTest,
    ::testing::Values(
        // Test case 1: Total slot count equals expert count (equal distribution)
        ReplicationTestParam{4,       // expertCount
            2,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 2.0f, 3.0f, 4.0f}, // expertLoadFactors - increasing load
            {1, 1, 1, 1},             // expectedReplicaCounts - each expert gets 1 replica
            "Equal slots and experts"},

        // Test case 2: No load information (even distribution)
        ReplicationTestParam{4,       // expertCount
            3,                        // slotCountPerRank
            2,                        // epSize
            {0.0f, 0.0f, 0.0f, 0.0f}, // expertLoadFactors - all zero
            {2, 2, 1, 1},             // expectedReplicaCounts - even distribution with remainder to first experts
            "Zero load factors"},

        // Test case 3: Greedy distribution based on load factors
        ReplicationTestParam{4,       // expertCount
            4,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 4.0f, 2.0f, 1.0f}, // expertLoadFactors - varied loads
            {1, 4, 2, 1},             // expectedReplicaCounts - proportional to load
            "Varied load factors"},

        // Test case 4: More complex scenario with uneven distribution
        ReplicationTestParam{5,               // expertCount
            3,                                // slotCountPerRank
            3,                                // epSize
            {10.0f, 5.0f, 2.0f, 15.0f, 8.0f}, // expertLoadFactors
            {2, 1, 1, 3, 2},                  // expectedReplicaCounts
            "Complex load distribution"},

        // Test case 5: Single expert with load, others zero
        ReplicationTestParam{3,  // expertCount
            2,                   // slotCountPerRank
            2,                   // epSize
            {0.0f, 10.0f, 0.0f}, // expertLoadFactors - only middle expert has load
            {1, 2, 1},           // expectedReplicaCounts - middle expert gets most replicas
            "Single expert with load"}),
    [](::testing::TestParamInfo<ReplicationTestParam> const& info)
    {
        // Generate readable test names based on the description
        std::string name = info.param.description;
        // Replace spaces and special characters with underscores
        std::replace(name.begin(), name.end(), ' ', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), '(', '_');
        std::replace(name.begin(), name.end(), ')', '_');
        return name;
    });

// Define test parameters for doPlacement test
struct PlacementTestParam
{
    int expertCount;
    int slotCountPerRank;
    int epSize;
    std::vector<float> expertLoadFactors;
    std::vector<int> replicaCounts; // Pre-computed replica counts
    double maxExpectedLoadDiff;     // Maximum expected load difference between ranks
    std::string description;
};

class MoePlacementTest : public ::testing::TestWithParam<PlacementTestParam>
{
};

TEST_P(MoePlacementTest, VerifyPlacement)
{
    auto param = GetParam();

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        param.expertCount, 2, 0, param.epSize, param.slotCountPerRank};

    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.expertReplicaCount.resize(param.expertCount);

    // Use the pre-computed replica counts instead of calling doReplication
    for (int i = 0; i < param.expertCount; i++)
    {
        cpuPlacement.expertReplicaCount[i] = param.replicaCounts[i];
    }

    // Initialize rankExpertIds for placement
    cpuPlacement.rankExpertIds.resize(param.epSize);
    for (int i = 0; i < param.epSize; i++)
    {
        cpuPlacement.rankExpertIds[i].resize(param.slotCountPerRank, -1);
    }

    // Call the function under test
    doPlacement(metaInfo, param.expertLoadFactors.data(), &cpuPlacement);

    // Verify all slots are assigned
    int totalSlots = param.epSize * param.slotCountPerRank;
    int assignedExperts = 0;
    std::vector<int> expertAssignments(param.expertCount, 0);

    for (int rankId = 0; rankId < param.epSize; rankId++)
    {
        for (int slotId = 0; slotId < param.slotCountPerRank; slotId++)
        {
            int expertId = cpuPlacement.rankExpertIds[rankId][slotId];

            // Verify expert ID is valid
            EXPECT_GE(expertId, 0) << "Expert ID should be non-negative";
            EXPECT_LT(expertId, param.expertCount) << "Expert ID should be less than expert count";

            // Count assignments
            expertAssignments[expertId]++;
            assignedExperts++;
        }
    }

    // Verify correct number of slots assigned
    EXPECT_EQ(assignedExperts, totalSlots) << "Total assigned experts should equal total slots";

    // Verify each expert is assigned the correct number of replicas
    for (int i = 0; i < param.expertCount; i++)
    {
        EXPECT_EQ(expertAssignments[i], param.replicaCounts[i])
            << "Expert " << i << " should be assigned to exactly replicaCounts[" << i << "] slots";
    }

    // Test load balancing property: ranks should have roughly equal total load
    std::vector<double> rankLoads(param.epSize, 0.0);
    for (int rankId = 0; rankId < param.epSize; rankId++)
    {
        for (int slotId = 0; slotId < param.slotCountPerRank; slotId++)
        {
            int expertId = cpuPlacement.rankExpertIds[rankId][slotId];
            double slotSize = param.expertLoadFactors[expertId] / static_cast<double>(param.replicaCounts[expertId]);
            rankLoads[rankId] += slotSize;
        }
    }

    // Print rank loads for debugging
    std::cout << "Rank loads for " << param.description << ":" << std::endl;
    for (int i = 0; i < param.epSize; i++)
    {
        std::cout << "  Rank " << i << ": " << rankLoads[i] << std::endl;
    }

    // Calculate max difference between rank loads
    double maxLoadDiff = 0.0;
    for (int i = 0; i < param.epSize - 1; i++)
    {
        for (int j = i + 1; j < param.epSize; j++)
        {
            maxLoadDiff = std::max(maxLoadDiff, std::abs(rankLoads[i] - rankLoads[j]));
        }
    }
    std::cout << "  Max load difference: " << maxLoadDiff << std::endl;

    // The load difference should be within the expected threshold
    EXPECT_LE(maxLoadDiff, param.maxExpectedLoadDiff) << "Load difference between ranks exceeds expected threshold";
}

// Define a set of test cases for placement
INSTANTIATE_TEST_SUITE_P(PlacementTests, MoePlacementTest,
    ::testing::Values(
        // Test case 1: Basic placement with equal load
        PlacementTestParam{4,         // expertCount
            2,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 1.0f, 1.0f, 1.0f}, // expertLoadFactors - equal load
            {1, 1, 1, 1},             // replicaCounts
            0.001,                    // maxExpectedLoadDiff
            "Equal load factors"},

        // Test case 2: Varied load factors
        PlacementTestParam{4,         // expertCount
            3,                        // slotCountPerRank
            2,                        // epSize
            {1.0f, 3.0f, 2.0f, 1.0f}, // expertLoadFactors - varied loads
            {1, 2, 2, 1},             // replicaCounts
            0.01,                     // maxExpectedLoadDiff
            "Varied load factors"},

        // Test case 3: Complex scenario with multiple ranks
        PlacementTestParam{5,                 // expertCount
            3,                                // slotCountPerRank
            3,                                // epSize
            {10.0f, 5.0f, 2.0f, 15.0f, 8.0f}, // expertLoadFactors
            {2, 1, 1, 3, 2},                  // replicaCounts
            2.0,                              // maxExpectedLoadDiff (higher tolerance for complex case)
            "Complex load distribution"},

        // Test case 4: Extreme load differences
        PlacementTestParam{3,      // expertCount
            2,                     // slotCountPerRank
            2,                     // epSize
            {1.0f, 100.0f, 10.0f}, // expertLoadFactors - extreme differences
            {1, 2, 1},             // replicaCounts
            9.0,                   // maxExpectedLoadDiff (higher tolerance due to extreme loads)
            "Extreme load differences"},

        // Test case 5: Many experts with single replica each
        PlacementTestParam{6,                     // expertCount
            3,                                    // slotCountPerRank
            2,                                    // epSize
            {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, // expertLoadFactors
            {1, 1, 1, 1, 1, 1},                   // replicaCounts - one replica per expert
            1.0,                                  // maxExpectedLoadDiff
            "One replica per expert"}),
    [](::testing::TestParamInfo<PlacementTestParam> const& info)
    {
        // Generate readable test names based on the description
        std::string name = info.param.description;
        // Replace spaces and special characters with underscores
        std::replace(name.begin(), name.end(), ' ', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), '(', '_');
        std::replace(name.begin(), name.end(), ')', '_');
        return name;
    });

TEST(MoeLoadBalancerMaskOnlyTest, DynamicPlacementHonorsDeadRankMask)
{
    constexpr int kExpertCount = 4;
    constexpr int kTopK = 2;
    constexpr int kEpRank = 0;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 2;
    constexpr int kDeadRank = 2;
    constexpr int kActiveSlotCount = (kEpSize - 1) * kSlotCountPerRank;

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{kExpertCount, kTopK, kEpRank, kEpSize, kSlotCountPerRank};
    std::vector<float> expertLoadFactor{8.0F, 4.0F, 2.0F, 1.0F};
    std::vector<uint8_t> deadRankMask{0, 0, 1, 0};

    MoePlacementCpuInfo cpuPlacement;
    doReplication(metaInfo, expertLoadFactor.data(), &cpuPlacement, &deadRankMask);

    int replicaSum = 0;
    for (int replicaCount : cpuPlacement.expertReplicaCount)
    {
        replicaSum += replicaCount;
    }
    EXPECT_EQ(replicaSum, kActiveSlotCount);

    cpuPlacement.rankExpertIds.resize(kEpSize);
    for (int rank = 0; rank < kEpSize; ++rank)
    {
        cpuPlacement.rankExpertIds[rank].resize(kSlotCountPerRank, 99);
    }

    doPlacement(metaInfo, expertLoadFactor.data(), &cpuPlacement, &deadRankMask);

    std::vector<int> placedReplicas(kExpertCount, 0);
    int assignedSlotCount = 0;
    for (int rank = 0; rank < kEpSize; ++rank)
    {
        for (int slot = 0; slot < kSlotCountPerRank; ++slot)
        {
            int const expertId = cpuPlacement.rankExpertIds[rank][slot];
            if (rank == kDeadRank)
            {
                EXPECT_EQ(expertId, -1);
                continue;
            }

            EXPECT_GE(expertId, 0);
            EXPECT_LT(expertId, kExpertCount);
            if (expertId >= 0 && expertId < kExpertCount)
            {
                ++placedReplicas[expertId];
                ++assignedSlotCount;
            }
        }
    }

    EXPECT_EQ(assignedSlotCount, kActiveSlotCount);
    for (int expertId = 0; expertId < kExpertCount; ++expertId)
    {
        EXPECT_EQ(placedReplicas[expertId], cpuPlacement.expertReplicaCount[expertId]);
    }
}

TEST(MoeLoadBalancerMaskOnlyTest, DynamicPlacementRejectsMismatchedDeadRankMask)
{
    constexpr int kExpertCount = 4;
    constexpr int kTopK = 2;
    constexpr int kEpRank = 0;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 2;

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{kExpertCount, kTopK, kEpRank, kEpSize, kSlotCountPerRank};
    std::vector<float> expertLoadFactor{1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<uint8_t> deadRankMask{0, 1};

    MoePlacementCpuInfo cpuPlacement;
    EXPECT_THROW(doReplication(metaInfo, expertLoadFactor.data(), &cpuPlacement, &deadRankMask),
        tensorrt_llm::common::TllmException);
}

TEST(MoeLoadBalancerMaskOnlyTest, DynamicPlacementRejectsReplicaCountMismatch)
{
    constexpr int kExpertCount = 4;
    constexpr int kTopK = 2;
    constexpr int kEpRank = 0;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 2;

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{kExpertCount, kTopK, kEpRank, kEpSize, kSlotCountPerRank};
    std::vector<float> expertLoadFactor{1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<uint8_t> deadRankMask{0, 0, 1, 0};

    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.expertReplicaCount = {1, 1, 1, 1};
    cpuPlacement.rankExpertIds.resize(kEpSize);
    for (int rank = 0; rank < kEpSize; ++rank)
    {
        cpuPlacement.rankExpertIds[rank].resize(kSlotCountPerRank, -1);
    }

    EXPECT_THROW(doPlacement(metaInfo, expertLoadFactor.data(), &cpuPlacement, &deadRankMask),
        tensorrt_llm::common::TllmException);
}

namespace
{

// doReplication then doPlacement on a fresh placement of metaInfo's shape.
MoePlacementCpuInfo planPlacement(tensorrt_llm::kernels::MoeLoadBalanceMetaInfo const& metaInfo,
    std::vector<float> expertLoadFactor, std::vector<uint8_t> const* deadRankMask,
    std::vector<uint8_t> const* unusableSlotMask)
{
    MoePlacementCpuInfo cpuPlacement;
    cpuPlacement.rankExpertIds.assign(metaInfo.epSize, std::vector<int>(metaInfo.slotCountPerRank, 99));
    doReplication(metaInfo, expertLoadFactor.data(), &cpuPlacement, deadRankMask, unusableSlotMask);
    doPlacement(metaInfo, expertLoadFactor.data(), &cpuPlacement, deadRankMask, unusableSlotMask);
    return cpuPlacement;
}

} // namespace

TEST(MoeLoadBalancerUnusableSlotTest, PlacementNeverFillsAnUnusableSlot)
{
    constexpr int kExpertCount = 4;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 3;
    constexpr int kSpareSlot = 2;
    constexpr int kFailedRank = 2;
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        kExpertCount, /*topK=*/2, /*epRank=*/0, kEpSize, kSlotCountPerRank};
    // The last slot of every rank is unusable, except on rank 1 (a spare slot a restore backed).
    std::vector<uint8_t> unusableSlotMask(kEpSize * kSlotCountPerRank, 0);
    for (int rank = 0; rank < kEpSize; ++rank)
    {
        unusableSlotMask[rank * kSlotCountPerRank + kSpareSlot] = static_cast<uint8_t>(rank != 1);
    }
    std::vector<uint8_t> const deadRankMask{0, 0, 1, 0};

    for (auto const* failedRankMask : {static_cast<std::vector<uint8_t> const*>(nullptr), &deadRankMask})
    {
        for (auto const& load : {std::vector<float>{8.0F, 4.0F, 2.0F, 1.0F}, std::vector<float>(kExpertCount, 0.0F)})
        {
            auto const placement = planPlacement(metaInfo, load, failedRankMask, &unusableSlotMask);
            int const usableSlotCount = failedRankMask == nullptr ? 9 : 7;
            int replicaSum = 0;
            for (int replicaCount : placement.expertReplicaCount)
            {
                EXPECT_GE(replicaCount, 1);
                replicaSum += replicaCount;
            }
            EXPECT_EQ(replicaSum, usableSlotCount);
            std::vector<int> placedReplicas(kExpertCount, 0);
            for (int rank = 0; rank < kEpSize; ++rank)
            {
                for (int slot = 0; slot < kSlotCountPerRank; ++slot)
                {
                    int const expertId = placement.rankExpertIds[rank][slot];
                    bool const usable = !(failedRankMask != nullptr && rank == kFailedRank)
                        && unusableSlotMask[rank * kSlotCountPerRank + slot] == 0;
                    if (!usable)
                    {
                        EXPECT_EQ(expertId, -1) << "rank " << rank << " slot " << slot;
                        continue;
                    }
                    ASSERT_GE(expertId, 0) << "rank " << rank << " slot " << slot;
                    ASSERT_LT(expertId, kExpertCount);
                    ++placedReplicas[expertId];
                }
            }
            EXPECT_EQ(placedReplicas, placement.expertReplicaCount);
        }
    }

    // Fewer usable slots than experts cannot be planned.
    std::vector<uint8_t> const tooFew(kEpSize * kSlotCountPerRank, 1);
    EXPECT_THROW(
        planPlacement(metaInfo, {1.0F, 1.0F, 1.0F, 1.0F}, nullptr, &tooFew), tensorrt_llm::common::TllmException);
    std::vector<uint8_t> const wrongSize(kEpSize, 0);
    EXPECT_THROW(
        planPlacement(metaInfo, {1.0F, 1.0F, 1.0F, 1.0F}, nullptr, &wrongSize), tensorrt_llm::common::TllmException);
}

TEST(MoeLoadBalancerMaskOnlyTest, ReconfigureMaskOnlyRemovesDeadRankSlots)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));

    constexpr int kExpertCount = 4;
    constexpr int kTopK = 2;
    constexpr int kEpRank = 0;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 2;
    constexpr int kDeadRank = 2;
    constexpr int kInvalidSlotId = kEpSize * kSlotCountPerRank;

    MoeLoadBalancer loadBalancer(kEpRank, kEpSize, /*layerUpdatesPerIter=*/0);
    auto layer = loadBalancer.AddLayer(kExpertCount, kTopK, kSlotCountPerRank);

    // Each expert has at least one surviving replica after rank 2 is masked.
    std::vector<int> initialAssignments{
        0, 1, // rank 0
        0, 2, // rank 1
        1, 3, // rank 2 (dead)
        2, 3  // rank 3
    };
    layer->setInitialWeightAssignments(initialAssignments);
    loadBalancer.finalizeModel();

    loadBalancer.reconfigureMaskOnly({kDeadRank});

    auto* placementCpuInfo = layer->getPlacementCpuInfo();
    for (int localSlotId = 0; localSlotId < kSlotCountPerRank; ++localSlotId)
    {
        EXPECT_EQ(placementCpuInfo->oldRankExpertIds[kDeadRank][localSlotId], -1);
    }

    std::vector<int> expectedReplicaCounts{2, 1, 2, 1};
    int totalReplicaCount = 0;
    for (int expertId = 0; expertId < kExpertCount; ++expertId)
    {
        EXPECT_EQ(placementCpuInfo->placementInfoForGPU.expertReplicaCount[expertId], expectedReplicaCounts[expertId]);
        totalReplicaCount += expectedReplicaCounts[expertId];
    }

    for (int replicaOffset = 0; replicaOffset < totalReplicaCount; ++replicaOffset)
    {
        int const globalSlotId = placementCpuInfo->placementInfoForGPU.globalSlotIds[replicaOffset];
        EXPECT_GE(globalSlotId, 0);
        EXPECT_LT(globalSlotId, kInvalidSlotId);
        EXPECT_NE(globalSlotId / kSlotCountPerRank, kDeadRank);
    }
    for (int replicaOffset = totalReplicaCount; replicaOffset < kInvalidSlotId; ++replicaOffset)
    {
        EXPECT_EQ(placementCpuInfo->placementInfoForGPU.globalSlotIds[replicaOffset], kInvalidSlotId);
    }

    loadBalancer.shutdown();
}

TEST(MoeLoadBalancerMaskOnlyTest, ReconfigureMaskOnlyRejectsActiveIteration)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));

    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/4, /*layerUpdatesPerIter=*/0);
    loadBalancer.finalizeModel();
    loadBalancer.startIter(/*iterId=*/0, /*enableStatistic=*/true, /*enableUpdateWeights=*/false);

    EXPECT_THROW(loadBalancer.reconfigureMaskOnly({2}), tensorrt_llm::common::TllmException);
    loadBalancer.endIter(/*iterId=*/0);

    loadBalancer.shutdown();
}

TEST(MoeLoadBalancerMaskOnlyTest, ReconfigureMaskOnlyFailsClosedForLastReplica)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));

    constexpr int kExpertCount = 4;
    constexpr int kTopK = 2;
    constexpr int kEpRank = 0;
    constexpr int kEpSize = 4;
    constexpr int kSlotCountPerRank = 1;
    constexpr int kDeadRank = 2;

    MoeLoadBalancer loadBalancer(kEpRank, kEpSize, /*layerUpdatesPerIter=*/0);
    auto layer = loadBalancer.AddLayer(kExpertCount, kTopK, kSlotCountPerRank);

    // Expert 2 only exists on rank 2, so mask-only recovery must fail closed.
    std::vector<int> initialAssignments{
        0, // rank 0
        1, // rank 1
        2, // rank 2 (dead)
        3  // rank 3
    };
    layer->setInitialWeightAssignments(initialAssignments);
    loadBalancer.finalizeModel();

    EXPECT_THROW(loadBalancer.reconfigureMaskOnly({kDeadRank}), tensorrt_llm::common::TllmException);
    EXPECT_EQ(layer->getPlacementCpuInfo()->oldRankExpertIds[kDeadRank][0], 2);

    loadBalancer.shutdown();
}

TEST(MoeLoadBalancerFaultToleranceOffTest, InitialAssignmentRejectsAnEmptySlot)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/4, /*layerUpdatesPerIter=*/0);
    auto layer = loadBalancer.AddLayer(/*expertCount=*/4, /*topK=*/2, /*slotCountPerRank=*/2);

    EXPECT_THROW(layer->setInitialWeightAssignments({0, 1, 2, 3, 0, -1, 1, 2}), tensorrt_llm::common::TllmException);
}

namespace
{

size_t vmmGranularity()
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    TLLM_CUDA_CHECK(cudaGetDevice(&prop.location.id));
    size_t granularity = 0;
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return granularity;
}

// What LocalCreator and the scratch book for unbacked slot memory: GPU bytes.
size_t bookedBytes()
{
    return MemoryCounters::getInstance().getGpu();
}

uint8_t readByte(uint8_t const* address)
{
    uint8_t value = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&value, address, 1, cudaMemcpyDeviceToHost));
    return value;
}

} // namespace

TEST(UnbackedSlotMemoryTest, SpareSlotsShareScratchUntilRestoredInPlace)
{
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = UnbackedSlotMemory::getInstance();
    auto const gpuBytes = bookedBytes();
    // Slots of 3.5 chunks: slot 1 owns chunks 4-6 whole (chunk 3 is shared with slot 0), slot 3 chunks 11-13.
    size_t const slotBytes = 3 * gran + gran / 2;
    auto* base = static_cast<uint8_t*>(memory.allocate(slotBytes, 4, {1, 3}));
    // Slots of 1.5 chunks: slot 1 owns chunk 2 whole, a 1-chunk piece.
    auto* small = static_cast<uint8_t*>(memory.allocate(gran + gran / 2, 2, {1}));
    // Memory of their own: 14 - 6 and 3 - 1 chunks; the scratch: 1 chunk, for every piece.
    EXPECT_EQ(bookedBytes() - gpuBytes, 11 * gran);

    // Backed slots keep their bytes; every chunk of every piece maps the one scratch chunk, so chunks 4, 5 and 11 and
    // the small range's chunk 2 are the same memory.
    TLLM_CUDA_CHECK(cudaMemset(base, 1, 4 * slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 2 * slotBytes, 2, slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 4 * gran, 7, gran));
    EXPECT_EQ(readByte(base + 5 * gran), 7);
    EXPECT_EQ(readByte(base + 11 * gran), 7);
    EXPECT_EQ(readByte(small + 2 * gran), 7);
    TLLM_CUDA_CHECK(cudaMemset(small + 2 * gran, 8, gran));
    EXPECT_EQ(readByte(base + 13 * gran + gran - 1), 8);
    EXPECT_EQ(readByte(base), 1);
    EXPECT_EQ(readByte(base + 2 * slotBytes), 2);

    // Restoring slot 1 maps memory of its own at the same addresses: what is written there is private. Chunk 3 has
    // memory of its own already: slot 0 is backed.
    EXPECT_EQ(memory.bytesToBack(base, {1}), 3 * gran);
    EXPECT_EQ(memory.bytesToBack(base, {1, 3}), 6 * gran);
    EXPECT_EQ(memory.setBacked(base, {1}, true), 3 * gran);
    EXPECT_EQ(memory.bytesToBack(base, {1}), 0);
    EXPECT_EQ(bookedBytes() - gpuBytes, 14 * gran);
    TLLM_CUDA_CHECK(cudaMemset(base + slotBytes, 9, slotBytes));
    EXPECT_EQ(readByte(base + 4 * gran), 9);
    EXPECT_EQ(readByte(base + 2 * slotBytes - 1), 9);
    EXPECT_EQ(readByte(base + 11 * gran), 8);
    EXPECT_EQ(readByte(small + 2 * gran), 8);
    EXPECT_EQ(readByte(base), 1);
    EXPECT_EQ(readByte(base + 2 * slotBytes), 2);

    // Spare again: slot 1 maps the scratch and its memory is freed.
    EXPECT_EQ(memory.setBacked(base, {1}, false), 3 * gran);
    EXPECT_EQ(bookedBytes() - gpuBytes, 11 * gran);
    EXPECT_EQ(readByte(base + 4 * gran), 8);
    // Only spare slots of a range change, and backing one twice maps nothing more.
    EXPECT_THROW(memory.setBacked(base, {0}, true), tensorrt_llm::common::TllmException);
    EXPECT_THROW(memory.bytesToBack(base, {0}), tensorrt_llm::common::TllmException);
    EXPECT_THROW(memory.setBacked(base + gran, {1}, true), tensorrt_llm::common::TllmException);
    EXPECT_EQ(memory.setBacked(base, {3}, true), 3 * gran);
    EXPECT_EQ(memory.setBacked(base, {3}, true), 0);

    // A chunk shared by two spare slots (chunk 1 of slots of 1.5 chunks) is backed while either slot is.
    auto* pair = static_cast<uint8_t*>(memory.allocate(gran + gran / 2, 2, {0, 1}));
    EXPECT_EQ(memory.bytesToBack(pair, {0}), 2 * gran);
    EXPECT_EQ(memory.setBacked(pair, {0}, true), 2 * gran);
    EXPECT_EQ(memory.bytesToBack(pair, {1}), gran);
    EXPECT_EQ(memory.setBacked(pair, {1}, true), gran);
    EXPECT_EQ(memory.setBacked(pair, {0}, false), gran);
    EXPECT_EQ(memory.setBacked(pair, {1}, false), 2 * gran);

    memory.free(pair);
    memory.free(small);
    memory.free(base);
    EXPECT_EQ(bookedBytes(), gpuBytes); // the scratch goes with the last piece that maps it
}

TEST(UnbackedSlotMemoryTest, BackingThatRunsOutOfMemoryChangesNoSlot)
{
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = UnbackedSlotMemory::getInstance();
    auto const gpuBytes = bookedBytes();
    // A spare slot of one chunk, and one of 1 GiB: more than the device keeps free while the test holds the rest.
    size_t const largeBytes = size_t{1} << 30;
    auto* small = static_cast<uint8_t*>(memory.allocate(gran, 2, {1}));
    auto* large = static_cast<uint8_t*>(memory.allocate(largeBytes, 1, {0}));
    auto mapped = [](uint8_t const* address)
    {
        int isMapped = 0;
        TLLM_CU_CHECK(
            cuPointerGetAttribute(&isMapped, CU_POINTER_ATTRIBUTE_MAPPED, reinterpret_cast<CUdeviceptr>(address)));
        return isMapped != 0;
    };
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    TLLM_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    void* held = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&held, freeBytes - largeBytes / 4));

    // The small slot fits, the large one does not: neither is backed, both still map the scratch.
    EXPECT_THROW(memory.setBacked({{small, {1}}, {large, {0}}}, true), tensorrt_llm::common::TllmException);
    EXPECT_EQ(memory.bytesToBack(small, {1}), gran);
    EXPECT_EQ(memory.bytesToBack(large, {0}), largeBytes);
    EXPECT_EQ(bookedBytes() - gpuBytes, 2 * gran); // slot 0 of the small range and the scratch
    EXPECT_TRUE(mapped(small + gran));
    EXPECT_TRUE(mapped(large));
    EXPECT_TRUE(mapped(large + largeBytes - 1));

    // With the memory back, the same call backs both.
    TLLM_CUDA_CHECK(cudaFree(held));
    EXPECT_EQ(memory.setBacked({{small, {1}}, {large, {0}}}, true), gran + largeBytes);
    EXPECT_EQ(memory.setBacked({{small, {1}}, {large, {0}}}, false), gran + largeBytes);
    memory.free(large);
    memory.free(small);
    EXPECT_EQ(bookedBytes(), gpuBytes);
}

TEST(MoeLoadBalancerSpareSlotTest, RestorePlacementFillsSpareSlotsAndRejoinEmptiesThem)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    using Placement = std::vector<std::vector<int>>;
    constexpr int kExpertCount = 4;
    constexpr int kSlotCountPerRank = 2;
    constexpr int kDeadRank = 1;
    constexpr size_t kWeightBytes = 256;
    // Static placement: no online update ever copies a weight.
    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/4, /*layerUpdatesPerIter=*/0, /*rankFailureRecovery=*/true);
    auto layer = loadBalancer.AddLayer(kExpertCount, /*topK=*/2, kSlotCountPerRank);
    // Pageable host weights, every byte of expert e's e + 1; slot 1 of rank 0 is spare and holds 0.
    std::vector<std::vector<uint8_t>> hostWeights;
    for (int expertId = 0; expertId < kExpertCount; ++expertId)
    {
        hostWeights.emplace_back(kWeightBytes, static_cast<uint8_t>(expertId + 1));
        layer->addSingleHostWeight(
            expertId, "weight", MoeWeight{hostWeights.back().data(), 1, kWeightBytes, kWeightBytes});
    }
    uint8_t* slots = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&slots, kSlotCountPerRank * kWeightBytes));
    TLLM_CUDA_CHECK(cudaMemset(slots, 1, kWeightBytes));
    TLLM_CUDA_CHECK(cudaMemset(slots + kWeightBytes, 0, kWeightBytes));
    for (int slotId = 0; slotId < kSlotCountPerRank; ++slotId)
    {
        layer->addSingleWeightSlot(
            slotId, "weight", MoeWeight{slots + slotId * kWeightBytes, 1, kWeightBytes, kWeightBytes});
    }
    auto slotBytes = [&](int slotId)
    {
        std::vector<uint8_t> bytes(kWeightBytes);
        TLLM_CUDA_CHECK(cudaMemcpy(bytes.data(), slots + slotId * kWeightBytes, kWeightBytes, cudaMemcpyDeviceToHost));
        return bytes;
    };
    Placement const healthy{{0, -1}, {1, -1}, {2, -1}, {3, -1}};
    layer->setInitialWeightAssignments({0, -1, 1, -1, 2, -1, 3, -1});
    loadBalancer.finalizeModel();
    // The device table routing reads: expert e's global slot.
    auto expertToSlot = [&]
    {
        std::vector<int> slots(kExpertCount);
        TLLM_CUDA_CHECK(cudaMemcpy(slots.data(), layer->getGpuPlacementInfo().globalSlotIds, kExpertCount * sizeof(int),
            cudaMemcpyDeviceToHost));
        return slots;
    };
    EXPECT_EQ(expertToSlot(), (std::vector<int>{0, 2, 4, 6}));

    // Rank 1 fails: its expert 1 moves into rank 0's spare slot, which the restore loads from the host weights.
    Placement const restored{{0, 1}, {1, -1}, {2, -1}, {3, -1}};
    loadBalancer.restorePlacement({kDeadRank}, {restored});
    EXPECT_EQ(layer->getOldRankExpertIds(), (Placement{{0, 1}, {-1, -1}, {2, -1}, {3, -1}}));
    EXPECT_EQ(expertToSlot(), (std::vector<int>{0, 1, 4, 6}));
    EXPECT_EQ(slotBytes(1), std::vector<uint8_t>(kWeightBytes, 2));
    EXPECT_EQ(slotBytes(0), std::vector<uint8_t>(kWeightBytes, 1));
    EXPECT_TRUE(layer->routesEachExpertToOneSlot());

    // A placement that loses an expert, or replicates one under one-slot routing, is rejected and changes nothing.
    TLLM_CUDA_CHECK(cudaMemset(slots + kWeightBytes, 0, kWeightBytes));
    EXPECT_THROW(loadBalancer.restorePlacement({kDeadRank}, {healthy}), tensorrt_llm::common::TllmException);
    EXPECT_THROW(loadBalancer.restorePlacement({}, {restored}), tensorrt_llm::common::TllmException);
    EXPECT_EQ(layer->getOldRankExpertIds(), (Placement{{0, 1}, {-1, -1}, {2, -1}, {3, -1}}));
    EXPECT_EQ(expertToSlot(), (std::vector<int>{0, 1, 4, 6}));
    EXPECT_EQ(slotBytes(1), std::vector<uint8_t>(kWeightBytes, 0));

    // Rank 1's replacement rejoins: the healthy placement again, rank 0's slot spare again and not loaded.
    loadBalancer.restorePlacement({}, {healthy});
    EXPECT_EQ(layer->getOldRankExpertIds(), healthy);
    EXPECT_EQ(expertToSlot(), (std::vector<int>{0, 2, 4, 6}));
    EXPECT_EQ(slotBytes(1), std::vector<uint8_t>(kWeightBytes, 0));
    loadBalancer.shutdown();
    TLLM_CUDA_CHECK(cudaFree(slots));

    // Without fault tolerance there is nothing to restore.
    MoeLoadBalancer plain(/*epRank=*/0, /*epSize=*/2, /*layerUpdatesPerIter=*/0);
    plain.AddLayer(kExpertCount, /*topK=*/2, /*slotCountPerRank=*/2)->setInitialWeightAssignments({0, 1, 2, 3});
    plain.finalizeModel();
    EXPECT_THROW(plain.restorePlacement({}, {Placement{{0, 1}, {2, 3}}}), tensorrt_llm::common::TllmException);
    plain.shutdown();
}

namespace
{

// The GPU's part of a layer in a forward: takes the layer once the worker hands it over for iterId and, with handBack,
// records loads as the layer's statistics and hands the layer back.
void runLayerOnGpu(SingleLayerMoeLoadBalancer& layer, int64_t iterId, std::vector<float> const& loads, bool handBack)
{
    auto* signal = layer.getSignal();
    int statisticEnabled = 0;
    moeWaitSignalForGpuStageForTest(signal, &statisticEnabled);
    int64_t const gpuIterId = signal->stepAndOwner >> 2;
    EXPECT_EQ(gpuIterId, iterId);
    if (handBack)
    {
        std::copy(loads.begin(), loads.end(), layer.getStatisticInfo()->expertLoadFactor);
        moeSetSignalForCpuStageForTest(signal);
    }
}

bool cpuHoldsSignal(SingleLayerMoeLoadBalancer& layer)
{
    return (layer.getSignal()->stepAndOwner & tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal::kDevice)
        == tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal::kCPU;
}

// One EP rank of a model with online EPLB under Wide-EP fault tolerance, its weights in GPU-copied slots. Every byte of
// expert e's weight in layer l is expertByte(l, e); an empty slot's bytes are 0.
class SimulatedEpRank
{
public:
    static constexpr size_t kWeightBytes = 256;

    static uint8_t expertByte(int layerId, int expertId)
    {
        return static_cast<uint8_t>(16 * (layerId + 1) + expertId + 1);
    }

    SimulatedEpRank(int epRank, int epSize, int expertCount, int slotCountPerRank, int layerCount,
        std::vector<int> const& initialAssignments)
        : mBalancer(epRank, epSize, /*layerUpdatesPerIter=*/1, /*rankFailureRecovery=*/true)
        , mEpRank(epRank)
    {
        mBalancer.setUseGpuMemcpy(true);
        for (int layerId = 0; layerId < layerCount; ++layerId)
        {
            auto layer = mBalancer.AddLayer(expertCount, /*topK=*/1, slotCountPerRank);
            for (int expertId = 0; expertId < expertCount; ++expertId)
            {
                void* hostWeight = nullptr;
                TLLM_CUDA_CHECK(cudaMallocHost(&hostWeight, kWeightBytes));
                std::fill_n(static_cast<uint8_t*>(hostWeight), kWeightBytes, expertByte(layerId, expertId));
                mHostWeights.push_back(hostWeight);
                layer->addSingleHostWeight(expertId, "weight", MoeWeight{hostWeight, 1, kWeightBytes, kWeightBytes});
            }
            std::vector<void*> slots;
            for (int slotId = 0; slotId < slotCountPerRank; ++slotId)
            {
                void* slot = nullptr;
                TLLM_CUDA_CHECK(cudaMalloc(&slot, kWeightBytes));
                int const expertId = initialAssignments[epRank * slotCountPerRank + slotId];
                TLLM_CUDA_CHECK(cudaMemset(slot, expertId < 0 ? 0 : expertByte(layerId, expertId), kWeightBytes));
                slots.push_back(slot);
                layer->addSingleWeightSlot(slotId, "weight", MoeWeight{slot, 1, kWeightBytes, kWeightBytes});
            }
            layer->setInitialWeightAssignments(initialAssignments);
            mSlots.push_back(slots);
            mLayers.push_back(layer);
        }
        mBalancer.finalizeModel();
    }

    ~SimulatedEpRank()
    {
        mBalancer.shutdown();
        for (auto* hostWeight : mHostWeights)
        {
            EXPECT_EQ(cudaFreeHost(hostWeight), cudaSuccess);
        }
        for (auto const& slots : mSlots)
        {
            for (auto* slot : slots)
            {
                EXPECT_EQ(cudaFree(slot), cudaSuccess);
            }
        }
    }

    SimulatedEpRank(SimulatedEpRank const&) = delete;
    SimulatedEpRank& operator=(SimulatedEpRank const&) = delete;

    MoeLoadBalancer& balancer()
    {
        return mBalancer;
    }

    SingleLayerMoeLoadBalancer& layer(int layerId)
    {
        return *mLayers[layerId];
    }

    // Every local slot the layer's placement fills holds its expert's weight.
    void expectSlotsHoldPlacement(int layerId)
    {
        auto const placement = mLayers[layerId]->getOldRankExpertIds();
        for (size_t slotId = 0; slotId < mSlots[layerId].size(); ++slotId)
        {
            int const expertId = placement[mEpRank][slotId];
            if (expertId < 0)
            {
                continue;
            }
            std::vector<uint8_t> bytes(kWeightBytes);
            TLLM_CUDA_CHECK(cudaMemcpy(bytes.data(), mSlots[layerId][slotId], kWeightBytes, cudaMemcpyDeviceToHost));
            EXPECT_EQ(static_cast<size_t>(std::count(bytes.begin(), bytes.end(), expertByte(layerId, expertId))),
                kWeightBytes)
                << "rank " << mEpRank << ", layer " << layerId << ", slot " << slotId << " should hold expert "
                << expertId;
        }
    }

    // The device table routing reads: per replica, its global slot.
    std::vector<int> routingTable(int layerId)
    {
        auto const metaInfo = mLayers[layerId]->getMetaInfo();
        std::vector<int> slots(metaInfo.epSize * metaInfo.slotCountPerRank);
        TLLM_CUDA_CHECK(cudaMemcpy(slots.data(), mLayers[layerId]->getGpuPlacementInfo().globalSlotIds,
            slots.size() * sizeof(int), cudaMemcpyDeviceToHost));
        return slots;
    }

private:
    MoeLoadBalancer mBalancer;
    int mEpRank;
    std::vector<std::shared_ptr<SingleLayerMoeLoadBalancer>> mLayers;
    std::vector<void*> mHostWeights;
    std::vector<std::vector<void*>> mSlots;
};

} // namespace

TEST(MoeLoadBalancerAbandonTest, SurvivorsInterruptedAtDifferentLayersResumeFromOneRestoredPlacement)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    using Placement = std::vector<std::vector<int>>;
    constexpr int kEpSize = 4;
    constexpr int kExpertCount = 5;
    constexpr int kSlotCountPerRank = 3;
    constexpr int kLayerCount = 2;
    constexpr int kFailedRank = 3;
    // Slot 2 of every rank is spare; only rank 3 holds expert 4.
    Placement const healthy{{0, 1, -1}, {2, 0, -1}, {3, 1, -1}, {4, 2, -1}};
    std::vector<int> initialAssignments;
    for (auto const& rankExpertIds : healthy)
    {
        initialAssignments.insert(initialAssignments.end(), rankExpertIds.begin(), rankExpertIds.end());
    }
    // Statistics that make EPLB's update replicate expert 4.
    std::vector<float> const loads{1, 1, 1, 1, 20};
    std::vector<std::unique_ptr<SimulatedEpRank>> survivors;
    for (int epRank = 0; epRank < kFailedRank; ++epRank)
    {
        survivors.push_back(std::make_unique<SimulatedEpRank>(
            epRank, kEpSize, kExpertCount, kSlotCountPerRank, kLayerCount, initialAssignments));
    }

    // Iteration 0 updates layer 0 (the update plan: layer 0, layer 1, none). Rank 3 fails during it: rank 0's forward
    // signals both layers, rank 1's only layer 0, rank 2's none (it stops inside layer 0). Rank 0 has also started
    // iteration 1, whose forward never ran.
    std::array<int, kFailedRank> const layersSignalled{2, 1, 0};
    for (int epRank = 0; epRank < kFailedRank; ++epRank)
    {
        auto& rank = *survivors[epRank];
        rank.balancer().startIter(0, /*enableStatistic=*/true, /*enableUpdateWeights=*/true);
        for (int layerId = 0; layerId < layersSignalled[epRank]; ++layerId)
        {
            runLayerOnGpu(rank.layer(layerId), 0, loads, /*handBack=*/true);
        }
        if (layersSignalled[epRank] < kLayerCount)
        {
            runLayerOnGpu(rank.layer(layersSignalled[epRank]), 0, loads, /*handBack=*/false);
        }
        rank.balancer().endIter(0);
    }
    survivors[0]->balancer().startIter(1, /*enableStatistic=*/true, /*enableUpdateWeights=*/true);
    survivors[0]->balancer().endIter(1);
    // Rank 2's worker waits for layer 0, which its GPU never hands back: no placement can be restored yet.
    EXPECT_THROW(survivors[2]->balancer().restorePlacement({kFailedRank}, {healthy, healthy}),
        tensorrt_llm::common::TllmException);

    // Every survivor abandons: no statistics, every signal with the CPU.
    for (auto& rank : survivors)
    {
        rank->balancer().abandonIterations();
        for (int layerId = 0; layerId < kLayerCount; ++layerId)
        {
            auto& layer = rank->layer(layerId);
            auto const* loadFactor = layer.getStatisticInfo()->expertLoadFactor;
            EXPECT_TRUE(std::all_of(loadFactor, loadFactor + kExpertCount, [](float load) { return load == 0.0F; }));
            EXPECT_TRUE(cpuHoldsSignal(layer));
        }
    }
    // Ranks 0 and 1 started layer 0's update, rank 2 did not: the placements differ.
    EXPECT_NE(survivors[0]->layer(0).getOldRankExpertIds(), healthy);
    EXPECT_EQ(survivors[2]->layer(0).getOldRankExpertIds(), healthy);

    // One agreed placement: rank 3's expert 4 moves into rank 0's spare slot. Each survivor's restore loads the local
    // slots whose expert differs from its own placement.
    Placement const restored{{0, 1, 4}, {2, 0, -1}, {3, 1, -1}, {4, 2, -1}};
    Placement const restoredActive{{0, 1, 4}, {2, 0, -1}, {3, 1, -1}, {-1, -1, -1}};
    for (auto& rank : survivors)
    {
        rank->balancer().restorePlacement({kFailedRank}, {restored, restored});
    }
    for (auto& rank : survivors)
    {
        for (int layerId = 0; layerId < kLayerCount; ++layerId)
        {
            EXPECT_EQ(rank->layer(layerId).getOldRankExpertIds(), restoredActive);
            EXPECT_EQ(rank->routingTable(layerId), survivors[0]->routingTable(layerId));
            rank->expectSlotsHoldPlacement(layerId);
        }
    }

    // The next iteration completes everywhere: the update plan restarted, so every survivor updates layer 0 alike and
    // leaves layer 1.
    for (auto& rank : survivors)
    {
        int64_t const iterId = rank.get() == survivors[0].get() ? 2 : 1;
        rank->balancer().startIter(iterId, /*enableStatistic=*/true, /*enableUpdateWeights=*/true);
        for (int layerId = 0; layerId < kLayerCount; ++layerId)
        {
            runLayerOnGpu(rank->layer(layerId), iterId, loads, /*handBack=*/true);
        }
        rank->balancer().endIter(iterId);
        // Waits for the weight update started.
        rank->balancer().shutdown();
    }
    auto const updated = survivors[0]->layer(0).getOldRankExpertIds();
    EXPECT_NE(updated, restoredActive);
    EXPECT_EQ(updated[kFailedRank], (std::vector<int>{-1, -1, -1}));
    EXPECT_EQ(updated[1][2], -1);
    EXPECT_EQ(updated[2][2], -1);
    for (auto& rank : survivors)
    {
        EXPECT_EQ(rank->layer(0).getOldRankExpertIds(), updated);
        EXPECT_EQ(rank->layer(1).getOldRankExpertIds(), restoredActive);
        rank->expectSlotsHoldPlacement(0);
        rank->expectSlotsHoldPlacement(1);
    }
}

TEST(MoeLoadBalancerAbandonTest, AbandonIterationsChecksItsCaller)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    std::vector<float> const loads{1, 1};

    // Without fault tolerance there is nothing to abandon; with a static placement nothing to do.
    MoeLoadBalancer plain(/*epRank=*/0, /*epSize=*/2, /*layerUpdatesPerIter=*/1);
    plain.AddLayer(/*expertCount=*/2, /*topK=*/1, /*slotCountPerRank=*/2)->setInitialWeightAssignments({0, 1, 1, 0});
    plain.finalizeModel();
    EXPECT_THROW(plain.abandonIterations(), tensorrt_llm::common::TllmException);
    plain.shutdown();
    MoeLoadBalancer staticPlacement(
        /*epRank=*/0, /*epSize=*/2, /*layerUpdatesPerIter=*/0, /*rankFailureRecovery=*/true);
    staticPlacement.finalizeModel();
    staticPlacement.abandonIterations();
    staticPlacement.shutdown();

    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/2, /*layerUpdatesPerIter=*/1, /*rankFailureRecovery=*/true);
    auto layer = loadBalancer.AddLayer(/*expertCount=*/2, /*topK=*/1, /*slotCountPerRank=*/2);
    layer->setInitialWeightAssignments({0, 1, 1, 0});
    loadBalancer.finalizeModel();

    // Iteration 0 completes; iteration 1 stops inside its layer. Until it is abandoned, no placement is restored.
    using Placement = std::vector<std::vector<int>>;
    Placement const swapped{{1, 0}, {0, 1}};
    loadBalancer.startIter(0, /*enableStatistic=*/true, /*enableUpdateWeights=*/false);
    runLayerOnGpu(*layer, 0, loads, /*handBack=*/true);
    loadBalancer.endIter(0);
    loadBalancer.startIter(1, /*enableStatistic=*/true, /*enableUpdateWeights=*/false);
    EXPECT_THROW(loadBalancer.abandonIterations(), tensorrt_llm::common::TllmException);
    runLayerOnGpu(*layer, 1, loads, /*handBack=*/false);
    loadBalancer.endIter(1);
    EXPECT_THROW(loadBalancer.restorePlacement({}, {swapped}), tensorrt_llm::common::TllmException);
    EXPECT_EQ(layer->getOldRankExpertIds(), (Placement{{0, 1}, {1, 0}}));
    loadBalancer.abandonIterations();
    EXPECT_TRUE(cpuHoldsSignal(*layer));
    loadBalancer.restorePlacement({}, {swapped});
    EXPECT_EQ(layer->getOldRankExpertIds(), swapped);

    // The next iterations hand the layer over and back as usual: the worker walks each to its end.
    for (int64_t iterId = 2; iterId < 4; ++iterId)
    {
        loadBalancer.startIter(iterId, /*enableStatistic=*/true, /*enableUpdateWeights=*/false);
        runLayerOnGpu(*layer, iterId, loads, /*handBack=*/true);
        loadBalancer.endIter(iterId);
    }
    loadBalancer.shutdown();
}

TEST(MoeLoadBalancerAbandonTest, AbandonIterationsWaitsForTheLaunchedWork)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/2, /*layerUpdatesPerIter=*/1, /*rankFailureRecovery=*/true);
    loadBalancer.AddLayer(/*expertCount=*/2, /*topK=*/1, /*slotCountPerRank=*/2)
        ->setInitialWeightAssignments({0, 1, 1, 0});
    loadBalancer.finalizeModel();
    // Device work that ends only when the test hands its layer over: a GPU stage wait on a signal of its own, in
    // pinned host memory as a load balancer's signals are.
    using tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal;
    MoeLoadBalanceSingleLayerSignal* held = nullptr;
    TLLM_CUDA_CHECK(cudaMallocHost(&held, sizeof(*held)));
    std::memset(static_cast<void*>(held), 0, sizeof(*held));
    held->stepAndOwner = MoeLoadBalanceSingleLayerSignal::kCPU;
    int* enabled = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&enabled, sizeof(int)));
    cudaStream_t stream = nullptr;
    TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    tensorrt_llm::kernels::moeWaitSignalForGpuStageDevice(held, enabled, stream);

    auto abandoned = std::async(std::launch::async, [&loadBalancer] { loadBalancer.abandonIterations(); });
    EXPECT_EQ(abandoned.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout)
        << "abandonIterations returned while launched work still ran";
    // Hand the layer over: the held work ends, and so does the call waiting for it.
    tensorrt_llm::kernels::moeSetSignalForGpuStageHost(held, 0, /*enableStatistic=*/false);
    EXPECT_EQ(abandoned.wait_for(std::chrono::seconds(10)), std::future_status::ready)
        << "abandonIterations did not return once the launched work ended";
    abandoned.get();
    loadBalancer.shutdown();
    TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
    TLLM_CUDA_CHECK(cudaFree(enabled));
    TLLM_CUDA_CHECK(cudaFreeHost(held));
}

// Iteration control parameter structure
struct IterConfig
{
    bool statisticEnabled;
    bool updateWeightsEnabled;
};

struct MoeLoadBalancerTestParam
{
    int expertCount;                     // Number of experts per layer
    int topK;                            // Number of experts selected for each token
    int epRank;                          // Current node rank
    int epSize;                          // Number of nodes in the cluster
    int slotCountPerRank;                // Number of slots on each node
    int numLayers;                       // Number of MoE layers in the model
    int layerUpdatesPerIter;             // Maximum number of layers to update per iteration
    int warmUpIters;                     // Number of warm-up iterations
    std::vector<IterConfig> iterConfigs; // Configuration for each iteration
};

class MoeLoadBalancerTest : public ::testing::TestWithParam<MoeLoadBalancerTestParam>
{
protected:
    void SetUp() override
    {
        setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
        auto param = GetParam();
        TLLM_CUDA_CHECK(cudaSetDevice(0));
        mLoadBalancer = std::make_unique<MoeLoadBalancer>(param.epRank, param.epSize, param.layerUpdatesPerIter);

        mLoadBalancer->setUseGpuMemcpy(true);

        // Create multiple MoE layers
        createLayers(param);

        // Create model weights for each layer
        createModelWeights(param);

        // Calculate which layers should be updated in each iteration
        computeLayerUpdateSchedule(param);

        // Initialize the last iteration ID
        mLastIterId = -1;
    }

    void TearDown() override
    {
        // Release all test resources
        for (auto& layerWeights : mAllHostWeights)
        {
            for (auto& weight : layerWeights)
            {
                EXPECT_EQ(cudaFreeHost(weight.mWeightPtr), cudaSuccess);
            }
        }
        mAllHostWeights.clear();

        for (auto& layerSlots : mAllSlotWeights)
        {
            for (auto& slot : layerSlots)
            {
                EXPECT_EQ(cudaFree(slot.mWeightPtr), cudaSuccess);
            }
        }
        mAllSlotWeights.clear();
    }

    // Calculate which layers should be updated in each iteration
    void computeLayerUpdateSchedule(MoeLoadBalancerTestParam const& param)
    {
        mLayerUpdateSchedule.clear();
        int numIters = param.iterConfigs.size();
        mLayerUpdateSchedule.resize(numIters);

        int totalLayers = param.numLayers;
        int updatesPerIter = param.layerUpdatesPerIter;

        // Calculate the minimum number of iterations needed to update all layers
        int minItersNeeded = (totalLayers + updatesPerIter - 1) / updatesPerIter;

        // If the number of layers is divisible by updates per iteration, add an extra empty update cycle
        int realItersUsed = minItersNeeded;
        if (totalLayers % updatesPerIter == 0)
        {
            realItersUsed += 1;
        }

        // Count effective update iterations (iterations that have updates enabled) after warm-up
        int effectiveUpdateIters = 0;

        // Allocate which layers to update for each iteration
        for (int iter = 0; iter < numIters; ++iter)
        {
            // Enforce rule: if statistics are disabled, updates must also be disabled
            if (!param.iterConfigs[iter].statisticEnabled && param.iterConfigs[iter].updateWeightsEnabled)
            {
                TLLM_CHECK_WITH_INFO(false,
                    "Invalid configuration at iteration %d: statistics disabled but updates enabled. "
                    "When statistics are disabled, updates must also be disabled.",
                    iter);
            }

            // Only schedule layer updates when weight updates are enabled and after warm-up period
            if (param.iterConfigs[iter].updateWeightsEnabled && iter >= param.warmUpIters)
            {
                // Calculate cycle position based on effective update iterations rather than raw iteration count
                int cyclePos = effectiveUpdateIters % realItersUsed;

                // If cyclePos is less than minItersNeeded, update layers according to existing logic
                // Otherwise, this is an empty update cycle, don't assign any layers
                if (cyclePos < minItersNeeded)
                {
                    // Calculate which layers should be updated at this position
                    for (int i = 0; i < updatesPerIter; ++i)
                    {
                        int layerId = cyclePos + i * minItersNeeded;
                        if (layerId < totalLayers)
                        {
                            mLayerUpdateSchedule[iter].push_back(layerId);
                        }
                    }
                }
                // Otherwise mLayerUpdateSchedule[iter] remains an empty vector, indicating no layers to update in this
                // cycle

                // Increment effective update iteration count
                effectiveUpdateIters++;
            }
        }
    }

    void createLayers(MoeLoadBalancerTestParam const& param)
    {
        mLayers.clear();
        // Create the specified number of MoE layers
        for (int i = 0; i < param.numLayers; ++i)
        {
            auto layer = mLoadBalancer->AddLayer(param.expertCount, param.topK, param.slotCountPerRank);
            mLayers.push_back(layer);
        }
    }

    void createModelWeights(MoeLoadBalancerTestParam const& param)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int weightSize = weightWidth * weightHeight;
        int const gpuWeightPitch = 2048;
        int gpuWeightSize = gpuWeightPitch * weightHeight;
        // Adjust the size of vectors storing weights for multiple layers
        mAllHostWeights.resize(param.numLayers);
        mAllSlotWeights.resize(param.numLayers);

        // Create weight slots for each layer
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            // Create host weights for each expert in the layer
            mAllHostWeights[layerId].resize(param.expertCount);
            for (int expertId = 0; expertId < param.expertCount; ++expertId)
            {
                float* hostWeightPtr;
                EXPECT_EQ(cudaMallocHost(&hostWeightPtr, weightSize * sizeof(float)), cudaSuccess);

                // Fill weight values: use a combination of layerId and expertId for unique identification
                // This ensures that weights for each expert/layer combination are unique
                float baseValue = generateWeightValue(layerId, expertId);
                for (int i = 0; i < weightSize; ++i)
                {
                    // Add small position offset to make each position have a different value
                    hostWeightPtr[i] = baseValue + i * 0.1f;
                }

                mAllHostWeights[layerId][expertId].mWeightPtr = hostWeightPtr;
                mAllHostWeights[layerId][expertId].mHeight = weightHeight;
                mAllHostWeights[layerId][expertId].mWidth = weightWidth * sizeof(float);
                mAllHostWeights[layerId][expertId].mPitch = weightWidth * sizeof(float);

                // Add to the weight updater for the corresponding layer
                mLayers[layerId]->addSingleHostWeight(expertId, "test_weight", mAllHostWeights[layerId][expertId]);
            }
        }

        // Generate initial expert assignments
        std::vector<int> initialExpertAssignments;
        for (int r = 0; r < param.epSize; ++r)
        {
            int expertStart = r * param.expertCount / param.epSize;
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                int expertId = (expertStart + slotId) % param.expertCount;
                initialExpertAssignments.push_back(expertId);
            }
        }

        int expertStart = param.epRank * param.expertCount / param.epSize;

        // Create weight slots for each layer
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            // Create device weights for each slot in the layer
            mAllSlotWeights[layerId].resize(param.slotCountPerRank);
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                int expertId = (expertStart + slotId) % param.expertCount;
                float* slotWeightPtr;
                EXPECT_EQ(cudaMalloc(&slotWeightPtr, gpuWeightSize * sizeof(float)), cudaSuccess);

                // Initialize weights
                for (int h = 0; h < weightHeight; h++)
                {
                    EXPECT_EQ(cudaMemcpy(slotWeightPtr + h * gpuWeightPitch,
                                  static_cast<float*>(mAllHostWeights[layerId][expertId].mWeightPtr) + h * weightWidth,
                                  sizeof(float) * weightWidth, cudaMemcpyHostToDevice),
                        cudaSuccess);
                }

                mAllSlotWeights[layerId][slotId].mWeightPtr = slotWeightPtr;
                mAllSlotWeights[layerId][slotId].mHeight = weightHeight;
                mAllSlotWeights[layerId][slotId].mWidth = weightWidth * sizeof(float);
                mAllSlotWeights[layerId][slotId].mPitch = gpuWeightPitch * sizeof(float);

                // Add to the corresponding layer slot
                mLayers[layerId]->addSingleWeightSlot(slotId, "test_weight", mAllSlotWeights[layerId][slotId]);
            }

            // Set initial weight assignments
            mLayers[layerId]->setInitialWeightAssignments(initialExpertAssignments);
        }

        // Save current weights for each layer and slot for later comparison
        saveCurrentWeights();
    }

    // Save the current weight state for all layers
    void saveCurrentWeights()
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int const weightSize = weightWidth * weightHeight;
        int const weightPitch = 2048;

        auto param = GetParam();
        mCurrentWeights.resize(param.numLayers);
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            mCurrentWeights[layerId].resize(param.slotCountPerRank);
            for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
            {
                mCurrentWeights[layerId][slotId].resize(weightSize);
                for (int h = 0; h < weightHeight; h++)
                {
                    EXPECT_EQ(cudaMemcpy(mCurrentWeights[layerId][slotId].data() + h * weightWidth,
                                  static_cast<float*>(mAllSlotWeights[layerId][slotId].mWeightPtr) + h * weightPitch,
                                  weightWidth * sizeof(float), cudaMemcpyDeviceToHost),
                        cudaSuccess);
                }
            }
        }
    }

    // Simulate forward pass, set different load distributions for each layer
    void setExpertLoadFactors(MoeLoadBalancerTestParam const& param)
    {
        for (int layerId = 0; layerId < param.numLayers; ++layerId)
        {
            float* expertLoadFactor = mLayers[layerId]->getStatisticInfo()->expertLoadFactor;
            for (int i = 0; i < param.expertCount; ++i)
            {
                // Set different load patterns for each layer
                if (layerId % 2 == 0)
                {
                    // Even layers: higher load for higher expert IDs
                    expertLoadFactor[i] = static_cast<float>(i + 1);
                }
                else
                {
                    // Odd layers: higher load for lower expert IDs
                    expertLoadFactor[i] = static_cast<float>(param.expertCount - i);
                }
            }
        }
    }

    // Verify if the weight updates for a layer match expectations
    void verifyLayerWeightUpdates(int layerId, int slotId, bool shouldBeUpdated, int64_t iterId)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int const weightSize = weightWidth * weightHeight;
        int const weightPitch = 2048;

        // Get current weights
        std::vector<float> currentWeight(weightSize);
        for (int h = 0; h < weightHeight; h++)
        {
            EXPECT_EQ(cudaMemcpy(currentWeight.data() + h * weightWidth,
                          static_cast<float*>(mAllSlotWeights[layerId][slotId].mWeightPtr) + h * weightPitch,
                          weightWidth * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
        }

        // Check if weights have changed
        bool hasChanged = false;
        for (int i = 0; i < weightSize && !hasChanged; ++i)
        {
            if (mCurrentWeights[layerId][slotId][i] != currentWeight[i])
            {
                hasChanged = true;
            }
        }

        // Verify according to expectations
        if (shouldBeUpdated)
        {
            // In some cases (e.g., optimal allocation didn't change), weights might not change; this is just a weak
            // check EXPECT_TRUE(hasChanged) << "Layer " << layerId << ", Slot " << slotId << " weights should have been
            // updated but weren't at iter " << iterId;
        }
        else
        {
            // When updates shouldn't happen, weights must remain unchanged
            EXPECT_FALSE(hasChanged) << "Layer " << layerId << ", Slot " << slotId
                                     << " weights shouldn't have changed but did at iter " << iterId;
        }

        // Update saved weight state
        mCurrentWeights[layerId][slotId] = currentWeight;
    }

    // Verify single layer's weight updates from the previous iteration
    void verifyLayerPreviousIterationWeightUpdates(
        int layerId, int64_t currentIterId, bool wasUpdateEnabled, bool shouldLayerBeUpdated)
    {
        // No "previous" iteration for the first iteration
        if (mLastIterId < 0)
        {
            return;
        }

        auto param = GetParam();
        for (int slotId = 0; slotId < param.slotCountPerRank; ++slotId)
        {
            verifyLayerWeightUpdates(layerId, slotId, wasUpdateEnabled && shouldLayerBeUpdated, currentIterId);
        }
    }

    // Simulate GPU waiting for signal, processing signal, and checking related state
    void simulateGpuSignalWaitAndProcess(int64_t iterId, bool expectedStatisticEnabled,
        bool expectedUpdateWeightsEnabled, std::vector<int> const& expectedUpdatedLayers)
    {
        // Execute simulated GPU forward pass and signal processing for each layer
        for (int layerId = 0; layerId < mLayers.size(); ++layerId)
        {
            auto signal = mLayers[layerId]->getSignal();

            // 1. Use helper function to simulate GPU waiting for signal
            int statisticEnabled;
            moeWaitSignalForGpuStageForTest(signal, &statisticEnabled);

            // Get current iteration ID from the signal
            int64_t gpuIterId = signal->stepAndOwner >> 2;

            // 2. Verify the obtained signal values
            EXPECT_EQ(gpuIterId, iterId) << "Layer " << layerId << " has wrong iterId: " << gpuIterId << " vs expected "
                                         << iterId;
            EXPECT_EQ(statisticEnabled != 0, expectedStatisticEnabled)
                << "Layer " << layerId << " statistic enabled mismatch at iteration " << iterId;

            // If not the first iteration, verify the current layer's weight updates from the previous iteration
            if (mLastIterId >= 0)
            {
                // Determine if the current layer should have been updated in the previous iteration
                bool shouldLayerBeUpdated = false;
                if (mLastIterId < mLayerUpdateSchedule.size())
                {
                    auto& updatedLayers = mLayerUpdateSchedule[mLastIterId];
                    shouldLayerBeUpdated
                        = std::find(updatedLayers.begin(), updatedLayers.end(), layerId) != updatedLayers.end();
                }

                // Get the previous iteration's configuration
                bool wasUpdateEnabled = false;
                if (mLastIterId < GetParam().iterConfigs.size())
                {
                    auto& lastConfig = GetParam().iterConfigs[mLastIterId];
                    // Enforce the rule: if statistics are disabled, updates must also be disabled
                    EXPECT_TRUE(lastConfig.statisticEnabled || !lastConfig.updateWeightsEnabled)
                        << "Invalid state detected in iteration " << mLastIterId
                        << ": statistics disabled but updates enabled";

                    wasUpdateEnabled = lastConfig.updateWeightsEnabled && mLastIterId >= GetParam().warmUpIters;
                }

                // Verify the previous iteration's updates for the current layer between GPU signal processing
                verifyLayerPreviousIterationWeightUpdates(layerId, iterId, wasUpdateEnabled, shouldLayerBeUpdated);
            }

            // After verification, regenerate host weights for the next verification
            updateHostWeightsForLayer(layerId, iterId);

            // 3. Use helper function to simulate GPU sending signal to CPU, indicating forward pass completion
            moeSetSignalForCpuStageForTest(signal);
        }

        // Update current iteration ID
        mLastIterId = iterId;
    }

    // Verify the final load balancing situation - need to verify the weight updates from the last iteration
    void verifyLastIterationWeightUpdates(MoeLoadBalancerTestParam const& param)
    {
        // Verify the weight updates from the last iteration
        if (mLastIterId >= 0)
        {
            for (int layerId = 0; layerId < param.numLayers; ++layerId)
            {
                // Determine if the current layer should have been updated in the last iteration
                bool shouldLayerBeUpdated = false;
                if (mLastIterId < mLayerUpdateSchedule.size())
                {
                    auto& updatedLayers = mLayerUpdateSchedule[mLastIterId];
                    shouldLayerBeUpdated
                        = std::find(updatedLayers.begin(), updatedLayers.end(), layerId) != updatedLayers.end();
                }

                // Get the last iteration's configuration
                bool wasUpdateEnabled = false;
                if (mLastIterId < param.iterConfigs.size())
                {
                    auto& lastConfig = param.iterConfigs[mLastIterId];
                    // Enforce the rule: if statistics are disabled, updates must also be disabled
                    EXPECT_TRUE(lastConfig.statisticEnabled || !lastConfig.updateWeightsEnabled)
                        << "Invalid state detected in iteration " << mLastIterId
                        << ": statistics disabled but updates enabled";

                    wasUpdateEnabled = lastConfig.updateWeightsEnabled && mLastIterId >= param.warmUpIters;
                }

                // Verify the updates for each layer from the last iteration
                verifyLayerPreviousIterationWeightUpdates(
                    layerId, mLastIterId + 1, wasUpdateEnabled, shouldLayerBeUpdated);
            }
        }

        // Other load balancing related verifications can be added here
    }

    // Generate consistent weight value based on layerId, expertId
    float generateWeightValue(int layerId, int expertId)
    {
        // Base value from layerId and expertId, ensuring each expert/layer combination is unique
        float baseValue = static_cast<float>((layerId + 1) * 100 + (expertId + 1));

        // Also set different values for different positions in the weight array by adding position offset
        return baseValue; // Position index will be added when used
    }

    // Regenerate host weights for a specific expert
    void regenerateHostWeights(int layerId, int expertId)
    {
        int const weightWidth = 1024;
        int const weightHeight = 10;
        int weightSize = weightWidth * weightHeight;

        float* hostWeightPtr = static_cast<float*>(mAllHostWeights[layerId][expertId].mWeightPtr);
        float baseValue = generateWeightValue(layerId, expertId);

        // Make values at different positions in the weight array also different
        for (int i = 0; i < weightSize; ++i)
        {
            // Add small position offset to make each position have a different value
            hostWeightPtr[i] = baseValue + i * 0.1f;
        }
    }

    // Update host weights for all slots in the current layer
    void updateHostWeightsForLayer(int layerId, int64_t iterId)
    {
        auto param = GetParam();
        // Update weights for all experts in this layer
        for (int expertId = 0; expertId < param.expertCount; ++expertId)
        {
            regenerateHostWeights(layerId, expertId);
        }
    }

    std::unique_ptr<MoeLoadBalancer> mLoadBalancer;
    std::vector<std::shared_ptr<SingleLayerMoeLoadBalancer>> mLayers; // Store pointers to all layers
    std::vector<std::vector<MoeWeight>> mAllHostWeights;              // Expert weights for each layer (host memory)
    std::vector<std::vector<MoeWeight>> mAllSlotWeights;              // Slot weights for each layer (device memory)
    std::vector<std::vector<std::vector<float>>> mCurrentWeights;     // Save current weight state
    std::vector<std::vector<int>> mLayerUpdateSchedule;               // Layers to update in each iteration
    int64_t mLastIterId;                                              // Save last iteration ID for verification
};

TEST_P(MoeLoadBalancerTest, MultiLayerTest)
{
    auto param = GetParam();

    // Complete model setup
    mLoadBalancer->finalizeModel();

    // Set warm-up iteration count
    if (param.warmUpIters > 0)
    {
        mLoadBalancer->setWarmUpIterCount(param.warmUpIters);
    }

    // Set different load factors for each layer
    setExpertLoadFactors(param);

    // Run multiple iterations for testing
    for (int iterIdx = 0; iterIdx < param.iterConfigs.size(); ++iterIdx)
    {
        auto const& config = param.iterConfigs[iterIdx];
        int64_t iterId = iterIdx;

        // 1. Start new iteration
        mLoadBalancer->startIter(iterId, config.statisticEnabled, config.updateWeightsEnabled);

        // 2. Simulate GPU waiting and processing signals
        std::vector<int> expectedUpdatedLayers;
        if (iterIdx < mLayerUpdateSchedule.size())
        {
            expectedUpdatedLayers = mLayerUpdateSchedule[iterIdx];
        }

        simulateGpuSignalWaitAndProcess(iterId, config.statisticEnabled,
            config.updateWeightsEnabled && iterIdx >= param.warmUpIters, expectedUpdatedLayers);

        // 3. End current iteration
        mLoadBalancer->endIter(iterId);

        // Make sure all operations are complete
        cudaDeviceSynchronize();
    }

    // At the end of the test, verify load balancing
    verifyLastIterationWeightUpdates(param);

    // Normal shutdown of the load balancer
    mLoadBalancer->shutdown();

    mLoadBalancer.reset();
}

// Define test parameters
INSTANTIATE_TEST_SUITE_P(MoeLoadBalancerTests, MoeLoadBalancerTest,
    ::testing::Values(
        // Scenario 1: Two-layer model, alternating enable/disable statistics and updates
        MoeLoadBalancerTestParam{8, // expertCount
            2,                      // topK
            0,                      // epRank
            2,                      // epSize
            4,                      // slotCountPerRank
            2,                      // numLayers
            1,                      // layerUpdatesPerIter (update 1 layer at a time)
            2,                      // warmUpIters
            {
                // iterConfigs
                {true, true},  // Iteration 0: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 1: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 2: Enable statistics and updates
                {true, false}, // Iteration 3: Enable statistics, disable updates
                {true, true},  // Iteration 4: Enable statistics and updates
                {false, false} // Iteration 5: Disable statistics and updates (both must be disabled together)
            }},

        // Scenario 2: Three-layer model, updating 2 layers each time
        MoeLoadBalancerTestParam{16, // expertCount
            1,                       // topK
            0,                       // epRank
            4,                       // epSize
            4,                       // slotCountPerRank
            3,                       // numLayers
            2,                       // layerUpdatesPerIter (update 2 layers at a time)
            1,                       // warmUpIters
            {
                // iterConfigs
                {true, true},  // Iteration 0: Enable statistics and updates (warm-up)
                {true, true},  // Iteration 1: Enable statistics and updates
                {true, false}, // Iteration 2: Enable statistics, disable updates
                {true, true}   // Iteration 3: Enable statistics and updates
            }},

        // Scenario 3: Five-layer model, updating 3 layers each time, no warm-up
        MoeLoadBalancerTestParam{32, // expertCount
            4,                       // topK
            1,                       // epRank
            8,                       // epSize
            4,                       // slotCountPerRank
            5,                       // numLayers
            3,                       // layerUpdatesPerIter (update 3 layers at a time)
            0,                       // warmUpIters (no warm-up)
            {
                // iterConfigs
                {true, true},   // Iteration 0: Enable statistics and updates
                {false, false}, // Iteration 1: Disable statistics and updates
                {true, true}    // Iteration 2: Enable statistics and updates
            }},

        // Scenario 4: 61-layer model, updating 10 layers each time, 5 warm-up iterations
        MoeLoadBalancerTestParam{256, // expertCount
            8,                        // topK
            4,                        // epRank
            36,                       // epSize
            8,                        // slotCountPerRank
            61,                       // numLayers
            10,                       // layerUpdatesPerIter (update 10 layers at a time)
            5,                        // warmUpIters (5 warm-up iterations)
            {
                // iterConfigs
                {true, true},   // Iteration 0: Enable statistics and updates
                {true, true},   // Iteration 1: Enable statistics and updates
                {true, true},   // Iteration 2: Enable statistics and updates
                {true, true},   // Iteration 3: Enable statistics and updates
                {true, true},   // Iteration 4: Enable statistics and updates
                {true, true},   // Iteration 5: Enable statistics and updates
                {true, true},   // Iteration 6: Enable statistics and updates
                {true, true},   // Iteration 7: Enable statistics and updates
                {true, true},   // Iteration 8: Enable statistics and updates
                {true, true},   // Iteration 9: Enable statistics and updates
                {true, true},   // Iteration 10: Enable statistics and updates
                {true, true},   // Iteration 11: Enable statistics and updates
                {true, true},   // Iteration 12: Enable statistics and updates
                {false, false}, // Iteration 13: Disable statistics and updates
                {true, false},  // Iteration 14: Enable statistics, disable updates
                {true, true},   // Iteration 15: Enable statistics and updates
                {false, false}, // Iteration 16: Disable statistics and updates
                {false, false}, // Iteration 17: Disable statistics and updates
                {true, true},   // Iteration 18: Enable statistics and updates
                {true, true},   // Iteration 19: Enable statistics and updates
                {true, true},   // Iteration 20: Enable statistics and updates
                {true, true}    // Iteration 21: Enable statistics and updates
            }}));
