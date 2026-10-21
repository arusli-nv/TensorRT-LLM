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
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

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
    constexpr int kGhostSlot = 2;
    constexpr int kDeadRank = 2;
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{
        kExpertCount, /*topK=*/2, /*epRank=*/0, kEpSize, kSlotCountPerRank};
    // The last slot of every rank is unusable, except on rank 1 (a ghost slot a restore backed).
    std::vector<uint8_t> unusableSlotMask(kEpSize * kSlotCountPerRank, 0);
    for (int rank = 0; rank < kEpSize; ++rank)
    {
        unusableSlotMask[rank * kSlotCountPerRank + kGhostSlot] = static_cast<uint8_t>(rank != 1);
    }
    std::vector<uint8_t> const deadRankMask{0, 0, 1, 0};

    for (auto const* dead : {static_cast<std::vector<uint8_t> const*>(nullptr), &deadRankMask})
    {
        for (auto const& load : {std::vector<float>{8.0F, 4.0F, 2.0F, 1.0F}, std::vector<float>(kExpertCount, 0.0F)})
        {
            auto const placement = planPlacement(metaInfo, load, dead, &unusableSlotMask);
            int const usableSlotCount = dead == nullptr ? 9 : 7;
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
                    bool const usable = !(dead != nullptr && rank == kDeadRank)
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

TEST(MoeLoadBalancerUnusableSlotTest, AllUsableMaskPlacesExactlyAsNoMask)
{
    // The placement main's planner makes without a mask: replicas by load, each rank's slots filled in order.
    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo small{/*expertCount=*/4, /*topK=*/2, /*epRank=*/0, /*epSize=*/2,
        /*slotCountPerRank=*/3};
    std::vector<uint8_t> const smallAllUsable(6, 0);
    for (auto const* mask : {static_cast<std::vector<uint8_t> const*>(nullptr), &smallAllUsable})
    {
        auto const placement = planPlacement(small, {8.0F, 4.0F, 2.0F, 1.0F}, nullptr, mask);
        EXPECT_EQ(placement.expertReplicaCount, (std::vector<int>{2, 2, 1, 1}));
        EXPECT_EQ(placement.rankExpertIds, (std::vector<std::vector<int>>{{0, 1, 2}, {0, 1, 3}}));
    }

    tensorrt_llm::kernels::MoeLoadBalanceMetaInfo metaInfo{/*expertCount=*/16, /*topK=*/4, /*epRank=*/1,
        /*epSize=*/8, /*slotCountPerRank=*/3};
    std::vector<uint8_t> const allUsable(metaInfo.epSize * metaInfo.slotCountPerRank, 0);
    std::vector<uint8_t> const deadRankMask{0, 0, 0, 1, 0, 0, 1, 0};
    std::vector<std::vector<float>> loads{std::vector<float>(metaInfo.expertCount, 0.0F)};
    for (int seed = 1; seed <= 3; ++seed)
    {
        std::vector<float> load(metaInfo.expertCount);
        for (int expertId = 0; expertId < metaInfo.expertCount; ++expertId)
        {
            load[expertId] = static_cast<float>((expertId * 7 * seed + 3) % 11) + 0.5F * seed;
        }
        loads.push_back(load);
    }
    for (auto const* dead : {static_cast<std::vector<uint8_t> const*>(nullptr), &deadRankMask})
    {
        for (auto const& load : loads)
        {
            auto const withoutMask = planPlacement(metaInfo, load, dead, nullptr);
            auto const withMask = planPlacement(metaInfo, load, dead, &allUsable);
            EXPECT_EQ(withMask.expertReplicaCount, withoutMask.expertReplicaCount);
            EXPECT_EQ(withMask.rankExpertIds, withoutMask.rankExpertIds);
        }
    }
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

namespace
{

// Sets TLLM_FAULT_TOLERANCE_MODE for its lifetime and restores the previous value.
class ScopedFaultToleranceMode
{
public:
    explicit ScopedFaultToleranceMode(bool enabled)
    {
        if (char const* previous = std::getenv("TLLM_FAULT_TOLERANCE_MODE"))
        {
            mPrevious = previous;
        }
        if (enabled)
        {
            setenv("TLLM_FAULT_TOLERANCE_MODE", "1", 1);
        }
        else
        {
            unsetenv("TLLM_FAULT_TOLERANCE_MODE");
        }
    }

    ~ScopedFaultToleranceMode()
    {
        if (mPrevious)
        {
            setenv("TLLM_FAULT_TOLERANCE_MODE", mPrevious->c_str(), 1);
        }
        else
        {
            unsetenv("TLLM_FAULT_TOLERANCE_MODE");
        }
    }

    ScopedFaultToleranceMode(ScopedFaultToleranceMode const&) = delete;
    ScopedFaultToleranceMode& operator=(ScopedFaultToleranceMode const&) = delete;

private:
    std::optional<std::string> mPrevious;
};

TEST(MoeLoadBalancerFaultToleranceOffTest, InitialAssignmentRejectsAnEmptySlot)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    ScopedFaultToleranceMode const faultToleranceMode(/*enabled=*/false);
    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/4, /*layerUpdatesPerIter=*/0);
    auto layer = loadBalancer.AddLayer(/*expertCount=*/4, /*topK=*/2, /*slotCountPerRank=*/2);

    EXPECT_THROW(layer->setInitialWeightAssignments({0, 1, 2, 3, 0, -1, 1, 2}), tensorrt_llm::common::TllmException);
}

TEST(MoeLoadBalancerFaultToleranceOffTest, FinalizeRejectsWeightSlotsWithoutHostWeights)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    ScopedFaultToleranceMode const faultToleranceMode(/*enabled=*/false);
    constexpr int kSlotCountPerRank = 2;
    constexpr size_t kBytes = 256;
    MoeLoadBalancer loadBalancer(/*epRank=*/0, /*epSize=*/4, /*layerUpdatesPerIter=*/1);
    auto layer = loadBalancer.AddLayer(/*expertCount=*/4, /*topK=*/2, kSlotCountPerRank);
    std::vector<void*> slots(kSlotCountPerRank, nullptr);
    for (int slotId = 0; slotId < kSlotCountPerRank; ++slotId)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&slots[slotId], kBytes));
        MoeWeight slot;
        slot.mWeightPtr = slots[slotId];
        slot.mHeight = 1;
        slot.mWidth = kBytes;
        slot.mPitch = kBytes;
        layer->addSingleWeightSlot(slotId, "w", slot);
    }
    layer->setInitialWeightAssignments({0, 1, 0, 1, 2, 3, 0, 1});

    EXPECT_THROW(loadBalancer.finalizeModel(), tensorrt_llm::common::TllmException);
    for (auto* slot : slots)
    {
        TLLM_CUDA_CHECK(cudaFree(slot));
    }
}

// One layer on EP rank 0 of 4 with 2 slots per rank; by default experts 2 and 3 live only on rank 2. With host weights,
// every expert's is filled with expertId + 1; every local slot starts with its expert's values (0 in a -1 slot).
// With ghostSlot1, the slots are one ghost range (GhostSlotMemory) whose slot 1 maps no memory, loaded by CPU copies
// where it can be host accessible.
class RestorePlacementLayer
{
public:
    static constexpr int kExpertCount = 4;
    static constexpr int kEpSize = 4;
    static constexpr int kSlotCountPerRank = 2;
    static constexpr int kDeadRank = 2;

    explicit RestorePlacementLayer(bool withHostWeights, int layerUpdatesPerIter = 1, size_t weightElems = 256,
        bool ghostSlot1 = false, std::vector<int> const& initialAssignments = {0, 1, 0, 1, 2, 3, 0, 1})
        : mLoadBalancer(/*epRank=*/0, kEpSize, layerUpdatesPerIter)
        , mWeightElems(weightElems)
    {
        bool const cpuCopies = ghostSlot1 && GhostSlotMemory::hostAccessibleSupported();
        mLoadBalancer.setUseGpuMemcpy(!cpuCopies);
        mLayer = mLoadBalancer.AddLayer(kExpertCount, /*topK=*/2, kSlotCountPerRank);
        size_t const bytes = mWeightElems * sizeof(float);
        for (int expertId = 0; withHostWeights && expertId < kExpertCount; ++expertId)
        {
            float* host = nullptr;
            TLLM_CUDA_CHECK(cudaMallocHost(&host, bytes));
            std::fill_n(host, mWeightElems, static_cast<float>(expertId + 1));
            mHostWeights.push_back(host);
            mLayer->addSingleHostWeight(expertId, "w", weight(host));
        }
        if (ghostSlot1)
        {
            mGhost = static_cast<float*>(
                GhostSlotMemory::getInstance().allocate(bytes, kSlotCountPerRank, {1}, cpuCopies));
        }
        for (int slotId = 0; slotId < kSlotCountPerRank; ++slotId)
        {
            float* slot = mGhost ? mGhost + slotId * mWeightElems : nullptr;
            if (!mGhost)
            {
                TLLM_CUDA_CHECK(cudaMalloc(&slot, bytes));
            }
            std::vector<float> const values(mWeightElems, static_cast<float>(initialAssignments[slotId] + 1));
            TLLM_CUDA_CHECK(cudaMemcpy(slot, values.data(), bytes, cudaMemcpyHostToDevice));
            mSlots.push_back(slot);
            mLayer->addSingleWeightSlot(slotId, "w", weight(slot));
        }
        mLayer->setInitialWeightAssignments(initialAssignments);
        mLoadBalancer.finalizeModel();
    }

    ~RestorePlacementLayer()
    {
        mLoadBalancer.shutdown();
        for (auto* host : mHostWeights)
        {
            cudaFreeHost(host);
        }
        if (mGhost)
        {
            GhostSlotMemory::getInstance().free(mGhost);
            return;
        }
        for (auto* slot : mSlots)
        {
            cudaFree(slot);
        }
    }

    // Runs one iteration whose EPLB update plans this layer from expertLoadFactor, and waits for it.
    void runUpdate(int64_t iterId, std::vector<float> const& expertLoadFactor)
    {
        std::copy(expertLoadFactor.begin(), expertLoadFactor.end(), mLayer->getStatisticInfo()->expertLoadFactor);
        mLoadBalancer.startIter(iterId, /*enableStatistic=*/true, /*enableUpdateWeights=*/true);
        int statisticEnabled = 0;
        moeWaitSignalForGpuStageForTest(mLayer->getSignal(), &statisticEnabled);
        moeSetSignalForCpuStageForTest(mLayer->getSignal());
        mLoadBalancer.endIter(iterId);
        ASSERT_TRUE(mLoadBalancer.quiesce(/*timeoutMs=*/10000));
    }

    void writeSlot(int slotId, std::vector<float> const& values)
    {
        TLLM_CUDA_CHECK(
            cudaMemcpy(mSlots[slotId], values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // The slot's value at element elem (default: the first).
    float slotValue(int slotId, size_t elem = 0) const
    {
        float value = 0;
        TLLM_CUDA_CHECK(cudaMemcpy(&value, mSlots[slotId] + elem, sizeof(float), cudaMemcpyDeviceToHost));
        return value;
    }

    // Empty (-1) slots and slots without host weights need fault tolerance.
    ScopedFaultToleranceMode mFaultToleranceMode{/*enabled=*/true};
    MoeLoadBalancer mLoadBalancer;
    std::shared_ptr<SingleLayerMoeLoadBalancer> mLayer;

private:
    MoeWeight weight(float* ptr) const
    {
        MoeWeight w;
        w.mWeightPtr = ptr;
        w.mHeight = 1;
        w.mWidth = mWeightElems * sizeof(float);
        w.mPitch = w.mWidth;
        return w;
    }

    size_t mWeightElems;
    std::vector<float*> mHostWeights;
    std::vector<float*> mSlots;
    float* mGhost = nullptr;
};

// GhostSlotMemoryTest runs on the memory online EPLB gets: host accessible where it can be.
bool hostAccessible()
{
    return GhostSlotMemory::hostAccessibleSupported();
}

void* allocateGhost(size_t slotBytes, int slotCount, std::vector<int> const& emptySlots)
{
    return GhostSlotMemory::getInstance().allocate(slotBytes, slotCount, emptySlots, hostAccessible());
}

// What LocalCreator books for ghost slot memory: GPU bytes.
size_t bookedBytes()
{
    return MemoryCounters::getInstance().getGpu();
}

size_t vmmGranularity()
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    TLLM_CUDA_CHECK(cudaGetDevice(&prop.location.id));
    prop.allocFlags.gpuDirectRDMACapable = hostAccessible();
    size_t granularity = 0;
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return granularity;
}

} // namespace

TEST(MoeLoadBalancerRestoreTest, OneSlotPerExpertRoutingKeepsOneReplicaPerExpert)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/true, /*layerUpdatesPerIter=*/0);
    // Experts 0 and 1 have two replicas: routing must choose between them.
    EXPECT_FALSE(setup.mLayer->routesEachExpertToOneSlot());

    // Every second copy emptied: one slot per expert.
    setup.mLoadBalancer.restorePlacement({}, {{{0, -1}, {1, -1}, {2, 3}, {-1, -1}}});
    EXPECT_TRUE(setup.mLayer->routesEachExpertToOneSlot());

    // Losing rank 2 moves its experts into empty slots: still one replica each.
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {{{0, 2}, {1, 3}, {-1, -1}, {-1, -1}}});
    EXPECT_TRUE(setup.mLayer->routesEachExpertToOneSlot());

    // A second copy would need the replica-choosing kernel, which captured graphs do not launch.
    EXPECT_THROW(
        setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {{{0, 2}, {1, 3}, {-1, -1}, {0, -1}}}),
        std::exception);
}

TEST(MoeLoadBalancerRestoreTest, RestorePlacementLoadsLostExpertsIntoSlots)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/true);

    // Rank 0 takes expert 2 over a replica of expert 0, rank 1 takes expert 3 over a replica of expert 1.
    std::vector<std::vector<int>> const placement{{2, 1}, {0, 3}, {-1, -1}, {0, 1}};
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement});

    auto* info = setup.mLayer->getPlacementCpuInfo();
    EXPECT_EQ(info->oldRankExpertIds[0], (std::vector<int>{2, 1}));
    EXPECT_EQ(info->oldRankExpertIds[RestorePlacementLayer::kDeadRank], (std::vector<int>{-1, -1}));
    std::vector<int> const expectedReplicaCounts{2, 2, 1, 1};
    for (int expertId = 0; expertId < RestorePlacementLayer::kExpertCount; ++expertId)
    {
        EXPECT_EQ(info->placementInfoForGPU.expertReplicaCount[expertId], expectedReplicaCounts[expertId]);
    }
    EXPECT_EQ(setup.slotValue(0), 3.0f); // expert 2, loaded from the host copy
    EXPECT_EQ(setup.slotValue(1), 2.0f); // expert 1, unchanged
}

TEST(MoeLoadBalancerRestoreTest, RestorePlacementRejectsPlacementMissingAnExpert)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/true);

    // Expert 3 has no replica outside the dead rank: nothing may change.
    std::vector<std::vector<int>> const placement{{2, 1}, {0, 1}, {-1, -1}, {0, 1}};
    EXPECT_THROW(setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement}),
        tensorrt_llm::common::TllmException);
    EXPECT_EQ(setup.mLayer->getPlacementCpuInfo()->oldRankExpertIds[0], (std::vector<int>{0, 1}));
    EXPECT_EQ(setup.slotValue(0), 1.0f);
}

TEST(MoeLoadBalancerRestoreTest, EmptySlotIsNotRoutedAndTakesALostExpert)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/true);
    auto* info = setup.mLayer->getPlacementCpuInfo();

    // Empty rank 0's slot 1 (a replica of expert 1): routing skips it, nothing is loaded.
    setup.mLoadBalancer.restorePlacement({}, {{{0, -1}, {0, 1}, {2, 3}, {0, 1}}});
    EXPECT_EQ(info->oldRankExpertIds[0], (std::vector<int>{0, -1}));
    EXPECT_EQ(info->placementInfoForGPU.expertReplicaCount[1], 2);
    EXPECT_EQ(setup.slotValue(1), 2.0f);

    // Losing rank 2: expert 2 goes into the empty slot and is loaded from its host copy.
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {{{0, 2}, {3, 1}, {-1, -1}, {0, 1}}});
    EXPECT_EQ(info->oldRankExpertIds[0], (std::vector<int>{0, 2}));
    std::vector<int> const expectedReplicaCounts{2, 2, 1, 1};
    for (int expertId = 0; expertId < RestorePlacementLayer::kExpertCount; ++expertId)
    {
        EXPECT_EQ(info->placementInfoForGPU.expertReplicaCount[expertId], expectedReplicaCounts[expertId]);
    }
    EXPECT_EQ(setup.slotValue(1), 3.0f);
}

TEST(MoeLoadBalancerRestoreTest, RestorePlacementNeedsHostWeights)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/false);

    std::vector<std::vector<int>> const placement{{2, 1}, {0, 3}, {-1, -1}, {0, 1}};
    EXPECT_THROW(setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement}),
        tensorrt_llm::common::TllmException);
}

TEST(MoeLoadBalancerRestoreTest, RestorePlacementPublishesSlotsTheCallerLoaded)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    RestorePlacementLayer setup(/*withHostWeights=*/false, /*layerUpdatesPerIter=*/0);

    // The caller copied expert 2 into rank 0's slot 0; the restore publishes it without loading anything.
    std::vector<float> const loaded(256, 7.0f);
    setup.writeSlot(0, loaded);
    std::vector<std::vector<int>> const placement{{2, 1}, {0, 3}, {-1, -1}, {0, 1}};
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement}, /*slotsLoaded=*/true);

    auto* info = setup.mLayer->getPlacementCpuInfo();
    EXPECT_EQ(info->oldRankExpertIds[0], (std::vector<int>{2, 1}));
    EXPECT_EQ(info->placementInfoForGPU.expertReplicaCount[2], 1);
    EXPECT_EQ(setup.slotValue(0), 7.0f);
    EXPECT_EQ(setup.slotValue(1), 2.0f);
}

TEST(GhostSlotMemoryTest, EmptySlotsShareScratchUntilBacked)
{
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    // Slots of 3.5 chunks: slot 1 owns chunks 4-6 whole (chunk 3 is shared with slot 0), slot 3 chunks 11-13.
    size_t const slotBytes = 3 * gran + gran / 2;
    auto& memory = GhostSlotMemory::getInstance();
    auto* base = static_cast<uint8_t*>(allocateGhost(slotBytes, 4, {1, 3}));
    auto slot = [&](int i) { return GhostSlotMemory::Range{base + i * slotBytes, slotBytes}; };
    // Two load balancers: each keeps or releases only the backing it asked for.
    int const owner = 0;
    int const other = 0;
    EXPECT_EQ(memory.unbackedBytes({slot(0)}), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 3 * gran);
    EXPECT_EQ(memory.unbackedBytes({slot(1), slot(3), slot(1)}), 6 * gran);
    EXPECT_EQ(memory.unbackedBytes({{&slotBytes, 1}}), 0u); // not ghost memory

    // Backed slots keep their bytes; the empty slots' whole chunks share scratch: chunks 4-6 and 11-13 each map a
    // 2-chunk and a 1-chunk scratch allocation, so chunks 4 and 11 are the same memory.
    TLLM_CUDA_CHECK(cudaMemset(base, 1, 4 * slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 2 * slotBytes, 2, slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 4 * gran, 7, gran));
    uint8_t value = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&value, base + 11 * gran, 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(value, 7);
    TLLM_CUDA_CHECK(cudaMemcpy(&value, base, 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(value, 1);
    TLLM_CUDA_CHECK(cudaMemcpy(&value, base + 2 * slotBytes, 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(value, 2);

    memory.back({slot(1)}, &owner);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot(3)}), 3 * gran);
    TLLM_CUDA_CHECK(cudaMemset(base + slotBytes, 5, slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 12 * gran, 9, gran));
    std::vector<uint8_t> restored(slotBytes);
    TLLM_CUDA_CHECK(cudaMemcpy(restored.data(), base + slotBytes, slotBytes, cudaMemcpyDeviceToHost));
    EXPECT_TRUE(std::all_of(restored.begin(), restored.end(), [](uint8_t b) { return b == 5; }));

    // Another owner's keep or release leaves it pending.
    memory.keepPending(&other);
    EXPECT_EQ(memory.releasePending(&other), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 0u);
    // A plan abandoned before its install gives the memory back: slot 1 maps scratch again (chunk 4 is chunk 11).
    EXPECT_EQ(memory.releasePending(&owner), 3 * gran);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 3 * gran);
    TLLM_CUDA_CHECK(cudaMemset(base + 11 * gran, 3, 1));
    TLLM_CUDA_CHECK(cudaMemcpy(&value, base + 4 * gran, 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(value, 3);
    // An installed plan keeps it.
    memory.back({slot(1)}, &owner);
    memory.keepPending(&owner);
    EXPECT_EQ(memory.releasePending(&owner), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 0u);
    memory.free(base);

    // Adjacent empty slots 1 and 2 are pieces of their own (chunks 4-6 and 7-9): backing slot 1 leaves slot 2's
    // chunks unbacked.
    base = static_cast<uint8_t*>(allocateGhost(slotBytes, 4, {1, 2}));
    EXPECT_EQ(memory.unbackedBytes({slot(1), slot(2)}), 6 * gran);
    memory.back({slot(1)}, &owner);
    EXPECT_EQ(memory.unbackedBytes({slot(1)}), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot(2)}), 3 * gran);
    TLLM_CUDA_CHECK(cudaMemset(base + slotBytes, 5, slotBytes));
    TLLM_CUDA_CHECK(cudaMemset(base + 7 * gran, 9, 3 * gran));
    TLLM_CUDA_CHECK(cudaMemcpy(restored.data(), base + slotBytes, slotBytes, cudaMemcpyDeviceToHost));
    EXPECT_TRUE(std::all_of(restored.begin(), restored.end(), [](uint8_t b) { return b == 5; }));
    memory.free(base);
}

TEST(GhostSlotMemoryTest, CapturedGraphReadsWhateverIsMappedBehindASlot)
{
    // The design's core claim: a CUDA graph captured over a ghost slot keeps working while the memory behind
    // it changes (scratch, then backed, then scratch again), without recapture.
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = GhostSlotMemory::getInstance();
    auto* base = static_cast<uint8_t*>(allocateGhost(2 * gran, 2, {1}));
    GhostSlotMemory::Range const slot1{base + 2 * gran, 2 * gran};
    uint8_t* out = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&out, 2 * gran));
    cudaStream_t stream = nullptr;
    TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    TLLM_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(out, base + 2 * gran, 2 * gran, cudaMemcpyDeviceToDevice, stream));
    TLLM_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    TLLM_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    auto replayAll = [&](uint8_t expected)
    {
        TLLM_CUDA_CHECK(cudaGraphLaunch(exec, stream));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> host(2 * gran);
        TLLM_CUDA_CHECK(cudaMemcpy(host.data(), out, host.size(), cudaMemcpyDeviceToHost));
        return std::all_of(host.begin(), host.end(), [&](uint8_t b) { return b == expected; });
    };
    replayAll(0); // scratch: no fault, contents unspecified

    int const owner = 0;
    memory.back({slot1}, &owner);
    TLLM_CUDA_CHECK(cudaMemset(base + 2 * gran, 7, 2 * gran));
    EXPECT_TRUE(replayAll(7));
    EXPECT_EQ(memory.releasePending(&owner), 2 * gran);
    replayAll(0); // scratch again: still no fault
    memory.back({slot1}, &owner);
    TLLM_CUDA_CHECK(cudaMemset(base + 2 * gran, 9, 2 * gran));
    EXPECT_TRUE(replayAll(9));

    TLLM_CUDA_CHECK(cudaGraphExecDestroy(exec));
    TLLM_CUDA_CHECK(cudaGraphDestroy(graph));
    TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
    TLLM_CUDA_CHECK(cudaFree(out));
    memory.keepPending(&owner);
    memory.free(base);
}

TEST(GhostSlotMemoryTest, BackingThatCannotBeAllocatedChangesNothing)
{
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = GhostSlotMemory::getInstance();
    auto release = [](uint8_t* ptr) { GhostSlotMemory::getInstance().free(ptr); };
    using Ghost = std::unique_ptr<uint8_t, decltype(release)>;
    // Empty slots map only the shared scratch, so neither range below uses device memory however large it is. The
    // second is larger than the whole device: no backing for it can be allocated, whatever else uses the GPU, so
    // backing both fails (after the first one's memory was allocated if the first range is processed first).
    size_t freeBytes = 0, totalBytes = 0;
    TLLM_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    size_t const hugeBytes = (totalBytes / gran + 1) * gran;
    Ghost small(static_cast<uint8_t*>(allocateGhost(gran, 2, {1})), release);
    Ghost huge(static_cast<uint8_t*>(allocateGhost(hugeBytes, 1, {0})), release);
    GhostSlotMemory::Range const smallSlot{small.get() + gran, gran};
    GhostSlotMemory::Range const hugeSlot{huge.get(), hugeBytes};
    auto const gpuBytes = bookedBytes();
    int const owner = 0;
    EXPECT_THROW(memory.back({smallSlot, hugeSlot}, &owner), tensorrt_llm::common::TllmException);
    EXPECT_EQ(memory.unbackedBytes({smallSlot}), gran);
    EXPECT_EQ(memory.unbackedBytes({hugeSlot}), hugeBytes);
    EXPECT_EQ(bookedBytes(), gpuBytes);
    EXPECT_EQ(memory.releasePending(&owner), 0u);
    // Every chunk of both ranges is still mapped (scratch): a full write does not fault.
    TLLM_CUDA_CHECK(cudaMemset(small.get(), 1, 2 * gran));
    TLLM_CUDA_CHECK(cudaMemset(huge.get(), 1, hugeBytes));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(GhostSlotMemoryTest, FreedAndFailedAllocationsLeaveNoMemoryBehind)
{
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = GhostSlotMemory::getInstance();
    auto freeDeviceBytes = []
    {
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        size_t freeBytes = 0, totalBytes = 0;
        TLLM_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        return freeBytes;
    };
    // A first range settles what the driver keeps for the process.
    memory.free(allocateGhost(gran, 2, {1}));
    size_t const freeBefore = freeDeviceBytes();
    auto const gpuBytes = bookedBytes();
    // Device bytes used since; under the 15 chunks of scratch, as the driver may keep some page tables.
    auto leaked = [&]
    {
        size_t const now = freeDeviceBytes();
        return now >= freeBefore ? 0 : freeBefore - now;
    };

    // The scratch is freed with the last range: freeing it gives back every byte, scratch included.
    void* range = allocateGhost(gran, 2, {1});
    EXPECT_LT(freeDeviceBytes(), freeBefore);
    memory.free(range);
    EXPECT_LT(leaked(), 4 * gran);

    // Slots of three quarters of the free memory: slot 0 maps its own memory and slot 1 maps scratch before the
    // memory of slot 2 fails to allocate. The failed allocation unmaps both and releases all the memory it made.
    size_t const slotBytes = freeBefore / 4 * 3 / gran * gran;
    EXPECT_THROW(allocateGhost(slotBytes, 3, {1}), tensorrt_llm::common::TllmException);
    EXPECT_EQ(bookedBytes(), gpuBytes);
    EXPECT_LT(leaked(), 4 * gran);
    // The same layout with slot 2 empty fits.
    range = allocateGhost(slotBytes, 3, {1, 2});
    TLLM_CUDA_CHECK(cudaMemset(range, 1, 3 * slotBytes));
    memory.free(range);
    EXPECT_EQ(bookedBytes(), gpuBytes);
    EXPECT_LT(leaked(), 4 * gran);
}

TEST(GhostSlotMemoryTest, SleepTagsNeverTouchGhostEntries)
{
    // Sleep and wakeup release and materialize entries of the process's CudaVirtualMemoryManager by tag. Ghost pieces
    // have tags of their own: a backed slot keeps its memory and contents, an empty one stays on scratch.
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = GhostSlotMemory::getInstance();
    auto* base = static_cast<uint8_t*>(allocateGhost(gran, 3, {1, 2}));
    GhostSlotMemory::Range const slot1{base + gran, gran};
    GhostSlotMemory::Range const slot2{base + 2 * gran, gran};
    int const owner = 0;
    memory.back({slot1}, &owner);
    memory.keepPending(&owner);
    TLLM_CUDA_CHECK(cudaMemset(base + gran, 7, gran));
    auto& manager = getVirtualMemoryManager();
    for (char const* tag : {"model", "model_weights", "kv_cache", "ghost", "ghost/"})
    {
        EXPECT_EQ(manager.releaseWithTag(tag), 0u) << tag;
        EXPECT_EQ(manager.materializeWithTag(tag), 0u) << tag;
    }
    EXPECT_EQ(memory.unbackedBytes({slot1}), 0u);
    EXPECT_EQ(memory.unbackedBytes({slot2}), gran);
    std::vector<uint8_t> host(gran);
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), base + gran, gran, cudaMemcpyDeviceToHost));
    EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t b) { return b == 7; }));
    TLLM_CUDA_CHECK(cudaMemset(base + 2 * gran, 1, gran)); // scratch: no fault
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    memory.free(base);
}

TEST(GhostSlotMemoryTest, FailedPieceIsDroppedAndEndsOnScratch)
{
    // A piece whose memory cannot be allocated is dropped by the manager (a bad handle): its range stays on scratch,
    // the piece is added again (a later restore retries it), and freeing the range leaves no memory behind.
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    auto& memory = GhostSlotMemory::getInstance();
    auto& manager = getVirtualMemoryManager();
    auto freeDeviceBytes = []
    {
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        size_t freeBytes = 0, totalBytes = 0;
        TLLM_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        return std::make_pair(freeBytes, totalBytes);
    };
    // A first range settles what the driver keeps for the process.
    memory.free(allocateGhost(gran, 2, {1}));
    auto const [freeBefore, totalBytes] = freeDeviceBytes();
    auto const gpuBytes = bookedBytes();
    // Larger than the whole device: its memory can never be allocated.
    size_t const hugeBytes = (totalBytes / gran + 1) * gran;
    auto* huge = static_cast<uint8_t*>(allocateGhost(hugeBytes, 1, {0}));
    GhostSlotMemory::Range const slot{huge, hugeBytes};
    int const owner = 0;
    manager.retrieveBadHandles();
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        EXPECT_THROW(memory.back({slot}, &owner), tensorrt_llm::common::TllmException);
        EXPECT_EQ(manager.retrieveBadHandles(), std::vector<uintptr_t>{reinterpret_cast<uintptr_t>(huge)});
        EXPECT_EQ(memory.unbackedBytes({slot}), hugeBytes);
        EXPECT_EQ(memory.releasePending(&owner), 0u);
    }
    TLLM_CUDA_CHECK(cudaMemset(huge, 1, hugeBytes)); // every chunk still maps scratch
    memory.free(huge);
    EXPECT_EQ(bookedBytes(), gpuBytes);
    auto const freeAfter = freeDeviceBytes().first;
    EXPECT_LT(freeAfter >= freeBefore ? size_t{0} : freeBefore - freeAfter, 4 * gran);
}

TEST(GhostSlotMemoryTest, CpuWritesReachResidentAndBackedSlots)
{
    // GDRCopy maps each resident run and each backed piece. A backed slot spans several of them.
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    if (!hostAccessible())
    {
        GTEST_SKIP() << "Ghost slot memory cannot be host accessible here";
    }
    size_t const gran = vmmGranularity();
    // Slots of 2.25 chunks with slot 1 empty: chunks 0-2 and 4-8 are resident runs, chunk 3 is slot 1's piece.
    // Slot 1 spans the end of the first run, the piece and the start of the second run.
    size_t const slotBytes = 2 * gran + gran / 4;
    auto& memory = GhostSlotMemory::getInstance();
    auto& allocator = HostAccessibleDeviceAllocator::getInstance();
    auto* base = static_cast<uint8_t*>(allocateGhost(slotBytes, 4, {1}));
    GhostSlotMemory::Range const slot1{base + slotBytes, slotBytes};
    auto readBack = [](uint8_t const* ptr, size_t size)
    {
        std::vector<uint8_t> host(size);
        TLLM_CUDA_CHECK(cudaMemcpy(host.data(), ptr, size, cudaMemcpyDeviceToHost));
        return host;
    };

    // Resident: written through its host pointer.
    std::vector<uint8_t> const ones(gran, 1);
    auto* resident = allocator.getHostPtr(base + 5 * gran);
    ASSERT_NE(resident, nullptr);
    allocator.memcpyToDevice(resident, ones.data(), gran);
    EXPECT_EQ(readBack(base + 5 * gran, gran), ones);
    // A piece on scratch has no host mapping.
    EXPECT_EQ(allocator.getHostPtr(base + 3 * gran), nullptr);

    int const owner = 0;
    memory.back({slot1}, &owner);
    memory.keepPending(&owner);
    std::vector<uint8_t> const twos(gran, 2);
    auto* piece = allocator.getHostPtr(base + 3 * gran);
    ASSERT_NE(piece, nullptr);
    allocator.memcpyToDevice(piece, twos.data(), gran);
    EXPECT_EQ(readBack(base + 3 * gran, gran), twos);
    // A whole slot by device address, across the three mappings.
    std::vector<uint8_t> pattern(slotBytes);
    for (size_t i = 0; i < slotBytes; ++i)
    {
        pattern[i] = static_cast<uint8_t>(i % 251);
    }
    allocator.memcpyToDeviceAddress(base + slotBytes, pattern.data(), slotBytes);
    EXPECT_EQ(readBack(base + slotBytes, slotBytes), pattern);

    memory.free(base);
    EXPECT_EQ(allocator.getHostPtr(base + 3 * gran), nullptr);
    EXPECT_EQ(allocator.getHostPtr(base), nullptr);
}

TEST(MoeLoadBalancerRestoreTest, RestoreMapsMemoryBehindAGhostSlot)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    size_t const gran = vmmGranularity();
    size_t const elems = 2 * gran / sizeof(float);
    RestorePlacementLayer setup(/*withHostWeights=*/true, /*layerUpdatesPerIter=*/1, elems, /*ghostSlot1=*/true);
    setup.mLoadBalancer.restorePlacement({}, {{{0, -1}, {0, 1}, {2, 3}, {0, 1}}});

    // Losing rank 2: expert 2 goes into the ghost slot, which first gets its two chunks.
    std::vector<std::vector<int>> const placement{{0, 2}, {3, 1}, {-1, -1}, {0, 1}};
    EXPECT_EQ(setup.mLoadBalancer.restoreBackingBytes({placement}), static_cast<int64_t>(2 * gran));
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement});
    EXPECT_EQ(setup.mLoadBalancer.restoreBackingBytes({placement}), 0);
    EXPECT_EQ(setup.slotValue(1), 3.0f);
    EXPECT_EQ(setup.slotValue(1, elems - 1), 3.0f);
    EXPECT_EQ(setup.slotValue(0, elems - 1), 1.0f);
}

TEST(MoeLoadBalancerRestoreTest, UpdatesFillAGhostSlotOnlyOnceARestoreLoadedIt)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    // Slot 1 of every rank is a ghost slot: -1 from the start, outside EPLB's updates.
    RestorePlacementLayer setup(/*withHostWeights=*/true, /*layerUpdatesPerIter=*/1, /*weightElems=*/256,
        /*ghostSlot1=*/false, {0, -1, 1, -1, 2, -1, 3, -1});
    auto* info = setup.mLayer->getPlacementCpuInfo();
    std::vector<float> const load{1.0F, 2.0F, 3.0F, 4.0F};

    // Four usable slots for four experts: the update moves experts between slots 0 and never writes a slot 1.
    setup.runUpdate(0, load);
    for (int rank = 0; rank < RestorePlacementLayer::kEpSize; ++rank)
    {
        EXPECT_GE(info->oldRankExpertIds[rank][0], 0) << "rank " << rank;
        EXPECT_EQ(info->oldRankExpertIds[rank][1], -1) << "rank " << rank;
    }
    EXPECT_EQ(setup.slotValue(0), static_cast<float>(info->oldRankExpertIds[0][0] + 1));
    EXPECT_EQ(setup.slotValue(1), 0.0F);

    // Losing rank 2: its expert goes into rank 0's ghost slot, which then joins EPLB's updates.
    auto placement = info->oldRankExpertIds;
    int const lostExpert = placement[RestorePlacementLayer::kDeadRank][0];
    placement[RestorePlacementLayer::kDeadRank] = {-1, -1};
    placement[0][1] = lostExpert;
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement});
    EXPECT_EQ(setup.slotValue(1), static_cast<float>(lostExpert + 1));
    setup.mLoadBalancer.resetUpdateSchedule();

    // Rank 0's two slots and slot 0 of ranks 1 and 3 hold the four experts; the other ghost slots stay empty.
    setup.runUpdate(1, load);
    EXPECT_GE(info->oldRankExpertIds[0][0], 0);
    EXPECT_GE(info->oldRankExpertIds[0][1], 0);
    EXPECT_EQ(info->oldRankExpertIds[RestorePlacementLayer::kDeadRank], (std::vector<int>{-1, -1}));
    for (int rank : {1, 3})
    {
        EXPECT_GE(info->oldRankExpertIds[rank][0], 0) << "rank " << rank;
        EXPECT_EQ(info->oldRankExpertIds[rank][1], -1) << "rank " << rank;
    }
    for (int expertId = 0; expertId < RestorePlacementLayer::kExpertCount; ++expertId)
    {
        EXPECT_EQ(info->placementInfoForGPU.expertReplicaCount[expertId], 1);
    }
    EXPECT_EQ(setup.slotValue(0), static_cast<float>(info->oldRankExpertIds[0][0] + 1));
    EXPECT_EQ(setup.slotValue(1), static_cast<float>(info->oldRankExpertIds[0][1] + 1));
}

TEST(MoeLoadBalancerRestoreTest, CpuCopiesLoadARestoredGhostSlotThroughEachMapping)
{
    setenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK", "1", 1);
    TLLM_CUDA_CHECK(cudaSetDevice(0));
    if (!GhostSlotMemory::hostAccessibleSupported())
    {
        GTEST_SKIP() << "Ghost slot memory cannot be host accessible here";
    }
    size_t const gran = vmmGranularity();
    // Slots of 2.25 chunks: ghost slot 1 spans the end of the resident run (chunk 2), its own piece (chunk 3) and the
    // piece it shares with the space past the slots (chunk 4); each is a mapping of its own.
    size_t const slotBytes = 2 * gran + gran / 4;
    size_t const elems = slotBytes / sizeof(float);
    RestorePlacementLayer setup(
        /*withHostWeights=*/true, /*layerUpdatesPerIter=*/1, elems, /*ghostSlot1=*/true, {0, -1, 1, -1, 2, -1, 3, -1});
    auto* info = setup.mLayer->getPlacementCpuInfo();
    auto expectSlotHolds = [&](int slotId, float value)
    {
        // The first and last elements of the slot and of each mapping in it.
        for (size_t byte : {size_t{0}, gran - gran / 4, 2 * gran - gran / 4, slotBytes - sizeof(float)})
        {
            EXPECT_EQ(setup.slotValue(slotId, byte / sizeof(float)), value) << "slot " << slotId << " byte " << byte;
        }
        EXPECT_EQ(setup.slotValue(slotId, (gran - gran / 4) / sizeof(float) - 1), value) << "slot " << slotId;
    };

    // Losing rank 2: its expert 2 goes into rank 0's ghost slot, whose pieces the restore maps and pins.
    std::vector<std::vector<int>> const placement{{0, 2}, {1, -1}, {-1, -1}, {3, -1}};
    EXPECT_EQ(setup.mLoadBalancer.restoreBackingBytes({placement}), static_cast<int64_t>(2 * gran));
    setup.mLoadBalancer.restorePlacement({RestorePlacementLayer::kDeadRank}, {placement});
    expectSlotHolds(1, 3.0F);

    // EPLB's updates then load both slots of rank 0 with CPU copies.
    setup.mLoadBalancer.resetUpdateSchedule();
    for (int64_t iterId = 0; iterId < 3; ++iterId)
    {
        std::vector<float> load{1.0F, 2.0F, 3.0F, 4.0F};
        std::rotate(load.begin(), load.begin() + iterId + 1, load.end());
        setup.runUpdate(iterId, load);
        expectSlotHolds(0, static_cast<float>(info->oldRankExpertIds[0][0] + 1));
        expectSlotHolds(1, static_cast<float>(info->oldRankExpertIds[0][1] + 1));
    }
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

TEST_P(MoeLoadBalancerTest, QuiesceWaitsForTheWorkerAndTheGpuStage)
{
    auto param = GetParam();
    if (param.iterConfigs.size() < 2)
    {
        GTEST_SKIP() << "needs two iterations";
    }
    // quiesce needs the worker of WideEP fault tolerance, fixed at finalizeModel.
    ScopedFaultToleranceMode const faultToleranceMode(/*enabled=*/true);
    mLoadBalancer->finalizeModel();
    if (param.warmUpIters > 0)
    {
        mLoadBalancer->setWarmUpIterCount(param.warmUpIters);
    }
    setExpertLoadFactors(param);
    for (int64_t iterId = 0; iterId < 2; ++iterId)
    {
        auto const& config = param.iterConfigs[iterId];
        mLoadBalancer->startIter(iterId, config.statisticEnabled, config.updateWeightsEnabled);
        mLoadBalancer->endIter(iterId);
        // The worker waits for the GPU stage of this iteration: not quiesced, and the schedule cannot be reset.
        EXPECT_FALSE(mLoadBalancer->quiesce(/*timeoutMs=*/100));
        EXPECT_THROW(mLoadBalancer->resetUpdateSchedule(), std::exception);
        simulateGpuSignalWaitAndProcess(iterId, config.statisticEnabled,
            config.updateWeightsEnabled && iterId >= param.warmUpIters,
            iterId < static_cast<int64_t>(mLayerUpdateSchedule.size()) ? mLayerUpdateSchedule[iterId]
                                                                       : std::vector<int>{});
        EXPECT_TRUE(mLoadBalancer->quiesce(/*timeoutMs=*/10000));
        cudaDeviceSynchronize();
    }
    mLoadBalancer->resetUpdateSchedule();
    for (auto const& layer : mLayers)
    {
        auto const* statistic = layer->getStatisticInfo();
        for (int expertId = 0; expertId < param.expertCount; ++expertId)
        {
            EXPECT_EQ(statistic->expertLoadFactor[expertId], 0.0f);
        }
    }
    mLoadBalancer->shutdown();
    mLoadBalancer.reset();
}

TEST_P(MoeLoadBalancerTest, QuiesceReleasesTheLayersAnInterruptedForwardDidNotReach)
{
    auto param = GetParam();
    if (mLayers.size() < 2 || param.iterConfigs.size() < 2)
    {
        GTEST_SKIP() << "needs two layers and two iterations";
    }
    // quiesce needs the worker of WideEP fault tolerance, fixed at finalizeModel.
    ScopedFaultToleranceMode const faultToleranceMode(/*enabled=*/true);
    mLoadBalancer->finalizeModel();
    if (param.warmUpIters > 0)
    {
        mLoadBalancer->setWarmUpIterCount(param.warmUpIters);
    }
    setExpertLoadFactors(param);
    auto runLayer = [this](int layerId, int64_t iterId)
    {
        auto* signal = mLayers[layerId]->getSignal();
        int statisticEnabled;
        moeWaitSignalForGpuStageForTest(signal, &statisticEnabled);
        EXPECT_EQ(static_cast<int64_t>(signal->stepAndOwner >> 2), iterId) << "layer " << layerId;
        moeSetSignalForCpuStageForTest(signal);
    };

    // Iteration 0's forward raises after its first layer: no later layer ever sets its CPU stage.
    auto const& first = param.iterConfigs[0];
    mLoadBalancer->startIter(0, first.statisticEnabled, first.updateWeightsEnabled);
    runLayer(0, 0);
    mLoadBalancer->endIter(0, /*interrupted=*/true);
    ASSERT_TRUE(mLoadBalancer->quiesce(/*timeoutMs=*/10000));
    using Signal = tensorrt_llm::kernels::MoeLoadBalanceSingleLayerSignal;
    for (size_t layerId = 0; layerId < mLayers.size(); ++layerId)
    {
        ASSERT_EQ(mLayers[layerId]->getSignal()->stepAndOwner & Signal::kDevice, Signal::kCPU)
            << "layer " << layerId << " is not released to the CPU";
    }
    mLoadBalancer->resetUpdateSchedule();

    // The next iteration runs every layer and quiesces as usual.
    auto const& second = param.iterConfigs[1];
    mLoadBalancer->startIter(1, second.statisticEnabled, second.updateWeightsEnabled);
    for (size_t layerId = 0; layerId < mLayers.size(); ++layerId)
    {
        runLayer(static_cast<int>(layerId), 1);
    }
    mLoadBalancer->endIter(1);
    EXPECT_TRUE(mLoadBalancer->quiesce(/*timeoutMs=*/10000));
    cudaDeviceSynchronize();
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
