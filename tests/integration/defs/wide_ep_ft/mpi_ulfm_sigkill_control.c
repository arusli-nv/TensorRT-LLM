/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <mpi.h>

#include <mpi-ext.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank = -1;
    int worldSize = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize != 3)
    {
        fprintf(stderr, "EXPECTED_THREE_RANKS got=%d\n", worldSize);
        MPI_Finalize();
        return 2;
    }

    int value = rank + 1;
    int healthySum = 0;
    int healthyCode = MPI_Allreduce(&value, &healthySum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("HEALTHY rank=%d pid=%d code=%d sum=%d\n", rank, getpid(), healthyCode, healthySum);
    fflush(stdout);
    if (healthyCode != MPI_SUCCESS || healthySum != 6)
    {
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1)
    {
        raise(SIGKILL);
        return 4;
    }

    MPI_Comm survivors = MPI_COMM_NULL;
    int shrinkCode = MPIX_Comm_shrink(MPI_COMM_WORLD, &survivors);
    if (shrinkCode != MPI_SUCCESS)
    {
        printf("SHRINK_ERROR rank=%d code=%d\n", rank, shrinkCode);
        fflush(stdout);
        return 5;
    }

    int survivorCount = -1;
    MPI_Comm_size(survivors, &survivorCount);
    int agreed = 1;
    int agreeCode = MPIX_Comm_agree(survivors, &agreed);
    int survivorSum = 0;
    int allreduceCode = MPI_Allreduce(&rank, &survivorSum, 1, MPI_INT, MPI_SUM, survivors);
    printf("SURVIVOR_AGREE old_rank=%d count=%d agree_code=%d agreed=%d allreduce_code=%d sum=%d\n", rank,
        survivorCount, agreeCode, agreed, allreduceCode, survivorSum);
    fflush(stdout);

    MPI_Comm_free(&survivors);
    MPI_Finalize();
    return survivorCount == 2 && agreeCode == MPI_SUCCESS && agreed == 1 && allreduceCode == MPI_SUCCESS
            && survivorSum == 2
        ? 0
        : 6;
}
