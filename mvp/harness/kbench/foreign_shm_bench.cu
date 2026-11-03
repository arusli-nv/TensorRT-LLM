// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Does a restore pay page faults on host copies another local rank wrote? A child process creates and fills a POSIX
// shm segment (as HostMoeTensorSharer does); this process maps it without touching it and copies it to the GPU in
// COPIES pageable cudaMemcpyAsync calls: cold, after madvise(MADV_POPULATE_READ), and warm (again).
// Usage: foreign_shm_bench [MB=1400] [COPIES=1872]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

static double ms(std::chrono::steady_clock::time_point a)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

static char* attach(char const* name, size_t bytes)
{
    int fd = shm_open(name, O_RDWR, 0600);
    auto* p = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    return p;
}

int main(int argc, char** argv)
{
    size_t const bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1400) << 20;
    int const copies = argc > 2 ? std::atoi(argv[2]) : 1872;
    char name[64];
    std::snprintf(name, sizeof(name), "/foreign_shm_bench_%d", getpid());
    if (fork() == 0)
    {
        int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        if (ftruncate(fd, bytes) != 0)
            _exit(1);
        auto* p = static_cast<char*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        std::memset(p, 7, bytes);
        _exit(0);
    }
    int status = 0;
    wait(&status);
    char* dev = nullptr;
    cudaMalloc(&dev, bytes);
    size_t const each = bytes / copies;
    auto copy = [&](char const* src)
    {
        auto t = std::chrono::steady_clock::now();
        for (int i = 0; i < copies; ++i)
            cudaMemcpyAsync(dev + i * each, src + i * each, each, cudaMemcpyHostToDevice, 0);
        cudaDeviceSynchronize();
        return ms(t);
    };
    char* cold = attach(name, bytes);
    std::printf("RESULT foreign cold_ms=%.2f\n", copy(cold));
    std::printf("RESULT foreign warm_ms=%.2f\n", copy(cold));
    char* fresh = attach(name, bytes);
    auto t = std::chrono::steady_clock::now();
    int rc = madvise(fresh, bytes, MADV_POPULATE_READ);
    std::printf("RESULT populate rc=%d ms=%.2f\n", rc, ms(t));
    std::printf("RESULT foreign populated_ms=%.2f\n", copy(fresh));
    shm_unlink(name);
    std::printf("FOREIGN_SHM_DONE\n");
    return 0;
}
