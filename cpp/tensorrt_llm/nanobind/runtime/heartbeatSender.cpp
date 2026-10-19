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

#include "heartbeatSender.h"

#include "sha256.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <nanobind/stl/string.h>
#if defined(__linux__)
#include <netdb.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::runtime
{
namespace
{

#if defined(__linux__)
constexpr std::size_t kSha256BlockBytes = 64;
using Tag = std::array<unsigned char, CSHA256::OUTPUT_SIZE>;

//! HMAC-SHA256 of `message` under `key` (RFC 2104).
Tag hmacSha256(std::string const& key, std::string const& message)
{
    std::array<unsigned char, kSha256BlockBytes> block{};
    if (key.size() > block.size())
    {
        CSHA256().Write(reinterpret_cast<unsigned char const*>(key.data()), key.size()).Finalize(block.data());
    }
    else
    {
        std::copy(key.begin(), key.end(), block.begin());
    }
    std::array<unsigned char, kSha256BlockBytes> pad{};
    std::transform(block.begin(), block.end(), pad.begin(),
        [](unsigned char byte) { return static_cast<unsigned char>(byte ^ 0x36); });
    Tag inner{};
    CSHA256()
        .Write(pad.data(), pad.size())
        .Write(reinterpret_cast<unsigned char const*>(message.data()), message.size())
        .Finalize(inner.data());
    std::transform(block.begin(), block.end(), pad.begin(),
        [](unsigned char byte) { return static_cast<unsigned char>(byte ^ 0x5c); });
    Tag tag{};
    CSHA256().Write(pad.data(), pad.size()).Write(inner.data(), inner.size()).Finalize(tag.data());
    return tag;
}

//! Sends a WideEP worker's heartbeat to the recovery authority from a native thread, without the Python GIL, so
//! that it keeps arriving while the process's Python threads are starved.
//!
//! Every interval, starting at construction, it sends one UDP datagram to the authority's host and port: the JSON
//! payload of a `NativeHeartbeat` message (rank, pid, a count of the datagrams sent and the last published executor
//! iteration) followed by its HMAC-SHA256 tag, as `tensorrt_llm.executor.recovery.protocol.encode_datagram` does. A
//! datagram that cannot be sent is dropped.
class HeartbeatSender
{
public:
    HeartbeatSender(
        std::string const& host, int port, nb::bytes const& key, int rank, std::int64_t pid, double intervalS)
        : mKey{key.c_str(), key.size()}
        , mRank{rank}
        , mPid{pid}
        , mInterval{std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(intervalS))}
    {
        TLLM_CHECK_WITH_INFO(intervalS > 0, "heartbeat interval must be positive");
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_DGRAM;
        addrinfo* addresses = nullptr;
        std::string const service = std::to_string(port);
        int const rc = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
        TLLM_CHECK_WITH_INFO(
            rc == 0, "cannot resolve the recovery authority %s:%d: %s", host.c_str(), port, ::gai_strerror(rc));
        for (addrinfo* address = addresses; address != nullptr && mSocket < 0; address = address->ai_next)
        {
            mSocket = ::socket(address->ai_family, address->ai_socktype | SOCK_CLOEXEC, address->ai_protocol);
            if (mSocket >= 0 && ::connect(mSocket, address->ai_addr, address->ai_addrlen) != 0)
            {
                ::close(mSocket);
                mSocket = -1;
            }
        }
        ::freeaddrinfo(addresses);
        TLLM_CHECK_WITH_INFO(
            mSocket >= 0, "cannot open a UDP socket to the recovery authority %s:%d", host.c_str(), port);
        mThread = std::thread([this] { run(); });
    }

    HeartbeatSender(HeartbeatSender const&) = delete;
    HeartbeatSender& operator=(HeartbeatSender const&) = delete;

    ~HeartbeatSender()
    {
        stop();
        ::close(mSocket);
    }

    //! Report `iteration` as the executor's progress from the next datagram on.
    void publish(std::int64_t iteration) noexcept
    {
        mIteration.store(iteration, std::memory_order_relaxed);
    }

    //! Stop sending; idempotent. The thread never takes the GIL, so a caller may hold it.
    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStopping = true;
        }
        mWake.notify_all();
        if (mThread.joinable())
        {
            mThread.join();
        }
    }

private:
    void run()
    {
        ::pthread_setname_np(::pthread_self(), "ft_heartbeat");
        std::unique_lock<std::mutex> lock(mMutex);
        for (std::int64_t beat = 1; !mStopping; ++beat)
        {
            send(beat);
            mWake.wait_for(lock, mInterval, [this] { return mStopping; });
        }
    }

    void send(std::int64_t beat)
    {
        std::array<char, 192> payload{};
        int const length = std::snprintf(payload.data(), payload.size(),
            R"({"beat": %lld, "iteration": %lld, "pid": %lld, "rank": %d, "type": "NativeHeartbeat"})",
            static_cast<long long>(beat), static_cast<long long>(mIteration.load(std::memory_order_relaxed)),
            static_cast<long long>(mPid), mRank);
        std::string datagram(payload.data(), static_cast<std::size_t>(length));
        // Signed over a zero byte, then the payload, so it never authenticates as a connection's message.
        Tag const tag = hmacSha256(mKey, std::string(1, '\0') + datagram);
        datagram.append(reinterpret_cast<char const*>(tag.data()), tag.size());
        if (::send(mSocket, datagram.data(), datagram.size(), MSG_DONTWAIT) < 0 && !mSendFailed)
        {
            mSendFailed = true;
            TLLM_LOG_WARNING("WideEP heartbeat to the recovery authority failed: %s", std::strerror(errno));
        }
    }

    std::string const mKey;
    int const mRank;
    std::int64_t const mPid;
    std::chrono::nanoseconds const mInterval;
    int mSocket{-1};
    std::atomic<std::int64_t> mIteration{0};
    bool mSendFailed{false};
    std::mutex mMutex;
    std::condition_variable mWake;
    bool mStopping{false};
    std::thread mThread;
};
#endif

} // namespace

void initHeartbeatSenderBindings(nb::module_& m)
{
#if defined(__linux__)
    nb::class_<HeartbeatSender>(m, "HeartbeatSender")
        .def(nb::init<std::string const&, int, nb::bytes const&, int, std::int64_t, double>(), nb::arg("host"),
            nb::arg("port"), nb::arg("key"), nb::arg("rank"), nb::arg("pid"), nb::arg("interval_s"))
        .def("publish", &HeartbeatSender::publish, nb::arg("iteration"))
        .def("stop", &HeartbeatSender::stop, nb::call_guard<nb::gil_scoped_release>());
#endif
}

} // namespace tensorrt_llm::nanobind::runtime
