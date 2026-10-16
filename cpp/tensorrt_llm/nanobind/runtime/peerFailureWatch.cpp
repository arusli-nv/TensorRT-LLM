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

#include "peerFailureWatch.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

#include <nanobind/stl/function.h>
#if ENABLE_MULTI_DEVICE
#include <mpi.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#endif

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::runtime
{
namespace
{

#if ENABLE_MULTI_DEVICE
//! Watches a private communicator for a failed peer from a native thread.
//!
//! It posts a receive that no peer matches; ULFM fails it once any process of the communicator fails. The thread
//! polls it with MPI_Test, which needs no GIL. On a failure it first sets `failed`, then stores 1 to the abort word
//! (the MoE all-to-all kernels' pinned abort request, if given), then calls `onFailure` with the MPI error class,
//! once. Any MPI error fails the watch: a watch that cannot detect is reported like a failure.
class PeerFailureWatch
{
public:
    using Callback = std::function<void(int)>;

    PeerFailureWatch(std::int64_t fortranComm, std::uintptr_t abortWord, Callback onFailure, double pollIntervalS)
        : mComm{MPI_Comm_f2c(static_cast<MPI_Fint>(fortranComm))}
        , mAbort{reinterpret_cast<std::int32_t*>(abortWord)}
        , mOnFailure{std::move(onFailure)}
        , mPollInterval{
              std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(pollIntervalS))}
    {
        TLLM_CHECK_WITH_INFO(pollIntervalS > 0, "poll interval must be positive");
        int const rc = MPI_Irecv(&mBuffer, 1, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, mComm, &mRequest);
        TLLM_CHECK_WITH_INFO(rc == MPI_SUCCESS, "MPI_Irecv for peer-failure detection failed (%d)", rc);
        mThread = std::thread([this] { watch(); });
    }

    PeerFailureWatch(PeerFailureWatch const&) = delete;
    PeerFailureWatch& operator=(PeerFailureWatch const&) = delete;

    ~PeerFailureWatch()
    {
        if (PyGILState_Check())
        {
            nb::gil_scoped_release release;
            stop();
        }
        else
        {
            stop();
        }
    }

    [[nodiscard]] bool failed() const noexcept
    {
        return mFailed.load();
    }

    //! Stop watching and release the receive; returns whether it did, after which the caller may free the
    //! communicator. Idempotent; never calls `onFailure` after it returns true. Called from `onFailure` itself, it
    //! only stops the watch and returns false: the thread cannot join itself, so a later call from another
    //! thread (at the latest the destructor's) joins and releases.
    bool stop()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStopping = true;
        }
        mWake.notify_all();
        if (std::this_thread::get_id() == mThread.get_id())
        {
            return false;
        }
        std::call_once(mReleased,
            [this]
            {
                mThread.join();
                // A failed peer may fail these too; the communicator is freed next either way.
                if (mRequest != MPI_REQUEST_NULL && MPI_Cancel(&mRequest) == MPI_SUCCESS)
                {
                    MPI_Wait(&mRequest, MPI_STATUS_IGNORE);
                }
            });
        return true;
    }

private:
    void watch()
    {
#if defined(__linux__)
        pthread_setname_np(pthread_self(), "peer_fail_watch");
#endif
        std::unique_lock<std::mutex> lock(mMutex);
        while (!mStopping)
        {
            int done = 0;
            int const rc = MPI_Test(&mRequest, &done, MPI_STATUS_IGNORE);
            // No peer sends on this communicator, so a completed receive means detection is gone (MPI_Test has
            // released the request): report it like a failure rather than polling a null request blind.
            if (rc != MPI_SUCCESS || done)
            {
                lock.unlock();
                report(rc != MPI_SUCCESS ? rc : MPI_ERR_OTHER);
                return;
            }
            mWake.wait_for(lock, mPollInterval);
        }
    }

    void report(int rc)
    {
        int errorClass = rc;
        MPI_Error_class(rc, &errorClass);
        mFailed.store(true);
        if (mAbort != nullptr)
        {
            __atomic_store_n(mAbort, 1, __ATOMIC_SEQ_CST);
        }
        try
        {
            // The nanobind function wrapper acquires the GIL for the call.
            mOnFailure(errorClass);
        }
        catch (nb::python_error& error)
        {
            nb::gil_scoped_acquire acquire;
            error.discard_as_unraisable("peer-failure watch callback");
        }
        // Anything else escaping a std::thread would terminate the process; `failed` and the abort word are
        // already set, which is what the serving loop acts on.
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("peer-failure watch callback failed: %s", error.what());
        }
        catch (...)
        {
            TLLM_LOG_ERROR("peer-failure watch callback failed with an unknown exception");
        }
    }

    MPI_Comm mComm;
    MPI_Request mRequest{MPI_REQUEST_NULL};
    char mBuffer{0};
    std::int32_t* mAbort;
    Callback mOnFailure;
    std::chrono::nanoseconds mPollInterval;
    std::atomic<bool> mFailed{false};
    std::mutex mMutex;
    std::condition_variable mWake;
    bool mStopping{false};
    std::once_flag mReleased;
    std::thread mThread;
};
#endif

} // namespace

void initPeerFailureWatchBindings(nb::module_& m)
{
#if ENABLE_MULTI_DEVICE
    nb::class_<PeerFailureWatch>(m, "PeerFailureWatch")
        .def(nb::init<std::int64_t, std::uintptr_t, PeerFailureWatch::Callback, double>(), nb::arg("fortran_comm"),
            nb::arg("abort_word"), nb::arg("on_failure"), nb::arg("poll_interval_s"))
        .def_prop_ro("failed", &PeerFailureWatch::failed)
        .def("stop", &PeerFailureWatch::stop, nb::call_guard<nb::gil_scoped_release>());
#endif
}

} // namespace tensorrt_llm::nanobind::runtime
