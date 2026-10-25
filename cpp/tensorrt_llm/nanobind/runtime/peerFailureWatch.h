/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace tensorrt_llm::nanobind::runtime
{

//! Bind PeerFailureWatch: a native thread that detects a failed MPI peer without the Python GIL.
void initPeerFailureWatchBindings(::nanobind::module_& m);

} // namespace tensorrt_llm::nanobind::runtime
