/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace tensorrt_llm::nanobind::runtime
{

//! Bind HeartbeatSender: a native thread that sends a WideEP worker's heartbeat to the recovery authority without
//! the Python GIL.
void initHeartbeatSenderBindings(::nanobind::module_& m);

} // namespace tensorrt_llm::nanobind::runtime
