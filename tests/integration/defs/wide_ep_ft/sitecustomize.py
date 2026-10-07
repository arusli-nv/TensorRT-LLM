# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in instrumentation for native MPI model workers only."""

import os
import sys

if os.environ.get("WIDEEP_FT_RUN_ID") and any(
    name in sys.orig_argv
    for name in ("tensorrt_llm.llmapi.mgmn_worker_node", "tensorrt_llm.llmapi.mgmn_leader_node")
):
    from fault_injector import install_worker_hook

    install_worker_hook()
