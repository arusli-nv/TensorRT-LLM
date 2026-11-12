# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accept or discard the executor step that a rank failure may have interrupted."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from .llm_request import LlmRequest, PyResult

if TYPE_CHECKING:
    from .py_executor import BatchState, PyExecutor
    from .scheduler import ScheduledRequests

# The context reuse commits of a step, held until the step is accepted.
ContextCommits = List[Tuple[KVCacheManagerV2, Dict[int, int]]]
# The stored context rows of a step's requests, rewound if the step is discarded.
ContextRows = List[Tuple[LlmRequest, PyResult.ContextRowsMark]]


def mark_context_rows(scheduled_batch: "ScheduledRequests") -> ContextRows:
    """Return the stored context rows of each request that returns context logits."""
    return [
        (request, request.py_result.mark_context_rows())
        for request in scheduled_batch.context_requests
        if request.py_return_context_logits
    ]


def commit_context_blocks(executor: "PyExecutor", batch_state: "BatchState") -> None:
    """Commit the context chunks of an accepted step for reuse."""
    if not any(commit_ends for _, commit_ends in batch_state.context_commits):
        return
    # A commit can rewrite host page-index rows that the current step's input copy may still be reading.
    executor._wait_for_model_engine_input_copy()
    for manager, commit_ends in batch_state.context_commits:
        manager.commit_context_blocks(batch_state.scheduled_requests, commit_ends)


def accept_previous_batch(executor: "PyExecutor") -> Optional["BatchState"]:
    """Return the executor's previous batch once rank failure recovery accepted it, or None."""
    batch = executor.previous_batch
    if batch is None:
        return None
    executor.rank_failure_recovery.drop_invalid_step()
    if executor.previous_batch is None:
        return None
    commit_context_blocks(executor, batch)
    return batch
