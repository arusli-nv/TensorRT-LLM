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
"""Requests that a rank failure ends under Wide-EP fault tolerance: their error and their resubmission."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .utils import RequestError

if TYPE_CHECKING:
    from .._torch.pyexecutor.llm_request import LlmResponse
    from .base_worker import BaseWorker
    from .request import GenerationRequest

# The error code that the serving frontends answer a RankFailureError with.
RANK_FAILURE_CODE = "rank_failure"


class RankFailureError(RequestError):
    """Error of a request that a Wide-EP rank failure ended, safe to retry."""

    def __init__(
        self, message: str = "A rank failure ended the request; sending it again can succeed"
    ):
        super().__init__(message)


class RankFailureRequests:
    """Resubmit the requests that a rank failure ended before the client received any output.

    The worker tracks a request from its submit until its first response, a cancel or a failed
    enqueue. The client has seen nothing of a request still tracked when its RankFailureError
    arrives, so the request runs again unnoticed.
    """

    def __init__(self, worker: "BaseWorker"):
        self._worker = worker
        # Each tracked request with its multimodal params and their multimodal input, which the
        # first enqueue sets to None after converting it.
        self._without_first_response: Dict[int, Tuple["GenerationRequest", Optional[Tuple]]] = {}

    def add(self, client_id: int, request: "GenerationRequest") -> None:
        """Start tracking a request. Call it before the first enqueue, which may answer at once."""
        params = request.multimodal_params
        multimodal = None if params is None else (params, params.multimodal_input)
        self._without_first_response[client_id] = (request, multimodal)

    def stop_tracking(self, client_id: int) -> None:
        """Stop tracking a request that got a response, was cancelled or was never enqueued."""
        self._without_first_response.pop(client_id, None)

    def resubmit_if_unanswered(self, response: "LlmResponse") -> bool:
        """Resubmit the request of ``response`` if a rank failure ended it before any response.

        Returns:
            Whether the request was resubmitted, in which case ``response`` must not be delivered.
        """
        client_id = response.client_id
        if not isinstance(response.error_msg, RankFailureError):
            return False
        try:
            request, multimodal = self._without_first_response[client_id]
        except KeyError:
            return False
        # Restore the multimodal input that the first enqueue cleared.
        if multimodal is not None:
            request.multimodal_params, multimodal_input = multimodal
            request.multimodal_params.multimodal_input = multimodal_input
        # A request that ends before the submit stored its request id fails, since the submit
        # would replace the resubmitted request's id.
        if not self._worker.resubmit(client_id, request):
            return False
        # A client cancel may run during the resubmit. abort_request stops tracking the request,
        # then reads its request id. If it read the id before the store above, it cancelled only the
        # old request and the request is no longer tracked, so cancel the new one here. If it reads
        # the id later, it cancels the new request itself.
        cancelled = client_id not in self._without_first_response
        self._without_first_response.pop(client_id, None)
        if cancelled:
            self._worker.abort_request(client_id)
        return True
