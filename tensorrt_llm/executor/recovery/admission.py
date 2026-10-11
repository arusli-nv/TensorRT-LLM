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
"""Frontend request admission driven by the recovery authority."""

import threading
from collections.abc import Callable

from .client import RecoveryAborted, RecoveryClient, RecoveryError
from .protocol import Abort, AdmissionClosed, CloseAdmission, ProtocolError, ReopenAdmission

__all__ = ["FrontendAdmission"]


class FrontendAdmission:
    """Closes and reopens frontend admission on authority command.

    Submitters must hold ``lock`` while checking ``closed`` and registering a request,
    so once admission is closed ``is_idle`` observes every admitted request.

    Args:
        client: Frontend connection to the authority.
        is_idle: Returns True when the frontend has no request in flight.
        on_failure: Called once, from the background thread, when recovery aborts or
            the authority exchange fails.
        poll_interval: Seconds between idle checks while closing.
    """

    def __init__(
        self,
        client: RecoveryClient,
        is_idle: Callable[[], bool],
        on_failure: Callable[[RecoveryError | ProtocolError], None],
        poll_interval: float = 0.05,
    ) -> None:
        self.lock = threading.Lock()
        self.closed = False
        self._client = client
        self._is_idle = is_idle
        self._on_failure = on_failure
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="recovery_admission")
        self._thread.start()

    def close(self) -> None:
        """Ask the background thread to exit; it closes the client within ``poll_interval``."""
        self._stop.set()

    def _run(self) -> None:
        try:
            self._serve()
        except (RecoveryError, ProtocolError) as error:
            self._on_failure(error)
        finally:
            self._client.close()

    def _serve(self) -> None:
        draining = False
        while not self._stop.is_set():
            message = self._client.recv(self._poll_interval)
            if isinstance(message, Abort):
                raise RecoveryAborted(message.reason)
            if isinstance(message, CloseAdmission) and not self.closed:
                with self.lock:
                    self.closed = True
                draining = True
            elif isinstance(message, ReopenAdmission) and self.closed and not draining:
                self.closed = False
            elif message is not None:
                raise RecoveryError(f"Unexpected {message} with admission closed={self.closed}")
            if draining and self._is_idle():
                self._client.send(AdmissionClosed())
                draining = False
