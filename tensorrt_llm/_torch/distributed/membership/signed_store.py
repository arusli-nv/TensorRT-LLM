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
"""Sign rendezvous store values with the Wide-EP deployment's shared secret."""

import hashlib
import hmac
import os
from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional

from tensorrt_llm._wide_ep_fault_tolerance import SHARED_SECRET_ENV

if TYPE_CHECKING:
    from torch.distributed import Store

_TAG_BYTES = hashlib.sha256().digest_size
SHARED_SECRET_BYTES = 32


class UnsignedValueError(ValueError):
    """A rendezvous store value is not signed with the deployment's shared secret."""


def parse_shared_secret(secret: str) -> bytes:
    """Parse the hex shared secret ``secret``.

    Raises:
        ValueError: ``secret`` is not hex or is shorter than ``SHARED_SECRET_BYTES``.
    """
    try:
        parsed = bytes.fromhex(secret)
    except ValueError:
        parsed = b""
    if len(parsed) < SHARED_SECRET_BYTES:
        raise ValueError(
            f"Wide-EP fault tolerance requires a shared secret of at least {SHARED_SECRET_BYTES} "
            f"bytes ({2 * SHARED_SECRET_BYTES} hex digits) in {SHARED_SECRET_ENV}"
        )
    return parsed


class SignedStore:
    """A rendezvous store client that signs the values it sets and verifies those it gets.

    Each value is prefixed with an HMAC-SHA256 tag over its key and bytes, keyed by a digest of
    the shared secret. A process without the secret cannot set a value that the deployment's
    processes accept. Counters (``add``) carry no tag.
    """

    def __init__(self, store: "Store", secret: Optional[str] = None):
        """Sign the values of ``store`` with ``secret`` (hex), by default ``SHARED_SECRET_ENV``.

        Raises:
            ValueError: ``secret`` is not a valid shared secret (:func:`parse_shared_secret`).
        """
        secret = os.environ[SHARED_SECRET_ENV] if secret is None else secret
        self._store = store
        self._key = hmac.new(
            parse_shared_secret(secret), b"rendezvous store values", hashlib.sha256
        ).digest()

    @property
    def host(self) -> str:
        return self._store.host

    @property
    def port(self) -> int:
        return self._store.port

    def set(self, key: str, value: bytes) -> None:
        self._store.set(key, self._tag(key, value) + value)

    def get(self, key: str) -> bytes:
        """Wait until ``key`` is set and return its value.

        Raises:
            UnsignedValueError: The value is not signed with the deployment's shared secret.
        """
        return self._verified(key, self._store.get(key))

    def multi_get(self, keys: List[str]) -> List[bytes]:
        """Return the values for ``keys``, as :meth:`get` does."""
        return [self._verified(k, v) for k, v in zip(keys, self._store.multi_get(keys))]

    def check(self, keys: List[str]) -> bool:
        return self._store.check(keys)

    def wait(self, keys: List[str], timeout: Optional[timedelta] = None) -> None:
        if timeout is None:
            self._store.wait(keys)
        else:
            self._store.wait(keys, timeout)

    def add(self, key: str, amount: int) -> int:
        return self._store.add(key, amount)

    def delete_key(self, key: str) -> bool:
        return self._store.delete_key(key)

    def _tag(self, key: str, value: bytes) -> bytes:
        encoded = key.encode()
        message = len(encoded).to_bytes(4, "big") + encoded + bytes(value)
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def _verified(self, key: str, signed: bytes) -> bytes:
        tag, value = signed[:_TAG_BYTES], signed[_TAG_BYTES:]
        if not hmac.compare_digest(tag, self._tag(key, value)):
            raise UnsignedValueError(
                f"Rendezvous store value {key!r} is not signed with this deployment's "
                f"{SHARED_SECRET_ENV}, so another process set it"
            )
        return value
