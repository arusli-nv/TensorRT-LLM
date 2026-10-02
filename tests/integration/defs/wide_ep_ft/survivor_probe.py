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
"""Validate diagnostic survivor replies without changing TRT-LLM membership."""

from __future__ import annotations

import hashlib
import json


def make_probe_proposal(world_size: int, target_rank: int, transaction_id: str) -> dict:
    if type(world_size) is not int or world_size < 2:
        raise ValueError("world_size must be an integer of at least two")
    if type(target_rank) is not int or not 0 < target_rank < world_size:
        raise ValueError("target_rank must be a nonzero world rank")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("transaction_id must be a nonempty string")
    return {
        "transaction_id": transaction_id,
        "world_size": world_size,
        "failed_rank": target_rank,
        "survivors": [rank for rank in range(world_size) if rank != target_rank],
        "base_generation": 0,
        "next_generation": 1,
    }


def validate_probe_proposal(proposal: dict, rank: int, expected_base_generation: int = 0) -> None:
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a mapping")
    expected_keys = {
        "transaction_id",
        "world_size",
        "failed_rank",
        "survivors",
        "base_generation",
        "next_generation",
    }
    if proposal.keys() != expected_keys:
        raise ValueError("proposal fields do not match the diagnostic schema")
    expected = make_probe_proposal(
        proposal["world_size"], proposal["failed_rank"], proposal["transaction_id"]
    )
    if proposal["survivors"] != expected["survivors"]:
        raise ValueError("survivor set does not match one failed rank")
    if type(proposal["base_generation"]) is not int or type(proposal["next_generation"]) is not int:
        raise ValueError("generations must be integers")
    if proposal["base_generation"] != expected_base_generation:
        raise ValueError("stale or unexpected base generation")
    if proposal["next_generation"] != expected_base_generation + 1:
        raise ValueError("next generation must be consecutive")
    if type(rank) is not int or rank not in proposal["survivors"]:
        raise ValueError("this rank is not a proposed survivor")


def probe_digest(proposal: dict) -> str:
    return hashlib.sha256(
        json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def summarize_probe_replies(
    proposal: dict, replies: dict[int, dict], rank_to_pid: dict[int, int]
) -> dict:
    expected_ranks = set(proposal["survivors"])
    digest = probe_digest(proposal)
    missing = sorted(expected_ranks - replies.keys())
    unexpected = sorted(replies.keys() - expected_ranks)
    invalid = sorted(
        rank
        for rank in expected_ranks & replies.keys()
        if not isinstance(replies[rank], dict)
        or type(replies[rank].get("rank")) is not int
        or replies[rank].get("rank") != rank
        or type(replies[rank].get("pid")) is not int
        or replies[rank]["pid"] != rank_to_pid.get(rank)
        or replies[rank].get("digest") != digest
        or type(replies[rank].get("base_generation")) is not int
        or replies[rank].get("base_generation") != proposal["base_generation"]
        or type(replies[rank].get("next_generation")) is not int
        or replies[rank].get("next_generation") != proposal["next_generation"]
    )
    return {
        "all_survivors_replied": not (missing or unexpected or invalid),
        "expected_digest": digest,
        "missing_ranks": missing,
        "unexpected_ranks": unexpected,
        "invalid_ranks": invalid,
    }
