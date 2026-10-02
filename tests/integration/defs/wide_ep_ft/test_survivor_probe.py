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
"""CPU-only checks for one-failure diagnostic survivor replies."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from ray_process_loss_probe import _submit_survivor_probes
from survivor_probe import (
    make_probe_proposal,
    probe_digest,
    summarize_probe_replies,
    validate_probe_proposal,
)
from survivor_probe_worker_extension import SurvivorProbeWorkerExtension


class SurvivorProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal = make_probe_proposal(4, 1, "run-1")

    def test_proposal_is_one_nonzero_failure(self) -> None:
        self.assertEqual(self.proposal["survivors"], [0, 2, 3])
        validate_probe_proposal(self.proposal, 2)
        for world_size, target_rank in ((1, 1), (4, 0), (4, 4)):
            with self.subTest(world_size=world_size, target_rank=target_rank):
                with self.assertRaises(ValueError):
                    make_probe_proposal(world_size, target_rank, "run-1")
        with self.assertRaisesRegex(ValueError, "not a proposed survivor"):
            validate_probe_proposal(self.proposal, 1)

    def test_stale_or_malformed_proposals_are_rejected(self) -> None:
        variants = (
            {**self.proposal, "base_generation": -1, "next_generation": 0},
            {**self.proposal, "next_generation": 2},
            {**self.proposal, "survivors": [0, 3]},
            {**self.proposal, "survivors": [0, 2, 2, 3]},
            {**self.proposal, "unexpected": True},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    validate_probe_proposal(variant, 0)

    def test_digest_is_canonical_and_transaction_specific(self) -> None:
        reordered = dict(reversed(list(self.proposal.items())))
        self.assertEqual(probe_digest(reordered), probe_digest(self.proposal))
        conflicting = {**self.proposal, "transaction_id": "run-2"}
        self.assertNotEqual(probe_digest(conflicting), probe_digest(self.proposal))

    def test_complete_replies_require_every_matching_survivor(self) -> None:
        digest = probe_digest(self.proposal)
        replies = {
            rank: {
                "rank": rank,
                "pid": 1000 + rank,
                "digest": digest,
                "base_generation": 0,
                "next_generation": 1,
            }
            for rank in self.proposal["survivors"]
        }
        rank_to_pid = {rank: 1000 + rank for rank in self.proposal["survivors"]}
        self.assertTrue(
            summarize_probe_replies(self.proposal, replies, rank_to_pid)["all_survivors_replied"]
        )
        missing = summarize_probe_replies(
            self.proposal,
            {rank: reply for rank, reply in replies.items() if rank != 2},
            rank_to_pid,
        )
        self.assertFalse(missing["all_survivors_replied"])
        self.assertEqual(missing["missing_ranks"], [2])
        invalid = summarize_probe_replies(
            self.proposal, {**replies, 2: {"status": "timeout"}}, rank_to_pid
        )
        self.assertFalse(invalid["all_survivors_replied"])
        self.assertEqual(invalid["invalid_ranks"], [2])
        wrong_pid = summarize_probe_replies(
            self.proposal, {**replies, 2: {**replies[2], "pid": 9999}}, rank_to_pid
        )
        self.assertEqual(wrong_pid["invalid_ranks"], [2])
        wrong_generation = summarize_probe_replies(
            self.proposal, {**replies, 2: {**replies[2], "next_generation": True}}, rank_to_pid
        )
        self.assertEqual(wrong_generation["invalid_ranks"], [2])
        unexpected = summarize_probe_replies(self.proposal, {**replies, 1: replies[0]}, rank_to_pid)
        self.assertEqual(unexpected["unexpected_ranks"], [1])

    def test_worker_reply_is_idempotent_but_rejects_conflict(self) -> None:
        distributed = types.ModuleType("torch.distributed")
        distributed.get_rank = lambda: 0
        torch = types.ModuleType("torch")
        torch.distributed = distributed
        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": distributed}):
            worker = SurvivorProbeWorkerExtension()
            first = worker.prepare_survivor_probe(self.proposal)
            self.assertEqual(worker.prepare_survivor_probe(self.proposal), first)
            self.assertEqual(first["digest"], probe_digest(self.proposal))
            with self.assertRaisesRegex(ValueError, "conflicting diagnostic proposal"):
                worker.prepare_survivor_probe({**self.proposal, "transaction_id": "run-2"})
            with self.assertRaisesRegex(ValueError, "stale"):
                worker.prepare_survivor_probe(
                    {**self.proposal, "base_generation": -1, "next_generation": 0}
                )

    def test_submission_failure_preserves_other_survivor_probes(self) -> None:
        workers = []
        for rank in range(4):

            def submit(method: str, proposal: dict, world_rank: int = rank):
                if world_rank == 2:
                    raise RuntimeError("actor unavailable")
                return world_rank

            workers.append(
                types.SimpleNamespace(call_worker_method=types.SimpleNamespace(remote=submit))
            )

        references, errors = _submit_survivor_probes(workers, self.proposal)
        self.assertEqual(references, {0: 0, 3: 3})
        self.assertEqual(errors[2]["status"], "submission_error")
        self.assertEqual(errors[2]["error"], "actor unavailable")


if __name__ == "__main__":
    unittest.main()
