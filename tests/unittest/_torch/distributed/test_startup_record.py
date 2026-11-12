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
"""Tests for the startup record of Wide-EP fault tolerance.

A replacement rank's startup collectives return what rank 0 received at the same calls. ``MnnvlDist``'s host
collectives run over a fake host exchange. Rank 0 records them, and a replacement of rank 2 replays them. The
replacement's own exchange reaches only itself.
"""

import contextlib
import os
import pickle
import resource
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist
from tensorrt_llm._torch.distributed.membership import REPLACEMENT_CANNOT_START
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead
from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_auto_host_tier_quota,
    _raise_on_every_rank,
    _sync_host_tier_quota,
    get_host_memory_limits,
)
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
REPLACED = 2
GIB = 1 << 30


class FakeHostExchange:
    """Fake the ``TPHostExchange`` of ``rank`` with the ranks in ``active_mask``.

    When this rank sends ``obj``, peer ``r`` sends ``peers(r, obj)``.
    """

    def __init__(self, rank, active_mask=(1 << WORLD) - 1, peers=lambda r, obj: obj):
        self.rank, self.active_mask, self.peers = rank, active_mask, peers
        self.calls = 0

    def _sent(self, r, obj):
        return obj if r == self.rank else self.peers(r, obj)

    def exchange(self, payload, readers=None, recipients=None, idle=False):
        self.calls += 1
        obj = pickle.loads(payload)
        read = range(WORLD) if readers is None else readers
        return [
            pickle.dumps(self._sent(r, obj)) if r in read and self.active_mask >> r & 1 else None
            for r in range(WORLD)
        ]

    def allgather_array(self, row):
        self.calls += 1
        return np.stack(
            [
                np.asarray(self._sent(r, row), dtype=np.int64)
                if self.active_mask >> r & 1
                else np.zeros_like(row)
                for r in range(WORLD)
            ]
        )


def _dist(rank, host_exchange):
    dist = MnnvlDist(
        SimpleNamespace(
            rank=rank, tp_rank=rank, tp_size=WORLD, cp_size=1, world_size=WORLD, dwdp_enabled=False
        )
    )
    dist._host_exchange = host_exchange
    return dist


@pytest.fixture(autouse=True)
def fresh_record(monkeypatch):
    """Give each test an empty startup record."""
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)


def _rank_0(peers=lambda r, obj: obj):
    """Return rank 0's ``MnnvlDist`` and start recording its startup."""
    startup_record.record()
    return _dist(0, FakeHostExchange(0, peers=peers))


def _publish() -> bytes:
    """End rank 0's startup and return its record as the rendezvous store carries it."""
    startup_record.close()
    return pickle.dumps(startup_record.published())


def _replacement(published: bytes):
    """Return a replacement of rank ``REPLACED`` that replays ``published``, whose host exchange reaches only itself."""
    for name, value in vars(StartupRecord()).items():
        setattr(startup_record, name, value)
    startup_record.replay(pickle.loads(published))
    return _dist(REPLACED, FakeHostExchange(REPLACED, active_mask=1 << REPLACED))


# The startup's host collectives. Each one runs at the same line on every rank.
def _gather(dist, obj):
    return dist.allgather(obj)


def _barrier(dist):
    dist.barrier()


def _startup(dist, max_tokens):
    flags = dist.tp_allgather(1)
    tactics = dist.tp_broadcast({"gemm": 3} if dist.mapping.tp_rank == 0 else None, root=0)
    max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
    rows = dist.tp_allgather_int64([8, 1])
    _barrier(dist)
    return flags, tactics, max_tokens, rows.tolist()


def _fewer_tokens(r, obj):
    """Give peer r 100 * r tokens fewer than this rank."""
    return obj - 100 * r if isinstance(obj, int) and obj > 1 else obj


def test_replacement_replays_rank_0_results():
    expected = ([1] * WORLD, {"gemm": 3}, 4096 - 300, [[8, 1]] * WORLD)
    assert _startup(_rank_0(_fewer_tokens), 4096) == expected
    replacement = _replacement(_publish())
    # The replacement has its predecessor's capacity of 4096 - 200, which the minimum bounds.
    assert _startup(replacement, 4096 - 200) == expected
    assert replacement.host_exchange.calls == 0
    startup_record.close()


def test_unbounded_minimum_stops_replacement():
    _startup(_rank_0(_fewer_tokens), 4096)
    published = _publish()
    # A capacity larger than the predecessor's 4096 - 200 takes the minimum.
    assert _startup(_replacement(published), 8192)[2] == 4096 - 300
    replacement = _replacement(published)
    with pytest.raises(
        SystemExit, match=f"allreduce in _startup .* is {4096 - 400}, beyond .* min {4096 - 300}"
    ) as exited:
        _startup(replacement, 4096 - 400)
    assert exited.value.code == REPLACEMENT_CANNOT_START


def test_paused_agreement_is_not_recorded():
    _rank_0()
    startup_record.pause()
    assert startup_record.agree("estimation", lambda: 1) == 1
    startup_record.resume()
    assert startup_record.values == {}


def test_divergent_replacement_cannot_start():
    rank_0 = _rank_0()
    _gather(rank_0, 1)
    _barrier(rank_0)
    replacement = _replacement(_publish())
    # A different collective stops the replacement.
    with pytest.raises(
        SystemExit, match="is barrier in _barrier .*, the original ranks' allgather in _gather"
    ):
        _barrier(replacement)
    # So does a different payload.
    with pytest.raises(
        SystemExit, match="allgather in _gather .* differs from the original rank's"
    ) as exited:
        _gather(replacement, 0)
    assert exited.value.code == REPLACEMENT_CANNOT_START
    assert _gather(replacement, 1) == [1] * WORLD
    # So does one collective fewer.
    with pytest.raises(
        SystemExit, match="makes 1 host collectives, the original ranks 2: the next one is barrier"
    ):
        startup_record.close()
    _barrier(replacement)
    with pytest.raises(SystemExit, match="one more than the original ranks made"):
        _barrier(replacement)
    startup_record.close()


def test_equal_payloads_pickled_differently_match():
    repeated = "payload"
    _gather(_rank_0(), [repeated, repeated])
    replacement = _replacement(_publish())
    # Two equal strings pickle differently from one string twice.
    assert _gather(replacement, ["payload", "".join(["pay", "load"])]) == [["payload"] * 2] * WORLD
    startup_record.close()


def test_bounded_status_reports_replacement_error(distributed):
    """Check that a replacement whose KV cache allocation failed raises that error."""
    mapping = SimpleNamespace(world_size=WORLD)
    distributed.dist = _rank_0()
    _raise_on_every_rank(None, mapping, "KV cache allocation")
    distributed.dist = _replacement(_publish())
    with pytest.raises(MemoryError, match="pool"):
        _raise_on_every_rank(MemoryError("pool"), mapping, "KV cache allocation")


def _quota(dist, quota):
    return startup_record.agree_rank_capacity(
        "kv_cache_device_quota", dist.mapping.rank, quota, lambda: dist.allgather(quota)
    )


def test_replacement_takes_predecessor_quota():
    """Check that a replacement of a rank with a smaller quota than rank 0 takes that quota."""

    def smaller(r, obj):
        return obj - r

    assert _quota(_rank_0(smaller), 100) == 100
    published = _publish()
    assert _quota(_replacement(published), 110) == 100 - REPLACED
    startup_record.close()
    replacement = _replacement(published)
    with pytest.raises(
        SystemExit, match=f"is {100 - REPLACED - 1}, below its predecessor's {100 - REPLACED}"
    ):
        _quota(replacement, 100 - REPLACED - 1)


class _ReadAheadLoader:
    """A checkpoint loader whose session setup and finalization gather across ranks, as rank-striped read-ahead does."""

    def __init__(self, dist):
        self.dist = dist

    @contextlib.contextmanager
    def open_weight_session(self, checkpoint_dir, **kwargs):
        self.dist.allgather("setup")
        yield {}
        self.dist.allgather("finalization")


def test_checkpoint_session_is_not_recorded():
    rank_0 = _rank_0()
    with model_loader._timed_checkpoint_weight_session(
        _ReadAheadLoader(rank_0), "checkpoint", {}, "preparation", "finalization"
    ):
        _barrier(rank_0)
    replacement = _replacement(_publish())
    _barrier(replacement)
    startup_record.close()


def _kv_load_sample(dist, sample):
    return dist.tp_allgather(sample)


# The allgather of the KV load sample, as PyExecutor._initialize_execution makes it.
_initialize_execution = FunctionType(
    _kv_load_sample.__code__.replace(co_filename="py_executor.py", co_name="_initialize_execution"),
    globals(),
)


def test_replacement_sends_own_kv_load_sample():
    _initialize_execution(_rank_0(), {"timestampUnixNanos": 1})
    replacement = _replacement(_publish())
    assert (
        _initialize_execution(replacement, {"timestampUnixNanos": 2})
        == [{"timestampUnixNanos": 1}] * WORLD
    )
    startup_record.close()


def _autotuner(dist):
    """Return an AutoTuner that tunes across the TP group of ``dist``."""
    with (
        patch.dict(os.environ, TLLM_PROFILING_TIMER="cuda_event"),
        patch("torch.cuda.get_device_name", return_value="GPU"),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        tuner = AutoTuner()
    rank = dist.mapping.rank
    tuner.setup_distributed_state(Mapping(world_size=WORLD, rank=rank, tp_size=WORLD), dist)
    return tuner


@pytest.fixture
def distributed(monkeypatch):
    """Make ``Distributed.get`` return the ``dist`` of the rank the test runs as."""
    ranks = SimpleNamespace(dist=None)
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda mapping: ranks.dist))
    return ranks


def _auto_host_tier_quota(monkeypatch, mem_available):
    """Return main's automatic host tier quota beside an 80 GiB device quota.

    The node has ``mem_available`` bytes free.
    """
    monkeypatch.setattr(MnnvlDist, "local_world_size", WORLD)
    monkeypatch.setattr(
        rank_striped_read_ahead, "effective_available_host_memory", lambda: mem_available
    )
    monkeypatch.setattr(
        resource, "getrlimit", lambda limit: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    mapping = SimpleNamespace(world_size=WORLD)
    limits = get_host_memory_limits(mapping)
    host_quota = _compute_auto_host_tier_quota(
        80 * GIB, limits.local_ranks, limits.mem_available, limits.memlock_limit
    )
    return _sync_host_tier_quota(host_quota, mapping)


def test_replacement_host_tier_needs_room(distributed, monkeypatch):
    """Check the quota when the survivors' pinned host tiers leave the replacement less memory free."""
    distributed.dist = _rank_0()
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB) == 80 * GIB
    published = _publish()
    distributed.dist = _replacement(published)
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB - 3 * 80 * GIB) == 80 * GIB
    startup_record.close()
    distributed.dist = _replacement(published)
    with pytest.raises(
        SystemExit,
        match=f"_sync_host_tier_quota .* is {60 * GIB}, beyond the original ranks' min {80 * GIB}",
    ) as exited:
        _auto_host_tier_quota(monkeypatch, 120 * GIB)
    assert exited.value.code == REPLACEMENT_CANNOT_START


def _gemm(num_tokens):
    """Return the profiling cache key of the GEMM for ``num_tokens`` tokens."""
    return ("gemm", "GemmRunner", "0", ((num_tokens,),))


class _DecoderEngine(SimpleNamespace):
    """Fake a decoder engine of ``dist`` without MXFP8 layers.

    Its forward runs a GEMM whose tactics every rank merges, as ``AutoTuner.choose_one`` does. A token count with no
    tactic yet is profiled at ``measured_ms`` and then merged.
    """

    def __init__(self, dist, measured_ms):
        tuner = _autotuner(dist)
        profiling_cache = tuner.profiling_cache
        profiling_cache.independent_op = {"gemm"}

        def forward(batch, *args, **kwargs):
            if _gemm(batch.num_tokens) not in profiling_cache.cache:
                profiling_cache.cache[_gemm(batch.num_tokens)] = (0, 3, measured_ms)
                tuner._merge_cache_data("gemm")

        super().__init__(
            _config=SimpleNamespace(
                enable_autotuner=True,
                torch_compile_enabled=False,
                max_num_tokens=16,
                max_batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                max_total_draft_tokens=0,
                is_spec_decode=False,
                spec_config=None,
            ),
            model=SimpleNamespace(modules=lambda: []),
            mapping=SimpleNamespace(tp_size=WORLD, has_pp=lambda: False),
            dist=dist,
            guided_decoder=None,
            _graphs=SimpleNamespace(enabled=False, no_cuda_graph=contextlib.nullcontext),
            _dummy_requests=SimpleNamespace(
                create_warmup_request=lambda resources, num_tokens, num_gen_requests: (
                    SimpleNamespace(num_tokens=num_tokens)
                ),
                release_batch_context=lambda batch, resources: contextlib.nullcontext(batch),
            ),
            _agreement=WarmupAgreement(dist, dist.mapping),
            _warmup_timer=_WarmupTimer(rank=dist.mapping.rank),
            _completed_autotuner_warmup_num_tokens=None,
            tactics_at_general_warmup=None,
            _release_megamoe_profiling_scratch=Mock(),
            _forward_warmup=Mock(side_effect=forward),
            profiling_cache=profiling_cache,
        )

    def _warmup_scheduled(self, resource_manager, kv_cache_manager):
        """Main's warmup phases, of which only the autotuner's runs here, after the general warmup's forwards."""
        self.tactics_at_general_warmup = dict(self.profiling_cache.cache)
        DecoderWarmup._run_autotuner_warmup(self, resource_manager)

    def autotuner_warmup(self, num_tokens):
        """Run main's warmup up to its autotuner phase, with KV cache room for ``num_tokens`` tokens."""
        kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: num_tokens)
        resource_manager = SimpleNamespace(get_resource_manager=lambda key: kv_cache_manager)
        with (
            patch.object(
                warmup.AutoTuner, "get", return_value=Mock(profiling_cache=self.profiling_cache)
            ),
            patch.object(warmup, "autotune", lambda **kwargs: contextlib.nullcontext()),
            patch.object(warmup, "warmup_named_allreduce_topologies"),
            patch.object(warmup, "clear_memory_buffers"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.synchronize"),
        ):
            DecoderWarmup.run(self, resource_manager)


def _original_rank_0(estimation_num_tokens, final_num_tokens):
    """Run rank 0's autotuner warmups and return its engine and published startup record.

    The warmup of the KV cache estimation is not recorded. The warmup of the final executor is.
    """
    engine = _DecoderEngine(_rank_0(), measured_ms=0.1)
    startup_record.pause()
    engine.autotuner_warmup(estimation_num_tokens)
    startup_record.resume()
    engine.autotuner_warmup(final_num_tokens)
    return engine, _publish()


def test_replacement_skips_autotuner_warmup_like_originals():
    """Check a final warmup with the estimation's token count, which the original ranks skip.

    The replacement skips it too and takes the tactics that rank 0 tuned in the estimation.
    """
    rank_0, published = _original_rank_0(16, 16)
    # Only the estimation's warmup ran.
    assert rank_0._forward_warmup.call_count == 2
    replacement = _DecoderEngine(_replacement(published), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    replacement._forward_warmup.assert_not_called()
    assert replacement.profiling_cache.cache == rank_0.profiling_cache.cache
    # Its general warmup forwards and CUDA graph captures already run rank 0's tactics.
    assert replacement.tactics_at_general_warmup == rank_0.profiling_cache.cache
    assert replacement.dist.host_exchange.calls == 0
    startup_record.close()


def test_replacement_adopts_independent_op_tactics():
    """Check that a replacement takes rank 0's tactics of an op that every rank tunes on its own."""
    rank_0 = _DecoderEngine(_rank_0(), measured_ms=0.1)
    bmm = ("bmm", "BmmRunner", "0", ((16,),))
    rank_0.profiling_cache.cache[bmm] = (0, 7, 0.2)
    startup_record.pause()
    rank_0.autotuner_warmup(16)
    startup_record.resume()
    rank_0.autotuner_warmup(16)
    replacement = _DecoderEngine(_replacement(_publish()), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    assert replacement.profiling_cache.cache.get(bmm) == (0, 7, 0.2)
    startup_record.close()
