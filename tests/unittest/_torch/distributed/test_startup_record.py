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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist, TPHostExchange
from tensorrt_llm._torch.distributed.membership import REPLACEMENT_CANNOT_START
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_auto_host_tier_quota,
    _sync_host_tier_quota,
    get_host_memory_limits,
)
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
REPLACED = 2
GIB = 1 << 30


def _in_mask(mask, rank):
    """Return whether the bit of ``rank`` is set in ``mask``."""
    return bool(mask & (1 << rank))


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
            pickle.dumps(self._sent(r, obj))
            if r in read and _in_mask(self.active_mask, r)
            else None
            for r in range(WORLD)
        ]

    def allgather_array(self, row):
        self.calls += 1
        return np.stack(
            [
                np.asarray(self._sent(r, row), dtype=np.int64)
                if _in_mask(self.active_mask, r)
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


def _replacement(published: bytes, host_exchange=False):
    """Return a replacement of rank ``REPLACED`` that replays ``published``.

    By default its host exchange reaches only itself.
    """
    for name, value in vars(StartupRecord()).items():
        setattr(startup_record, name, value)
    startup_record.replay(pickle.loads(published))
    if host_exchange is False:
        host_exchange = FakeHostExchange(REPLACED, active_mask=1 << REPLACED)
    return _dist(REPLACED, host_exchange)


# The startup's host collectives. Each one runs at the same line on every rank.
def _gather(dist, obj):
    return dist.allgather(obj)


def _barrier(dist):
    dist.barrier()


def _rows(dist):
    return dist.tp_allgather_int64([8, 1])


def _startup(dist, max_tokens):
    flags = dist.tp_allgather(1)
    tactics = dist.tp_broadcast({"gemm": 3} if dist.mapping.tp_rank == 0 else None, root=0)
    max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
    rows = _rows(dist)
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


def test_paused_collectives_are_not_recorded():
    rank_0 = _rank_0()
    startup_record.pause()
    assert rank_0.allgather("estimation") == ["estimation"] * WORLD
    startup_record.resume()
    _barrier(rank_0)
    replacement = _replacement(_publish())
    # This replays rank 0's barrier, not its paused gather.
    _barrier(replacement)
    assert replacement.host_exchange.calls == 0
    startup_record.pause()
    # A paused collective runs over the replacement's own host exchange.
    assert replacement.allgather("alone") == [None, None, "alone", None]
    startup_record.resume()
    assert replacement.host_exchange.calls == 1
    startup_record.close()


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


def _tuner(dist, cache):
    """Return an autotuner of ``dist`` that tuned ``cache``, and the dict it merges into its cache."""
    tuner = _autotuner(dist)
    merged = {}
    tuner.profiling_cache = SimpleNamespace(cache=cache, merge_cache_data=merged.update)
    return tuner, merged


def test_replacement_merges_original_ranks_tactics():
    """Check that a replacement merges the original ranks' tactics, not its own measurements."""

    def slower(r, obj):
        return [
            {key: (tactic, r, time + r) for key, (tactic, _, time) in cache.items()}
            for cache in obj
        ]

    tuner, merged = _tuner(_rank_0(slower), {("gemm", 8): (3, 0, 0.1)})
    tuner.post_tune_merge_tactics()
    assert merged == {("gemm", 8): (3, 0, 0.1)}
    tuner, merged = _tuner(_replacement(_publish()), {("gemm", 8): (5, 2, 0.05)})
    tuner.post_tune_merge_tactics()
    assert merged == {("gemm", 8): (3, 0, 0.1)}
    startup_record.close()


@pytest.fixture
def distributed(monkeypatch):
    """Make ``Distributed.get`` return the ``dist`` of the rank the test runs as."""
    ranks = SimpleNamespace(dist=None)
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda mapping: ranks.dist))
    return ranks


def _resolve(monkeypatch, target):
    """Run modeling_v2's resolution on all ranks, with this rank resolving ``target``."""
    monkeypatch.setattr(_router_index, "modeling_v2_resolve", lambda config, llm_args: target)
    config = SimpleNamespace(mapping=SimpleNamespace(rank=0, world_size=WORLD))
    return _router_index.modeling_v2_resolve_on_all_ranks(
        config, SimpleNamespace(modeling_v2="auto")
    )


def _main_startup(monkeypatch, host_quota, target=None):
    mapping = SimpleNamespace(world_size=WORLD)
    return _sync_host_tier_quota(host_quota, mapping), _resolve(monkeypatch, target)


def test_replacement_takes_main_startup_agreements(distributed, monkeypatch):
    """Check KV cache manager V2's host quota minimum and modeling_v2's resolution on a replacement."""

    def less_host_memory(r, obj):
        return obj - 8 * GIB * r if isinstance(obj, int) else obj

    distributed.dist = _rank_0(less_host_memory)
    assert _main_startup(monkeypatch, 64 * GIB) == (40 * GIB, None)
    published = _publish()
    distributed.dist = _replacement(published)
    assert _main_startup(monkeypatch, 48 * GIB) == (40 * GIB, None)
    startup_record.close()
    distributed.dist = _replacement(published)
    # A host quota below the minimum stops the replacement.
    with pytest.raises(SystemExit, match="allreduce in _sync_host_tier_quota"):
        _main_startup(monkeypatch, 32 * GIB)
    distributed.dist = _replacement(published)
    # So does resolving another implementation.
    with pytest.raises(
        SystemExit, match="allgather in modeling_v2_resolve_on_all_ranks .* differs"
    ):
        _main_startup(monkeypatch, 48 * GIB, target=AutoTuner)


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


def test_replacement_takes_the_recorded_host_tier(distributed, monkeypatch):
    """Check the quota when the survivors' pinned host tiers leave the replacement less memory free."""
    distributed.dist = _rank_0()
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB) == 80 * GIB
    distributed.dist = _replacement(_publish())
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB - 3 * 80 * GIB) == 80 * GIB
    startup_record.close()


def test_replacement_lacks_host_tier_room(distributed, monkeypatch):
    distributed.dist = _rank_0()
    _auto_host_tier_quota(monkeypatch, 720 * GIB)
    distributed.dist = _replacement(_publish())
    with pytest.raises(
        SystemExit,
        match=f"_sync_host_tier_quota .* is {60 * GIB}, beyond the original ranks' min {80 * GIB}",
    ) as exited:
        _auto_host_tier_quota(monkeypatch, 120 * GIB)
    assert exited.value.code == REPLACEMENT_CANNOT_START


def test_replacement_takes_main_warmup_agreement():
    """Check WarmupAgreement when rank 1 opts out of a phase and proposes fewer warmup shapes."""

    def rank_1_opts_out(r, obj):
        if r != 1:
            return obj
        return 0 if isinstance(obj, int) else obj[:1]

    def warm_up(dist):
        agreement = WarmupAgreement(dist, dist.mapping)
        return agreement.agree_flag(True), agreement.agree_shapes([(1, 0), (2, 0)])

    assert warm_up(_rank_0(rank_1_opts_out)) == (False, [(1, 0)])
    replacement = _replacement(_publish())
    assert warm_up(replacement) == (False, [(1, 0)])
    assert replacement.host_exchange.calls == 0
    startup_record.close()


def _vote(dist, flag):
    return WarmupAgreement(dist, dist.mapping).agree_flag(flag)


def test_other_warmup_votes_must_match():
    """Check that only the autotuner warmup's skip vote may differ from the predecessor's."""
    _vote(_rank_0(), True)
    with pytest.raises(SystemExit, match="agree_flag in _vote .* differs"):
        _vote(_replacement(_publish()), False)


def test_replayed_votes_count_every_original_rank():
    _rows(_rank_0())
    replacement = _replacement(_publish())
    rows = _rows(replacement)
    assert replacement.tp_active_rows(rows).tolist() == [[8, 1]] * WORLD
    startup_record.close()
    # After its startup, only its own row is active until it is admitted.
    assert replacement.tp_active_rows(rows).tolist() == [[8, 1]]


def test_host_exchange_created_at_first_collective(monkeypatch):
    """Check that a replacement creates its host exchange at the same call as the original ranks.

    The creation maps memory in a collective of the replacement's own group.
    """
    _barrier(_rank_0())
    published = _publish()
    host_exchange = FakeHostExchange(REPLACED, active_mask=1 << REPLACED)
    created = []
    monkeypatch.setattr(
        TPHostExchange,
        "create",
        staticmethod(lambda mapping: (created.append(mapping), host_exchange)[1]),
    )
    replacement = _replacement(published, host_exchange=None)
    assert created == []
    _barrier(replacement)
    assert created == [replacement.mapping]
    assert host_exchange.calls == 0
    startup_record.close()


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
    assert replacement.dist.host_exchange.calls == 0
    startup_record.close()


def test_replacement_runs_autotuner_warmup_like_originals():
    """Check a final warmup with more tokens than the estimation, which the original ranks tune.

    The replacement makes the same collectives, and its merge takes their tactics, not its own measurement.
    """
    rank_0, published = _original_rank_0(8, 16)
    assert rank_0._forward_warmup.call_count == 4
    replacement = _DecoderEngine(_replacement(published), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    assert replacement._forward_warmup.call_count == 2
    assert replacement.profiling_cache.cache == rank_0.profiling_cache.cache
    assert replacement.profiling_cache.cache[_gemm(16)] == (0, 3, 0.1)
    assert replacement.dist.host_exchange.calls == 0
    # The replacement made every collective of the original ranks.
    startup_record.close()


def test_replacement_warmup_uses_rank_0_tactics():
    """Check that a replacement's warmup forwards and CUDA graph captures run rank 0's tactics."""
    rank_0, published = _original_rank_0(16, 16)
    replacement = _DecoderEngine(_replacement(published), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    assert replacement.tactics_at_general_warmup == rank_0.profiling_cache.cache
    startup_record.close()
