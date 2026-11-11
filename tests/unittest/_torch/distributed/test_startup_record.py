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
"""A replacement rank's startup collectives return what rank 0 received at the same calls (the startup record of
Wide-EP fault tolerance). ``MnnvlDist``'s host collectives run over a fake host exchange: rank 0 records them, then a
replacement of rank 2, whose own exchange reaches only itself, replays them."""

import contextlib
import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch.autotuner import AutoTuner, AutoTunerProfilingCache
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist, TPHostExchange
from tensorrt_llm._torch.distributed.membership import REPLACEMENT_CANNOT_START
from tensorrt_llm._torch.distributed.startup_record import StartupRecord, startup_record
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import _sync_host_tier_quota
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

pytestmark = pytest.mark.cpu_only

WORLD = 4
REPLACED = 2
GIB = 1 << 30


class FakeHostExchange:
    """The ``TPHostExchange`` of ``rank`` with the ranks in ``active_mask``, where a peer ``r`` sends ``peers(r, obj)``
    when this rank sends ``obj``."""

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
    dist = MnnvlDist.__new__(MnnvlDist)
    dist.mapping = SimpleNamespace(
        rank=rank, tp_rank=rank, tp_size=WORLD, cp_size=1, world_size=WORLD, dwdp_enabled=False
    )
    dist._host_exchange = host_exchange
    dist._active_rows_index = (None, None)
    return dist


@pytest.fixture(autouse=True)
def fresh_record(monkeypatch):
    """This process's startup record, empty."""
    for name, value in vars(StartupRecord()).items():
        monkeypatch.setattr(startup_record, name, value)


def _rank_0(peers=lambda r, obj: obj):
    """Rank 0, recording its startup."""
    startup_record.record()
    return _dist(0, FakeHostExchange(0, peers=peers))


def _publish() -> bytes:
    """The end of rank 0's startup: its record, as the rendezvous store carries it."""
    startup_record.close()
    return pickle.dumps(startup_record.published())


def _replacement(published: bytes, host_exchange=False):
    """A replacement of rank ``REPLACED`` that replays ``published``; its host exchange reaches only itself."""
    for name, value in vars(StartupRecord()).items():
        setattr(startup_record, name, value)
    startup_record.replay(pickle.loads(published))
    if host_exchange is False:
        host_exchange = FakeHostExchange(REPLACED, active_mask=1 << REPLACED)
    return _dist(REPLACED, host_exchange)


# The startup's host collectives: each at the same line on every rank.
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
    """Rank r's peers have 100 * r tokens fewer than it."""
    return obj - 100 * r if isinstance(obj, int) and obj > 1 else obj


def test_a_replacement_replays_what_rank_0_received():
    expected = ([1] * WORLD, {"gemm": 3}, 4096 - 300, [[8, 1]] * WORLD)
    assert _startup(_rank_0(_fewer_tokens), 4096) == expected
    replacement = _replacement(_publish())
    # Its own capacity is its predecessor's (4096 - 200): the minimum bounds it.
    assert _startup(replacement, 4096 - 200) == expected
    assert replacement.host_exchange.calls == 0
    startup_record.close()


def test_a_minimum_that_does_not_bound_the_replacement_s_own_value_stops_it():
    _startup(_rank_0(_fewer_tokens), 4096)
    published = _publish()
    # A larger capacity than its predecessor's (4096 - 200) takes the minimum.
    assert _startup(_replacement(published), 8192)[2] == 4096 - 300
    replacement = _replacement(published)
    with pytest.raises(SystemExit, match="allreduce in _startup .* differs") as exited:
        _startup(replacement, 4096 - 400)
    assert exited.value.code == REPLACEMENT_CANNOT_START


def test_paused_collectives_are_neither_recorded_nor_replayed():
    rank_0 = _rank_0()
    startup_record.pause()
    assert rank_0.allgather("estimation") == ["estimation"] * WORLD
    startup_record.resume()
    _barrier(rank_0)
    replacement = _replacement(_publish())
    _barrier(replacement)  # rank 0's barrier, not its paused gather
    assert replacement.host_exchange.calls == 0
    startup_record.pause()
    assert replacement.allgather("alone") == [None, None, "alone", None]  # its own host exchange
    startup_record.resume()
    assert replacement.host_exchange.calls == 1
    startup_record.close()


def test_a_replacement_that_does_not_start_as_the_original_ranks_cannot_start():
    rank_0 = _rank_0()
    _gather(rank_0, 1)
    _barrier(rank_0)
    replacement = _replacement(_publish())
    with pytest.raises(
        SystemExit, match="is barrier in _barrier .*, the original ranks' allgather in _gather"
    ):
        _barrier(replacement)  # another collective
    with pytest.raises(
        SystemExit, match="allgather in _gather .* differs from the original rank's"
    ) as exited:
        _gather(replacement, 0)  # another payload
    assert exited.value.code == REPLACEMENT_CANNOT_START
    assert _gather(replacement, 1) == [1] * WORLD
    with pytest.raises(
        SystemExit, match="makes 1 host collectives, the original ranks 2: the next one is barrier"
    ):
        startup_record.close()  # one collective fewer
    _barrier(replacement)
    with pytest.raises(SystemExit, match="one more than the original ranks made"):
        _barrier(replacement)
    startup_record.close()


def _tuner(dist, cache):
    """An autotuner of ``dist`` that tuned ``cache``; returns it and what it merges into its cache."""
    tuner = AutoTuner.__new__(AutoTuner)
    tuner._dist = dist
    tuner._is_distributed = lambda: True
    merged = {}
    tuner.profiling_cache = SimpleNamespace(cache=cache, merge_cache_data=merged.update)
    return tuner, merged


def test_a_replacement_merges_the_original_ranks_tactics_not_its_own():
    """Each rank's tuned tactics are its own measurements: the replacement's differ from its predecessor's, and it
    merges the original ranks' (RANK_VARYING_CALLERS)."""

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
    """``Distributed.get`` returns the ``dist`` of the rank the test runs as."""
    ranks = SimpleNamespace(dist=None)
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda mapping: ranks.dist))
    return ranks


def _resolve(monkeypatch, target):
    """Main's modeling_v2 resolution on all ranks, with this rank resolving ``target``."""
    monkeypatch.setattr(_router_index, "modeling_v2_resolve", lambda config, llm_args: target)
    config = SimpleNamespace(mapping=SimpleNamespace(rank=0, world_size=WORLD))
    return _router_index.modeling_v2_resolve_on_all_ranks(
        config, SimpleNamespace(modeling_v2="auto")
    )


def _main_startup(monkeypatch, host_quota, target=None):
    mapping = SimpleNamespace(world_size=WORLD)
    return _sync_host_tier_quota(host_quota, mapping), _resolve(monkeypatch, target)


def test_main_s_startup_agreements_take_the_original_ranks_results_on_a_replacement(
    distributed, monkeypatch
):
    """The KV cache manager V2's host quota (the fleet minimum) and modeling_v2's resolution on all ranks, as main
    runs them."""

    def less_host_memory(r, obj):
        return obj - 8 * GIB * r if isinstance(obj, int) else obj

    distributed.dist = _rank_0(less_host_memory)
    assert _main_startup(monkeypatch, 64 * GIB) == (40 * GIB, None)
    published = _publish()
    distributed.dist = _replacement(published)
    assert _main_startup(monkeypatch, 48 * GIB) == (40 * GIB, None)
    startup_record.close()
    distributed.dist = _replacement(published)
    with pytest.raises(SystemExit, match="allreduce in _sync_host_tier_quota"):
        _main_startup(monkeypatch, 32 * GIB)  # less than the minimum
    distributed.dist = _replacement(published)
    with pytest.raises(
        SystemExit, match="allgather in modeling_v2_resolve_on_all_ranks .* differs"
    ):
        _main_startup(monkeypatch, 48 * GIB, target=AutoTuner)  # builds another implementation


def test_main_s_warmup_agreement_on_a_replacement_decides_as_the_original_ranks():
    """Main's WarmupAgreement: rank 1 opts out of a phase and proposes fewer warmup shapes."""

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


def test_votes_on_a_replacement_count_every_original_rank_during_the_replay():
    _rows(_rank_0())
    replacement = _replacement(_publish())
    rows = _rows(replacement)
    assert replacement.tp_active_rows(rows).tolist() == [[8, 1]] * WORLD
    startup_record.close()
    # Past its startup, only its own row is active until it is admitted.
    assert replacement.tp_active_rows(rows).tolist() == [[8, 1]]


def test_a_replacement_creates_its_host_exchange_at_its_first_replayed_collective(monkeypatch):
    """At the same call as the original ranks: its creation (a collective of its isolated group) maps memory."""
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
    """The profiling cache key of the GEMM for ``num_tokens`` tokens."""
    return ("gemm", "GemmRunner", "0", ((num_tokens,),))


class _DecoderEngine(SimpleNamespace):
    """A decoder engine of ``dist`` (no MXFP8 layer) whose forward runs a GEMM tuned by merging every rank's
    tactics, as main's ``AutoTuner.choose_one`` does: it profiles a token count it has no tactic for, measuring
    ``measured_ms``, then merges."""

    def __init__(self, dist, measured_ms):
        profiling_cache = AutoTunerProfilingCache.__new__(AutoTunerProfilingCache)
        profiling_cache.cache, profiling_cache.excluded_op = {}, set()
        profiling_cache.independent_op = {"gemm"}  # the ops whose tactics every rank shares
        tuner = AutoTuner.__new__(AutoTuner)
        tuner._dist, tuner.profiling_cache = dist, profiling_cache
        tuner._is_distributed = lambda: True

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
            _release_megamoe_profiling_scratch=Mock(),
            _forward_warmup=Mock(side_effect=forward),
            profiling_cache=profiling_cache,
        )

    def autotuner_warmup(self, num_tokens):
        """Main's autotuner warmup with KV cache room for ``num_tokens`` tokens."""
        kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: num_tokens)
        resource_manager = SimpleNamespace(get_resource_manager=lambda key: kv_cache_manager)
        with (
            patch.object(
                warmup.AutoTuner, "get", return_value=Mock(profiling_cache=self.profiling_cache)
            ),
            patch.object(warmup, "autotune", lambda **kwargs: contextlib.nullcontext()),
            patch.object(warmup, "clear_memory_buffers"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.synchronize"),
        ):
            DecoderWarmup._run_autotuner_warmup(self, resource_manager)


def _original_rank_0(estimation_num_tokens, final_num_tokens):
    """Rank 0's autotuner warmups, of the KV cache estimation (not recorded) and of the final executor; returns its
    engine and the startup record it publishes."""
    engine = _DecoderEngine(_rank_0(), measured_ms=0.1)
    startup_record.pause()
    engine.autotuner_warmup(estimation_num_tokens)
    startup_record.resume()
    engine.autotuner_warmup(final_num_tokens)
    return engine, _publish()


def test_a_replacement_skips_the_autotuner_warmup_the_original_ranks_skip_with_rank_0_s_tactics():
    """The final warmup has the estimation's token count: the original ranks skip it, and so does a replacement,
    which skipped the estimation, with the tactics rank 0 tuned in it."""
    rank_0, published = _original_rank_0(16, 16)
    assert rank_0._forward_warmup.call_count == 2  # the estimation's warmup only
    replacement = _DecoderEngine(_replacement(published), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    replacement._forward_warmup.assert_not_called()
    assert replacement.profiling_cache.cache == rank_0.profiling_cache.cache
    assert replacement.dist.host_exchange.calls == 0
    startup_record.close()


def test_a_replacement_runs_the_autotuner_warmup_the_original_ranks_run_and_tunes_what_they_tune():
    """The final warmup has more tokens than the estimation's: the original ranks tune the GEMM for them, and so does
    a replacement, starting from the tactics rank 0 holds there: it makes their collectives, and its merge takes
    their tactics, not its own measurement."""
    rank_0, published = _original_rank_0(8, 16)
    assert rank_0._forward_warmup.call_count == 4
    replacement = _DecoderEngine(_replacement(published), measured_ms=0.05)
    replacement.autotuner_warmup(16)
    assert replacement._forward_warmup.call_count == 2
    assert replacement.profiling_cache.cache == rank_0.profiling_cache.cache
    assert replacement.profiling_cache.cache[_gemm(16)] == (0, 3, 0.1)
    assert replacement.dist.host_exchange.calls == 0
    startup_record.close()  # it made every collective of the original ranks
