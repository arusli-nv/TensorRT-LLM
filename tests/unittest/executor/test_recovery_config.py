# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideEP fault-tolerance settings: one validated object, read once per process."""

import pytest

from tensorrt_llm.executor.recovery import config as config_module
from tensorrt_llm.executor.recovery.config import CONFIG_ENV, FaultToleranceConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def load(monkeypatch, tmp_path):
    def load(text=None):
        config_module.fault_tolerance_config.cache_clear()
        if text is None:
            monkeypatch.delenv(CONFIG_ENV, raising=False)
        else:
            path = tmp_path / "ft.yaml"
            path.write_text(text)
            monkeypatch.setenv(CONFIG_ENV, str(path))
        return config_module.fault_tolerance_config()

    yield load
    config_module.fault_tolerance_config.cache_clear()


def test_defaults_without_a_file(load):
    assert load() == FaultToleranceConfig()
    assert (
        load().extra_slots == "empty"
        and load().failure_domain == "node"
        and not load().hang_detection
    )


def test_file_fields_override_the_defaults(load):
    config = load("extra_slots: borrowed\nhang_detection: true\nhang_confirm_s: 10\n")
    assert (config.extra_slots, config.hang_detection, config.hang_confirm_s) == (
        "borrowed",
        True,
        10.0,
    )


@pytest.mark.parametrize(
    "text, match",
    [
        ("extra_slots: replicas\n", "extra_slots"),
        ("failure_domain: rack\n", "failure_domain"),
        ("spare_slots: empty\n", "spare_slots"),
        ("recovery_timeout_s: 10\nauthority_timeout_s: 20\n", "authority_timeout_s"),
        ("heartbeat_s: 5\n", "heartbeat_s"),
        # The runtime fence must end before the fence command starts.
        ("heartbeat_s: 3\n", "heartbeat_s"),
        ("hang_confirm_s: -1\n", "hang_confirm_s"),
        ("- empty\n", "mapping"),
    ],
)
def test_invalid_settings_are_refused(load, text, match):
    with pytest.raises(ValueError, match=match):
        load(text)


def test_hang_resolution_bounds_confirm_and_fence():
    config = FaultToleranceConfig(hang_suspect_s=2, hang_confirm_s=10)
    assert config.hang_resolution_s() == 2 + 10 + config_module.COMMAND_FENCE_GRACE_S


def test_every_field_has_a_description():
    assert all(field.description for field in FaultToleranceConfig.model_fields.values())


def test_digest_is_stable_and_changes_with_any_field():
    digest = FaultToleranceConfig().digest()
    assert digest == FaultToleranceConfig().digest() == FaultToleranceConfig(heartbeat_s=1).digest()
    changed = {
        "extra_slots": "borrowed",
        "failure_domain": "rank",
        "recovery_timeout_s": 40.0,
        "authority_timeout_s": 10.0,
        "hang_detection": True,
        "hang_suspect_s": 3.0,
        "hang_confirm_s": 10.0,
        "heartbeat_s": 0.5,
        "heartbeat_timeout_s": 6.0,
    }
    assert changed.keys() == FaultToleranceConfig.model_fields.keys()
    digests = {FaultToleranceConfig(**{name: value}).digest() for name, value in changed.items()}
    assert digest not in digests and len(digests) == len(changed)
