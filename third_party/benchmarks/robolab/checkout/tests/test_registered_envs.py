# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify auto_register_droid_envs() populates the env registry.

Replaces the legacy scripts/check_registered_envs.py.
"""

import gymnasium as gym

from robolab.core.environments.factory import get_envs
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs


def test_auto_register_runs():
    auto_register_droid_envs()


def test_envs_registered():
    auto_register_droid_envs()
    envs = get_envs()
    assert len(envs) >= 100, f"Expected ≥100 registered envs, got {len(envs)}"


def test_canonical_env_in_gym_registry():
    auto_register_droid_envs()
    envs = get_envs(task="BananaInBowlTask")
    assert envs, "No registered env found for BananaInBowlTask"
    assert envs[0] in gym.envs.registry, (
        f"{envs[0]} reported by factory but missing from gym.envs.registry"
    )


def test_no_duplicate_registrations():
    auto_register_droid_envs()
    envs = get_envs()
    assert len(envs) == len(set(envs)), "Duplicate env names found in registration"
