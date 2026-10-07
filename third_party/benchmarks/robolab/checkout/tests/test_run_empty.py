# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a minimal empty episode end-to-end.

Defaults to BananaInBowlTask + the droid registration. Override
via `pytest --task <TaskName>` or `pytest --env-name <FullEnvName>` to
verify a different task or robot setup.
"""

import torch

from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs


def _resolve_env_name(task_arg, env_name_arg):
    if env_name_arg:
        return env_name_arg
    auto_register_droid_envs()
    envs = get_envs(task=task_arg)
    assert envs, f"No registered env found for task '{task_arg}'"
    return envs[0]


def test_run_empty_episode(task_arg, env_name_arg):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA device required for run_empty episode")

    env_name = _resolve_env_name(task_arg, env_name_arg)

    env, env_cfg = create_env(env_name, num_envs=1, use_fabric=True)
    try:
        obs, _ = env.reset()
        assert obs is not None

        action_dim = env.action_manager.total_action_dim
        actions = torch.zeros((1, action_dim), device=env.device)
        for _ in range(5):
            obs, _, term, trunc, info = env.step(actions)
            assert obs is not None
            assert term is not None
            assert trunc is not None

        end_episode(env)
    finally:
        env.close()
