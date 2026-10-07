# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the toss-into-box task.

The default ``RewardsCfg`` in the env config is intentionally empty (task
success is checked via terminations).  This helper is provided in case a
future RL setup wants a simple phase-based shaping signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.task_phase.float() / 2.0
