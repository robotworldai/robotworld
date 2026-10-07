# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whack-a-Mole termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """All popup windows have been processed (task_phase == 4)."""
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.task_phase == 4


def is_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check whether the episode had any successful hits (for metrics/logging)."""
    if not hasattr(env, "valid_hits"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.valid_hits > 0


def get_success_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode success rate: valid_hits / total_popups. (N,)"""
    if not hasattr(env, "success_rate"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return env.success_rate