# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for pick-place from conveyor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_x_range: tuple = (-0.2, 0.2),
    box_y_range: tuple = (-0.6, -0.3),
    box_z_range: tuple = (0.0, 0.08),
) -> torch.Tensor:
    """True when object is inside the box region (env-local coords).

    Uses the same coordinate system and thresholds as events.py and rewards.py
    to ensure consistent success detection across the entire codebase.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    in_x = (pos_local[:, 0] >= box_x_range[0]) & (pos_local[:, 0] <= box_x_range[1])
    in_y = (pos_local[:, 1] >= box_y_range[0]) & (pos_local[:, 1] <= box_y_range[1])
    in_z = (pos_local[:, 2] >= box_z_range[0]) & (pos_local[:, 2] <= box_z_range[1])
    return in_x & in_y & in_z


def task_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success: terminate when task phase reaches 4 (object in box)."""
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.task_phase == 4
