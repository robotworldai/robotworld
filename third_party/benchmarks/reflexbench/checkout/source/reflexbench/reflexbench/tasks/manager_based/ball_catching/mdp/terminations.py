# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_in_catch_zone(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    catch_radius: float = 0.065,
    catch_depth: float = 0.13,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ball_pos_w = ball.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    xy_distance = torch.norm(ball_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)
    within_radius = xy_distance <= catch_radius
    within_depth = (ball_pos_w[:, 2] <= ee_pos_w[:, 2]) & (ball_pos_w[:, 2] >= ee_pos_w[:, 2] - catch_depth)
    return within_radius & within_depth


def ball_on_ground(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_height: float = 0.03,
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    launched = getattr(env, "launch_detected", None)
    if launched is None:
        launched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return launched & (ball.data.root_pos_w[:, 2] < min_height)


def task_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.task_phase == 4
