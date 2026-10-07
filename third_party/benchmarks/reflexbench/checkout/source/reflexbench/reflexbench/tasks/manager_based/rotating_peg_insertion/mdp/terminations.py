# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_PEG_Z_OFFSET = -0.045
_PEG_LENGTH = 0.10
_HOLE_RADIUS = 0.022
_PEG_RADIUS = 0.005
_HOLE_CLEAR_RADIUS = _HOLE_RADIUS
_DISC_HEIGHT = 0.01
_INSERTION_DEPTH_THRESHOLD = 0.02


def peg_in_hole(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hole_radius: float = _HOLE_CLEAR_RADIUS,
    insertion_depth: float = _INSERTION_DEPTH_THRESHOLD,
) -> torch.Tensor:
    if not hasattr(env, "current_hole_world_pos") or not hasattr(env, "disc_center_local"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    peg = env.scene["peg"]
    peg_pos_w = peg.data.root_pos_w[:, :3]
    peg_pos_local = peg_pos_w - env.scene.env_origins
    peg_tip = peg_pos_local.clone()
    peg_tip[:, 2] = peg_pos_local[:, 2] - _PEG_LENGTH * 0.5

    hole_world = env.current_hole_world_pos
    xy_dist = torch.norm(peg_tip[:, :2] - hole_world[:, :2], dim=1)
    in_radius = xy_dist <= hole_radius

    disc_top_z = env.disc_center_local[:, 2] + _DISC_HEIGHT * 0.5
    peg_depth_below_disc = disc_top_z - peg_tip[:, 2]
    deep_enough = peg_depth_below_disc >= insertion_depth

    return in_radius & deep_enough


def task_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.task_phase == 4


def peg_dropped(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_peg_deviation: float = 0.15,
    min_peg_height: float = 0.10,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_local = ee_pos_w - env.scene.env_origins

    too_low = ee_pos_local[:, 2] < min_peg_height
    return too_low
