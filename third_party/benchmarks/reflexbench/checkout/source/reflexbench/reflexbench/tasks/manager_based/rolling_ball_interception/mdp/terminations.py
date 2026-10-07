# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternion (w, x, y, z)."""
    quat_xyz = quat[..., 1:]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(quat_xyz, t, dim=-1)


def _catcher_mouth_center_w(
    catcher_pos_w: torch.Tensor,
    catcher_quat_w: torch.Tensor,
    mouth_center_rel_pos_to_catcher: tuple[float, float, float],
) -> torch.Tensor:
    local_offset = torch.tensor(
        mouth_center_rel_pos_to_catcher,
        dtype=torch.float32,
        device=catcher_pos_w.device,
    ).expand_as(catcher_pos_w)
    return catcher_pos_w + _quat_apply(catcher_quat_w, local_offset)


def ball_in_catch_zone(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    catcher_cfg: SceneEntityCfg = SceneEntityCfg("catcher"),
    catch_radius: float = 0.08,
    catch_depth: float = 0.12,
    mouth_center_rel_pos_to_catcher: tuple[float, float, float] = (0.0, 0.0, 0.05),
) -> torch.Tensor:
    """Check whether the ball is inside the virtual catch zone below the cup mouth."""
    ball: RigidObject = env.scene[ball_cfg.name]
    catcher: RigidObject = env.scene[catcher_cfg.name]

    ball_pos_w = ball.data.root_pos_w[:, :3]
    catcher_pos_w = catcher.data.root_pos_w[:, :3]
    catcher_quat_w = catcher.data.root_quat_w[:, :]
    mouth_pos_w = _catcher_mouth_center_w(
        catcher_pos_w,
        catcher_quat_w,
        mouth_center_rel_pos_to_catcher,
    )

    xy_distance = torch.norm(ball_pos_w[:, :2] - mouth_pos_w[:, :2], dim=1)
    within_radius = xy_distance <= catch_radius
    within_depth = (ball_pos_w[:, 2] <= mouth_pos_w[:, 2]) & (
        ball_pos_w[:, 2] >= mouth_pos_w[:, 2] - catch_depth
    )
    return within_radius & within_depth


def ball_on_ground(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_height: float = 0.04,
) -> torch.Tensor:
    """Terminate when the ball has been released and falls below *min_height*."""
    ball: RigidObject = env.scene[ball_cfg.name]
    released = getattr(env, "rolling_detected", None)
    if released is None:
        released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return released & (ball.data.root_pos_w[:, 2] < min_height)


def task_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.task_phase == 4
