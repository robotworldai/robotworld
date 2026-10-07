# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whack-a-Mole observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .events import BOARD_SURFACE_Z, MOLE_DOWN_Z, MOLE_UP_Z, MOLE_XY_POSITIONS, NUM_MOLES, NUM_WINDOWS


# -- Helpers ------------------------------------------------------------------

def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q (w, x, y, z)."""
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = torch.cross(q_vec, v, dim=1)
    b = torch.cross(q_vec, a, dim=1)
    return v + 2.0 * (-q_w * a + b)


# -- Standard proprioception (same API as ball_catching) ----------------------

def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in robot root frame. (N, 3)."""
    robot = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w,
    )
    return ee_pos_b


def gripper_width_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalised gripper finger width ∈ [0, 1]. (N, 1)."""
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    normalised = torch.clamp(finger_width / 0.08, 0.0, 1.0)
    return normalised.unsqueeze(1)


def task_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalised task phase ∈ [0, 1]. (N, 1)."""
    if not hasattr(env, "task_phase"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.task_phase.unsqueeze(1).float() / 4.0


# -- Mole-specific observations -----------------------------------------------

def mole_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """All 5 mole positions in robot root frame. (N, 15)."""
    if not hasattr(env, "mole_positions_local"):
        return torch.zeros((env.num_envs, NUM_MOLES * 3), device=env.device)

    robot = env.scene[robot_cfg.name]

    # Build world-frame positions from fixed XY + current mole_heights
    mole_pos_w = env.mole_positions_local.clone()  # (N, 5, 3)
    mole_pos_w[:, :, :2] += env.scene.env_origins[:, :2].unsqueeze(1)
    mole_pos_w[:, :, 2] = env.scene.env_origins[:, 2:3] + env.mole_heights

    # Transform each mole position into robot root frame
    all_pos_b = torch.zeros_like(mole_pos_w)
    for i in range(NUM_MOLES):
        pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, mole_pos_w[:, i, :],
        )
        all_pos_b[:, i, :] = pos_b

    return all_pos_b.reshape(env.num_envs, -1)  # (N, 15)


def mole_heights_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalised mole heights (0 = down, 1 = up). (N, 5)."""
    if not hasattr(env, "mole_heights"):
        return torch.zeros((env.num_envs, NUM_MOLES), device=env.device)
    span = max(MOLE_UP_Z - MOLE_DOWN_Z, 1e-6)
    normalised = (env.mole_heights - MOLE_DOWN_Z) / span
    return torch.clamp(normalised, 0.0, 1.0)


def active_mole_index_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Active mole index normalised ∈ [0, 1]. Inactive -> 0. (N, 1)."""
    if not hasattr(env, "active_mole_id"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    idx = env.active_mole_id.float()
    # Map -1 -> 0, else (idx+1)/NUM_MOLES so that mole 0 -> 0.2, mole 4 -> 1.0
    idx = torch.where(idx < 0, torch.zeros_like(idx), (idx + 1.0) / NUM_MOLES)
    return idx.unsqueeze(1)


def active_mole_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the currently-active mole in robot root frame. (N, 3). Zeros if none."""
    if not hasattr(env, "active_mole_id") or not hasattr(env, "mole_positions_local"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    robot = env.scene[robot_cfg.name]
    active_mid = env.active_mole_id.long()  # (N,)
    has_active = active_mid >= 0

    result = torch.zeros((env.num_envs, 3), device=env.device)
    if not has_active.any():
        return result

    active_ids = torch.arange(env.num_envs, device=env.device)[has_active]
    mid = active_mid[active_ids]

    # Build world position for the active mole of each env
    batch_idx = torch.arange(len(active_ids), device=env.device)
    local_xy = env.mole_positions_local[active_ids][batch_idx, mid, :2]
    local_z = env.mole_heights[active_ids][batch_idx, mid]

    mole_pos_w = torch.zeros(len(active_ids), 3, device=env.device)
    mole_pos_w[:, :2] = local_xy + env.scene.env_origins[active_ids, :2]
    mole_pos_w[:, 2] = local_z + env.scene.env_origins[active_ids, 2]

    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[active_ids],
        robot.data.root_quat_w[active_ids],
        mole_pos_w,
    )
    result[active_ids] = pos_b
    return result


def window_remaining_time_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining time (seconds) in the current active window. (N, 1)."""
    if not hasattr(env, "window_active"):
        return torch.zeros((env.num_envs, 1), device=env.device)

    result = torch.zeros((env.num_envs, 1), device=env.device)
    active = env.window_active
    if not active.any():
        return result

    active_ids = torch.arange(env.num_envs, device=env.device)[active]
    win_idx = env.current_window_idx[active_ids].long()
    start_t = env.popup_start_times[active_ids].gather(1, win_idx.unsqueeze(1)).squeeze(1)
    dur = env.popup_durations[active_ids].gather(1, win_idx.unsqueeze(1)).squeeze(1)
    remaining = (start_t + dur) - env.episode_timer[active_ids]
    result[active_ids, 0] = torch.clamp(remaining, min=0.0)
    return result


def valid_hits_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalised valid-hit count ∈ [0, 1]. (N, 1)."""
    if not hasattr(env, "valid_hits"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return (env.valid_hits.float() / NUM_WINDOWS).unsqueeze(1)


def current_window_index_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalised current window index ∈ [0, 1]. (N, 1)."""
    if not hasattr(env, "current_window_idx"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return (env.current_window_idx.float() / NUM_WINDOWS).unsqueeze(1)