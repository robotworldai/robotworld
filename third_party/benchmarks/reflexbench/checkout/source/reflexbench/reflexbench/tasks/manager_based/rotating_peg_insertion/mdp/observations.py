# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = torch.cross(q_vec, v, dim=1)
    b = torch.cross(q_vec, a, dim=1)
    return v + 2 * (-q_w * a + b)


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def peg_tip_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    peg_z_offset: float = -0.045,
    peg_length: float = 0.10,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    peg_tip_w = ee_pos_w.clone()
    peg_tip_w[:, 2] = ee_pos_w[:, 2] + peg_z_offset - peg_length * 0.5

    peg_tip_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, peg_tip_w
    )
    return peg_tip_b


def disc_angle_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "disc_angle"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    angle = env.disc_angle
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def disc_angular_velocity_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "disc_angular_velocity"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    max_omega = math.radians(60.0)
    return (env.disc_angular_velocity / max_omega).unsqueeze(1)


def hole_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not hasattr(env, "current_hole_world_pos"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    robot: RigidObject = env.scene[robot_cfg.name]
    hole_w = env.current_hole_world_pos + env.scene.env_origins
    hole_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, hole_w)
    return hole_b


def predicted_next_alignment_time_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "predicted_next_alignment_time"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return torch.clamp(env.predicted_next_alignment_time / 5.0, 0.0, 1.0).unsqueeze(1)


def gripper_width_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    normalized = torch.clamp(finger_width / 0.08, 0.0, 1.0)
    return normalized.unsqueeze(1)


def task_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "task_phase"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.task_phase.unsqueeze(1).float() / 4.0
