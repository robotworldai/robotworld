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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = torch.cross(q_vec, v, dim=1)
    b = torch.cross(q_vec, a, dim=1)
    return v + 2 * (-q_w * a + b)


def ball_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_pos_w[:, :3]
    ball_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ball_pos_w)
    return ball_pos_b


def ball_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel_w = ball.data.root_lin_vel_w[:, :3]
    return _quat_rotate_inverse(robot.data.root_quat_w, ball_vel_w)


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


def launch_detected_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "launch_detected"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.launch_detected.unsqueeze(1).float()


def predicted_intercept_position_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "predicted_intercept_pos"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    return env.predicted_intercept_pos


def predicted_intercept_time_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "predicted_intercept_time"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.predicted_intercept_time.unsqueeze(1)
