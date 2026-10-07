# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the toss-into-box task (ball pre-grasped in gripper)."""

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


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_vel_w = obj.data.root_lin_vel_w[:, :3]
    object_vel_b = _quat_rotate_inverse(robot.data.root_quat_w, object_vel_w)
    return object_vel_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )
    return object_quat_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    return ee_pos_b


def gripper_width_obs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalised finger width in [0, 1] (1.0 == fully open at 0.04 m each)."""
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    normalized = torch.clamp(finger_width / 0.08, 0.0, 1.0)
    return normalized.unsqueeze(1)


def task_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Task phase normalised to [0, 1].

    Phase 0: ball held / pre-release
    Phase 1: ball released (in flight)
    Phase 2: ball in box (success)
    """
    if not hasattr(env, "task_phase"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.task_phase.unsqueeze(1).float() / 2.0


def box_target_position_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "box_target_position"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    return env.box_target_position
