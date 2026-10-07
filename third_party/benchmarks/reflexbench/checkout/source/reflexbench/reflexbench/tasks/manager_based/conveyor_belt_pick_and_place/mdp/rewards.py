# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for pick-place from conveyor.

Phase 0: reaching_object (tanh dense)
Phase 1: stable_grasp + lifting_object (linear) + lift_milestone (one-shot)
Phase 2: object_goal_tracking (linear) + object_goal_tracking_fine (tanh) + arrive_box_vertical (one-shot)
Phase 3: gripper_open_release + object_near_target_phase3 + object_in_box (one-shot)
Global: orientation_alignment, action_rate_l2, joint_vel_l2, joint_limit, joint_posture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .terminations import object_in_box

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _is_grasping(
    env: ManagerBasedRLEnv,
    ee_pos_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    grasp_threshold: float,
    robot_cfg_name: str = "robot",
) -> torch.Tensor:
    """Grasp detection: distance close AND gripper closed."""
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg_name]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    is_close = ee_object_distance < grasp_threshold

    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    is_gripper_closed = finger_width < 0.05

    return is_close & is_gripper_closed


def _to_local(env: ManagerBasedRLEnv, pos_w: torch.Tensor) -> torch.Tensor:
    """Convert world coords to env-local coords (subtract env_origins)."""
    return pos_w - env.scene.env_origins[:, :3]


def _is_object_in_box(
    env: ManagerBasedRLEnv,
    object_cfg_name: str = "object",
    box_x_range: tuple = (-0.2, 0.2),
    box_y_range: tuple = (-0.6, -0.3),
    box_z_range: tuple = (0.0, 0.08),
) -> torch.Tensor:
    """Check if object is inside the box region (local coords)."""
    object_asset: RigidObject = env.scene[object_cfg_name]
    pos = _to_local(env, object_asset.data.root_pos_w[:, :3])
    in_x = (pos[:, 0] >= box_x_range[0]) & (pos[:, 0] <= box_x_range[1])
    in_y = (pos[:, 1] >= box_y_range[0]) & (pos[:, 1] <= box_y_range[1])
    in_z = (pos[:, 2] >= box_z_range[0]) & (pos[:, 2] <= box_z_range[1])
    return in_x & in_y & in_z


# ------------------------------------------------------------------ #
# Phase 0: Approaching object (tanh-kernel dense reward)
# ------------------------------------------------------------------ #

def reaching_object_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Approach reward: 1 - tanh(dist / std). Active in Phase 0, 1."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    reward = 1.0 - torch.tanh(distance / std)
    if hasattr(env, "task_phase"):
        is_active = (env.task_phase == 0) | (env.task_phase == 1)
        return torch.where(is_active, reward, torch.zeros_like(reward))
    return reward


def grasp_success_reward(
    env: ManagerBasedRLEnv,
    bonus: float = 1000.0,
    grasp_threshold: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """One-shot grasp bonus on first successful grasp. Phase 0 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold)

    if not hasattr(env, "_grasp_success_given"):
        env._grasp_success_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    new_grasp = is_grasping & (~env._grasp_success_given)
    env._grasp_success_given = env._grasp_success_given | is_grasping
    reward = torch.where(new_grasp, bonus, 0.0)

    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 0, reward, torch.zeros_like(reward))
    return reward


# ------------------------------------------------------------------ #
# Phase 1/2: Stable grasp + lift
# ------------------------------------------------------------------ #

def stable_grasp_reward(
    env: ManagerBasedRLEnv,
    grasp_threshold: float = 0.03,
    robot_cfg_name: str = "robot",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Phase 1/2 stable grasp: 1.0 when grasping, 0.0 otherwise."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold, robot_cfg_name)
    reward = is_grasping.float()
    if hasattr(env, "task_phase"):
        is_active = (env.task_phase == 1) | (env.task_phase == 2)
        return torch.where(is_active, reward, torch.zeros_like(reward))
    return reward


def lifting_object_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.15,
    max_height: float = 0.30,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_threshold: float = 0.03,
) -> torch.Tensor:
    """Linear lift reward: higher object = more reward. Capped at max_height. Phase 1 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold)

    height_reward = (height - minimal_height) / (max_height - minimal_height)
    height_reward = torch.clamp(height_reward, min=0.0, max=1.0)
    is_valid_height = (height >= minimal_height) & (height <= max_height)
    reward = torch.where(is_grasping & is_valid_height, height_reward, torch.zeros_like(height_reward))

    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 1, reward, torch.zeros_like(reward))
    return reward


def lift_milestone_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.35,
    bonus: float = 1000.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-time bonus when object height first reaches threshold. Phase 1 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    is_lifted = height >= threshold
    if not hasattr(env, "_lift_milestone_given"):
        env._lift_milestone_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    new_milestone = is_lifted & (~env._lift_milestone_given)
    env._lift_milestone_given = env._lift_milestone_given | is_lifted
    reward = torch.where(
        new_milestone, torch.tensor(bonus, device=env.device), torch.zeros_like(height)
    )
    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 1, reward, torch.zeros_like(reward))
    return reward


# ------------------------------------------------------------------ #
# Phase 2: Goal tracking (transport to box)
# ------------------------------------------------------------------ #

def object_goal_tracking_reward(
    env: ManagerBasedRLEnv,
    max_distance: float = 1.2,
    target_pos: tuple = (0.0, -0.4, 0.3),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_threshold: float = 0.03,
) -> torch.Tensor:
    """Linear goal tracking: 1 - dist/max_distance while grasping. Phase 2 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold)

    object_pos_local = _to_local(env, object_pos_w)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    distance = torch.norm(object_pos_local - target, dim=1)
    reward = is_grasping.float() * torch.clamp(1.0 - distance / max_distance, min=0.0)

    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 2, reward, torch.zeros_like(reward))
    return reward


def object_goal_tracking_fine_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    target_pos: tuple = (0.0, -0.4, 0.3),
    vertical_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_threshold: float = 0.03,
) -> torch.Tensor:
    """Fine goal tracking (tanh) with vertical gripper requirement. Phase 2 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold)

    # Gripper Z-axis: pointing down -> gz_z close to -1
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    x, y = ee_quat[:, 1], ee_quat[:, 2]
    gz_z = 1.0 - 2.0 * (x * x + y * y)
    is_vertical = gz_z < (-1.0 + vertical_threshold)

    object_pos_local = _to_local(env, object_pos_w)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    distance = torch.norm(object_pos_local - target, dim=1)
    reward = (is_grasping & is_vertical).float() * (1.0 - torch.tanh(distance / std))

    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 2, reward, torch.zeros_like(reward))
    return reward


def arrive_box_vertical_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple = (0.0, -0.4, 0.3),
    position_threshold: float = 0.05,
    vertical_threshold: float = 0.15,
    bonus: float = 2000.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_threshold: float = 0.03,
) -> torch.Tensor:
    """One-shot bonus: arrived above box with vertical gripper. Phase 2 only."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    is_grasping = _is_grasping(env, ee_pos_w, object_pos_w, grasp_threshold)

    ee_pos_local = _to_local(env, ee_pos_w)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    is_at_target = torch.norm(ee_pos_local - target, dim=1) < position_threshold

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    x, y = ee_quat[:, 1], ee_quat[:, 2]
    gz_z = 1.0 - 2.0 * (x * x + y * y)
    is_vertical = gz_z < (-1.0 + vertical_threshold)

    meets_all = is_grasping & is_at_target & is_vertical
    if not hasattr(env, "_arrive_box_vertical_given"):
        env._arrive_box_vertical_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    new_arrival = meets_all & (~env._arrive_box_vertical_given)
    env._arrive_box_vertical_given = env._arrive_box_vertical_given | meets_all
    reward = torch.where(
        new_arrival, torch.tensor(bonus, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )
    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 2, reward, torch.zeros_like(reward))
    return reward


# ------------------------------------------------------------------ #
# Phase 3: Release into box
# ------------------------------------------------------------------ #

def object_near_target_phase3_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple = (0.0, -0.4, 0.3),
    proximity_threshold: float = 0.05,
    vertical_threshold: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Phase 3 position-hold: reward when EE near target and vertical."""
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_local = _to_local(env, ee_pos_w)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    distance = torch.norm(ee_pos_local - target, dim=1)
    is_near = distance < proximity_threshold

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    x, y = ee_quat[:, 1], ee_quat[:, 2]
    gz_z = 1.0 - 2.0 * (x * x + y * y)
    is_vertical = gz_z < (-1.0 + vertical_threshold)

    reward = (is_near & is_vertical).float()
    return torch.where(env.task_phase == 3, reward, torch.zeros_like(reward))


def object_in_box_reward(
    env: ManagerBasedRLEnv,
    bonus: float = 5000.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-shot bonus when object falls into box. Phase 3."""
    if not hasattr(env, "task_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    is_release_phase = env.task_phase == 3
    in_box = _is_object_in_box(env, object_cfg.name)
    if not hasattr(env, "_object_in_box_given"):
        env._object_in_box_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    new_in_box = is_release_phase & in_box & (~env._object_in_box_given)
    env._object_in_box_given = env._object_in_box_given | (is_release_phase & in_box)
    return torch.where(new_in_box, bonus, 0.0)


def gripper_open_release_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple = (0.0, -0.45, 0.25),
    release_radius: float = 0.1,
    max_finger_width: float = 0.08,
    robot_cfg_name: str = "robot",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Phase 3 release: reward gripper opening near target. Prevents early release."""
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg_name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_local = _to_local(env, ee_pos_w)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    dist_to_target = torch.norm(ee_pos_local - target, dim=1)
    is_near_target = dist_to_target < release_radius

    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    open_reward = torch.clamp(finger_width / max_finger_width, 0.045, 1.0)

    reward = torch.where(is_near_target, open_reward, torch.zeros_like(open_reward))
    if hasattr(env, "task_phase"):
        return torch.where(env.task_phase == 3, reward, torch.zeros_like(reward))
    return reward


# ------------------------------------------------------------------ #
# Task completion (kept for backward compat with V2 reward configs)
# ------------------------------------------------------------------ #

def task_completion_reward(
    env: ManagerBasedRLEnv,
    completion_bonus: float = 5000.0,
) -> torch.Tensor:
    """One-time bonus when object has fallen into the box (success)."""
    is_completed = object_in_box(env)
    if not hasattr(env, "_completion_reward_given"):
        env._completion_reward_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    new_completion = is_completed & (~env._completion_reward_given)
    env._completion_reward_given = env._completion_reward_given | is_completed
    return torch.where(new_completion, completion_bonus, 0.0)


# ------------------------------------------------------------------ #
# Global penalties
# ------------------------------------------------------------------ #

def orientation_alignment_penalty(
    env: ManagerBasedRLEnv,
    penalty_scale: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE-object yaw misalignment penalty. Phase 0, 1 active."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_quat = obj.data.root_quat_w
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    object_yaw = torch.atan2(
        2 * (object_quat[:, 0] * object_quat[:, 3] + object_quat[:, 1] * object_quat[:, 2]),
        1 - 2 * (object_quat[:, 2] ** 2 + object_quat[:, 3] ** 2),
    )
    ee_yaw = torch.atan2(
        2 * (ee_quat[:, 0] * ee_quat[:, 3] + ee_quat[:, 1] * ee_quat[:, 2]),
        1 - 2 * (ee_quat[:, 2] ** 2 + ee_quat[:, 3] ** 2),
    )
    angle_diff = torch.abs(object_yaw - ee_yaw)
    angle_diff = torch.min(angle_diff, 2 * 3.14159 - angle_diff)
    penalty = penalty_scale * angle_diff
    if hasattr(env, "task_phase"):
        is_rl_phase = (env.task_phase == 0) | (env.task_phase == 1)
        return torch.where(is_rl_phase, penalty, torch.zeros_like(penalty))
    return penalty


def joint_limit_penalty(
    env: ManagerBasedRLEnv,
    margin: float = 0.1,
    penalty_scale: float = -1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint limit penalty: increases when near limits. All phases."""
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[robot_cfg.name]
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_lower = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 0]
    joint_upper = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 1]
    dist_to_lower = joint_pos - joint_lower
    dist_to_upper = joint_upper - joint_pos
    lower_penalty = torch.clamp(margin - dist_to_lower, min=0.0) / margin
    upper_penalty = torch.clamp(margin - dist_to_upper, min=0.0) / margin
    total_penalty = (lower_penalty + upper_penalty).sum(dim=1)
    return penalty_scale * total_penalty


def joint_posture_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint posture regularization: penalize deviation from default.

    Phase 2 exempt: transport requires large joint reconfiguration.
    """
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[asset_cfg.name]
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, arm_joint_ids]
    deviation = torch.sum((joint_pos - default_joint_pos) ** 2, dim=1)
    if hasattr(env, "task_phase"):
        is_transport = env.task_phase == 2
        return torch.where(is_transport, torch.zeros_like(deviation), deviation)
    return deviation
