# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# base_height_reward函数未使用，使用base_height_reward_simple，增加对高度的跟踪奖励
def base_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基于命令的基座高度跟踪奖励（使用指数核）。

    奖励条件：
    1. 基座高度接近目标 → exp(-error²/σ²)
    2. 身体直立（重力对齐）→ 乘以重力投影系数
    3. 计算左右腿的不对称程度
    防止"趴在地上碰巧高度对"的 reward hacking。
    """
    if not hasattr(env, "base_height") or not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name)
    height_target = cmd[:, 2]
    error = env.base_height - height_target
    # 1. 高度误差
    reward = torch.exp(-torch.square(error) / 0.05)
    # 2. 重力对齐：身体越直立，系数越接近 1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # 3. 计算左右腿的不对称程度
    theta0 = env.theta0  # (num_envs, 2)
    L0 = env.L0          # (num_envs, 2)

    # 归一化不对称度（范围 0~1，0=完全对称，1=完全不对称）
    theta_asym = torch.square(theta0[:, 0] - theta0[:, 1])  # 角度不对称
    L0_asym = torch.square(L0[:, 0] - L0[:, 1])             # 长度不对称

    # 组合不对称度（可调权重）
    asym = theta_asym + 0.5 * L0_asym  # L0 的权重可以调低一些

    # 用指数衰减将不对称度转换为协调因子
    # exp(-asym / sigma^2)
    # sigma 控制“容忍度”：sigma 越小，要求越严格
    coordination_factor = torch.exp(-asym / 0.05)  # 0.05 是经验值，可调
    reward *= coordination_factor
    return reward


def base_height_reward_simple(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """纯粹的高度跟踪奖励（不含不对称惩罚，不对称由 nominal_state 单独处理）。

    使用指数核 exp(-error²/σ²)，配合重力对齐条件，防止 reward hacking。
    """
    if not hasattr(env, "base_height"):
        return torch.zeros(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name)
    height_target = cmd[:, 2]
    error = env.base_height - height_target
    # 指数核，默认 σ=0.05，使 ±5 cm 误差时奖励约为 0.37。
    reward = torch.exp(-torch.square(error) / std**2)
    # 重力对齐条件：身体必须直立才能获得高度奖励
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """关节加速度惩罚（使用 asset.data.joint_acc）"""
    asset = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return reward


def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    动作二阶差分惩罚（仅作用于腿部动作）。
    需要环境维护 _action_history（最近3步动作）。
    """
    if not hasattr(env, "_action_history"):
        # 若环境未提供历史，返回零惩罚（不惩罚）
        return torch.zeros(env.num_envs, device=env.device)

    history = env._action_history  # (num_envs, 3, num_actions)
    a_t = history[:, 2, :]         # 当前步动作
    a_t1 = history[:, 1, :]        # 上一步动作
    a_t2 = history[:, 0, :]        # 上上步动作

    # 仅腿部动作索引（根据你的动作顺序：0,1,3,4 为腿部）
    leg_indices = [0, 1, 3, 4]
    diff = a_t[:, leg_indices] - 2 * a_t1[:, leg_indices] + a_t2[:, leg_indices]
    reward = torch.sum(torch.square(diff), dim=1)
    return reward


def nominal_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚左右虚拟腿不对称。"""
    if not hasattr(env, "theta0"):
        return torch.zeros(env.num_envs, device=env.device)
    theta0 = env.theta0  # (num_envs, 2)
    L0 = env.L0  # (num_envs, 2)
    # theta0 对称性
    theta0_asym = torch.square(theta0[:, 0] - theta0[:, 1])
    # 长度误差按 0.1 m 归一化，使它与角度误差量级相近。
    L0_asym = torch.square((L0[:, 0] - L0[:, 1]) / 0.1)
    return theta0_asym + 0.25 * L0_asym


def leg_posture_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the open-chain legs near their nominal IK branch.

    Virtual-leg length and angle alone do not identify the two-link IK branch. This
    term makes the nominal branch the unique low-cost solution without preventing
    the smaller motions needed for balancing and height control.
    """
    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(error), dim=1)


def track_heading_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    heading 角度跟踪奖励（指数核）。
    直接惩罚当前 yaw 与目标 heading 的误差，解决纯角速度跟踪的稳态误差问题。
    """
    from isaaclab.utils.math import wrap_to_pi
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    # 计算当前 yaw 角
    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    forward_w = quat_apply(asset.data.root_quat_w, forward_vec)
    yaw_current = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    # heading 误差
    heading_error = wrap_to_pi(cmd[:, 3] - yaw_current)
    reward = torch.exp(-torch.square(heading_error) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_lin_vel_xy_exp_wl(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    轮腿机器人线速度跟踪奖励（仅 x 方向），使用指数核。
    注意：此函数与通用版本分开，避免名称覆盖。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_lin_vel_x = cmd[:, 0]
    base_lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    error_sq = torch.square(cmd_lin_vel_x - base_lin_vel_x)
    reward = torch.exp(-error_sq / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp_wl(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    轮腿机器人角速度跟踪奖励（yaw 方向），使用指数核。
    注意：此函数与通用版本分开，避免名称覆盖。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_ang_vel = cmd[:, 1]
    base_ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    error_sq = torch.square(cmd_ang_vel - base_ang_vel_z)
    reward = torch.exp(-error_sq / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# 增强型跟踪奖励 — 修正为正向奖励（值域 [0, 1]）
def track_lin_vel_enhance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    error_sq = torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])
    # 使用 0.5 倍 std 的核函数（而非 1/√10），对精确跟踪给予更高奖励，同时保留有效梯度
    reward = torch.exp(-error_sq / (std**2 / 4))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_ang_vel_enhance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    error_sq = torch.square(cmd[:, 1] - asset.data.root_ang_vel_b[:, 2])
    # 使用 0.5 倍 std 的核函数（而非 1/√10），对精确跟踪给予更高奖励，同时保留有效梯度
    reward = torch.exp(-error_sq / (std**2 / 4))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def soft_joint_limits_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.15,  # 距离软限位多远开始惩罚（弧度）
) -> torch.Tensor:
    """
    改进版限位惩罚：在接近限位 margin 弧度时就开始惩罚，
    采用二次函数，越接近限位，惩罚越大。
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    # 距离下限的余量
    lower_margin = pos - lower
    upper_margin = upper - pos

    # 当余量 < margin 时，产生惩罚（二次）
    # penalty_lower = torch.clamp(margin - lower_margin, min=0.0) ** 2
    # penalty_upper = torch.clamp(margin - upper_margin, min=0.0) ** 2

    # 当余量 < margin 时，产生惩罚（一次）
    penalty_lower = torch.clamp(margin - lower_margin, min=0.0)
    penalty_upper = torch.clamp(margin - upper_margin, min=0.0)

    return torch.sum(penalty_lower + penalty_upper, dim=1)


def recovery_upright_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "wheel_legged_commands",
    tilt_std: float = 0.30,
    height_std: float = 0.07,
    active_only: bool = True,
) -> torch.Tensor:
    """Reward returning to a commanded upright height after recoverable disturbances."""
    gravity_b = env.scene["robot"].data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
    target_height = env.command_manager.get_command(command_name)[:, 2]
    height_error = env.base_height - target_height
    reward = torch.exp(-torch.square(tilt / tilt_std)) * torch.exp(
        -torch.square(height_error / height_std)
    )
    if active_only and hasattr(env, "_recovery_reward_mask"):
        reward = reward * env._recovery_reward_mask
    return reward
