# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def push_robots_by_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    push_interval_s: float = 7.0,
    max_push_vel_xy: float = 2.0,
    force_scale_z: float = 0.5,
):
    """随机施加外力推动机器人（模拟原 Isaac Gym 的 _push_robots）"""
    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # 获取机器人资产
    asset = env.scene[asset_cfg.name]
    # 计算最大推力（基于基座质量和目标速度变化）
    base_mass = asset.data.default_mass[:, 0]  # 假设第一个身体是基座
    max_push_force = (base_mass.mean() * max_push_vel_xy) / env.physics_dt
    # 生成随机力（在水平面内，均匀分布）
    forces_w = torch.rand(len(env_ids), 3, device=env.device) * 2 - 1  # [-1,1]
    forces_w[:, :2] *= max_push_force
    forces_w[:, 2] *= max_push_force * force_scale_z
    # 将力从基座坐标系旋转到世界坐标系（原代码 quat_rotate）
    base_quat = asset.data.root_quat_w[env_ids]
    forces_w = math_utils.quat_apply(base_quat, forces_w)
    # 应用瞬时力到基座（body index 0）
    asset.instantaneous_wrench_composer.add_forces_and_torques(
        forces=forces_w.unsqueeze(1),  # (num_envs, 1, 3)
        torques=torch.zeros_like(forces_w).unsqueeze(1),
        body_ids=[0],
        env_ids=env_ids,
    )
    # Recovery tasks use the push as a well-defined evaluation trial.  Keep the
    # event generic so the flat/jump environments do not need the extra state.
    start_recovery = getattr(env, "start_recovery_attempt", None)
    if start_recovery is not None:
        start_recovery(env_ids)

def randomize_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_range: tuple[float, float],
):
    """在启动时随机偏移默认关节位置"""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=env.device)
    else:
        joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
    # 只随机化 SceneEntityCfg 指定的关节，避免把连续轮关节的默认角度也偏移。
    low, high = offset_range
    offset = torch.rand(len(env_ids), len(joint_ids), device=env.device) * (high - low) + low
    # 修改默认位置（只修改 data 缓存，物理引擎会使用）
    asset.data.default_joint_pos[env_ids[:, None], joint_ids] += offset
