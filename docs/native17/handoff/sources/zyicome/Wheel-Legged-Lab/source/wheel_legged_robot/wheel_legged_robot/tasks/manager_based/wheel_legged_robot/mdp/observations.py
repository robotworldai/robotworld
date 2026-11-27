# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def theta0(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 环境初始化阶段（load_managers）也会调用此函数来检查输出形状，
    # 此时 env.theta0 尚未设置，返回零张量即可。
    if not hasattr(env, "theta0"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.theta0  # 返回形状为 (num_envs, 2)

def theta0_dot(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "theta0_dot"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.theta0_dot  # 返回形状为 (num_envs, 2)

def L0(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.L0  # 返回形状为 (num_envs, 2)

def L0_dot(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "L0_dot"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.L0_dot  # 返回形状为 (num_envs, 2)

def joint_wheel_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The wheel joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return joint_wheel_pos

def joint_wheel_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The wheel joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return joint_wheel_vel

def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint accelerations of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return joint_acc

def wheel_legged_commands(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: tuple[float, float, float] = (2.0, 0.25, 5.0),
) -> torch.Tensor:
    """VMC 策略的命令观测：线速度 x、角速度 yaw、高度目标（取前 3 个命令）。"""
    # 1. 获取命令张量（形状：num_envs, 4）
    commands = env.command_manager.get_command(command_name)  # 假设有 4 列
    # 2. 只取前 3 列：lin_vel_x, ang_vel_yaw, height
    commands = commands[:, :3]
    # 3. 应用缩放
    scale_tensor = torch.tensor(scale, device=env.device)
    return commands * scale_tensor


def obstacle_forward_scan(
    env: ManagerBasedRLEnv,
    distances: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
) -> torch.Tensor:
    """Robot-yaw-aligned forward height scan of the active obstacle geometry."""
    count = len(distances)
    if not hasattr(env, "obstacle_position_w"):
        return torch.zeros(env.num_envs, count, device=env.device)
    samples = torch.tensor(distances, device=env.device).unsqueeze(0)
    root_delta = env._robot.data.root_pos_w[:, :2] - env.obstacle_position_w
    obstacle_from_root = -torch.sum(
        root_delta * env.obstacle_forward_w, dim=1, keepdim=True
    )
    half_width = 0.5 * env.obstacle_width.unsqueeze(1)
    hit = (samples - obstacle_from_root).abs() <= half_width
    return torch.where(hit, env.obstacle_height.unsqueeze(1), 0.0)


def depth_camera_obstacle_scan(
    env: ManagerBasedRLEnv,
    sample_count: int = 10,
) -> torch.Tensor:
    """Return the deployable obstacle-height profile extracted from depth.

    The perceptive environment updates this cache once per control step after
    deprojecting the calibrated depth image into world-space points. Keeping
    the output shape identical to :func:`obstacle_forward_scan` allows an
    Oracle checkpoint to initialize the perceptive policy without changing
    the actor architecture.
    """
    scan = getattr(env, "obstacle_depth_scan", None)
    if scan is None:
        return torch.zeros(env.num_envs, sample_count, device=env.device)
    if scan.shape[1] != sample_count:
        raise RuntimeError(
            f"Depth obstacle scan has {scan.shape[1]} samples, expected {sample_count}."
        )
    return scan

def base_mass(env) -> torch.Tensor:
    """返回基座质量减去所有环境均值，形状 (num_envs, 1)"""
    if hasattr(env, "_base_mass"):
        mass = env._base_mass  # (num_envs,)
    else:
        # 若尚未随机化，从 URDF 读取默认质量
        asset = env.scene["robot"]
        mass = asset.data.default_mass[:, 0].to(device=env.device)  # 基座质量（确保在正确设备）
    # 计算均值并中心化
    mass_centered = mass - mass.mean()
    return mass_centered.unsqueeze(-1)  # (num_envs, 1)

def base_com(env) -> torch.Tensor:
    """返回基座质心偏移（x,y,z）"""
    if hasattr(env, "_base_com"):
        return env._base_com  # 形状 (num_envs, 3)
    else:
        return torch.zeros(env.num_envs, 3, device=env.device)

def default_joint_offset(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回默认关节位置的随机偏移量（仅匹配 asset_cfg 指定的关节）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "_default_joint_offset"):
        return env._default_joint_offset[:, asset_cfg.joint_ids]
    else:
        return torch.zeros(env.num_envs, len(asset_cfg.joint_ids), device=env.device)

def friction_coef(env) -> torch.Tensor:
    """返回地面摩擦系数（标量）"""
    if hasattr(env, "_friction_coef"):
        return env._friction_coef.unsqueeze(-1)
    else:
        return torch.ones(env.num_envs, 1, device=env.device) * 0.5

def restitution_coef(env) -> torch.Tensor:
    """返回地面恢复系数（标量）"""
    if hasattr(env, "_restitution_coef"):
        return env._restitution_coef.unsqueeze(-1)
    else:
        return torch.zeros(env.num_envs, 1, device=env.device) * 0.5


def compact_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    reference_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_base"),
    longitudinal_samples: int = 5,
    lateral_samples: int = 3,
) -> torch.Tensor:
    """Return a compact forward terrain grid relative to ground under the body.

    The full ray grid is reduced to ``longitudinal_samples × lateral_samples``
    points while preserving its two-dimensional layout.  Positive values mean
    terrain above the ground directly below the chassis.  This is deployable
    with a depth/LiDAR preprocessing layer that emits the same local grid.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits_z)
    reference_sensor = env.scene.sensors[reference_sensor_cfg.name]
    reference_hits = reference_sensor.data.ray_hits_w[..., 2]
    reference_valid = torch.isfinite(reference_hits)
    reference = torch.where(
        reference_valid, reference_hits, torch.zeros_like(reference_hits)
    ).sum(dim=1, keepdim=True) / reference_valid.sum(dim=1, keepdim=True).clamp(min=1)
    relative = torch.where(valid, hits_z - reference, torch.zeros_like(hits_z))

    pattern = sensor.cfg.pattern_cfg
    nx = round(float(pattern.size[0]) / float(pattern.resolution)) + 1
    ny = round(float(pattern.size[1]) / float(pattern.resolution)) + 1
    sample_count = longitudinal_samples * lateral_samples
    if nx * ny != relative.shape[1] or pattern.ordering != "xy":
        indices = torch.linspace(
            0, relative.shape[1] - 1, sample_count, device=env.device
        ).round().long()
        return relative.index_select(1, indices)

    grid = relative.reshape(env.num_envs, ny, nx)
    cache_key = (
        sensor_cfg.name,
        nx,
        ny,
        longitudinal_samples,
        lateral_samples,
    )
    cache = getattr(env, "_compact_height_scan_index_cache", {})
    if cache_key not in cache:
        x_start = -0.5 * float(pattern.size[0]) + float(sensor.cfg.offset.pos[0])
        x_end = 0.5 * float(pattern.size[0]) + float(sensor.cfg.offset.pos[0])
        y_start = -0.5 * float(pattern.size[1]) + float(sensor.cfg.offset.pos[1])
        y_end = 0.5 * float(pattern.size[1]) + float(sensor.cfg.offset.pos[1])
        x = torch.linspace(x_start, x_end, nx, device=env.device)
        y = torch.linspace(y_start, y_end, ny, device=env.device)
        target_x = torch.linspace(
            max(0.10, x_start), x_end, longitudinal_samples, device=env.device
        )
        lateral_extent = min(0.20, 0.5 * float(pattern.size[1]))
        target_y = torch.linspace(
            -lateral_extent, lateral_extent, lateral_samples, device=env.device
        )
        cache[cache_key] = (
            (x[:, None] - target_x[None, :]).abs().argmin(dim=0),
            (y[:, None] - target_y[None, :]).abs().argmin(dim=0),
        )
        env._compact_height_scan_index_cache = cache
    x_indices, y_indices = cache[cache_key]
    # Output order is near-to-far, and within each distance right/center/left.
    sampled = grid[:, y_indices[:, None], x_indices[None, :]].transpose(1, 2)
    return sampled.reshape(env.num_envs, sample_count)
