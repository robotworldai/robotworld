# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from isaaclab.utils.math import wrap_to_pi, quat_apply

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class WheelLeggedCommand(CommandTerm):
    """轮腿机器人的命令生成器：生成 [vx, ang_vel_yaw, height, heading]"""
    cfg: WheelLeggedCommandCfg

    def __init__(self, cfg: WheelLeggedCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # 初始化命令缓冲区 (num_envs, 4)
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        # 标记是否采用朝向命令
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 本体前向向量 (用于计算世界系下的航向)
        self._forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # 初始化指标缓冲区
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        """对指定的环境重新采样命令"""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.hold_command_during_jump and hasattr(self._env, "jump_phase"):
            # A velocity-command discontinuity between takeoff and recovery
            # creates an impossible target and corrupts moving-jump credit.
            # Postpone the sample for active environments until their next
            # regular resampling window.
            env_ids = env_ids[self._env.jump_phase[env_ids] == 0]
        if len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        # 1. 线速度 x
        self._command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # 2. 角速度 yaw (如果使用朝向控制，这个值会在 _update_command 中被覆盖)
        self._command[env_ids, 1] = r.uniform_(*self.cfg.ranges.ang_vel_yaw)
        # 3. 高度目标
        self._command[env_ids, 2] = r.uniform_(*self.cfg.ranges.height)
        # 4. 朝向目标 (如果启用)
        if self.cfg.heading_command:
            self._command[env_ids, 3] = r.uniform_(*self.cfg.ranges.heading)
            # 按配置比例选择 heading 控制环境；其余环境保留直接采样的 yaw-rate。
            self._is_heading_env[env_ids] = (
                torch.rand(len(env_ids), device=self.device) < self.cfg.rel_heading_envs
            )
        # 重置跟踪误差指标（新命令周期重新开始统计）
        for key in self.metrics:
            self.metrics[key][env_ids] = 0.0

    def _update_command(self):
        """后处理：若启用朝向控制，计算角速度命令"""
        if self.cfg.heading_command:
            # 获取机器人当前朝向（yaw角）
            # 通过 quat_apply 将本体前向向量旋转到世界系，再计算航向角
            asset = self._env.scene[self.cfg.asset_name]
            quat_w = asset.data.root_quat_w
            forward_w = quat_apply(quat_w, self._forward_vec)
            yaw_current = torch.atan2(forward_w[:, 1], forward_w[:, 0])
            # 计算航向误差
            heading_error = wrap_to_pi(self._command[:, 3] - yaw_current)
            # 仅对启用朝向控制的环境计算角速度
            env_ids = self._is_heading_env.nonzero(as_tuple=False).flatten()
            # 比例控制
            ang_vel_cmd = self.cfg.heading_control_stiffness * heading_error[env_ids]
            ang_vel_cmd = torch.clip(ang_vel_cmd, self.cfg.ranges.ang_vel_yaw[0], self.cfg.ranges.ang_vel_yaw[1])
            self._command[env_ids, 1] = ang_vel_cmd

    def _update_metrics(self):
        """更新日志指标：线速度、角速度、高度跟踪误差。"""
        # 获取机器人当前状态
        asset = self._env.scene[self.cfg.asset_name]
        # 时间归一化因子
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # 线速度 x 跟踪误差
        self.metrics["error_vel_x"] += (
            torch.abs(self._command[:, 0] - asset.data.root_lin_vel_b[:, 0]) / max_command_step
        )
        # 角速度 yaw 跟踪误差
        self.metrics["error_vel_yaw"] += (
            torch.abs(self._command[:, 1] - asset.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        # 高度跟踪误差
        self.metrics["error_height"] += (
            torch.abs(self._command[:, 2] - asset.data.root_pos_w[:, 2]) / max_command_step
        )
        # 角度误差（仅在启用朝向命令时计算）
        if self.cfg.heading_command:
            quat_w = asset.data.root_quat_w
            forward_w = quat_apply(quat_w, self._forward_vec)
            yaw_current = torch.atan2(forward_w[:, 1], forward_w[:, 0])
            heading_error = wrap_to_pi(self._command[:, 3] - yaw_current)
            self.metrics["error_heading"] += torch.abs(heading_error) / max_command_step

    # ======================================================================
    # Debug Visualization（覆写 CommandTerm 的方法）
    # ======================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_heading_vis"):
                # 自定义箭头：细长型 (x=长, y/z=细)
                arrow_scale = (0.1, 0.1, 1.0)
                goal_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                goal_arrow_cfg.prim_path="/Visuals/Command/goal_heading"
                goal_arrow_cfg.markers["arrow"].scale = arrow_scale
                self._goal_heading_vis = VisualizationMarkers(goal_arrow_cfg)

                current_arrow_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/current_heading")
                current_arrow_cfg.markers["arrow"].scale = arrow_scale
                self._current_heading_vis = VisualizationMarkers(current_arrow_cfg)
                self._goal_height_vis = VisualizationMarkers(
                    SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_height")
                )
                self._goal_heading_vis.set_visibility(True)
                self._current_heading_vis.set_visibility(True)
                self._goal_height_vis.set_visibility(True)
        else:
            if hasattr(self, "_goal_heading_vis"):
                self._goal_heading_vis.set_visibility(False)
                self._current_heading_vis.set_visibility(False)
                self._goal_height_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        # CommandManager may deliver one final render event while the
        # environment is being torn down. At that point DirectRLEnv has
        # already deleted ``scene`` and the visualization must become a no-op.
        try:
            scene = self._env.scene
        except (AttributeError, ReferenceError):
            return
        asset = scene[self.cfg.asset_name]
        if not asset.is_initialized:
            return

        root_pos = asset.data.root_pos_w
        root_quat = asset.data.root_quat_w

        # 当前航向角
        forward_w = quat_apply(root_quat, self._forward_vec)
        yaw_current = torch.atan2(forward_w[:, 1], forward_w[:, 0])

        # --- 目标朝向（绿色箭头，默认沿 X 轴，绕 Z 旋转 heading 角度） ---
        zeros = torch.zeros(self.num_envs, device=self.device)
        goal_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self._command[:, 3])
        goal_pos = root_pos.clone()
        goal_pos[:, 2] += 0.5
        self._goal_heading_vis.visualize(goal_pos, goal_quat)

        # --- 当前朝向（蓝色箭头） ---
        current_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_current)
        current_pos = root_pos.clone()
        current_pos[:, 2] += 0.3
        self._current_heading_vis.visualize(current_pos, current_quat)

        # --- 目标高度（红色球体） ---
        height_pos = root_pos.clone()
        height_pos[:, 2] = self._command[:, 2]
        self._goal_height_vis.visualize(height_pos)

# 以下参数需在实际机器人cfg配置文件中进行设置，在这设置会被覆盖
@configclass
class WheelLeggedCommandCfg(CommandTermCfg):
    """配置：轮腿机器人的 4 项命令 (vx, ang_vel_yaw, height, heading)"""
    class_type: type[CommandTerm] = WheelLeggedCommand

    asset_name: str = "robot"
    """机器人资产名称。"""

    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    """命令重采样的时间间隔范围（秒）。"""

    heading_command: bool = True
    """是否启用朝向命令。若为 True，则角速度由朝向误差计算得出。"""
    heading_control_stiffness: float = 1.5
    """朝向控制的刚度系数。"""
    rel_heading_envs: float = 1.0
    """采用朝向命令的环境比例（仅当 heading_command=True 时有效）。"""
    hold_command_during_jump: bool = False
    """跳跃状态机非 IDLE 时暂缓重采样速度命令。"""

    @configclass
    class Ranges:
        """命令采样范围"""
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)   # 线速度 x (m/s)
        ang_vel_yaw: tuple[float, float] = (-5.0, 5.0) # 角速度 yaw (rad/s)
        height: tuple[float, float] = (0.1, 0.25)      # 基座高度目标 (m)
        heading: tuple[float, float] = (-0.01, 0.01)  # 朝向目标 (rad)

    ranges: Ranges = Ranges()
    """命令采样范围。"""
