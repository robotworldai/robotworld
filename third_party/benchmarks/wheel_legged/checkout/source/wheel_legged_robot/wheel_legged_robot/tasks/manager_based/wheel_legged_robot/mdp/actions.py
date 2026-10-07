# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class VMCAction(ActionTerm):
    """
    VMC（虚拟模型控制）动作项。

    动作向量（6维）：
        [theta0_ref_L, L0_ref_L, wheel_vel_L, theta0_ref_R, L0_ref_R, wheel_vel_R]
    """
    cfg: VMCActionCfg

    def __init__(self, cfg: VMCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = env.num_envs
        self._pi = torch.acos(torch.zeros(1, device=self._device)) * 2
        self._joint_ids, resolved_joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if resolved_joint_names != list(cfg.joint_names):
            raise ValueError(
                f"VMCAction could not resolve all required joints. Expected {list(cfg.joint_names)}, "
                f"resolved {resolved_joint_names} from {list(self._asset.joint_names)}."
            )

        # 动作缓冲区（6维：theta0_ref_L, L0_ref_L, wheel_vel_L, theta0_ref_R, L0_ref_R, wheel_vel_R）
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._torque_saturation = torch.zeros(self.num_envs, device=self.device)
        # Per-environment output gate used by power-on and controller hand-off
        # state machines.  Ordinary tasks remain fully enabled by default.
        self._motor_enable_scale = torch.ones(self.num_envs, device=self.device)

        # 缓存上一时刻的运动学状态（供观测使用）
        self._theta0 = torch.zeros(self._num_envs, 2, device=self._device)
        self._L0 = torch.zeros(self._num_envs, 2, device=self._device)

        # 轮速误差积分缓冲（用于消除轮速稳态误差）
        self._wheel_vel_integral = torch.zeros(self._num_envs, 2, device=self._device)

        # ----- 启动时随机化（与 _init_buffers 对应）-----
        # 1. VMC 增益随机化
        if cfg.randomize_kp:
            scale = torch.rand(self._num_envs, 2, device=self._device) * 0.2 + 0.9  # [0.9, 1.1]
            self._theta_kp = cfg.kp_theta * scale
            self._l0_kp = cfg.kp_l0 * scale
        else:
            self._theta_kp = torch.full((self._num_envs, 2), cfg.kp_theta, device=self._device)
            self._l0_kp = torch.full((self._num_envs, 2), cfg.kp_l0, device=self._device)

        # 2. VMC 阻尼随机化
        if cfg.randomize_kd:
            scale = torch.rand(self._num_envs, 2, device=self._device) * 0.2 + 0.9  # [0.9, 1.1]
            self._theta_kd = cfg.kd_theta * scale
            self._l0_kd = cfg.kd_l0 * scale
        else:
            self._theta_kd = torch.full((self._num_envs, 2), cfg.kd_theta, device=self._device)
            self._l0_kd = torch.full((self._num_envs, 2), cfg.kd_l0, device=self._device)

        # 3. 力矩缩放随机化
        if cfg.randomize_torque_scale:
            self._torque_scale = torch.rand(self._num_envs, 6, device=self._device) * 0.2 + 0.9  # [0.9, 1.1]
        else:
            self._torque_scale = torch.ones(self._num_envs, 6, device=self._device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_ids(self) -> list[int]:
        """Articulation indices in VMC logical order (left leg/wheel, right leg/wheel)."""
        return self._joint_ids

    @property
    def torque_saturation(self) -> torch.Tensor:
        """Per-environment fraction of motors whose requested torque was clipped."""
        return self._torque_saturation

    @property
    def motor_enable_scale(self) -> torch.Tensor:
        """Current per-environment actuator-output scale in ``[0, 1]``."""
        return self._motor_enable_scale

    def set_motor_enable_scale(
        self,
        scale: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set a safe actuator-output ramp without changing policy actions.

        A zero scale represents an unpowered robot.  Clearing the wheel PI
        integral while disabled prevents wind-up during passive settling.
        """
        if env_ids is None:
            env_ids = slice(None)
            target = self._motor_enable_scale
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            target = self._motor_enable_scale[env_ids]
        scale_tensor = torch.as_tensor(scale, dtype=target.dtype, device=self.device)
        if scale_tensor.ndim == 0:
            scale_tensor = scale_tensor.expand_as(target)
        else:
            scale_tensor = torch.broadcast_to(scale_tensor, target.shape)
        self._motor_enable_scale[env_ids] = scale_tensor.clamp(0.0, 1.0)
        disabled = self._motor_enable_scale[env_ids] <= 1.0e-4
        if torch.any(disabled):
            if isinstance(env_ids, slice):
                self._wheel_vel_integral[disabled] = 0.0
            else:
                self._wheel_vel_integral[env_ids[disabled]] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # 保留网络原始输出用于诊断，但控制器只接收有界动作。不能依赖 PPO
        # distribution 自行限幅：Normal distribution 的采样理论上是无界的。
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._theta0[env_ids] = 0.0
        self._L0[env_ids] = 0.0
        self._wheel_vel_integral[env_ids] = 0.0
        self._torque_saturation[env_ids] = 0.0
        self._motor_enable_scale[env_ids] = 1.0

    def _compute_kinematics(self):
        """Compute and publish the current virtual-leg state from joint state."""
        # Reorder articulation state into the controller's logical order. USD/PhysX
        # traversal order is not the same as URDF declaration or action order.
        dof_pos = self._asset.data.joint_pos[:, self._joint_ids]  # (num_envs, 6)
        dof_vel = self._asset.data.joint_vel[:, self._joint_ids]  # (num_envs, 6)

        # 运动学计算（左腿：0,1,2；右腿：3,4,5）
        # theta1 = [q0, -q3]     (左髋, 右髋)
        theta1 = torch.stack([dof_pos[:, 0], -dof_pos[:, 3]], dim=-1)
        # theta2 = [q1 + pi/2, -q4 + pi/2]   (左膝, 右膝)
        theta2 = torch.stack([dof_pos[:, 1] + self._pi/2, -dof_pos[:, 4] + self._pi/2], dim=-1)

        # 前向运动学：计算 L0, theta0
        L0, theta0 = self._forward_kinematics(theta1, theta2)

        # 更新缓存的状态（供 VMC 映射和观测使用）
        # Keep the buffers allocated during construction. Rebinding them to a
        # clone created under ``torch.inference_mode()`` turns them into
        # inference tensors, which cannot later be cleared by an interactive
        # Play callback running outside that context (for example K power-off).
        self._theta0.copy_(theta0)
        self._L0.copy_(L0)

        # 3. 计算导数 — 使用关节速度 + FK（与原始 Isaac Gym 一致，比位置差分更准确）
        # 原始代码：leg_post_physics_step() 中 dt=0.001
        theta1_dot = torch.stack([dof_vel[:, 0], -dof_vel[:, 3]], dim=-1)
        theta2_dot = torch.stack([dof_vel[:, 1], -dof_vel[:, 4]], dim=-1)
        dt_deriv = 0.001  # 与原始 Isaac Gym 一致
        L0_temp, theta0_temp = self._forward_kinematics(
            theta1 + theta1_dot * dt_deriv, theta2 + theta2_dot * dt_deriv
        )
        theta0_dot = (theta0_temp - theta0) / dt_deriv
        L0_dot = (L0_temp - L0) / dt_deriv

        # 同步写入环境属性（供观测和奖励使用）
        if hasattr(self._env, "theta0"):
            self._env.theta0[:] = theta0
            self._env.theta0_dot[:] = theta0_dot
            self._env.L0[:] = L0
            self._env.L0_dot[:] = L0_dot
        return dof_pos, dof_vel, theta1, theta2, L0, theta0, theta0_dot, L0_dot

    def update_kinematics(self):
        """Refresh observations after the last physics step or after a reset."""
        self._compute_kinematics()

    def apply_actions(self):
        # 1-3. 读取关节状态并更新虚拟腿运动学
        dof_pos, dof_vel, theta1, theta2, L0, theta0, theta0_dot, L0_dot = self._compute_kinematics()

        # 4. 解析动作（6维）
        actions = self._processed_actions  # (num_envs, 6)
        # 左腿：0,1,2；右腿：3,4,5
        theta0_ref_L = actions[:, 0] * self.cfg.action_scale_theta
        L0_ref_L = torch.clamp(
            actions[:, 1] * self.cfg.action_scale_l0 + self.cfg.l0_offset,
            self.cfg.l0_min,
            self.cfg.l0_max,
        )
        wheel_vel_ref_L = actions[:, 2] * self.cfg.action_scale_vel

        theta0_ref_R = actions[:, 3] * self.cfg.action_scale_theta
        L0_ref_R = torch.clamp(
            actions[:, 4] * self.cfg.action_scale_l0 + self.cfg.l0_offset,
            self.cfg.l0_min,
            self.cfg.l0_max,
        )
        wheel_vel_ref_R = actions[:, 5] * self.cfg.action_scale_vel

        # 组装为 (num_envs, 2) 方便批量计算
        theta0_ref = torch.stack([theta0_ref_L, theta0_ref_R], dim=-1)
        L0_ref = torch.stack([L0_ref_L, L0_ref_R], dim=-1)
        wheel_vel_ref = torch.stack([wheel_vel_ref_L, wheel_vel_ref_R], dim=-1)

        # 5. VMC 控制力计算
        # 力矩（任务空间）：theta 和 L0 的 PD 控制
        torque_theta = self._theta_kp * (theta0_ref - theta0) - self._theta_kd * theta0_dot
        l0_kp, l0_kd, feedforward_force = self._get_l0_control_parameters()
        force_L0 = l0_kp * (L0_ref - L0) - l0_kd * L0_dot
        # 加上前馈力
        force_L0 += feedforward_force

        # 轮子速度控制（PI 控制，带积分项消除稳态误差）
        wheel_vel = dof_vel[:, [2, 5]]  # 左轮、右轮速度
        wheel_vel_error = wheel_vel_ref - wheel_vel
        # 积分项，带限幅防止 windup
        # apply_actions 每个 physics step 都会调用，因此必须使用 physics_dt，
        # 而不是 control step_dt（decimation > 1 时后者会重复积分）。
        dt = self._env.physics_dt if hasattr(self._env, "physics_dt") else 0.005
        self._wheel_vel_integral += (
            wheel_vel_error
            * dt
            * self.cfg.kp_wheel_integral
            * self._motor_enable_scale.unsqueeze(-1)
        )
        # 积分贡献限制为轮电机力矩上限的一部分，避免长时间饱和后 wind-up。
        wheel_effort_limits = self._asset.data.joint_effort_limits[:, self._joint_ids][:, [2, 5]]
        integral_limit = wheel_effort_limits * self.cfg.wheel_integral_limit_ratio
        self._wheel_vel_integral.copy_(
            torch.clamp(
                self._wheel_vel_integral,
                -integral_limit,
                integral_limit,
            )
        )
        torque_wheel = self.cfg.kp_wheel * wheel_vel_error + self._wheel_vel_integral

        # 6. 通过 VMC 雅可比映射到关节力矩
        T1, T2 = self._vmc_mapping(theta1, theta2, L0, theta0, force_L0, torque_theta)

        # 7. 组合完整力矩（6维：左髋、左膝、左轮、右髋、右膝、右轮）
        # 注意：右腿符号需要根据你的 VMC 映射决定，此处取负（与原代码一致）
        torques = torch.cat([
            T1[:, 0:1],           # 左髋
            T2[:, 0:1],           # 左膝
            torque_wheel[:, 0:1], # 左轮
            -T1[:, 1:2],          # 右髋（取反）
            -T2[:, 1:2],          # 右膝（取反）
            torque_wheel[:, 1:2], # 右轮
        ], dim=-1)

        # 应用力矩缩放
        torques = torques * self._torque_scale
        torques = torques * self._motor_enable_scale.unsqueeze(-1)

        # 获取每个关节的力矩限位（如果多个环境共享同一限位，可取第一行）
        effort_limits = self._asset.data.joint_effort_limits[:, self._joint_ids]  # (num_envs, 6)
        # 记录策略是否长期依赖电机饱和，再对每个环境裁剪。
        self._torque_saturation[:] = torch.mean((torch.abs(torques) > effort_limits).float(), dim=1)
        torques = torch.clamp(torques, -effort_limits, effort_limits)

        # 8. 写入仿真
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    def _get_l0_control_parameters(self):
        """Return per-environment virtual-leg gains and feed-forward force.

        Subclasses may override this hook for phase-dependent control.  The
        default returns the original parameters exactly, so ordinary locomotion
        tasks are unaffected.
        """
        feedforward_force = torch.full_like(self._l0_kp, self.cfg.feedforward_force)
        return self._l0_kp, self._l0_kd, feedforward_force

    # ---------- 辅助方法 ----------
    def _forward_kinematics(self, theta1, theta2):
        """计算虚拟腿长度 L0 和角度 theta0（相对于垂直方向）。"""
        # theta1, theta2: (num_envs, 2)  左、右腿
        offset = self.cfg.offset
        l1 = self.cfg.l1
        l2 = self.cfg.l2

        # 末端位置（在基座坐标系中）
        end_x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
        end_y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)

        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.atan2(end_y, end_x) - self._pi / 2
        return L0, theta0

    def _vmc_mapping(self, theta1, theta2, L0, theta0, force, torque):
        """
        VMC 映射：将任务空间力/力矩转换为关节力矩。

        参数：
            theta1, theta2: (num_envs, 2)
            L0: (num_envs, 2)
            theta0: (num_envs, 2) 虚拟腿角度（相对于垂直方向）
            force: (num_envs, 2)  虚拟腿上的力（正为拉伸）
            torque: (num_envs, 2) 虚拟腿上的力矩（正为逆时针）

        返回：
            T1, T2: (num_envs, 2)  髋关节和膝关节力矩
        """
        theta0_vmc = theta0 + self._pi / 2

        t11 = self.cfg.l1 * torch.sin(theta0_vmc - theta1) - self.cfg.l2 * torch.sin(theta1 + theta2 - theta0_vmc)
        safe_L0 = torch.clamp(L0, min=self.cfg.singularity_epsilon)
        t12 = (
            self.cfg.l1 * torch.cos(theta0_vmc - theta1)
            + self.cfg.l2 * torch.cos(theta1 + theta2 - theta0_vmc)
        ) / safe_L0
        t21 = -self.cfg.l2 * torch.sin(theta1 + theta2 - theta0_vmc)
        t22 = self.cfg.l2 * torch.cos(theta1 + theta2 - theta0_vmc) / safe_L0

        T1 = t11 * force + t12 * torque
        T2 = t21 * force + t22 * torque
        return T1, T2

@configclass
class VMCActionCfg(ActionTermCfg):
    """VMC 动作配置。"""
    class_type: type[ActionTerm] = VMCAction  # 指向实现类

    asset_name: str = "robot"
    joint_names: tuple[str, ...] = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )

    # === VMC 参数 ===
    action_scale_theta: float = 0.5
    action_scale_l0: float = 0.1
    action_scale_vel: float = 10.0
    action_clip: float = 1.0

    l0_offset: float = 0.237
    l0_min: float = 0.18
    l0_max: float = 0.30
    singularity_epsilon: float = 0.05
    feedforward_force: float = 40.0

    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 20.0
    kp_wheel: float = 10.0
    kp_wheel_integral: float = 1.0
    wheel_integral_limit_ratio: float = 0.25

    offset: float = 0.054
    l1: float = 0.15
    l2: float = 0.25

    # 域随机化开关
    randomize_kp: bool = True
    randomize_kp_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = True
    randomize_kd_range: tuple[float, float] = (0.9, 1.1)
    randomize_torque_scale: bool = True
    randomize_torque_scale_range: tuple[float, float] = (0.9, 1.1)
