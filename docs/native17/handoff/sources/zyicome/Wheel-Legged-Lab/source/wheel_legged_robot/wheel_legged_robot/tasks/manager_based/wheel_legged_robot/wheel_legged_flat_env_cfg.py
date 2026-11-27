# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 zyicome

import math
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp
from .mdp import VMCActionCfg
from .assets.wheellegged import WHEEL_LEGGED_CFG

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##

@configclass
class WheelLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##
class WheelLeggedVMCEnv(ManagerBasedRLEnv):
    """
    轮腿机器人 VMC 环境类。

    维护以下状态：
    - 运动学状态（theta0, theta0_dot, L0, L0_dot）——由 VMCAction 填充
    - 基座相对高度（base_height）
    - 动作历史（用于 action_smooth 奖励）
    - 自定义缓冲区（用于奖励计算）
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._robot = self.scene["robot"]
        self._height_scanner = self.scene.sensors.get("height_scanner")

        # 获取关节数量（用于缓冲区初始化）
        self.num_dof = self._robot.num_joints

        # ========== 运动学状态（由 VMCAction 填充） ==========
        self.theta0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self.L0 = torch.zeros(self.num_envs, 2, device=self.device)
        self.L0_dot = torch.zeros(self.num_envs, 2, device=self.device)

        # ========== 基座相对高度 ==========
        self.base_height = torch.zeros(self.num_envs, device=self.device)

        # ========== 动作历史（最近3步，用于 action_smooth 奖励） ==========
        num_actions = self.action_manager.action.shape[1]  # 通常为6
        self._action_history = torch.zeros(self.num_envs, 3, num_actions, device=self.device)

        # ========== 自定义缓冲区（原 Isaac Gym 中的重置相关） ==========
        self._last_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self._feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)  # 4个足端（如有需要）
        self._last_base_position = torch.zeros(self.num_envs, 3, device=self.device)

        # ========== 域随机化捕获标记（第一次 reset 后才捕获，确保 startup 事件已执行） ==========
        self._domain_rand_captured = False

    def _capture_domain_rand(self):
        """读取物理引擎中已随机化的值（须在 startup 事件之后调用），存储供 critic privileged observations 使用。"""
        asset = self._robot

        # base_link 质量（body 0）—— root_physx_view 返回 CPU tensor，需转到 device
        masses = asset.root_physx_view.get_masses().to(self.device)  # (num_envs, num_bodies)
        self._base_mass = masses[:, 0].clone()

        # base_link 质心偏移 — 健壮处理不同 get_coms() 返回形状
        try:
            coms = asset.root_physx_view.get_coms().to(self.device)
            if coms.dim() == 3 and coms.shape[2] == 3:
                self._base_com = coms[:, 0, :].clone()  # (num_envs, num_bodies, 3)
            elif coms.dim() == 2 and coms.shape[1] >= 3:
                # 可能是 (num_envs, num_bodies*3)，取 base_link 的 xyz
                self._base_com = coms[:, :3].clone()
            else:
                self._base_com = coms[:, :3].reshape(self.num_envs, 3).clone()
        except Exception:
            self._base_com = torch.zeros(self.num_envs, 3, device=self.device)

        # 关节默认位置偏移 — 只取前 6 个 actuated 关节（跳过 fixed joints）
        default_pos = asset.data.default_joint_pos[:, :6]  # (num_envs, 6)
        self._default_joint_offset = (default_pos - default_pos.mean(dim=0, keepdim=True)).clone()

        # 摩擦/恢复系数：尝试从物理引擎读取已随机化的值
        try:
            # root_physx_view.get_materials() 返回 (num_envs, num_bodies) 的摩擦/恢复
            materials = asset.root_physx_view.get_materials().to(self.device)
            # PhysX material layout: static friction, dynamic friction, restitution.
            if materials.dim() == 3 and materials.shape[2] >= 3:
                self._friction_coef = materials[:, 0, 0].clone()     # base_link 的静摩擦
                self._restitution_coef = materials[:, 0, 2].clone()  # base_link 的恢复系数
            else:
                raise RuntimeError("Unexpected material shape")
        except Exception:
            # 若无法读取，回退到默认值
            self._friction_coef = torch.ones(self.num_envs, device=self.device) * 0.5
            self._restitution_coef = torch.ones(self.num_envs, device=self.device) * 0.5

    def step(self, action: torch.Tensor):
        """覆写 step，在正确的时间点调用 _update_base_height（在 reward 计算之前）。"""
        # Normal policy distributions are unbounded. Store and penalize the same bounded
        # command that the VMC controller actually executes.
        action = torch.clamp(action.to(self.device), -1.0, 1.0)
        self.action_manager.process_action(action)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # ===== 在 reward 计算之前刷新运动学、基座高度和动作历史 =====
        self.action_manager.get_term("vmc").update_kinematics()
        self._update_base_height()
        self._update_action_history()
        self._pre_reward_update()

        # post-step counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # reward computation（此时 base_height 已更新）
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # recorder
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # reset terminated envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # commands & interval events
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # ===== 记录调试指标 =====
        self._log_debug_metrics()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_reward_update(self):
        """Hook for task-specific state updates after physics and before rewards."""
        pass

    def _log_debug_metrics(self):
        """记录每步的调试指标到 extras['log']，供 cusrl Metrics 采集。"""
        if not hasattr(self, "theta0"):
            return

        log = {}

        # 运动学状态（所有环境平均）
        log["theta0_L"] = self.theta0[:, 0].mean().item()
        log["theta0_R"] = self.theta0[:, 1].mean().item()
        log["L0_L"] = self.L0[:, 0].mean().item()
        log["L0_R"] = self.L0[:, 1].mean().item()
        log["base_height"] = self.base_height.mean().item()

        # 关节位置（所有环境平均）
        if hasattr(self, "_robot"):
            vmc_term = self.action_manager.get_term("vmc")
            joint_pos = self._robot.data.joint_pos[:, vmc_term.joint_ids]
            log["rf0_pos"] = joint_pos[:, 3].mean().item()
            log["lf0_pos"] = joint_pos[:, 0].mean().item()
            log["rf1_pos"] = joint_pos[:, 4].mean().item()
            log["lf1_pos"] = joint_pos[:, 1].mean().item()
            leg_local_ids = [0, 1, 3, 4]
            leg_asset_ids = [vmc_term.joint_ids[index] for index in leg_local_ids]
            leg_limits = self._robot.data.soft_joint_pos_limits[:, leg_asset_ids]
            leg_pos = joint_pos[:, leg_local_ids]
            joint_margin = torch.minimum(leg_pos - leg_limits[..., 0], leg_limits[..., 1] - leg_pos)
            log["joint_margin_min"] = joint_margin.min(dim=1).values.mean().item()
            log["torque_saturation"] = vmc_term.torque_saturation.mean().item()
            log["motor_enable_scale"] = vmc_term.motor_enable_scale.mean().item()

        # 命令值（所有环境平均）
        if hasattr(self, "command_manager"):
            cmd = self.command_manager.get_command("wheel_legged_commands")  # (num_envs, 4)
            log["cmd_vx"] = cmd[:, 0].mean().item()
            log["cmd_vx_abs"] = cmd[:, 0].abs().mean().item()  # 绝对值均值，反映实际命令幅度
            cmd_term = self.command_manager.get_term("wheel_legged_commands")
            log["cmd_vx_limit"] = float(cmd_term.cfg.ranges.lin_vel_x[1])
            if hasattr(self, "_command_curriculum_last_score"):
                log["curriculum_score"] = self._command_curriculum_last_score.item()
            log["cmd_wz_limit"] = float(cmd_term.cfg.ranges.ang_vel_yaw[1])
            if hasattr(self, "_command_curriculum_last_angular_score"):
                log["curriculum_angular_score"] = (
                    self._command_curriculum_last_angular_score.item()
                )
            log["cmd_wz"] = cmd[:, 1].mean().item()
            log["cmd_wz_abs"] = cmd[:, 1].abs().mean().item()
            log["cmd_height"] = cmd[:, 2].mean().item()
            log["cmd_heading"] = cmd[:, 3].mean().item()
            # 瞬时跟踪误差（直接计算，不依赖累加 metrics）
            if hasattr(self, "_robot"):
                vel_x = self._robot.data.root_lin_vel_b[:, 0]
                cmd_vx = cmd[:, 0]
                log["vx_err_inst"] = torch.abs(cmd_vx - vel_x).mean().item()
                log["vel_x_abs"] = vel_x.abs().mean().item()
                # 对称采样时 signed mean 会抵消；gain=1 表示整体速度幅值和方向均正确。
                cmd_energy = torch.mean(cmd_vx.square()).clamp_min(1.0e-6)
                log["vx_tracking_gain"] = (torch.mean(cmd_vx * vel_x) / cmd_energy).item()
                moving_cmd = cmd_vx.abs() > 0.05
                if moving_cmd.any():
                    log["vx_sign_match"] = (
                        (cmd_vx[moving_cmd] * vel_x[moving_cmd] > 0).float().mean().item()
                    )
                omega_z = self._robot.data.root_ang_vel_b[:, 2]
                cmd_wz = cmd[:, 1]
                log["wz_err_inst"] = torch.abs(cmd_wz - omega_z).mean().item()
                log["omega_z_abs"] = omega_z.abs().mean().item()
                wz_energy = torch.mean(cmd_wz.square()).clamp_min(1.0e-6)
                log["wz_tracking_gain"] = (
                    torch.mean(cmd_wz * omega_z) / wz_energy
                ).item()
                log["h_err_inst"] = torch.abs(cmd[:, 2] - self._robot.data.root_pos_w[:, 2]).mean().item()

        # 实际运动状态（对比命令值，判断跟踪效果）
        if hasattr(self, "_robot"):
            # 实际线速度（本体坐标系 x 方向）
            log["vel_x"] = self._robot.data.root_lin_vel_b[:, 0].mean().item()
            # 实际角速度（本体坐标系 z 方向，yaw rate）
            log["omega_z"] = self._robot.data.root_ang_vel_b[:, 2].mean().item()
            # 实际航向角（yaw）
            from isaaclab.utils.math import quat_apply
            forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            forward_w = quat_apply(self._robot.data.root_quat_w, forward_vec)
            yaw = torch.atan2(forward_w[:, 1], forward_w[:, 0])
            log["yaw"] = yaw.mean().item()
            heading_error = torch.atan2(
                torch.sin(cmd[:, 3] - yaw),
                torch.cos(cmd[:, 3] - yaw),
            )
            log["heading_err_abs"] = heading_error.abs().mean().item()

        # 基座姿态
        if hasattr(self, "_robot"):
            gravity_b = self._robot.data.projected_gravity_b
            log["tilt_angle"] = torch.acos(
                torch.clamp(-gravity_b[:, 2], -1.0, 1.0)
            ).mean().item()

        # 动作统计
        if hasattr(self, "action_manager"):
            action = self.action_manager.action
            log["action_mean"] = action.mean().item()
            log["action_std"] = action.std().item()
            log["action_abs_max"] = action.abs().max().item()
            vmc_term = self.action_manager.get_term("vmc")
            wheel_action = vmc_term.processed_actions[:, [2, 5]]
            log["wheel_action_abs"] = wheel_action.abs().mean().item()
            log["wheel_action_clip_fraction"] = (
                wheel_action.abs() >= 0.99 * vmc_term.cfg.action_clip
            ).float().mean().item()
            wheel_asset_ids = [vmc_term.joint_ids[2], vmc_term.joint_ids[5]]
            log["wheel_vel_abs"] = (
                self._robot.data.joint_vel[:, wheel_asset_ids].abs().mean().item()
            )

        self.extras["log"] = {**self.extras.get("log", {}), **log}

    def _update_base_height(self):
        """计算基座相对高度（基座高度 - 地面高度），逐环境处理 NaN/Inf。"""
        if self._height_scanner is not None:
            ray_hits = self._height_scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
            # 逐环境计算地面高度，对每个环境单独处理 NaN/Inf
            valid_mask = ~(torch.isnan(ray_hits) | torch.isinf(ray_hits))
            # 对每个环境，只对有效射线取平均；若某环境全无效，则 ground_height=0
            safe_hits = torch.where(valid_mask, ray_hits, torch.zeros_like(ray_hits))
            ground_height = safe_hits.sum(dim=1) / valid_mask.float().sum(dim=1).clamp(min=1)
        else:
            ground_height = torch.zeros(self.num_envs, device=self.device)

        root_height = self._robot.data.root_pos_w[:, 2]
        self.base_height = root_height - ground_height

    def _update_action_history(self):
        """更新动作历史缓冲区（保留最近3步）。"""
        current_action = self.action_manager.action
        self._action_history = torch.cat(
            [self._action_history[:, 1:], current_action.unsqueeze(1)],
            dim=1
        )

    def _reset_idx(self, env_ids):
        """
        重置指定环境。
        扩展了父类重置逻辑，增加自定义缓冲区的清零。
        注意：父类 _reset_idx 已处理 episode_sums 等基础 buffer 的清零。
        """
        # 1. 调用父类重置（处理基础的 buffer 清零，包括 episode_sums）
        #    注意：父类的 _reset_idx 会触发 EventManager 执行 startup 事件（仅在首次全局 reset 时）
        super()._reset_idx(env_ids)

        # 2. 首次 reset 时，startup 事件已执行完毕，此时捕获域随机化值才是正确的
        if not self._domain_rand_captured:
            # 检查是否所有环境都在被重置（首次 reset 一定是全局的）
            if len(env_ids) == self.num_envs:
                self._capture_domain_rand()
                self._domain_rand_captured = True

        # 3. 清零自定义缓冲区
        self._action_history[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        self._last_base_position[env_ids] = self._robot.data.root_pos_w[env_ids]

        # 4. 重置运动学状态（避免观测读取到旧值）
        self.theta0[env_ids] = 0.0
        self.theta0_dot[env_ids] = 0.0
        self.L0[env_ids] = 0.0
        self.L0_dot[env_ids] = 0.0
        # Reset 后 observation 会立刻计算；不能让第一帧看到虚假的全零虚拟腿。
        self.action_manager.get_term("vmc").update_kinematics()


@configclass
class VmcActionsCfg:
    """VMC 动作配置（仅包含 VMC 动作项）。"""
    vmc = VMCActionCfg(
        asset_name="robot",
        action_scale_theta=0.35,
        action_scale_l0=0.06,
        # r=0.0675 m、半轮距=0.25 m。vx=1 m/s 且 heading 误差为 pi 时，
        # wz≈0.6*pi，快侧轮约需 21.8 rad/s；24 留出约 10% 余量。
        action_scale_vel=24.0,
        action_clip=1.0,
        l0_offset=0.237,
        l0_min=0.18,
        l0_max=0.30,
        feedforward_force=60.0,
        kp_theta=60.0,
        kd_theta=3.0,
        kp_l0=600.0,
        kd_l0=20.0,
        kp_wheel=0.5,
        kp_wheel_integral=0.2,
        offset=0.054,
        l1=0.15,
        l2=0.25,
    )

@configclass
class WheelleggedObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        velocity_commands = ObsTerm(
            func=mdp.wheel_legged_commands,
            params={
                "command_name": "wheel_legged_commands",
                "scale": (2.0, 0.25, 5.0),
            },
            clip=(-100.0, 100.0),
            scale=1.0,  # 注意：这里不要重复缩放，已在函数内部处理
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        leg_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-30.0, 30.0),
            scale=0.05,
        )
        theta0 = ObsTerm(
            func=mdp.theta0,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        theta0_dot = ObsTerm(
            func=mdp.theta0_dot,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        L0 = ObsTerm(
            func=mdp.L0,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        L0_dot = ObsTerm(
            func=mdp.L0_dot,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        joint_wheel_vel = ObsTerm(
            func=mdp.joint_wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        # 3 + 3 + 3 + 2 + 2 + 2 + 2 + 2 + 2 + 6 = 27

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        # policy observations
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        velocity_commands = ObsTerm(
            func=mdp.wheel_legged_commands,
            params={
                "command_name": "wheel_legged_commands",
                "scale": (2.0, 0.25, 5.0),
            },
            clip=(-100.0, 100.0),
            scale=1.0,  # 注意：这里不要重复缩放，已在函数内部处理
        ) #3
        # joint_pos_rel = ObsTerm(
        #     func=mdp.joint_pos_rel_without_wheel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # ) #6
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        #     clip=(-100.0, 100.0),
        #     scale=1.0,
        # ) #6
        theta0 = ObsTerm(
            func=mdp.theta0,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        theta0_dot = ObsTerm(
            func=mdp.theta0_dot,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        L0 = ObsTerm(
            func=mdp.L0,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        L0_dot = ObsTerm(
            func=mdp.L0_dot,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        joint_wheel_vel = ObsTerm(
            func=mdp.joint_wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_Joint", "r_wheel_Joint"], preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #2
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=3,
            flatten_history_dim=True,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        # 3 + 3 + 3 + 2 + 2 + 2 + 2 + 2 + 2 + 3x6 = 39
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        ) # 77
        torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        base_mass = ObsTerm(
            func=mdp.base_mass,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #1
        base_com = ObsTerm(
            func=mdp.base_com,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #3
        default_joint_offset = ObsTerm(
            func=mdp.default_joint_offset,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #6
        friction_coef = ObsTerm(
            func=mdp.friction_coef,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #1
        restitution_coef = ObsTerm(
            func=mdp.restitution_coef,
            clip=(-100.0, 100.0),
            scale=1.0,
        ) #1
        # 3 + 39 + 6 + 6 + 6 + 77 + 6 + 1 + 3 + 6 + 1 + 1 = 155

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WheelLeggedEventCfg:
    """Configuration for events."""
        # ========== 1. 物理参数随机化（启动时执行一次） ==========
    # 对应原 _process_rigid_body_props

    # 摩擦/恢复系数随机化（所有身体）
    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": False,
        },
    )

    # 基座质量随机化（add 模式）
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 0.01,
        },
    )

    # 其他身体质量随机化（scale 模式，排除 base_link 避免与 randomize_base_mass 双重随机化）
    randomize_other_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="(?!base_link).*"),  # 排除 base_link
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 0.01,
        },
    )

    # 基座质心随机化
    randomize_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )

    randomize_joints_pos = EventTermCfg(
        func=mdp.randomize_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            ),
            "offset_range": (-0.02, 0.02),
        },
    )

    # ========== 2. 初始状态重置（每次 reset 时执行） ==========

    # 重置关节位置（使用默认位置 + 可选的随机缩放）
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,  # 官方函数
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (1.0, 1.0),  # 1.0 = 完全使用默认位置
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    # ========== 3. 外部扰动（间隔执行） ==========
    # 对应原 _push_robots

    push_robot = EventTermCfg(
        func=mdp.push_robots_by_force,
        mode="interval",
        interval_range_s=(7.0, 10.0),  # 每隔 7~10s 触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "push_interval_s": 7.0,  # 原配置
            "max_push_vel_xy": 0.5,
            "force_scale_z": 0.0,
        },
    )

    # ========== 4. 驱动器参数随机化（VMC 模式下禁用） ==========
    # 在action.py中已实现随机化开关

@configclass
class WheelLeggedRewardsCfg:
    lin_vel_z = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    orientation = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        # 轮式平衡加速需要短时俯仰，不能用过强姿态惩罚把策略锁死。
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewardTermCfg(
        func=mdp.base_height_reward_simple,
        weight=2.5,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    dof_vel_legs = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                preserve_order=True,
            )
        },
    )
    dof_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        # 历史日志显示该项曾是最大负奖励；过大会阻止快速平衡纠偏。
        weight=-5e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.soft_joint_limits_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                preserve_order=True,
            )
        },
    )
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )
    action_smooth = RewardTermCfg(
        func=mdp.action_smooth_l2,
        weight=-0.005,
    )
    collision = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 0.1},
    )
    # 腿杆（大小腿）与地面碰撞惩罚 — 提高到与 base 碰撞同级
    leg_collision = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-2.0,  # 从 -0.3 提高，腿拖地很严重
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["lf0_Link", "lf1_Link", "rf0_Link", "rf1_Link"]), "threshold": 0.1},
    )
    nominal_state = RewardTermCfg(
        func=mdp.nominal_state,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    leg_posture = RewardTermCfg(
        func=mdp.leg_posture_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                preserve_order=True,
            )
        },
    )
    # 初始命令仅为 ±0.1 m/s；std 过宽会让原地站立也获得近满分。
    # 0.25 仍保留有效梯度，同时能明确区分“站住”和“跟随速度”。
    track_lin_vel = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp_wl,
        weight=3.0,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp_wl,
        weight=1.0,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # heading 角度跟踪奖励（消除纯角速度跟踪的稳态误差）
    track_heading = RewardTermCfg(
        func=mdp.track_heading_exp,
        weight=0.25,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    termination = RewardTermCfg(
        func=mdp.is_terminated,
        # RewardManager 会乘 step_dt=0.01，因此该事件实际约为 -1.0。
        weight=-100.0,
    )

    # is_terminated = RewardTermCfg(
    #     func=mdp.is_terminated,
    #     weight=0.0,
    # )

@configclass
class WheelLeggedTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )
    joint_limits = TerminationTermCfg(
        func=mdp.joint_pos_out_of_limit,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                preserve_order=True,
            )
        },
    )
    minimum_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.12},
    )

    # illegal_contact = TerminationTermCfg(
    #     func=mdp.illegal_contact,
    #     time_out=False,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
    #         "threshold": 10.0,
    #     },
    # )

    # terrain_out_of_bounds = TerminationTermCfg(
    #     func=mdp.terrain_out_of_bounds,
    #     time_out=True,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    # )


@configclass
class WheelLeggedCommandsCfg:
    """命令配置"""
    # 注意：这里的属性名 "wheel_legged_commands" 就是奖励函数中要引用的 command_name
    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),       # 每 5 秒重采样一次
        heading_command=True,                   # 启用朝向控制
        heading_control_stiffness=0.6,          # 从 1.5 降低，避免产生过大的瞬态角速度命令
        rel_heading_envs=1.0,                   # 所有环境都使用朝向控制
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            # heading 模式下 wz=0.6*wrap_to_pi(error)，理论最大约 1.885 rad/s。
            # ±2.0 已完整覆盖；设置成 ±5.0 不会产生更快的实际命令。
            ang_vel_yaw=(-2.0, 2.0),
            # 必须高于 minimum_height=0.12；否则准确跟踪低高度命令会触发终止。
            height=(0.16, 0.26),
            heading=(-3.14, 3.14),
        ),
    )

@configclass
class WheelLeggedCurriculumCfg:
    """课程配置"""

    # 命令课程（线速度范围逐步扩大）
    command_levels = CurriculumTermCfg(
        func=mdp.wheel_legged_command_curriculum,
        params={
            "reward_term_name": "track_lin_vel",           # 奖励项名称
            "command_name": "wheel_legged_commands",       # 命令项名称
            "angular_reward_term_name": "track_ang_vel",   # yaw 课程奖励项
            "range_multiplier": (0.1, 1.0),                # 初始 10%，最终 100%
            "threshold": 0.9,                              # 完整时长归一化得分阈值
            "step_size": 0.05,                             # 每次扩大 0.05 m/s
            "initial_ang_vel_limit": 0.5,                  # 初始最大 0.5 rad/s
            "final_ang_vel_limit": 2.0,                    # 覆盖 0.6*pi
            "angular_threshold": 0.85,
            "angular_step_size": 0.25,
            "min_episodes": 256,                           # 异步完成的 episode 统计窗口
        },
    )

@configclass
class WheelLeggedFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelLeggedRobotSceneCfg = WheelLeggedRobotSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: WheelleggedObservationsCfg = WheelleggedObservationsCfg()
    actions: VmcActionsCfg = VmcActionsCfg()
    commands: WheelLeggedCommandsCfg = WheelLeggedCommandsCfg()
    rewards: WheelLeggedRewardsCfg = WheelLeggedRewardsCfg()
    terminations: WheelLeggedTerminationsCfg = WheelLeggedTerminationsCfg()
    events: WheelLeggedEventCfg = WheelLeggedEventCfg()
    curriculum: WheelLeggedCurriculumCfg = WheelLeggedCurriculumCfg()

    base_link_name = "base_link"
    # fmt: off
    joint_names = [
        "lf0_Joint",   # 索引 0
        "lf1_Joint",   # 索引 1
        "l_wheel_Joint", # 索引 2
        "rf0_Joint",   # 索引 3
        "rf1_Joint",   # 索引 4
        "r_wheel_Joint", # 索引 5
    ]
    # fmt: on

    env_class = WheelLeggedVMCEnv

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ===== 仿真参数 =====
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.num_position_iterations = 8
        self.sim.physx.num_velocity_iterations = 4

        # ------------------------------Sence------------------------------
        self.scene.robot = WHEEL_LEGGED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)) # 传感器相对于 base_link 的位置偏移，z=0.5m
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.6]) # 传感器扫描范围为 1.0m x 0.6m
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)) # 传感器相对于 base_link 的位置偏移，z=0.5m
        self.scene.height_scanner_base.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.6]) # 传感器扫描范围为 1.0m x 0.6m

        # ===== 传感器更新周期 =====
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # ------------------------------Observations------------------------------
        self.observations = WheelleggedObservationsCfg()
        # policy observations
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.theta0.scale = 1.0
        self.observations.policy.theta0_dot.scale = 0.05
        self.observations.policy.L0.scale = 5.0
        self.observations.policy.L0_dot.scale = 0.25
        self.observations.policy.joint_wheel_vel.scale = 0.05
        # critic observations
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.projected_gravity.scale = 1.0
        self.observations.critic.theta0.scale = 1.0
        self.observations.critic.theta0_dot.scale = 0.05
        self.observations.critic.L0.scale = 5.0
        self.observations.critic.L0_dot.scale = 0.25
        self.observations.critic.joint_wheel_vel.scale = 0.05
        self.observations.critic.joint_acc.scale = 0.0025
        self.observations.critic.joint_pos_rel.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.height_scan.scale = 5.0
        self.observations.critic.torques.scale = 0.05

        # ------------------------------Actions------------------------------
        self.actions = VmcActionsCfg()

        # ------------------------------Events------------------------------
        self.events = WheelLeggedEventCfg()

        # ------------------------------Rewards------------------------------
        self.rewards = WheelLeggedRewardsCfg()

        # ------------------------------Terminations------------------------------
        self.terminations = WheelLeggedTerminationsCfg()

        # ------------------------------Curriculums------------------------------
        self.curriculum = WheelLeggedCurriculumCfg()

        # ------------------------------Commands------------------------------
        self.commands = WheelLeggedCommandsCfg()
