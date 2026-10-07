"""Core Isaac Lab environment for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from .assets import DRONE_ROPE_PLANK_CFG
from .evaluation import (
    TargetPositionEvaluator,
    TargetPositionEvaluatorCfg,
    TrajectoryTrackingEvaluator,
    TrajectoryTrackingEvaluatorCfg,
)
from .interfaces import (
    PositionInterface,
    PositionInterfaceCfg,
    ThrustInterface,
    ThrustInterfaceCfg,
    VelocityInterface,
    VelocityInterfaceCfg,
)
from .robustness import RobustnessManager, RobustnessManagerCfg
from .tasks import TargetPositionTask, TargetPositionTaskCfg, TrajectoryTrackingTask, TrajectoryTrackingTaskCfg


@configclass
class AerialBalanceEnvCfg(DirectRLEnvCfg):
    """Configuration for the modular Aerial-Balance-Bench environment."""

    decimation: int = 3
    episode_length_s: float = 10.0
    seed: int = 2025

    # physics_cfg: PhysxCfg = PhysxCfg(
    #     solver_type=0,
    #     min_position_iteration_count=128,
    #     min_velocity_iteration_count=32,
    # )
    physics_cfg: PhysxCfg = PhysxCfg(
        solver_type=0,
        min_position_iteration_count=255,
        min_velocity_iteration_count=255,
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 180,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=physics_cfg,
    )

    action_space = spaces.Box(
        low=np.array([-0.5 * decimation * (1 / 180)], dtype=np.float32),
        high=np.array([0.5 * decimation * (1 / 180)], dtype=np.float32),
        dtype=np.float32,
    )
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

    robot_cfg: ArticulationCfg = DRONE_ROPE_PLANK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drone_bodies: list[str] = ["base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3"]
    holder_bodies: list[str] = ["holder"]
    holder_plank_joints: list[str] = ["holder_plank_joint"]
    slider_holder_joints: list[str] = ["slider_holder_joint"]

    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            visible=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=255,
                solver_velocity_iteration_count=64,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            radius=0.023,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=0,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)

    plank_length: float = 1.06
    plank_slide_length: float = 0.03
    beam_block_offset: float = 0.30
    r_holder: float = 0.02
    rope_length: float = 0.9
    initial_ball_z: float = 2.03
    beam_position_min: float = 0.0
    beam_position_max: float = 0.70
    max_theta: float = 50.0 * np.pi / 180.0

    task_name: str = "target_position"
    target_position_task: TargetPositionTaskCfg = TargetPositionTaskCfg()
    trajectory_tracking_task: TrajectoryTrackingTaskCfg = TrajectoryTrackingTaskCfg()
    interface_name: str = "velocity"
    velocity_interface: VelocityInterfaceCfg = VelocityInterfaceCfg()
    position_interface: PositionInterfaceCfg = PositionInterfaceCfg()
    thrust_interface: ThrustInterfaceCfg = ThrustInterfaceCfg()
    robustness: RobustnessManagerCfg = RobustnessManagerCfg()
    target_position_evaluator: TargetPositionEvaluatorCfg = TargetPositionEvaluatorCfg()
    trajectory_tracking_evaluator: TrajectoryTrackingEvaluatorCfg = TrajectoryTrackingEvaluatorCfg()


class AerialBalanceEnv(DirectRLEnv):
    """Core benchmark environment with pluggable task/interface hooks."""

    cfg: AerialBalanceEnvCfg

    def __init__(self, cfg: AerialBalanceEnvCfg, render_mode: str | None = None, **kwargs):
        self._configure_spaces(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self._find_scene_handles()
        self.task, self.evaluator = self._create_task_and_evaluator(cfg)
        self.control_interface = self._create_control_interface(cfg)
        self.robustness = RobustnessManager(cfg.robustness, self.num_envs, self.device)
        self._allocate_state_buffers()
        self.sim.set_camera_view(eye=[0.5, 3.0, 3.2], target=[0.6, 0.0, 2.3])

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        self.evaluator.reset_all()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_idx(env_ids)
        self.obs_buf = self._get_observations()
        self.extras = {"step": self._step_extras(), "benchmark": self.evaluator.get_metrics()}
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor):
        action = action.to(device=self.device, dtype=torch.float32)
        action = self.robustness.before_action(self, action)
        command = self._pre_physics_step(action)
        self.robustness.after_command_update(self, command)

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.episode_physics_length_buf += 1
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and (
                self.sim.has_gui() or self.sim.has_rtx_sensors()
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self.robustness.after_physics_step(self)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._update_state_from_sim()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reward_buf = self._get_rewards()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        state = self._state_dict()
        benchmark_metrics = self.evaluator.update(
            state,
            self.reset_terminated,
            self.reset_time_outs,
            self.episode_length_buf,
        )
        self.extras = {"step": self._step_extras(), "benchmark": benchmark_metrics}

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            self._reset_idx(reset_env_ids)

        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _setup_scene(self):
        self.drone_rope_plank = Articulation(self.cfg.robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.articulations["drone_rope_plank"] = self.drone_rope_plank
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        super()._reset_idx(env_ids)
        self._reset_robot_state(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()

        self.task.sample_reset(self, env_ids)
        self.control_interface.reset(env_ids, env=self)
        self.robustness.reset(self, env_ids)
        self.episode_physics_length_buf[env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()

        self.initial_drone_z[env_ids] = self._compute_drone_z()[env_ids]
        self._update_state_from_sim(env_ids=env_ids, zero_derivatives=True)
        self.evaluator.reset_episode(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        return self.control_interface.pre_physics_step(actions)

    def _apply_action(self):
        self.control_interface.apply(self)

    def _get_observations(self):
        pg, _ = self.task.get_reference(self.episode_length_buf)
        obs = torch.stack(
            [
                self.pb,
                self.vb,
                self.ab,
                self.theta,
                self.omega,
                self.alpha,
                self.drz,
                self.vrz,
                self.arz,
                pg,
                self.control_interface.last_action.squeeze(-1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        reward_kwargs = {}
        if self.cfg.task_name == "target_position":
            reward_kwargs["terminated"] = self.reset_terminated
        return self.task.compute_reward(
            self._state_dict(),
            self.control_interface.get_command_state(),
            self.control_interface.last_action,
            **reward_kwargs,
        )

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length
        theta_terminal = torch.abs(self.theta) > self.cfg.max_theta
        ball_terminal = (self.pb < self.cfg.beam_position_min) | (self.pb > self.cfg.beam_position_max)
        task_terminal = self.task.compute_task_dones(self._state_dict())
        return theta_terminal | ball_terminal | task_terminal, time_out

    def set_ball_position_along_beam(self, env_ids: torch.Tensor, ball_position: torch.Tensor):
        """Place the ball at a benchmark beam coordinate for selected envs."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ball_position = ball_position.to(device=self.device, dtype=torch.float32)

        holder_pos = self.drone_rope_plank.data.body_pos_w[env_ids, self.holder_body_ids[0]]
        holder_pos = holder_pos - self.scene.env_origins[env_ids]

        beam_coordinate_offset = (
            self.cfg.plank_length
            + self.cfg.r_holder
            - self.cfg.plank_slide_length
            - self.cfg.beam_block_offset
        )
        ball_state = torch.zeros((env_ids.numel(), 13), device=self.device)
        ball_state[:, 0] = holder_pos[:, 0] + beam_coordinate_offset - ball_position
        ball_state[:, 1] = holder_pos[:, 1]
        ball_state[:, 2] = self.cfg.initial_ball_z
        ball_state[:, :3] += self.scene.env_origins[env_ids]
        ball_state[:, 3] = 1.0
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)

    def _configure_spaces(self, cfg: AerialBalanceEnvCfg):
        step_dt = cfg.decimation * cfg.sim.dt
        if cfg.interface_name == "velocity":
            action_limit = cfg.velocity_interface.max_acc * step_dt
        elif cfg.interface_name == "position":
            action_limit = 0.5 * cfg.position_interface.max_acc * step_dt**2
        elif cfg.interface_name == "thrust":
            action_limit = cfg.thrust_interface.max_delta_force
        else:
            raise ValueError(
                f"Unsupported interface_name '{cfg.interface_name}'. "
                "Expected 'velocity', 'position', or 'thrust'."
            )
        cfg.action_space = spaces.Box(
            low=np.array([-action_limit], dtype=np.float32),
            high=np.array([action_limit], dtype=np.float32),
            dtype=np.float32,
        )
        cfg.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        cfg.target_position_evaluator.episode_length_s = cfg.episode_length_s
        cfg.trajectory_tracking_evaluator.episode_length_s = cfg.episode_length_s

    def _create_control_interface(self, cfg: AerialBalanceEnvCfg):
        if cfg.interface_name == "velocity":
            return VelocityInterface(
                cfg.velocity_interface,
                self.num_envs,
                self.device,
                self.step_dt,
                cfg.sim.gravity,
            )
        if cfg.interface_name == "position":
            return PositionInterface(
                cfg.position_interface,
                self.num_envs,
                self.device,
                self.step_dt,
                cfg.sim.gravity,
            )
        if cfg.interface_name == "thrust":
            return ThrustInterface(
                cfg.thrust_interface,
                self.num_envs,
                self.device,
                self.step_dt,
                cfg.sim.gravity,
            )
        raise ValueError(
            f"Unsupported interface_name '{cfg.interface_name}'. "
            "Expected 'velocity', 'position', or 'thrust'."
        )

    def _create_task_and_evaluator(self, cfg: AerialBalanceEnvCfg):
        if cfg.task_name == "target_position":
            return (
                TargetPositionTask(cfg.target_position_task, self.num_envs, self.device),
                TargetPositionEvaluator(cfg.target_position_evaluator, self.num_envs, self.device, self.step_dt),
            )
        if cfg.task_name == "trajectory_tracking":
            return (
                TrajectoryTrackingTask(cfg.trajectory_tracking_task, self.num_envs, self.device, self.step_dt),
                TrajectoryTrackingEvaluator(
                    cfg.trajectory_tracking_evaluator,
                    self.num_envs,
                    self.device,
                    self.step_dt,
                ),
            )
        raise ValueError(
            f"Unsupported task_name '{cfg.task_name}'. "
            "Expected 'target_position' or 'trajectory_tracking'."
        )

    def _find_scene_handles(self):
        self.drone_body_ids = self.drone_rope_plank.find_bodies(self.cfg.drone_bodies)[0]
        self.holder_body_ids = self.drone_rope_plank.find_bodies(self.cfg.holder_bodies)[0]
        self.holder_plank_joint_ids = self.drone_rope_plank.find_joints(self.cfg.holder_plank_joints)[0]
        self.slider_holder_joint_ids = self.drone_rope_plank.find_joints(self.cfg.slider_holder_joints)[0]

    def _allocate_state_buffers(self):
        self.pb = torch.zeros(self.num_envs, device=self.device)
        self.vb = torch.zeros(self.num_envs, device=self.device)
        self.ab = torch.zeros(self.num_envs, device=self.device)
        self.theta = torch.zeros(self.num_envs, device=self.device)
        self.omega = torch.zeros(self.num_envs, device=self.device)
        self.alpha = torch.zeros(self.num_envs, device=self.device)
        self.drz = torch.zeros(self.num_envs, device=self.device)
        self.vrz = torch.zeros(self.num_envs, device=self.device)
        self.arz = torch.zeros(self.num_envs, device=self.device)
        self.initial_drone_z = torch.zeros(self.num_envs, device=self.device)
        self.theta_limit = torch.full((self.num_envs,), self.cfg.max_theta, device=self.device)
        self.episode_physics_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _reset_robot_state(self, env_ids: torch.Tensor):
        joint_pos = self.drone_rope_plank.data.default_joint_pos[env_ids]
        joint_vel = self.drone_rope_plank.data.default_joint_vel[env_ids]
        root_state = self.drone_rope_plank.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.drone_rope_plank.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.drone_rope_plank.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        forces, torques = self.control_interface.initial_forces_and_torques(env_ids)
        self.drone_rope_plank.set_external_force_and_torque(
            forces,
            torques,
            body_ids=self.drone_body_ids,
            env_ids=env_ids,
        )

    def _compute_theta(self) -> torch.Tensor:
        return -self.drone_rope_plank.data.joint_pos[:, self.holder_plank_joint_ids[0]]

    def _compute_ball_position(self, theta: torch.Tensor) -> torch.Tensor:
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        holder_pos = self.drone_rope_plank.data.body_pos_w[:, self.holder_body_ids[0]] - self.scene.env_origins

        rel_x = ball_pos[:, 0] - holder_pos[:, 0]
        rel_z = ball_pos[:, 2] - holder_pos[:, 2]
        ball_position = (
            self.cfg.plank_length
            + self.cfg.r_holder
            - (rel_x / torch.cos(theta))
            - self.cfg.plank_slide_length
            + (rel_z - torch.tan(theta) * rel_x) * torch.sin(theta)
            - self.cfg.beam_block_offset
        )
        return ball_position

    def _compute_drone_z(self) -> torch.Tensor:
        drone_pos = self.drone_rope_plank.data.body_pos_w[:, self.drone_body_ids[0]]
        return drone_pos[:, 2] - self.scene.env_origins[:, 2]

    def _update_state_from_sim(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        zero_derivatives: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        theta = self._compute_theta()
        pb = self._compute_ball_position(theta)
        drone_z = self._compute_drone_z()

        if zero_derivatives:
            self.vb[env_ids] = 0.0
            self.ab[env_ids] = 0.0
            self.omega[env_ids] = 0.0
            self.alpha[env_ids] = 0.0
            self.vrz[env_ids] = 0.0
            self.arz[env_ids] = 0.0
        else:
            new_vb = (pb[env_ids] - self.pb[env_ids]) / self.step_dt

            self.ab[env_ids] = (new_vb - self.vb[env_ids]) / self.step_dt
            self.vb[env_ids] = new_vb
            self.omega[env_ids] = -self.drone_rope_plank.data.joint_vel[:, self.holder_plank_joint_ids[0]]
            self.alpha[env_ids] = -self.drone_rope_plank.data.joint_acc[:, self.holder_plank_joint_ids[0]]
            self.vrz[env_ids] = self.drone_rope_plank.data.body_lin_vel_w[:, self.drone_body_ids[0], 2]
            self.arz[env_ids] = self.drone_rope_plank.data.body_lin_acc_w[:, self.drone_body_ids[0], 2]

        self.pb[env_ids] = pb[env_ids]
        self.theta[env_ids] = theta[env_ids]
        self.drz[env_ids] = drone_z[env_ids] - self.initial_drone_z[env_ids]

    def _state_dict(self) -> dict[str, torch.Tensor]:
        pg, vg = self.task.get_reference(self.episode_length_buf)
        return {
            "pb": self.pb,
            "vb": self.vb,
            "ab": self.ab,
            "theta": self.theta,
            "omega": self.omega,
            "alpha": self.alpha,
            "drz": self.drz,
            "vrz": self.vrz,
            "arz": self.arz,
            "pg": pg,
            "vg": vg,
            "theta_limit": self.theta_limit,
        }

    def _step_extras(self) -> dict[str, torch.Tensor]:
        command = self.control_interface.get_command_state()
        pg, vg = self.task.get_reference(self.episode_length_buf)
        extras = {
            "pb": self.pb,
            "pg": pg,
            "vg": vg,
            "command_z": command["command_z"],
            "executed_command_z": command["executed_command_z"],
            "last_action": command["last_action"].squeeze(-1),
        }
        for key in (
            "vrz_cmd",
            "executed_vrz_cmd",
            "drz_cmd",
            "executed_drz_cmd",
            "target_position_z",
            "frz_cmd",
            "executed_frz_cmd",
            "delta_frz_cmd",
            "hover_force",
            "target_pitch",
            "target_height_acc",
            "beta",
        ):
            if key in command:
                extras[key] = command[key]
        extras.update(self.robustness.get_state())
        if hasattr(self.task, "get_task_info"):
            extras.update(self.task.get_task_info())
        return extras
