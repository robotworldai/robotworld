"""Vertical-thrust command interface for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
import yaml
from gymnasium import spaces
from omni.isaac.lab.utils import configclass

try:
    from ...utils.drone_models import AttitudeController, PropulsorModel
    from ...utils.paths import resource_path
except ImportError:  # Allows importing this module as top-level ``environments``.
    from utils.drone_models import AttitudeController, PropulsorModel
    from utils.paths import resource_path


@configclass
class ThrustInterfaceCfg:
    """Configuration for the vertical-thrust abstraction."""

    max_delta_force: float = 0.25
    max_pitch: float = 35.0 * math.pi / 180.0
    rope_length: float = 0.9
    uav_params_path: str = str(resource_path("robots", "hummingbird.yaml"))
    controller_params_path: str = str(resource_path("robots", "hummingbird_controller_params_attitude.yaml"))

    init_throttle_min: float = 0.541
    init_throttle_max: float = 0.542
    init_single_propeller_force_min: float = 1.760
    init_single_propeller_force_max: float = 1.762
    init_single_propeller_torque_min: tuple[float, float, float, float] = (-0.0283, 0.0281, -0.0283, 0.0281)
    init_single_propeller_torque_max: tuple[float, float, float, float] = (-0.0281, 0.0283, -0.0281, 0.0283)
    propulsor_throttle_noise_scale: float = 0.0
    eps: float = 1e-6


class ThrustInterface:
    """Incremental vertical thrust command interface."""

    def __init__(
        self,
        cfg: ThrustInterfaceCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        gravity: Sequence[float],
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt

        self.action_space = spaces.Box(
            low=np.array([-cfg.max_delta_force], dtype=np.float32),
            high=np.array([cfg.max_delta_force], dtype=np.float32),
            dtype=np.float32,
        )

        self.uav_params = self._load_yaml(cfg.uav_params_path)
        self.controller_params = self._load_yaml(cfg.controller_params_path)
        self.mass = torch.tensor(float(self.uav_params["mass"]), device=self.device)
        self.gravity_z = torch.tensor(float(abs(gravity[2])), device=self.device)
        self.hover_force = self.mass * self.gravity_z

        self.command_force_z = torch.full((num_envs,), self.hover_force.item(), device=self.device)
        self.executed_force_z = torch.full((num_envs,), self.hover_force.item(), device=self.device)
        self.target_pitch = torch.zeros(num_envs, device=self.device)
        self.target_height_acc = torch.zeros(num_envs, device=self.device)
        self.geometric_height_acc = torch.zeros(num_envs, device=self.device)
        self.beta = torch.zeros(num_envs, device=self.device)
        self.last_action = torch.zeros((num_envs, 1), device=self.device)

        self.rotor_model = PropulsorModel(
            self.uav_params["rotor_configuration"],
            num_envs,
            noise_scale=cfg.propulsor_throttle_noise_scale,
            random_propulsor_tau=False,
        ).to(device=self.device)
        self.attitude_controller = AttitudeController(
            g=torch.as_tensor(gravity, dtype=torch.float32).abs(),
            uav_params=self.uav_params,
            controller_params=self.controller_params,
            n_envs=num_envs,
            target_value_compute=False,
        ).to(device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None, env=None):
        """Reset command state and low-level model state."""
        del env
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.command_force_z[env_ids] = self.hover_force
        self.executed_force_z[env_ids] = self.hover_force
        self.target_pitch[env_ids] = 0.0
        self.target_height_acc[env_ids] = 0.0
        self.geometric_height_acc[env_ids] = 0.0
        self.beta[env_ids] = 0.0
        self.last_action[env_ids] = 0.0

        init_throttles = torch.rand((env_ids.numel(), 4), device=self.device)
        init_throttles = init_throttles * (self.cfg.init_throttle_max - self.cfg.init_throttle_min)
        init_throttles = init_throttles + self.cfg.init_throttle_min
        self.rotor_model.reset(init_throttles=init_throttles, env_ids=env_ids)
        self.attitude_controller.reset(env_ids=env_ids)

    def initial_forces_and_torques(self, env_ids: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial rotor forces and torques matching the old hover reset."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_reset_envs = env_ids.numel()

        forces = torch.zeros((num_reset_envs, 5, 3), device=self.device)
        forces[:, 1:, 2] = torch.rand((num_reset_envs, 4), device=self.device)
        forces[:, 1:, 2] *= self.cfg.init_single_propeller_force_max - self.cfg.init_single_propeller_force_min
        forces[:, 1:, 2] += self.cfg.init_single_propeller_force_min

        torque_low = torch.as_tensor(self.cfg.init_single_propeller_torque_min, device=self.device)
        torque_high = torch.as_tensor(self.cfg.init_single_propeller_torque_max, device=self.device)
        torques = torch.zeros_like(forces)
        torques[:, 1:, 2] = torch.rand((num_reset_envs, 4), device=self.device)
        torques[:, 1:, 2] *= torque_high - torque_low
        torques[:, 1:, 2] += torque_low
        torques[:, 0, 2] = torques[:, 1:, 2].sum(dim=-1)
        return forces, torques

    def pre_physics_step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a force increment action into a vertical thrust command."""
        action = action.to(device=self.device, dtype=torch.float32)
        clipped_action = torch.clamp(action, -self.cfg.max_delta_force, self.cfg.max_delta_force)
        self.last_action = clipped_action.clone()

        self.command_force_z = self.command_force_z + clipped_action.squeeze(-1)
        self.executed_force_z = self.command_force_z.clone()
        return self.get_command_state()

    def apply(self, env):
        """Apply the current vertical thrust command through the attitude controller."""
        theta = env.theta
        omega = env.omega
        alpha = env.alpha
        self.beta, self.geometric_height_acc = self._compute_geometric_vertical_acceleration(
            theta,
            omega,
            alpha,
            beam_length=env.cfg.plank_length,
            rope_length=getattr(env.cfg, "rope_length", self.cfg.rope_length),
        )
        self.target_pitch = self._compute_target_pitch(
            self.executed_force_z,
            self.beta,
            self.geometric_height_acc,
        )
        self.target_height_acc = self.executed_force_z / self.mass - self.gravity_z

        drone_state_w = env.drone_rope_plank.data.body_state_w[:, env.drone_body_ids[0]]
        target_roll = torch.zeros(self.num_envs, device=self.device)
        target_yaw = torch.zeros(self.num_envs, device=self.device)
        # target_height = drone_state_w[:, 2]
        # target_height_vel = drone_state_w[:, 9]
        throttle_cmds = self.attitude_controller(
            (env.episode_physics_length_buf + 1) * env.physics_dt,
            drone_state_w,
            target_roll,
            self.target_pitch,
            target_yaw,
            None,
            None,
            self.target_height_acc,
        )
        forces, torques = self._forces_and_torques_from_throttle(throttle_cmds)
        env.drone_rope_plank.set_external_force_and_torque(forces, torques, body_ids=env.drone_body_ids)

    def get_command_state(self) -> dict[str, torch.Tensor]:
        """Return interface-specific command state."""
        force_error = self.command_force_z - self.hover_force
        executed_force_error = self.executed_force_z - self.hover_force
        return {
            "command_z": force_error,
            "executed_command_z": executed_force_error,
            "frz_cmd": self.command_force_z,
            "executed_frz_cmd": self.executed_force_z,
            "delta_frz_cmd": force_error,
            "hover_force": self.hover_force.repeat(self.num_envs),
            "target_pitch": self.target_pitch,
            "target_height_acc": self.target_height_acc,
            "geometric_height_acc": self.geometric_height_acc,
            "beta": self.beta,
            "last_action": self.last_action,
        }

    def get_delay_command(self) -> torch.Tensor:
        """Return the full command tensor used by command-level action delay."""
        return self.command_force_z.unsqueeze(-1)

    def set_executed_delay_command(self, command: torch.Tensor):
        """Set the delayed command that will be executed by the low-level controller."""
        self.executed_force_z = command.to(device=self.device, dtype=torch.float32).squeeze(-1).clone()

    def _compute_target_pitch(
        self,
        force_z: torch.Tensor,
        beta: torch.Tensor,
        geometric_height_acc: torch.Tensor,
    ) -> torch.Tensor:
        denominator = torch.where(
            force_z.abs() < self.cfg.eps,
            torch.full_like(force_z, self.cfg.eps),
            force_z,
        )
        tan_pitch = ((force_z - self.mass * (geometric_height_acc + self.gravity_z)) * torch.tan(beta)) / denominator
        pitch = torch.atan(tan_pitch)
        return torch.clamp(pitch, -self.cfg.max_pitch, self.cfg.max_pitch)

    def _compute_geometric_vertical_acceleration(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        alpha: torch.Tensor,
        beam_length: float,
        rope_length: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beam_length_t = torch.tensor(float(beam_length), device=self.device, dtype=theta.dtype)
        rope_length_t = torch.tensor(float(rope_length), device=self.device, dtype=theta.dtype)
        ratio = beam_length_t / rope_length_t

        sin_beta = ratio * (1.0 - torch.cos(theta))
        sin_beta = torch.clamp(sin_beta, -1.0 + self.cfg.eps, 1.0 - self.cfg.eps)
        beta = torch.asin(sin_beta)
        cos_beta = torch.clamp(torch.cos(beta), min=self.cfg.eps)

        sin_beta_dot = ratio * torch.sin(theta) * omega
        sin_beta_ddot = ratio * (torch.cos(theta) * omega.square() + torch.sin(theta) * alpha)
        beta_dot = sin_beta_dot / cos_beta
        beta_ddot = sin_beta_ddot / cos_beta + sin_beta * sin_beta_dot.square() / cos_beta.pow(3)

        geometric_height_acc = -rope_length_t * (
            torch.cos(beta) * beta_dot.square() + torch.sin(beta) * beta_ddot
        )
        geometric_height_acc += beam_length_t * (torch.sin(theta) * omega.square() - torch.cos(theta) * alpha)
        return beta, geometric_height_acc

    def _forces_and_torques_from_throttle(self, throttle_cmds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        thrusts, moments, _ = self.rotor_model(throttle_cmds)
        forces = torch.zeros((self.num_envs, 5, 3), device=self.device)
        forces[:, 1:, 2] = thrusts

        torques = torch.zeros_like(forces)
        torques[:, 0, 2] = moments.sum(dim=-1)
        torques[:, 1:, 2] = moments
        return forces, torques

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, encoding="utf-8") as stream:
            return yaml.safe_load(stream)
