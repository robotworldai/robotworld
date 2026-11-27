"""Velocity-command interface for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import yaml
from gymnasium import spaces
from omni.isaac.lab.utils import configclass

try:
    from ...utils.drone_models import PropulsorModel, VelocityController
    from ...utils.paths import resource_path
except ImportError:  # Allows importing this module as top-level ``environments``.
    from utils.drone_models import PropulsorModel, VelocityController
    from utils.paths import resource_path


@configclass
class VelocityInterfaceCfg:
    """Configuration for the velocity-command abstraction."""

    max_acc: float = 0.5
    max_velocity: float = 0.0
    uav_params_path: str = str(resource_path("robots", "hummingbird.yaml"))
    controller_params_path: str = str(resource_path("robots", "hummingbird_controller_params_vel.yaml"))

    init_throttle_min: float = 0.541
    init_throttle_max: float = 0.542
    init_single_propeller_force_min: float = 1.760
    init_single_propeller_force_max: float = 1.762
    init_single_propeller_torque_min: tuple[float, float, float, float] = (-0.0283, 0.0281, -0.0283, 0.0281)
    init_single_propeller_torque_max: tuple[float, float, float, float] = (-0.0281, 0.0283, -0.0281, 0.0283)
    propulsor_throttle_noise_scale: float = 0.0


class VelocityInterface:
    """Incremental Z-velocity command interface."""

    def __init__(
        self,
        cfg: VelocityInterfaceCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        gravity: Sequence[float],
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt

        self.delta_velocity_limit = cfg.max_acc * step_dt
        self.action_space = spaces.Box(
            low=np.array([-self.delta_velocity_limit], dtype=np.float32),
            high=np.array([self.delta_velocity_limit], dtype=np.float32),
            dtype=np.float32,
        )

        self.command_velocity = torch.zeros((num_envs, 3), device=self.device)
        self.last_action = torch.zeros((num_envs, 1), device=self.device)
        self.executed_velocity = torch.zeros((num_envs, 3), device=self.device)

        self.uav_params = self._load_yaml(cfg.uav_params_path)
        self.controller_params = self._load_yaml(cfg.controller_params_path)
        self.rotor_model = PropulsorModel(
            self.uav_params["rotor_configuration"],
            num_envs,
            noise_scale=cfg.propulsor_throttle_noise_scale,
            random_propulsor_tau=False,
        ).to(device=self.device)
        self.velocity_controller = VelocityController(
            g=torch.as_tensor(gravity, dtype=torch.float32).abs(),
            uav_params=self.uav_params,
            controller_params=self.controller_params,
            n_envs=num_envs,
            target_value_compute=False,
            random_sample_vel_gain=False,
            random_mass=False,
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

        self.command_velocity[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.executed_velocity[env_ids] = 0.0

        init_throttles = torch.rand((env_ids.numel(), 4), device=self.device)
        init_throttles = init_throttles * (self.cfg.init_throttle_max - self.cfg.init_throttle_min)
        init_throttles = init_throttles + self.cfg.init_throttle_min
        self.rotor_model.reset(init_throttles=init_throttles, env_ids=env_ids)
        self.velocity_controller.reset(env_ids=env_ids)

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
        """Convert a velocity increment action into a clipped command."""
        action = action.to(device=self.device, dtype=torch.float32)
        clipped_action = torch.clamp(action, -self.delta_velocity_limit, self.delta_velocity_limit)
        self.last_action = clipped_action.clone()

        next_vrz = self.command_velocity[:, 2:3] + clipped_action
        if self.cfg.max_velocity > 0.0:
            next_vrz = torch.clamp(next_vrz, -self.cfg.max_velocity, self.cfg.max_velocity)
        self.command_velocity[:, 2:3] = next_vrz
        self.executed_velocity = self.command_velocity.clone()
        return self.get_command_state()

    def apply(self, env):
        """Apply the current velocity command through the low-level drone model."""
        target_yaw = torch.zeros(self.num_envs, device=self.device)
        drone_state_w = env.drone_rope_plank.data.body_state_w[:, env.drone_body_ids[0]]
        throttle_cmds = self.velocity_controller(
            (env.episode_physics_length_buf + 1) * env.physics_dt,
            drone_state_w,
            target_yaw,
            self.executed_velocity,
        )
        forces, torques = self._forces_and_torques_from_throttle(throttle_cmds)
        env.drone_rope_plank.set_external_force_and_torque(forces, torques, body_ids=env.drone_body_ids)

    def get_command_state(self) -> dict[str, torch.Tensor]:
        """Return interface-specific command state."""
        return {
            "target_velocity": self.command_velocity,
            "executed_velocity": self.executed_velocity,
            "command_z": self.command_velocity[:, 2],
            "executed_command_z": self.executed_velocity[:, 2],
            "vrz_cmd": self.command_velocity[:, 2],
            "executed_vrz_cmd": self.executed_velocity[:, 2],
            "last_action": self.last_action,
        }

    def get_delay_command(self) -> torch.Tensor:
        """Return the full command tensor used by command-level action delay."""
        return self.command_velocity

    def set_executed_delay_command(self, command: torch.Tensor):
        """Set the delayed command that will be executed by the low-level controller."""
        self.executed_velocity = command.to(device=self.device, dtype=torch.float32).clone()

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
