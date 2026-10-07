"""Trajectory-tracking task for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from omni.isaac.lab.utils import configclass


TRAJECTORY_TYPE_TO_ID = {
    "sine": 0,
    "triangle": 1,
    "trapezoid": 2,
}


@configclass
class TrajectoryTrackingTaskCfg:
    """Configuration for trajectory tracking."""

    center_position: float = 0.35
    initial_ball_position: float = 0.35
    trajectory_type: str = "sine"
    amplitude: float = 0.20
    period: float = 8.0
    amplitude_range: tuple[float, float] = (0.05, 0.25)
    period_range: tuple[float, float] = (6.0, 10.0)
    random_amplitude: bool = False
    random_period: bool = False

    position_weight: float = 5.0
    velocity_weight: float = 0.5
    command_weight: float = 0.5
    action_weight: float = 1.0
    failure_penalty: float = 500.0
    progress_weight: float = 1.0
    max_error_for_failure: float = 0.5


class TrajectoryTrackingTask:
    """Time-varying reference tracking task for the rolling ball."""

    def __init__(
        self,
        cfg: TrajectoryTrackingTaskCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt

        self.trajectory_type_id = torch.full(
            (num_envs,),
            self._trajectory_type_to_id(cfg.trajectory_type),
            dtype=torch.long,
            device=self.device,
        )
        self.amplitude = torch.full((num_envs,), cfg.amplitude, device=self.device)
        self.period = torch.full((num_envs,), cfg.period, device=self.device)
        self.previous_abs_error = torch.zeros(num_envs, device=self.device)

    def sample_reset(self, env, env_ids: Sequence[int] | torch.Tensor):
        """Sample trajectory parameters and reset the ball at the center."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.trajectory_type_id[env_ids] = self._sample_trajectory_type(env_ids.numel())
        self.amplitude[env_ids] = self._sample_parameter(
            fixed_value=self.cfg.amplitude,
            value_range=self.cfg.amplitude_range,
            sample_random=self.cfg.random_amplitude or self.cfg.trajectory_type == "random",
            size=env_ids.numel(),
        )
        self.period[env_ids] = self._sample_parameter(
            fixed_value=self.cfg.period,
            value_range=self.cfg.period_range,
            sample_random=self.cfg.random_period or self.cfg.trajectory_type == "random",
            size=env_ids.numel(),
        )

        initial_position = torch.full((env_ids.numel(),), self.cfg.initial_ball_position, device=self.device)
        env.set_ball_position_along_beam(env_ids, initial_position)

        pg, _ = self.get_reference(env.episode_length_buf)
        self.previous_abs_error[env_ids] = torch.abs(initial_position - pg[env_ids])

    def get_reference(self, step: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return desired ball position and finite-difference reference velocity."""
        if step is None:
            step = torch.zeros(self.num_envs, device=self.device)
        step = torch.as_tensor(step, device=self.device, dtype=torch.float32)
        if step.ndim == 0:
            step = step.repeat(self.num_envs)
        t = step * self.step_dt
        pg = self._reference_position(t)
        pg_next = self._reference_position(t + self.step_dt)
        vg = (pg_next - pg) / self.step_dt
        return pg, vg

    def compute_reward(self, state: dict[str, torch.Tensor], command: dict[str, torch.Tensor], action: torch.Tensor):
        """Compute the trajectory-tracking reward."""
        error = state["pb"] - state["pg"]
        velocity_error = state["vb"] - state["vg"]
        theta = state["theta"]
        command_z = command["command_z"]
        action_z = action.squeeze(-1)
        abs_error = torch.abs(error)

        object_reward = -self.cfg.position_weight * error.square()
        object_reward -= self.cfg.velocity_weight * velocity_error.square()

        control_reward = -self.cfg.command_weight * command_z.square()
        control_reward -= self.cfg.action_weight * action_z.square()

        failure_mask = (torch.abs(theta) > state["theta_limit"]) | (abs_error > self.cfg.max_error_for_failure)
        failure_reward = -self.cfg.failure_penalty * failure_mask.float()

        progress_reward = self.cfg.progress_weight * (self.previous_abs_error - abs_error)
        self.previous_abs_error = abs_error.detach().clone()

        return object_reward + control_reward + failure_reward + progress_reward

    def compute_task_dones(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return task-specific failure flags."""
        return torch.abs(state["pb"] - state["pg"]) > self.cfg.max_error_for_failure

    def get_task_info(self) -> dict[str, torch.Tensor]:
        """Return trajectory parameters for logging."""
        return {
            "trajectory_type_id": self.trajectory_type_id,
            "trajectory_amplitude": self.amplitude,
            "trajectory_period": self.period,
        }

    def _reference_position(self, t: torch.Tensor) -> torch.Tensor:
        wave = torch.zeros_like(t)
        sine_mask = self.trajectory_type_id == TRAJECTORY_TYPE_TO_ID["sine"]
        triangle_mask = self.trajectory_type_id == TRAJECTORY_TYPE_TO_ID["triangle"]
        trapezoid_mask = self.trajectory_type_id == TRAJECTORY_TYPE_TO_ID["trapezoid"]

        phase = 2.0 * math.pi * t / self.period
        if torch.any(sine_mask):
            wave[sine_mask] = self.amplitude[sine_mask] * torch.sin(phase[sine_mask])
        if torch.any(triangle_mask):
            triangle = self._triangle_wave(phase[triangle_mask])
            wave[triangle_mask] = self.amplitude[triangle_mask] * triangle
        if torch.any(trapezoid_mask):
            triangle = self._triangle_wave(phase[trapezoid_mask])
            trapezoid = torch.clamp(triangle * 2.0, -1.0, 1.0)
            wave[trapezoid_mask] = self.amplitude[trapezoid_mask] * trapezoid
        return self.cfg.center_position + wave

    def _sample_trajectory_type(self, size: int) -> torch.Tensor:
        if self.cfg.trajectory_type == "random":
            return torch.randint(0, len(TRAJECTORY_TYPE_TO_ID), (size,), device=self.device)
        trajectory_id = self._trajectory_type_to_id(self.cfg.trajectory_type)
        return torch.full((size,), trajectory_id, dtype=torch.long, device=self.device)

    def _sample_parameter(
        self,
        fixed_value: float,
        value_range: tuple[float, float],
        sample_random: bool,
        size: int,
    ) -> torch.Tensor:
        if sample_random:
            low, high = value_range
            return torch.rand(size, device=self.device) * (high - low) + low
        return torch.full((size,), fixed_value, device=self.device)

    @staticmethod
    def _trajectory_type_to_id(trajectory_type: str) -> int:
        if trajectory_type == "random":
            return TRAJECTORY_TYPE_TO_ID["sine"]
        if trajectory_type not in TRAJECTORY_TYPE_TO_ID:
            raise ValueError(
                f"Unsupported trajectory_type '{trajectory_type}'. "
                f"Expected one of {sorted(TRAJECTORY_TYPE_TO_ID)} or 'random'."
            )
        return TRAJECTORY_TYPE_TO_ID[trajectory_type]

    @staticmethod
    def _triangle_wave(phase: torch.Tensor) -> torch.Tensor:
        shifted_phase = torch.remainder(phase + math.pi / 2.0, 2.0 * math.pi)
        rising = shifted_phase < math.pi
        return torch.where(
            rising,
            2.0 * shifted_phase / math.pi - 1.0,
            3.0 - 2.0 * shifted_phase / math.pi,
        )
