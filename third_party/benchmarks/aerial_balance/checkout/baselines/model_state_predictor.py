"""Policy-side model state predictors for action-delay compensation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


@dataclass
class VelocityModelStatePredictorCfg:
    """Configuration for the velocity-interface model state predictor."""

    enabled: bool = False
    delay_step: int | str = 0
    solver: str = "rk4"
    step_dt: float | str = 0.0
    max_acc: float | str = 0.5
    max_velocity: float | str = 0.0
    plank_length: float | str = 1.06
    rope_length: float | str = 0.9
    gravity: float | str = 9.81
    ball_mass: float | str = 0.0005
    ball_radius: float | str = 0.023
    ball_inertia_ratio: float = 0.4
    epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "VelocityModelStatePredictorCfg":
        """Build a predictor config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


class VelocityModelStatePredictor:
    """Predict the future 11-D benchmark observation under velocity commands."""

    def __init__(self, cfg: VelocityModelStatePredictorCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.enabled = bool(cfg.enabled)

        self.delay_step = self._as_int(cfg.delay_step, "delay_step")
        self.step_dt = self._as_float(cfg.step_dt, "step_dt")
        self.max_acc = self._as_float(cfg.max_acc, "max_acc")
        self.max_velocity = self._as_float(cfg.max_velocity, "max_velocity")
        self.plank_length = self._as_float(cfg.plank_length, "plank_length")
        self.rope_length = self._as_float(cfg.rope_length, "rope_length")
        self.gravity = abs(self._as_float(cfg.gravity, "gravity"))
        self.ball_mass = self._as_float(cfg.ball_mass, "ball_mass")
        self.ball_radius = self._as_float(cfg.ball_radius, "ball_radius")
        self.ball_inertia_ratio = float(cfg.ball_inertia_ratio)
        self.epsilon = float(cfg.epsilon)

        if self.step_dt <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires step_dt > 0.")
        if self.max_acc <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires max_acc > 0.")
        if self.plank_length <= 0.0 or self.rope_length <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive plank_length and rope_length.")
        if self.ball_mass <= 0.0 or self.ball_radius <= 0.0:
            raise ValueError("VelocityModelStatePredictor requires positive ball_mass and ball_radius.")
        if self.ball_inertia_ratio < 0.0:
            raise ValueError("VelocityModelStatePredictor requires non-negative ball_inertia_ratio.")
        if self.solver not in {"euler", "rk4"}:
            raise ValueError("VelocityModelStatePredictor solver must be 'euler' or 'rk4'.")

        self.command_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_observation = torch.zeros((self.num_envs, 11), device=self.device)
        self.predicted_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.predicted_error_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.read_index = 0
        if self.delay_step > 0:
            self.command_queue = torch.zeros((self.delay_step, self.num_envs, 1), device=self.device)
        else:
            self.command_queue = torch.zeros((0, self.num_envs, 1), device=self.device)

    @property
    def solver(self) -> str:
        return str(self.cfg.solver).lower()

    @property
    def active(self) -> bool:
        return self.enabled and self.delay_step > 0

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset command and queue state for all envs or a selected subset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.command_z[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.predicted_observation[env_ids] = 0.0
        self.predicted_error[env_ids] = 0.0
        self.predicted_error_dot[env_ids] = 0.0
        self.predicted_error_ddot[env_ids] = 0.0
        if self.delay_step > 0:
            self.command_queue[:, env_ids] = 0.0

    def predict(self, observation: torch.Tensor, error_prev1: torch.Tensor | None = None) -> torch.Tensor:
        """Return a future 11-D observation after the configured delay horizon."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                "VelocityModelStatePredictor expects observation shape "
                f"({self.num_envs}, 11), got {tuple(observation.shape)}."
            )
        current_error = (
            observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
            - observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        )
        if not self.active:
            self.predicted_observation.copy_(observation)
            self.predicted_error.copy_(current_error)
            self.predicted_error_dot.zero_()
            self.predicted_error_ddot.zero_()
            return observation

        predicted = observation.clone()
        commands = self._pending_commands()
        pb = predicted[:, ObservationIndex.PB].clone()
        vb = predicted[:, ObservationIndex.VB].clone()
        theta = predicted[:, ObservationIndex.THETA].clone()
        drz = predicted[:, ObservationIndex.DRZ].clone()
        omega_prev = predicted[:, ObservationIndex.OMEGA].clone()
        vrz_prev = predicted[:, ObservationIndex.VRZ].clone()

        error_curr = current_error[:, 0].clone()
        error_prev = self._prepare_error_prev1(error_prev1, current_error)[:, 0].clone()
        predicted_error_dot = torch.zeros_like(error_curr)
        predicted_error_ddot = torch.zeros_like(error_curr)
        pg = observation[:, ObservationIndex.PG].clone()

        ab = predicted[:, ObservationIndex.AB].clone()
        omega = omega_prev.clone()
        alpha = predicted[:, ObservationIndex.ALPHA].clone()
        vrz = vrz_prev.clone()
        arz = predicted[:, ObservationIndex.ARZ].clone()

        for step_id in range(commands.shape[0]):
            vrz = commands[step_id, :, 0]
            pb, vb, theta, ab, omega = self._integrate_one_step(pb, vb, theta, vrz)
            alpha = (omega - omega_prev) / self.step_dt
            arz = (vrz - vrz_prev) / self.step_dt
            drz = drz + vrz * self.step_dt
            omega_prev = omega
            vrz_prev = vrz
            new_error = pb - pg
            predicted_error_dot = new_error - error_curr
            predicted_error_ddot = predicted_error_dot - (error_curr - error_prev)
            error_prev = error_curr
            error_curr = new_error

        predicted[:, ObservationIndex.PB] = pb
        predicted[:, ObservationIndex.VB] = vb
        predicted[:, ObservationIndex.AB] = ab
        predicted[:, ObservationIndex.THETA] = theta
        predicted[:, ObservationIndex.OMEGA] = omega
        predicted[:, ObservationIndex.ALPHA] = alpha
        predicted[:, ObservationIndex.DRZ] = drz
        predicted[:, ObservationIndex.VRZ] = vrz
        predicted[:, ObservationIndex.ARZ] = arz
        predicted[:, ObservationIndex.PG] = observation[:, ObservationIndex.PG]
        predicted[:, ObservationIndex.A_PREV] = observation[:, ObservationIndex.A_PREV]

        self.predicted_observation.copy_(predicted)
        self.predicted_error.copy_(error_curr.unsqueeze(-1))
        self.predicted_error_dot.copy_(predicted_error_dot.unsqueeze(-1))
        self.predicted_error_ddot.copy_(predicted_error_ddot.unsqueeze(-1))
        return predicted

    def get_error_prediction(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted error and its horizon-local finite differences."""
        return self.predicted_error, self.predicted_error_dot, self.predicted_error_ddot

    def update_after_action(self, action: torch.Tensor):
        """Mirror VelocityInterface command accumulation and enqueue the command."""
        action = action.to(device=self.device, dtype=torch.float32)
        clipped_action = torch.clamp(action, -self.max_acc * self.step_dt, self.max_acc * self.step_dt)
        self.last_action.copy_(clipped_action)
        self.command_z += clipped_action
        if self.max_velocity > 0.0:
            self.command_z.clamp_(min=-self.max_velocity, max=self.max_velocity)

        if self.delay_step <= 0:
            return
        self.command_queue[self.read_index] = self.command_z
        self.read_index = (self.read_index + 1) % self.delay_step

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return predictor state for rollout diagnostics."""
        state = {
            "policy_predictor_enabled": torch.full(
                (self.num_envs,),
                float(self.active),
                device=self.device,
            ),
            "policy_predictor_command_z": self.command_z[:, 0],
            "policy_predictor_error": self.predicted_error[:, 0],
            "policy_predictor_error_dot": self.predicted_error_dot[:, 0],
            "policy_predictor_error_ddot": self.predicted_error_ddot[:, 0],
        }
        names = (
            "pb",
            "vb",
            "ab",
            "theta",
            "omega",
            "alpha",
            "drz",
            "vrz",
            "arz",
            "pg",
            "a_prev",
        )
        for index, name in enumerate(names):
            state[f"policy_predicted_{name}"] = self.predicted_observation[:, index]
        return state

    def to(self, device: str | torch.device):
        """Move predictor buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self

    def _pending_commands(self) -> torch.Tensor:
        if self.delay_step <= 0:
            return self.command_queue
        ordered_ids = (torch.arange(self.delay_step, device=self.device) + self.read_index) % self.delay_step
        return self.command_queue[ordered_ids].clone()

    def _prepare_error_prev1(self, error_prev1: torch.Tensor | None, current_error: torch.Tensor) -> torch.Tensor:
        if error_prev1 is None:
            return current_error.clone()
        error_prev1 = error_prev1.to(device=self.device, dtype=torch.float32)
        if error_prev1.shape == (self.num_envs,):
            error_prev1 = error_prev1.unsqueeze(-1)
        if error_prev1.shape != (self.num_envs, 1):
            raise ValueError(
                "VelocityModelStatePredictor expects error_prev1 shape "
                f"({self.num_envs}, 1), got {tuple(error_prev1.shape)}."
            )
        return error_prev1

    def _integrate_one_step(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        vrz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.solver == "euler":
            derivatives = self._derivatives(pb, vb, theta, vrz)
            pb_next = pb + derivatives[:, 0] * self.step_dt
            vb_next = vb + derivatives[:, 1] * self.step_dt
            theta_next = theta + derivatives[:, 2] * self.step_dt
        else:
            state = torch.stack((pb, vb, theta), dim=-1)
            k1 = self._derivatives_from_state(state, vrz)
            k2 = self._derivatives_from_state(state + 0.5 * self.step_dt * k1, vrz)
            k3 = self._derivatives_from_state(state + 0.5 * self.step_dt * k2, vrz)
            k4 = self._derivatives_from_state(state + self.step_dt * k3, vrz)
            next_state = state + (self.step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pb_next = next_state[:, 0]
            vb_next = next_state[:, 1]
            theta_next = next_state[:, 2]

        final_derivatives = self._derivatives(pb_next, vb_next, theta_next, vrz)
        ab_next = final_derivatives[:, 1]
        omega_next = final_derivatives[:, 2]
        return pb_next, vb_next, theta_next, ab_next, omega_next

    def _derivatives_from_state(self, state: torch.Tensor, vrz: torch.Tensor) -> torch.Tensor:
        return self._derivatives(state[:, 0], state[:, 1], state[:, 2], vrz)

    def _derivatives(
        self,
        pb: torch.Tensor,
        vb: torch.Tensor,
        theta: torch.Tensor,
        vrz: torch.Tensor,
    ) -> torch.Tensor:
        beta = self._beta(theta)
        denominator = self.plank_length * self._safe_denominator(torch.cos(beta - theta))
        omega = -vrz * torch.cos(beta) / denominator
        ball_inertia = self.ball_inertia_ratio * self.ball_mass * self.ball_radius**2
        effective_mass = ball_inertia / self.ball_radius**2 + self.ball_mass
        ab = self.ball_mass * (pb + 0.33 - self.plank_length) * omega.square()
        ab -= self.ball_mass * self.gravity * torch.sin(theta)
        ab = ab / effective_mass
        return torch.stack((vb, ab, omega), dim=-1)

    def _beta(self, theta: torch.Tensor) -> torch.Tensor:
        sin_beta = (self.plank_length / self.rope_length) * (1.0 - torch.cos(theta))
        sin_beta = torch.clamp(sin_beta, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
        return torch.asin(sin_beta)

    def _safe_denominator(self, value: torch.Tensor) -> torch.Tensor:
        sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
        return torch.where(torch.abs(value) < self.epsilon, sign * self.epsilon, value)

    @staticmethod
    def _as_float(value, field_name: str) -> float:
        if isinstance(value, str):
            if value.lower() == "auto":
                raise ValueError(f"VelocityModelStatePredictorCfg.{field_name} must be resolved before use.")
            return float(value)
        return float(value)

    @staticmethod
    def _as_int(value, field_name: str) -> int:
        if isinstance(value, str):
            if value.lower() == "auto":
                raise ValueError(f"VelocityModelStatePredictorCfg.{field_name} must be resolved before use.")
            return int(value)
        return int(value)
