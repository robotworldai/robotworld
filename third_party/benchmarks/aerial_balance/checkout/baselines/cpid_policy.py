"""Cascaded incremental PID baseline for target-position balancing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex
from .model_state_predictor import VelocityModelStatePredictor, VelocityModelStatePredictorCfg


@dataclass
class AnglePIDCfg:
    """Outer-loop angle PID configuration."""

    kp: float = 0.5
    ti: float = 1.5
    td: float = 0.6
    max_theta_change: float = 0.1 / 180.0 * math.pi


@dataclass
class VelocityPIDCfg:
    """Inner-loop vertical velocity PID configuration."""

    kp: float = 10.0
    ti: float = 1000.0
    td: float = 0.0
    max_acc: float = 0.5


@dataclass
class CPIDPolicyCfg(BasePolicyCfg):
    """Configuration for the CPID high-level baseline."""

    name: str = "cpid"
    angle_pid: AnglePIDCfg = field(default_factory=AnglePIDCfg)
    velocity_pid: VelocityPIDCfg = field(default_factory=VelocityPIDCfg)
    state_predictor: VelocityModelStatePredictorCfg = field(default_factory=VelocityModelStatePredictorCfg)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "CPIDPolicyCfg":
        """Build a CPID config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg

        policy_data = data.get("cpid_policy", data.get("policy", data))
        if "name" in policy_data:
            cfg.name = str(policy_data["name"])
        if "policy_name" in policy_data:
            cfg.name = str(policy_data["policy_name"])

        angle_data = policy_data.get("angle_pid", policy_data.get("angle_pid_config", {}))
        velocity_data = policy_data.get("velocity_pid", policy_data.get("vel_pid_config", {}))
        predictor_data = policy_data.get("state_predictor", {})
        _update_pid_cfg(cfg.angle_pid, angle_data)
        _update_pid_cfg(cfg.velocity_pid, velocity_data)
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(predictor_data)
        return cfg


class CPIDPolicy(BasePolicy):
    """Cascaded incremental PID policy for the benchmark velocity interface."""

    cfg: CPIDPolicyCfg

    def __init__(
        self,
        cfg: CPIDPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        super().__init__(cfg, num_envs, device)
        if step_dt <= 0.0:
            raise ValueError("CPIDPolicy requires a positive step_dt.")
        self.step_dt = float(step_dt)
        self.max_vel_change = float(cfg.velocity_pid.max_acc) * self.step_dt
        self._resolve_predictor_cfg_defaults()
        self.state_predictor = VelocityModelStatePredictor(cfg.state_predictor, num_envs, self.device)

        self.theta_ref = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.history_needs_init = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

        self.raw_error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_dot = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_theta_ddot = torch.zeros((self.num_envs, 1), device=self.device)
        self.delta_theta = torch.zeros((self.num_envs, 1), device=self.device)
        self.delta_vrz = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 1), device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset CPID internal state for all envs or selected envs."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        for buffer in (
            self.theta_ref,
            self.error_prev1,
            self.error_prev2,
            self.error_theta_prev1,
            self.error_theta_prev2,
            self.raw_error,
            self.error,
            self.error_dot,
            self.error_ddot,
            self.error_theta,
            self.error_theta_dot,
            self.error_theta_ddot,
            self.delta_theta,
            self.delta_vrz,
            self.last_action,
        ):
            buffer[env_ids] = 0.0
        self.history_needs_init[env_ids] = True
        self.state_predictor.reset(env_ids)

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute the physical velocity increment action ``delta_vrz``."""
        del extras
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.shape[-1] < 11:
            raise ValueError(f"CPIDPolicy expects an 11-D observation, got shape {tuple(raw_obs.shape)}.")
        if raw_obs.shape[0] != self.num_envs:
            raise ValueError(f"CPIDPolicy expected {self.num_envs} envs, got {raw_obs.shape[0]}.")
        raw_obs = raw_obs[:, :11]
        raw_pb = raw_obs[:, ObservationIndex.PB : ObservationIndex.PB + 1]
        raw_pg = raw_obs[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        self.raw_error.copy_(raw_pb - raw_pg)

        init_envs = self.history_needs_init.clone()
        if torch.any(init_envs):
            self.error_prev1[init_envs] = self.raw_error[init_envs]
            self.error_prev2[init_envs] = self.raw_error[init_envs]

        obs = self.state_predictor.predict(raw_obs, error_prev1=self.error_prev1)

        current_theta = obs[:, ObservationIndex.THETA : ObservationIndex.THETA + 1]

        if self.state_predictor.active:
            error, error_dot, error_ddot = self.state_predictor.get_error_prediction()
            self.error.copy_(error)
            self.error_dot.copy_(error_dot)
            self.error_ddot.copy_(error_ddot)
        else:
            self.error.copy_(self.raw_error)
            self.error_dot.copy_(self.error - self.error_prev1)
            self.error_ddot.copy_(self.error - 2.0 * self.error_prev1 + self.error_prev2)

        self.delta_theta.copy_(self._angle_increment(self.error, self.error_dot, self.error_ddot))
        self.theta_ref += self.delta_theta

        self.error_theta.copy_(self.theta_ref - current_theta)
        if torch.any(init_envs):
            self.error_theta_prev1[init_envs] = self.error_theta[init_envs]
            self.error_theta_prev2[init_envs] = self.error_theta[init_envs]
        self.error_theta_dot.copy_(self.error_theta - self.error_theta_prev1)
        self.error_theta_ddot.copy_(self.error_theta - 2.0 * self.error_theta_prev1 + self.error_theta_prev2)

        self.delta_vrz.copy_(
            self._velocity_increment(
                self.error_theta,
                self.error_theta_dot,
                self.error_theta_ddot,
            )
        )
        self.last_action.copy_(torch.clamp(self.delta_vrz.neg(), -self.max_vel_change, self.max_vel_change))

        self.error_prev2.copy_(self.error_prev1)
        self.error_prev1.copy_(self.raw_error)
        self.error_theta_prev2.copy_(self.error_theta_prev1)
        self.error_theta_prev1.copy_(self.error_theta)
        if torch.any(init_envs):
            self.history_needs_init[init_envs] = False
        self.state_predictor.update_after_action(self.last_action)
        return self.last_action.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return CPID state useful for rollout diagnostics."""
        return {
            "policy_raw_error": self.raw_error[:, 0],
            "policy_error": self.error[:, 0],
            "policy_error_dot": self.error_dot[:, 0],
            "policy_error_ddot": self.error_ddot[:, 0],
            "policy_theta_ref": self.theta_ref[:, 0],
            "policy_error_theta": self.error_theta[:, 0],
            "policy_delta_vrz": self.last_action[:, 0],
            **self.state_predictor.get_state(),
        }

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.state_predictor.to(device)
        self.device = device
        return self

    def _resolve_predictor_cfg_defaults(self):
        predictor_cfg = self.cfg.state_predictor
        if _is_auto(predictor_cfg.delay_step):
            predictor_cfg.delay_step = 0
        if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
            predictor_cfg.step_dt = self.step_dt
        if _is_auto(predictor_cfg.max_acc) or float(predictor_cfg.max_acc) <= 0.0:
            predictor_cfg.max_acc = self.cfg.velocity_pid.max_acc
        if _is_auto(predictor_cfg.max_velocity):
            predictor_cfg.max_velocity = 0.0
        if _is_auto(predictor_cfg.plank_length):
            predictor_cfg.plank_length = 1.06
        if _is_auto(predictor_cfg.rope_length):
            predictor_cfg.rope_length = 0.9
        if _is_auto(predictor_cfg.gravity):
            predictor_cfg.gravity = 9.81
        if _is_auto(predictor_cfg.ball_mass):
            predictor_cfg.ball_mass = 0.0005
        if _is_auto(predictor_cfg.ball_radius):
            predictor_cfg.ball_radius = 0.023

    def _angle_increment(self, error: torch.Tensor, error_dot: torch.Tensor, error_ddot: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.angle_pid
        delta_theta = cfg.kp * error_dot
        delta_theta += self._safe_incremental_i_term(cfg.kp, cfg.ti, error)
        delta_theta += cfg.kp * cfg.td / self.step_dt * error_ddot
        return torch.clamp(delta_theta, -float(cfg.max_theta_change), float(cfg.max_theta_change))

    def _velocity_increment(
        self,
        error_theta: torch.Tensor,
        error_theta_dot: torch.Tensor,
        error_theta_ddot: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg.velocity_pid
        delta_vrz = cfg.kp * error_theta_dot
        delta_vrz += self._safe_incremental_i_term(cfg.kp, cfg.ti, error_theta)
        delta_vrz += cfg.kp * cfg.td / self.step_dt * error_theta_ddot
        return torch.clamp(delta_vrz, -self.max_vel_change, self.max_vel_change)

    def _safe_incremental_i_term(self, kp: float, ti: float, error: torch.Tensor) -> torch.Tensor:
        if float(ti) == 0.0:
            return torch.zeros_like(error)
        return float(kp) * self.step_dt / float(ti) * error


def _update_pid_cfg(target, values: Mapping | None):
    if not values:
        return
    aliases = {
        "Kp": "kp",
        "Ti": "ti",
        "Td": "td",
    }
    valid_keys = set(target.__dataclass_fields__)
    for key, value in values.items():
        normalized_key = aliases.get(key, key)
        if normalized_key in valid_keys:
            setattr(target, normalized_key, float(value))


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
