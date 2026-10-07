"""NMPC baseline policy for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base_policy import BasePolicy, BasePolicyCfg, ObservationIndex
from .model_state_predictor import VelocityModelStatePredictor, VelocityModelStatePredictorCfg
from .nmpc_core import (
    NMPCConstraintsCfg,
    NMPCControllerCfg,
    NMPCControllerPool,
    NMPCObjectiveCfg,
    NMPCSolverCfg,
)


@dataclass
class NMPCPolicyCfg(BasePolicyCfg):
    """Configuration for target-position NMPC policy evaluation."""

    name: str = "nmpc"
    gravity: float | str = "auto"
    plank_length: float | str = "auto"
    rope_length: float | str = "auto"
    ball_mass: float | str = "auto"
    ball_radius: float | str = "auto"
    ball_inertia_ratio: float = 0.4
    max_acc: float | str = "auto"
    max_velocity: float | str = "auto"
    t_step: float | str = "auto"
    use_multiprocessing: bool = True
    objective: NMPCObjectiveCfg = field(default_factory=NMPCObjectiveCfg)
    constraints: NMPCConstraintsCfg = field(default_factory=NMPCConstraintsCfg)
    solver: NMPCSolverCfg = field(default_factory=NMPCSolverCfg)
    state_predictor: VelocityModelStatePredictorCfg = field(default_factory=VelocityModelStatePredictorCfg)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "NMPCPolicyCfg":
        cfg = cls()
        if not data:
            return cfg

        policy_data = data.get("nmpc_policy", data.get("policy", data))
        aliases = {
            "policy_name": "name",
            "beam_length": "plank_length",
            "n_horizon": "solver",
        }
        for key, value in policy_data.items():
            normalized_key = aliases.get(key, key)
            if normalized_key in {"objective", "constraints", "solver", "state_predictor"}:
                continue
            if normalized_key == "use_multiprocessing":
                cfg.use_multiprocessing = bool(value)
            elif hasattr(cfg, normalized_key):
                setattr(cfg, normalized_key, value)

        _update_dataclass(cfg.objective, policy_data.get("objective", {}))
        _update_dataclass(cfg.constraints, policy_data.get("constraints", {}))
        _update_dataclass(cfg.solver, policy_data.get("solver", {}))
        if "n_horizon" in policy_data:
            cfg.solver.n_horizon = int(policy_data["n_horizon"])
        if "t_step" in policy_data:
            cfg.t_step = policy_data["t_step"]
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(policy_data.get("state_predictor", {}))
        return cfg


class NMPCPolicy(BasePolicy):
    """do-mpc based high-level velocity-interface policy."""

    cfg: NMPCPolicyCfg

    def __init__(
        self,
        cfg: NMPCPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)
        if self.step_dt <= 0.0:
            raise ValueError("NMPCPolicy requires a positive step_dt.")
        self._resolve_cfg_defaults()

        self.max_acc = float(cfg.max_acc)
        self.delta_vrz_limit = self.max_acc * self.step_dt
        self.max_velocity = float(cfg.max_velocity)
        self.state_predictor = VelocityModelStatePredictor(cfg.state_predictor, num_envs, self.device)

        controller_cfg = self._controller_cfg()
        self.controller_pool = NMPCControllerPool(
            controller_cfg,
            num_envs,
            use_multiprocessing=cfg.use_multiprocessing,
        )

        self.nmpc_state = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal = torch.zeros((self.num_envs,), device=self.device)
        self.accel_cmd = torch.zeros((self.num_envs, 1), device=self.device)
        self.delta_vrz = torch.zeros((self.num_envs, 1), device=self.device)
        self.compute_time = torch.zeros((self.num_envs,), device=self.device)
        self.solver_success = torch.zeros((self.num_envs,), device=self.device)
        self.last_solver_status = ["unknown" for _ in range(self.num_envs)]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset selected NMPC solvers and predictor state."""
        env_ids_tensor = self._env_ids_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        env_ids_list = [int(env_id) for env_id in env_ids_tensor.detach().cpu().tolist()]
        self.controller_pool.reset(env_ids_list)
        self.state_predictor.reset(env_ids_tensor)
        self.nmpc_state[env_ids_tensor] = 0.0
        self.goal[env_ids_tensor] = 0.0
        self.accel_cmd[env_ids_tensor] = 0.0
        self.delta_vrz[env_ids_tensor] = 0.0
        self.compute_time[env_ids_tensor] = 0.0
        self.solver_success[env_ids_tensor] = 0.0
        for env_id in env_ids_list:
            self.last_solver_status[env_id] = "reset"

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute physical velocity-increment actions from 11-D benchmark observations."""
        del extras
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.shape != (self.num_envs, 11):
            raise ValueError(f"NMPCPolicy expects observation shape ({self.num_envs}, 11), got {tuple(raw_obs.shape)}.")

        model_obs = self.state_predictor.predict(raw_obs)
        state = torch.stack(
            [
                model_obs[:, ObservationIndex.PB],
                model_obs[:, ObservationIndex.VB],
                model_obs[:, ObservationIndex.THETA],
                model_obs[:, ObservationIndex.VRZ],
            ],
            dim=-1,
        )
        goal = model_obs[:, ObservationIndex.PG]

        u_np, compute_time_np, success_np, status = self.controller_pool.make_step(
            state.detach().cpu().numpy(),
            goals=goal.detach().cpu().numpy(),
        )
        accel_cmd = torch.as_tensor(u_np, dtype=torch.float32, device=self.device)
        delta_vrz = torch.clamp(accel_cmd * self.step_dt, -self.delta_vrz_limit, self.delta_vrz_limit)

        self.nmpc_state.copy_(state)
        self.goal.copy_(goal)
        self.accel_cmd.copy_(accel_cmd)
        self.delta_vrz.copy_(delta_vrz)
        self.compute_time.copy_(torch.as_tensor(compute_time_np, dtype=torch.float32, device=self.device))
        self.solver_success.copy_(torch.as_tensor(success_np, dtype=torch.float32, device=self.device))
        self.last_solver_status = list(status)
        self.state_predictor.update_after_action(self.delta_vrz)
        return self.delta_vrz.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return NMPC diagnostics for rollout logging."""
        return {
            "nmpc_state_pb": self.nmpc_state[:, 0],
            "nmpc_state_vb": self.nmpc_state[:, 1],
            "nmpc_state_theta": self.nmpc_state[:, 2],
            "nmpc_state_vrz": self.nmpc_state[:, 3],
            "nmpc_goal": self.goal,
            "nmpc_accel_cmd": self.accel_cmd[:, 0],
            "nmpc_delta_vrz": self.delta_vrz[:, 0],
            "nmpc_compute_time": self.compute_time,
            "nmpc_solver_success": self.solver_success,
            **self.state_predictor.get_state(),
        }

    def close(self):
        """Close worker processes if multiprocessing is enabled."""
        self.controller_pool.close()

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.state_predictor.to(device)
        self.device = device
        return self

    def _resolve_cfg_defaults(self):
        cfg = self.cfg
        if _is_auto(cfg.t_step):
            cfg.t_step = self.step_dt
        cfg.solver.t_step = float(cfg.t_step)
        if _is_auto(cfg.max_acc):
            cfg.max_acc = 0.5
        if _is_auto(cfg.max_velocity):
            cfg.max_velocity = 0.0
        if _is_auto(cfg.gravity):
            cfg.gravity = 9.81
        if _is_auto(cfg.plank_length):
            cfg.plank_length = 1.06
        if _is_auto(cfg.rope_length):
            cfg.rope_length = 0.9
        if _is_auto(cfg.ball_mass):
            cfg.ball_mass = 0.0005
        if _is_auto(cfg.ball_radius):
            cfg.ball_radius = 0.023
        cfg.constraints.acceleration_min = -float(cfg.max_acc)
        cfg.constraints.acceleration_max = float(cfg.max_acc)
        if float(cfg.max_velocity) > 0.0:
            cfg.constraints.velocity_min = -float(cfg.max_velocity)
            cfg.constraints.velocity_max = float(cfg.max_velocity)

        predictor_cfg = cfg.state_predictor
        if _is_auto(predictor_cfg.delay_step):
            predictor_cfg.delay_step = 0
        if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
            predictor_cfg.step_dt = self.step_dt
        if _is_auto(predictor_cfg.max_acc) or float(predictor_cfg.max_acc) <= 0.0:
            predictor_cfg.max_acc = float(cfg.max_acc)
        if _is_auto(predictor_cfg.max_velocity):
            predictor_cfg.max_velocity = float(cfg.max_velocity)
        if _is_auto(predictor_cfg.plank_length):
            predictor_cfg.plank_length = float(cfg.plank_length)
        if _is_auto(predictor_cfg.rope_length):
            predictor_cfg.rope_length = float(cfg.rope_length)
        if _is_auto(predictor_cfg.gravity):
            predictor_cfg.gravity = float(cfg.gravity)
        if _is_auto(predictor_cfg.ball_mass):
            predictor_cfg.ball_mass = float(cfg.ball_mass)
        if _is_auto(predictor_cfg.ball_radius):
            predictor_cfg.ball_radius = float(cfg.ball_radius)

    def _controller_cfg(self) -> NMPCControllerCfg:
        return NMPCControllerCfg(
            gravity=float(self.cfg.gravity),
            plank_length=float(self.cfg.plank_length),
            rope_length=float(self.cfg.rope_length),
            ball_mass=float(self.cfg.ball_mass),
            ball_radius=float(self.cfg.ball_radius),
            ball_inertia_ratio=float(self.cfg.ball_inertia_ratio),
            initial_goal=0.35,
            objective=self.cfg.objective,
            constraints=self.cfg.constraints,
            solver=self.cfg.solver,
        )


def _update_dataclass(target: Any, values: Mapping | None):
    if not values:
        return
    for key, value in values.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if current is None:
                setattr(target, key, None if value is None else float(value))
            elif isinstance(current, bool):
                setattr(target, key, bool(value))
            elif isinstance(current, int):
                setattr(target, key, int(value))
            elif isinstance(current, float):
                setattr(target, key, float(value))
            else:
                setattr(target, key, value)


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
