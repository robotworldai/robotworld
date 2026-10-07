"""RL baseline policy wrappers for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from .base_policy import BasePolicy, BasePolicyCfg
from .model_state_predictor import VelocityModelStatePredictor, VelocityModelStatePredictorCfg
from .rl_models import MLPNetworkCfg, make_agent_class_and_cfg, make_models
from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg


@dataclass
class RLPolicyCfg(BasePolicyCfg):
    """Configuration for RL policy deployment/evaluation."""

    name: str = "rl"
    algorithm: str = "rpo"
    observation_mode: str = "legacy8"
    checkpoint_path: str | None = None
    load_checkpoint: bool = True
    deterministic: bool = True
    physical_action_limit: float | str = "auto"
    network: MLPNetworkCfg = field(default_factory=MLPNetworkCfg)
    agent: dict[str, Any] = field(default_factory=dict)
    state_predictor: VelocityModelStatePredictorCfg = field(default_factory=VelocityModelStatePredictorCfg)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLPolicyCfg":
        """Build an RL policy config from a YAML dictionary."""
        cfg = cls()
        if not data:
            return cfg

        policy_data = data.get("rl_policy", data.get("policy", data))
        if "name" in policy_data:
            cfg.name = str(policy_data["name"])
        if "policy_name" in policy_data:
            cfg.name = str(policy_data["policy_name"])
        if "algorithm" in policy_data:
            cfg.algorithm = str(policy_data["algorithm"]).lower()
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"]).lower()
        if "checkpoint_path" in policy_data:
            checkpoint_path = policy_data["checkpoint_path"]
            cfg.checkpoint_path = None if checkpoint_path in (None, "null", "") else str(checkpoint_path)
        if "load_checkpoint" in policy_data:
            cfg.load_checkpoint = bool(policy_data["load_checkpoint"])
        if "deterministic" in policy_data:
            cfg.deterministic = bool(policy_data["deterministic"])
        if "physical_action_limit" in policy_data:
            cfg.physical_action_limit = policy_data["physical_action_limit"]

        cfg.network = MLPNetworkCfg.from_dict(policy_data.get("network", policy_data.get("model", {})))
        cfg.agent = dict(policy_data.get("agent", {}))
        cfg.state_predictor = VelocityModelStatePredictorCfg.from_dict(policy_data.get("state_predictor", {}))
        return cfg


class RLPolicy(BasePolicy):
    """skrl-backed high-level RL policy."""

    cfg: RLPolicyCfg

    def __init__(
        self,
        cfg: RLPolicyCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        physical_action_limit: float,
    ):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)
        self.physical_action_limit = float(physical_action_limit)
        if self.step_dt <= 0.0:
            raise ValueError("RLPolicy requires a positive step_dt.")
        if self.physical_action_limit <= 0.0:
            raise ValueError("RLPolicy requires a positive physical_action_limit.")

        self._resolve_predictor_cfg_defaults()
        adapter_cfg = RLObservationAdapterCfg(observation_mode=cfg.observation_mode)
        self.observation_adapter = RLObservationAdapter(adapter_cfg, num_envs, self.device)
        self.state_predictor = VelocityModelStatePredictor(cfg.state_predictor, num_envs, self.device)

        self.normalized_action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_adapter.input_dim,),
            dtype=np.float32,
        )
        self.models = make_models(
            cfg.algorithm,
            cfg.network,
            self.policy_observation_space,
            self.normalized_action_space,
            self.device,
        )
        agent_cls, agent_cfg = make_agent_class_and_cfg(
            cfg.algorithm,
            cfg.agent,
            self.policy_observation_space,
            self.device,
        )
        agent_cfg.setdefault("experiment", {})
        agent_cfg["experiment"]["wandb"] = False
        self.agent_cfg = agent_cfg
        self.agent = agent_cls(
            models=self.models,
            memory=None,
            cfg=agent_cfg,
            observation_space=self.policy_observation_space,
            action_space=self.normalized_action_space,
            device=self.device,
        )
        if cfg.load_checkpoint:
            if not cfg.checkpoint_path:
                raise ValueError("RLPolicyCfg.load_checkpoint=True requires checkpoint_path.")
            self.agent.load(str(Path(cfg.checkpoint_path).expanduser()))

        self.timestep = 0
        self.policy_input = torch.zeros((self.num_envs, self.observation_adapter.input_dim), device=self.device)
        self.normalized_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.physical_action = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def observation_mode(self) -> str:
        return self.observation_adapter.observation_mode

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset RL policy-side history for all envs or selected envs."""
        env_ids = self._env_ids_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self.observation_adapter.reset(env_ids)
        self.state_predictor.reset(env_ids)
        self.normalized_action[env_ids] = 0.0
        self.physical_action[env_ids] = 0.0
        self.policy_input[env_ids] = 0.0
        if env_ids.numel() == self.num_envs:
            self.timestep = 0

    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute a physical velocity-increment action from environment observations."""
        del extras
        raw_obs = self._extract_policy_observation(observations)
        if raw_obs.shape != (self.num_envs, 11):
            raise ValueError(f"RLPolicy expects observation shape ({self.num_envs}, 11), got {tuple(raw_obs.shape)}.")

        model_obs = self.state_predictor.predict(raw_obs)
        self.policy_input.copy_(self.observation_adapter.transform(model_obs, update_history=True))

        outputs = self.agent.act(self.policy_input, timestep=self.timestep, timesteps=self.timestep)
        action, info = self._select_normalized_action(outputs)
        del info
        self.normalized_action.copy_(torch.clamp(action, -1.0, 1.0))
        self.physical_action.copy_(self.normalized_action * self.physical_action_limit)
        self.state_predictor.update_after_action(self.physical_action)
        self.timestep += 1
        return self.physical_action.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return RL policy diagnostics."""
        state = {
            "rl_normalized_action": self.normalized_action[:, 0],
            "rl_physical_action": self.physical_action[:, 0],
            **self.observation_adapter.get_state(),
            **self.state_predictor.get_state(),
        }
        return state

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.observation_adapter.to(device)
        self.state_predictor.to(device)
        self.device = device
        return self

    def _select_normalized_action(self, outputs) -> tuple[torch.Tensor, dict]:
        info = outputs[-1] if isinstance(outputs, tuple) and isinstance(outputs[-1], dict) else {}
        if self.cfg.deterministic and "no_pert_mean_actions" in info:
            action = info["no_pert_mean_actions"]
        elif isinstance(outputs, tuple):
            action = outputs[0]
        else:
            action = outputs
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape != (self.num_envs, 1):
            raise ValueError(f"RLPolicy expected normalized action shape ({self.num_envs}, 1), got {tuple(action.shape)}.")
        return action, info

    def _resolve_predictor_cfg_defaults(self):
        predictor_cfg = self.cfg.state_predictor
        if _is_auto(predictor_cfg.delay_step):
            predictor_cfg.delay_step = 0
        if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
            predictor_cfg.step_dt = self.step_dt
        if _is_auto(predictor_cfg.max_acc) or float(predictor_cfg.max_acc) <= 0.0:
            predictor_cfg.max_acc = self.physical_action_limit / self.step_dt
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


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"
