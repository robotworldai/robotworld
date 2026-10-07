"""skrl-compatible MLP models used by RL baselines."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

try:  # skrl is an optional dependency unless RL training/evaluation is used.
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

    _SKRL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local Isaac/skrl environment.
    _SKRL_IMPORT_ERROR = exc

    class Model:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()

    class GaussianMixin:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()

    class DeterministicMixin:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_skrl()


@dataclass
class MLPNetworkCfg:
    """Configuration for actor and critic MLPs."""

    layer_num: int = 3
    hidden_dim: int = 256
    clip_actions: bool = True
    clip_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    reduction: str = "sum"
    pid_obs: bool = False

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "MLPNetworkCfg":
        cfg = cls()
        if not data:
            return cfg
        aliases = {
            "inner_dimention": "hidden_dim",
            "inner_dimension": "hidden_dim",
            "model_inner_dimention": "hidden_dim",
            "model_inner_dimension": "hidden_dim",
            "model_layer_num": "layer_num",
        }
        for key, value in data.items():
            normalized_key = aliases.get(key, key)
            if hasattr(cfg, normalized_key):
                setattr(cfg, normalized_key, value)
        cfg.layer_num = int(cfg.layer_num)
        cfg.hidden_dim = int(cfg.hidden_dim)
        cfg.clip_actions = bool(cfg.clip_actions)
        cfg.clip_log_std = bool(cfg.clip_log_std)
        cfg.min_log_std = float(cfg.min_log_std)
        cfg.max_log_std = float(cfg.max_log_std)
        cfg.pid_obs = bool(cfg.pid_obs)
        return cfg


class MLPActor(GaussianMixin, Model):
    """Stochastic MLP actor with a tanh-normalized mean action."""

    def __init__(
        self,
        observation_space,
        action_space,
        pid_obs: bool = False,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.pid_obs = bool(pid_obs)
        input_dim = 3 if self.pid_obs else self.num_observations
        self.net = _make_mlp(input_dim, self.num_actions, int(layer_num), int(hidden_dim), output_tanh=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.pid_obs:
            states = states[:, :3]
        actor_action = self.net(states)
        return actor_action, self.log_std_parameter, {}


class RPOMLPActor(GaussianMixin, Model):
    """RPO actor: MLP mean plus the training-time uniform mean perturbation."""

    def __init__(
        self,
        observation_space,
        action_space,
        pid_obs: bool = False,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.pid_obs = bool(pid_obs)
        input_dim = 3 if self.pid_obs else self.num_observations
        self.net = _make_mlp(input_dim, self.num_actions, int(layer_num), int(hidden_dim), output_tanh=True)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.pid_obs:
            states = states[:, :3]
        mean_action = self.net(states)
        rpo_alpha = inputs.get("alpha", 0.0)
        if not torch.is_tensor(rpo_alpha):
            rpo_alpha = torch.as_tensor(rpo_alpha, device=mean_action.device, dtype=mean_action.dtype)
        perturbation = torch.zeros_like(mean_action).uniform_(-float(rpo_alpha), float(rpo_alpha))
        actor_action = mean_action + perturbation
        return actor_action, self.log_std_parameter, {"no_pert_mean_actions": mean_action}


class MLPCritic(DeterministicMixin, Model):
    """Deterministic MLP critic."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cuda:0",
        layer_num: int = 3,
        hidden_dim: int = 256,
        clip_actions: bool = False,
    ):
        require_skrl()
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = _make_mlp(
            self.num_observations,
            1,
            int(layer_num),
            int(hidden_dim),
            output_tanh=False,
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def require_skrl():
    """Raise a clear error if skrl is unavailable."""
    if _SKRL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RL baselines require the optional 'skrl' package. Run RL scripts inside the Isaac/skrl "
            "environment used for training, or install skrl>=1.3.0."
        ) from _SKRL_IMPORT_ERROR


def make_models(
    algorithm: str,
    network_cfg: MLPNetworkCfg,
    observation_space,
    action_space,
    device: str | torch.device,
) -> dict[str, Model]:
    """Build skrl policy/value models for PPO or RPO."""
    require_skrl()
    algorithm = str(algorithm).lower()
    actor_cls = RPOMLPActor if algorithm == "rpo" else MLPActor
    actor_observation_space = observation_space
    if network_cfg.pid_obs:
        actor_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    return {
        "policy": actor_cls(
            observation_space=actor_observation_space,
            action_space=action_space,
            device=device,
            pid_obs=network_cfg.pid_obs,
            layer_num=network_cfg.layer_num,
            hidden_dim=network_cfg.hidden_dim,
            clip_actions=network_cfg.clip_actions,
            clip_log_std=network_cfg.clip_log_std,
            min_log_std=network_cfg.min_log_std,
            max_log_std=network_cfg.max_log_std,
            reduction=network_cfg.reduction,
        ),
        "value": MLPCritic(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            layer_num=network_cfg.layer_num,
            hidden_dim=network_cfg.hidden_dim,
        ),
    }


def make_agent_class_and_cfg(
    algorithm: str,
    overrides: Mapping | None,
    observation_space,
    device: str | torch.device,
):
    """Return the selected skrl agent class and a configured agent dictionary."""
    require_skrl()
    algorithm = str(algorithm).lower()
    if algorithm == "rpo":
        from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

        agent_cls = RPO
        cfg = copy.deepcopy(RPO_DEFAULT_CONFIG)
    elif algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

        agent_cls = PPO
        cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    else:
        raise ValueError("RL algorithm must be 'rpo' or 'ppo'.")

    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveRL

    cfg.setdefault("experiment", {})
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

    _deep_update(cfg, overrides or {})
    return agent_cls, cfg


def _make_mlp(input_dim: int, output_dim: int, layer_num: int, hidden_dim: int, output_tanh: bool) -> nn.Sequential:
    net = nn.Sequential()
    current_dim = int(input_dim)
    for _ in range(1, int(layer_num)):
        net.append(nn.Linear(current_dim, int(hidden_dim)))
        net.append(nn.ReLU())
        current_dim = int(hidden_dim)
    net.append(nn.Linear(current_dim, int(output_dim)))
    if output_tanh:
        net.append(nn.Tanh())
    return net


def _deep_update(target: dict, values: Mapping):
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
