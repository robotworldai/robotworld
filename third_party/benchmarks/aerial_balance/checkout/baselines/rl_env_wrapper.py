"""Environment wrappers used for RL training."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import Wrapper, spaces

from .rl_observation_adapter import RLObservationAdapter, RLObservationAdapterCfg


TARGET_POSITION_BENCHMARK_METRICS = (
    "success_rate",
    "steady_state_error",
    "convergence_time",
    "climbing_time",
)


class NormalizedRLTrainingWrapper(Wrapper):
    """Expose normalized actions and adapted observations to skrl trainers."""

    def __init__(
        self,
        env,
        adapter_cfg: RLObservationAdapterCfg,
        physical_action_limit: float,
    ):
        super().__init__(env)
        self.physical_action_limit = float(physical_action_limit)
        if self.physical_action_limit <= 0.0:
            raise ValueError("NormalizedRLTrainingWrapper requires a positive physical_action_limit.")
        self.adapter = RLObservationAdapter(adapter_cfg, env.unwrapped.num_envs, env.unwrapped.device)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        policy_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.adapter.input_dim,),
            dtype=np.float32,
        )
        # skrl's IsaacLab wrapper reads observation spaces from
        # env.unwrapped.single_observation_space["policy"].  Keep the exposed
        # space aligned with the adapted observation returned by reset/step.
        self.observation_space = {"policy": policy_observation_space}
        self.single_action_space = self.action_space
        self.single_observation_space = {"policy": policy_observation_space}
        self.last_normalized_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_physical_action = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def unwrapped(self):
        """Expose this wrapper to skrl so adapted spaces are used."""
        return self

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def max_episode_length(self) -> int:
        return self.env.unwrapped.max_episode_length

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.adapter.reset()
        self.last_normalized_action.zero_()
        self.last_physical_action.zero_()
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": self.adapter.transform(observations["policy"], update_history=True)}, infos

    def step(self, action):
        if torch.is_tensor(action):
            normalized_action = action.to(device=self.device, dtype=torch.float32)
        else:
            normalized_action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if normalized_action.ndim == 1:
            normalized_action = normalized_action.unsqueeze(-1)
        normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
        physical_action = normalized_action * self.physical_action_limit
        observations, rewards, terminated, truncated, infos = self.env.step(physical_action)

        done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if done_env_ids.numel() > 0:
            self.adapter.reset(done_env_ids)

        self.last_normalized_action.copy_(normalized_action)
        self.last_physical_action.copy_(physical_action)
        adapted_observation = self.adapter.transform(observations["policy"], update_history=True)
        infos.setdefault("rl", {})
        infos["rl"]["normalized_action"] = self.last_normalized_action
        infos["rl"]["physical_action"] = self.last_physical_action
        infos["rl"]["policy_input"] = self.adapter.last_policy_input
        self._copy_benchmark_metrics_to_episode_info(infos)
        return {"policy": adapted_observation}, rewards, terminated, truncated, infos

    def _copy_benchmark_metrics_to_episode_info(self, infos: dict):
        """Expose benchmark metrics through skrl's default environment-info hook."""
        benchmark = infos.get("benchmark")
        if not isinstance(benchmark, dict):
            return
        episode_info = infos.setdefault("episode", {})
        for key in TARGET_POSITION_BENCHMARK_METRICS:
            value = benchmark.get(key)
            if value is not None:
                episode_info[key] = value
