"""Observation adapters used by RL baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base_policy import ObservationIndex


@dataclass
class RLObservationAdapterCfg:
    """Configuration for mapping benchmark observations to RL inputs."""

    observation_mode: str = "legacy8"

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "RLObservationAdapterCfg":
        cfg = cls()
        if not data:
            return cfg
        policy_data = data.get("rl_policy", data.get("policy", data))
        if "observation_mode" in policy_data:
            cfg.observation_mode = str(policy_data["observation_mode"])
        return cfg


class RLObservationAdapter:
    """Convert the 11-D benchmark observation to an RL policy input."""

    LEGACY8_FIELDS = (
        "error",
        "error_d1",
        "error_d2",
        "vb",
        "theta",
        "omega",
        "vrz",
        "a_prev",
    )
    FULL11_FIELDS = (
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

    def __init__(self, cfg: RLObservationAdapterCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.observation_mode = str(cfg.observation_mode).lower()
        if self.observation_mode not in {"legacy8", "full11"}:
            raise ValueError("RL observation_mode must be 'legacy8' or 'full11'.")

        self.error_prev1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_prev2 = torch.zeros((self.num_envs, 1), device=self.device)
        self.history_needs_init = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_policy_input = torch.zeros((self.num_envs, self.input_dim), device=self.device)
        self.error = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d1 = torch.zeros((self.num_envs, 1), device=self.device)
        self.error_d2 = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def input_dim(self) -> int:
        if self.observation_mode == "legacy8":
            return 8
        return 11

    @property
    def field_names(self) -> tuple[str, ...]:
        if self.observation_mode == "legacy8":
            return self.LEGACY8_FIELDS
        return self.FULL11_FIELDS

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset history for all envs or a selected subset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.error_prev1[env_ids] = 0.0
        self.error_prev2[env_ids] = 0.0
        self.error[env_ids] = 0.0
        self.error_d1[env_ids] = 0.0
        self.error_d2[env_ids] = 0.0
        self.last_policy_input[env_ids] = 0.0
        self.history_needs_init[env_ids] = True

    def transform(self, observation: torch.Tensor, update_history: bool = True) -> torch.Tensor:
        """Map a benchmark observation tensor to the configured RL input."""
        observation = observation.to(device=self.device, dtype=torch.float32)
        if observation.shape != (self.num_envs, 11):
            raise ValueError(
                f"RLObservationAdapter expects observation shape ({self.num_envs}, 11), "
                f"got {tuple(observation.shape)}."
            )

        if self.observation_mode == "full11":
            self.last_policy_input.copy_(observation)
            return observation

        pb = observation[:, ObservationIndex.PB : ObservationIndex.PB + 1]
        pg = observation[:, ObservationIndex.PG : ObservationIndex.PG + 1]
        error = pb - pg

        if update_history:
            init_envs = self.history_needs_init.clone()
            if torch.any(init_envs):
                self.error_prev1[init_envs] = error[init_envs]
                self.error_prev2[init_envs] = error[init_envs]

        error_d1 = error - self.error_prev1
        error_d2 = error - 2.0 * self.error_prev1 + self.error_prev2
        policy_input = torch.cat(
            [
                error,
                error_d1,
                error_d2,
                observation[:, ObservationIndex.VB : ObservationIndex.VB + 1],
                observation[:, ObservationIndex.THETA : ObservationIndex.THETA + 1],
                observation[:, ObservationIndex.OMEGA : ObservationIndex.OMEGA + 1],
                observation[:, ObservationIndex.VRZ : ObservationIndex.VRZ + 1],
                observation[:, ObservationIndex.A_PREV : ObservationIndex.A_PREV + 1],
            ],
            dim=-1,
        )

        self.error.copy_(error)
        self.error_d1.copy_(error_d1)
        self.error_d2.copy_(error_d2)
        self.last_policy_input.copy_(policy_input)

        if update_history:
            self.error_prev2.copy_(self.error_prev1)
            self.error_prev1.copy_(error)
            if torch.any(self.history_needs_init):
                self.history_needs_init[:] = False
        return policy_input

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return adapter diagnostics."""
        state = {
            "rl_adapter_error": self.error[:, 0],
            "rl_adapter_error_d1": self.error_d1[:, 0],
            "rl_adapter_error_d2": self.error_d2[:, 0],
            "rl_policy_input": self.last_policy_input,
        }
        return state

    def to(self, device: str | torch.device):
        """Move adapter buffers to a device and return self."""
        device = torch.device(device)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device))
        self.device = device
        return self
