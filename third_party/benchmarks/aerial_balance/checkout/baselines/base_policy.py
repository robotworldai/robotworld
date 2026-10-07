"""Common policy interfaces for benchmark baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class BasePolicyCfg:
    """Base configuration shared by benchmark policies."""

    name: str = "base"


class ObservationIndex:
    """Indices of the 11-D benchmark observation vector."""

    PB = 0
    VB = 1
    AB = 2
    THETA = 3
    OMEGA = 4
    ALPHA = 5
    DRZ = 6
    VRZ = 7
    ARZ = 8
    PG = 9
    A_PREV = 10


class BasePolicy(ABC):
    """Minimal interface implemented by all high-level baseline policies."""

    cfg: BasePolicyCfg

    def __init__(self, cfg: BasePolicyCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Reset policy state for all envs or a selected subset."""
        return None

    @abstractmethod
    def act(self, observations: dict[str, torch.Tensor] | torch.Tensor, extras: dict | None = None) -> torch.Tensor:
        """Compute a high-level action from the current environment observation."""

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return policy-local state for logging."""
        return {}

    def to(self, device: str | torch.device):
        """Move policy buffers to a device and return self."""
        self.device = torch.device(device)
        return self

    def _extract_policy_observation(self, observations: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(observations, dict):
            observations = observations["policy"]
        return observations.to(device=self.device, dtype=torch.float32)

    def _env_ids_tensor(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
