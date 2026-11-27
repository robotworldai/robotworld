"""Target-position benchmark metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from omni.isaac.lab.utils import configclass


@configclass
class TargetPositionEvaluatorCfg:
    """Configuration for target-position benchmark evaluation."""

    episode_length_s: float = 10.0
    num_eval_episodes: int = 1000
    target_zone: float = 0.01
    steady_state_window_s: float = 1.0


class TargetPositionEvaluator:
    """Accumulates SR/SE/CONT/CLIT for target-position balancing."""

    def __init__(
        self,
        cfg: TargetPositionEvaluatorCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt
        self.timeout_value = cfg.episode_length_s + step_dt
        self.steady_state_window_steps = max(1, int(math.ceil(cfg.steady_state_window_s / step_dt)))
        self.env_indices = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self.reset_all()

    def reset_all(self):
        """Clear aggregate and per-episode metric buffers."""
        self.completed_episodes = torch.zeros((), device=self.device)
        self.success_episodes = torch.zeros((), device=self.device)

        self.steady_state_error_sum = torch.zeros((), device=self.device)
        self.steady_state_error_sum_sq = torch.zeros((), device=self.device)
        self.converge_time_sum = torch.zeros((), device=self.device)
        self.converge_time_sum_sq = torch.zeros((), device=self.device)
        self.climbing_time_sum = torch.zeros((), device=self.device)
        self.climbing_time_sum_sq = torch.zeros((), device=self.device)

        self.climbing_time = torch.full((self.num_envs,), self.timeout_value, device=self.device)
        self.converge_time = torch.full((self.num_envs,), self.timeout_value, device=self.device)
        self.steady_state_error_buffer = torch.zeros(
            (self.num_envs, self.steady_state_window_steps),
            device=self.device,
        )
        self.steady_state_error_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steady_state_error_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset_episode(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset per-episode metric buffers for selected environments."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.climbing_time[env_ids] = self.timeout_value
        self.converge_time[env_ids] = self.timeout_value
        self.steady_state_error_buffer[env_ids] = 0.0
        self.steady_state_error_count[env_ids] = 0
        self.steady_state_error_index[env_ids] = 0

    def update(
        self,
        state: dict[str, torch.Tensor],
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update per-step and completed-episode statistics."""
        error = torch.abs(state["pb"] - state["pg"])
        in_target_zone = error <= self.cfg.target_zone
        current_time = episode_length_buf.to(self.device) * self.step_dt
        self._record_steady_state_error(error)

        first_entry = in_target_zone & (self.climbing_time >= self.timeout_value)
        self.climbing_time[first_entry] = current_time[first_entry]

        new_convergence = in_target_zone & (self.converge_time >= self.timeout_value)
        self.converge_time[new_convergence] = current_time[new_convergence]
        self.converge_time[~in_target_zone] = self.timeout_value

        done = terminated | time_outs
        if torch.any(done):
            done_count = done.sum().float()
            steady_state_error = self._get_steady_state_error(done)
            convergence_time = self.converge_time[done]
            climbing_time = self.climbing_time[done]

            self.completed_episodes += done_count
            self.success_episodes += (convergence_time < self.cfg.episode_length_s).sum().float()
            self.steady_state_error_sum += steady_state_error.sum()
            self.steady_state_error_sum_sq += steady_state_error.square().sum()
            self.converge_time_sum += convergence_time.sum()
            self.converge_time_sum_sq += convergence_time.square().sum()
            self.climbing_time_sum += climbing_time.sum()
            self.climbing_time_sum_sq += climbing_time.square().sum()

        return self.get_metrics()

    def get_metrics(self) -> dict[str, torch.Tensor]:
        """Return aggregate benchmark metrics."""
        denom = torch.clamp(self.completed_episodes, min=1.0)
        steady_state_error = self.steady_state_error_sum / denom
        convergence_time = self.converge_time_sum / denom
        climbing_time = self.climbing_time_sum / denom
        return {
            "completed_episodes": self.completed_episodes,
            "evaluation_complete": self.completed_episodes >= self.cfg.num_eval_episodes,
            "success_rate": self.success_episodes / denom,
            "steady_state_error": steady_state_error,
            "steady_state_error_std": self._population_std(
                self.steady_state_error_sum_sq,
                steady_state_error,
            ),
            "convergence_time": convergence_time,
            "convergence_time_std": self._population_std(
                self.converge_time_sum_sq,
                convergence_time,
            ),
            "climbing_time": climbing_time,
            "climbing_time_std": self._population_std(
                self.climbing_time_sum_sq,
                climbing_time,
            ),
        }

    def _record_steady_state_error(self, error: torch.Tensor):
        self.steady_state_error_buffer[self.env_indices, self.steady_state_error_index] = error
        self.steady_state_error_index = (self.steady_state_error_index + 1) % self.steady_state_window_steps
        self.steady_state_error_count = torch.clamp(
            self.steady_state_error_count + 1,
            max=self.steady_state_window_steps,
        )

    def _get_steady_state_error(self, done: torch.Tensor) -> torch.Tensor:
        sample_count = torch.clamp(self.steady_state_error_count[done], min=1).to(dtype=torch.float32)
        return self.steady_state_error_buffer[done].sum(dim=-1) / sample_count

    def _population_std(
        self,
        metric_sum_sq: torch.Tensor,
        metric_mean: torch.Tensor,
    ) -> torch.Tensor:
        denom = torch.clamp(self.completed_episodes, min=1.0)
        variance = metric_sum_sq / denom - metric_mean.square()
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.where(self.completed_episodes > 1.0, std, torch.zeros_like(std))
