"""Trajectory-tracking benchmark metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omni.isaac.lab.utils import configclass


@configclass
class TrajectoryTrackingEvaluatorCfg:
    """Configuration for trajectory-tracking benchmark evaluation."""

    episode_length_s: float = 10.0
    num_eval_episodes: int = 1000


class TrajectoryTrackingEvaluator:
    """Accumulates MAE/RMSE/MAXE for trajectory tracking."""

    def __init__(
        self,
        cfg: TrajectoryTrackingEvaluatorCfg,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt
        self.reset_all()

    def reset_all(self):
        """Clear aggregate and per-episode metric buffers."""
        self.completed_episodes = torch.zeros((), device=self.device)
        self.mae_sum = torch.zeros((), device=self.device)
        self.rmse_sum = torch.zeros((), device=self.device)
        self.maxe_sum = torch.zeros((), device=self.device)

        self.episode_abs_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_squared_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_abs_error = torch.zeros(self.num_envs, device=self.device)
        self.episode_samples = torch.zeros(self.num_envs, device=self.device)

    def reset_episode(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset per-episode metric buffers for selected environments."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self.episode_abs_error_sum[env_ids] = 0.0
        self.episode_squared_error_sum[env_ids] = 0.0
        self.episode_max_abs_error[env_ids] = 0.0
        self.episode_samples[env_ids] = 0.0

    def update(
        self,
        state: dict[str, torch.Tensor],
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update per-step and completed-episode tracking metrics."""
        del episode_length_buf
        abs_error = torch.abs(state["pb"] - state["pg"])
        self.episode_abs_error_sum += abs_error
        self.episode_squared_error_sum += abs_error.square()
        self.episode_max_abs_error = torch.maximum(self.episode_max_abs_error, abs_error)
        self.episode_samples += 1.0

        done = terminated | time_outs
        if torch.any(done):
            done_count = done.sum().float()
            sample_count = torch.clamp(self.episode_samples[done], min=1.0)
            mae = self.episode_abs_error_sum[done] / sample_count
            rmse = torch.sqrt(self.episode_squared_error_sum[done] / sample_count)
            maxe = self.episode_max_abs_error[done]

            self.completed_episodes += done_count
            self.mae_sum += mae.sum()
            self.rmse_sum += rmse.sum()
            self.maxe_sum += maxe.sum()

        return self.get_metrics()

    def get_metrics(self) -> dict[str, torch.Tensor]:
        """Return aggregate benchmark metrics."""
        denom = torch.clamp(self.completed_episodes, min=1.0)
        return {
            "completed_episodes": self.completed_episodes,
            "evaluation_complete": self.completed_episodes >= self.cfg.num_eval_episodes,
            "mean_absolute_error": self.mae_sum / denom,
            "root_mean_square_error": self.rmse_sum / denom,
            "maximum_absolute_error": self.maxe_sum / denom,
        }
