"""Target-position balancing task for Aerial-Balance-Bench."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omni.isaac.lab.utils import configclass


@configclass
class TargetPositionTaskCfg:
    """Configuration for target-position balancing."""

    ball_position_min: float = 0.10
    ball_position_max: float = 0.60
    goal_position_min: float = 0.10
    goal_position_max: float = 0.60
    random_goal: bool = True
    fixed_goal_position: float = 0.35
    min_initial_goal_distance: float = 0.0

    position_weight: float = 5.0
    ball_velocity_weight: float = 0.5
    command_weight: float = 0.5
    action_weight: float = 1.0
    failure_penalty: float = 500.0
    goal_bonus: float = 5.0
    goal_radius: float = 0.05
    goal_position_decay: float = 60.0
    goal_velocity_decay: float = 1.0
    max_error_for_failure: float = 0.7


class TargetPositionTask:
    """Set-point regulation task for the rolling ball."""

    def __init__(self, cfg: TargetPositionTaskCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.goal_position = torch.full((num_envs,), cfg.fixed_goal_position, device=self.device)
        self.reference_velocity = torch.zeros(num_envs, device=self.device)
        self.initial_ball_position = torch.zeros(num_envs, device=self.device)

    def sample_reset(self, env, env_ids: Sequence[int] | torch.Tensor):
        """Sample target-position task state and reset the ball on the beam."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        if self.cfg.random_goal:
            goal = self._sample_uniform(
                self.cfg.goal_position_min,
                self.cfg.goal_position_max,
                env_ids.numel(),
            )
        else:
            goal = torch.full((env_ids.numel(),), self.cfg.fixed_goal_position, device=self.device)

        ball_position = self._sample_uniform(
            self.cfg.ball_position_min,
            self.cfg.ball_position_max,
            env_ids.numel(),
        )
        if self.cfg.min_initial_goal_distance > 0.0:
            for _ in range(10):
                too_close = torch.abs(ball_position - goal) < self.cfg.min_initial_goal_distance
                if not torch.any(too_close):
                    break
                ball_position[too_close] = self._sample_uniform(
                    self.cfg.ball_position_min,
                    self.cfg.ball_position_max,
                    int(too_close.sum().item()),
                )

        self.goal_position[env_ids] = goal
        self.reference_velocity[env_ids] = 0.0
        self.initial_ball_position[env_ids] = ball_position
        env.set_ball_position_along_beam(env_ids, ball_position)

    def get_reference(self, step: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return desired ball position and velocity for the current control step."""
        return self.goal_position, self.reference_velocity

    def compute_reward(
        self,
        state: dict[str, torch.Tensor],
        command: dict[str, torch.Tensor],
        action: torch.Tensor,
        terminated: torch.Tensor | None = None,
    ):
        """Compute the target-position reward from the benchmark definition."""
        error = state["pb"] - self.goal_position
        ball_velocity = state["vb"]
        theta = state["theta"]
        command_z = command["command_z"]
        action_z = action.squeeze(-1)

        object_reward = -self.cfg.position_weight * error.square()
        object_reward -= self.cfg.ball_velocity_weight * ball_velocity.square()

        control_reward = -self.cfg.command_weight * command_z.square()
        control_reward -= self.cfg.action_weight * action_z.square()

        if terminated is None:
            failure_mask = (torch.abs(theta) > state["theta_limit"]) | (
                torch.abs(error) > self.cfg.max_error_for_failure
            )
        else:
            failure_mask = terminated
        failure_reward = -self.cfg.failure_penalty * failure_mask.float()

        near_goal = torch.abs(error) < self.cfg.goal_radius
        goal_reward = torch.full_like(error, self.cfg.goal_bonus) - self.cfg.goal_position_decay * torch.abs(error)
        goal_reward *= torch.exp(-self.cfg.goal_velocity_decay * torch.abs(ball_velocity))
        goal_reward = goal_reward * near_goal.float()

        return object_reward + control_reward + failure_reward + goal_reward

    def compute_task_dones(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return task-specific failure flags."""
        error = state["pb"] - self.goal_position
        return torch.abs(error) > self.cfg.max_error_for_failure

    def update_metrics(self, *_, **__):
        """Reserved for task-local metrics; target-position metrics live in TargetPositionEvaluator."""
        return {}

    def _sample_uniform(self, low: float, high: float, size: int) -> torch.Tensor:
        return torch.rand(size, device=self.device) * (high - low) + low
