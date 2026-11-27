"""Curriculum functions: independent linear ramps for impulse and sustained forces.

Also provides two lightweight diagnostic terms (push_impulse_trigger_count,
push_sustained_trigger_count) that return the cumulative per-env trigger sum so
it is visible in TensorBoard logs during smoke tests and full training.

Both ramp functions follow the same linear shape but support a delayed start
(curriculum_start_step > 0) so the sustained load curriculum can begin after the
policy has had time to learn basic locomotion without wind-load disturbances.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _linear_ramp_curriculum(
    env: ManagerBasedRLEnv,
    event_term_name: str,
    start_force: float,
    end_force: float,
    curriculum_start_step: int,
    num_curriculum_steps: int,
) -> torch.Tensor:
    """Shared ramp logic: linearly interpolate max_force over a step window.

    Progress is clamped to [0, 1].  Steps before curriculum_start_step yield
    start_force; steps after curriculum_start_step + num_curriculum_steps yield
    end_force.

    Mutates env.event_manager.get_term_cfg(event_term_name).params["max_force"]
    so the next event trigger picks up the updated value.

    Returns:
        Current max_force as a scalar tensor (logged by CurriculumManager).
    """
    elapsed = float(env.common_step_counter) - float(curriculum_start_step)
    progress = min(max(elapsed / float(num_curriculum_steps), 0.0), 1.0)
    current_max_force = start_force + progress * (end_force - start_force)
    env.event_manager.get_term_cfg(event_term_name).params["max_force"] = current_max_force
    return torch.tensor(current_max_force, device=env.device)


def push_impulse_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str,
    start_force: float,
    end_force: float,
    num_curriculum_steps: int,
) -> torch.Tensor:
    """Linearly ramp impulse max_force from start_force to end_force.

    Ramp begins at training step 0.

    Typical values: 30 N → 120 N over 800 PPO iterations × 24 steps = 19 200 steps.

    Args:
        event_term_name: EventTerm name for the push impulse (e.g. "push_robot_impulse").
        start_force: Force at step 0 (N).
        end_force: Force at step num_curriculum_steps (N).
        num_curriculum_steps: Total common steps for the ramp.

    Returns:
        Current max_force as scalar tensor.
    """
    return _linear_ramp_curriculum(
        env, event_term_name,
        start_force=start_force,
        end_force=end_force,
        curriculum_start_step=0,
        num_curriculum_steps=num_curriculum_steps,
    )


def push_sustained_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str,
    start_force: float,
    end_force: float,
    curriculum_start_step: int,
    num_curriculum_steps: int,
) -> torch.Tensor:
    """Linearly ramp sustained-load max_force, with a delayed start.

    Ramp begins at curriculum_start_step (not 0), so the policy can stabilise
    on basic locomotion before sustained loads are introduced.

    Typical values: 10 N → 40 N over steps 4 800–28 800 (iter 200–1000 at 24 steps/iter).

    Args:
        event_term_name: EventTerm name for the sustained load (e.g. "push_robot_sustained").
        start_force: Force at curriculum_start_step (N).
        end_force: Force at curriculum_start_step + num_curriculum_steps (N).
        curriculum_start_step: Common step at which ramp begins (= start_iter × steps_per_iter).
        num_curriculum_steps: Total common steps for the ramp.

    Returns:
        Current max_force as scalar tensor.
    """
    return _linear_ramp_curriculum(
        env, event_term_name,
        start_force=start_force,
        end_force=end_force,
        curriculum_start_step=curriculum_start_step,
        num_curriculum_steps=num_curriculum_steps,
    )


# ---------------------------------------------------------------------------
# Diagnostic terms — report cumulative trigger counts for TensorBoard logging
# ---------------------------------------------------------------------------


def push_impulse_trigger_count(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Return total impulse triggers across all envs (sum of trigger_count).

    Logged as Curriculum/push_impulse_trigger_count.  Useful during smoke tests
    to confirm the impulse event is actually firing.
    """
    from . import push_events  # late import to avoid circular dependency

    if push_events._impulse_state.initialized:
        return push_events._impulse_state.trigger_count.sum().float().to(env.device)
    return torch.tensor(0.0, device=env.device)


def push_sustained_trigger_count(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Return total sustained-load triggers across all envs.

    Logged as Curriculum/push_sustained_trigger_count.
    """
    from . import push_events  # late import to avoid circular dependency

    if push_events._sustained_state.initialized:
        return push_events._sustained_state.trigger_count.sum().float().to(env.device)
    return torch.tensor(0.0, device=env.device)
