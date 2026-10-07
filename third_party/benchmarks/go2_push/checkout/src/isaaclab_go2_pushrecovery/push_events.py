"""Event functions: brief horizontal impulses and sustained lateral loads.

Design
------
Isaac Lab's EventManager does not support timed duration natively.  We implement
duration tracking with module-level state objects (_PushState) and a *short*
EventTerm interval that fires frequently enough to catch expiry within one interval
window of the intended duration:

  - Impulse event: interval_range_s ≈ duration_range_s (~0.2 s).
    Each env is checked ~every 0.2 s.  Force is applied for 0.15–0.25 s then zeroed.
  - Sustained event: interval_range_s=(1.0, 2.0) s (check granularity ~1 s).
    Force is applied for 8–12 s then zeroed.

Per-env state is stored in two _PushState singletons.  A companion "reset" mode
EventTerm (reset_push_states) keeps state consistent after episode resets.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

# ---------------------------------------------------------------------------
# Per-env state container
# ---------------------------------------------------------------------------


class _PushState:
    """Bookkeeping for one time-limited force event type."""

    def __init__(self):
        self.active: torch.Tensor | None = None         # bool  (num_envs,)
        self.end_step: torch.Tensor | None = None       # long  (num_envs,) — step to zero force
        self.next_trigger_step: torch.Tensor | None = None  # long (num_envs,) — step for next push
        self.trigger_count: torch.Tensor | None = None  # long  (num_envs,) — diagnostics

    @property
    def initialized(self) -> bool:
        return self.active is not None

    def init(self, num_envs: int, device: str, step_dt: float,
             trigger_interval_range_s: tuple[float, float]) -> None:
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.end_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Stagger first triggers uniformly over [0, lo_t) so events begin
        # promptly from training start without a full-interval cold delay.
        lo_t, _ = trigger_interval_range_s
        init_delay_s = torch.rand(num_envs, device=device) * lo_t
        self.next_trigger_step = (init_delay_s / step_dt).long()
        self.trigger_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset_envs(self, env_ids: torch.Tensor, current_step: int, step_dt: float,
                   trigger_interval_range_s: tuple[float, float]) -> None:
        """Clear active state and schedule first post-reset trigger with a random delay."""
        self.active[env_ids] = False
        lo_t, hi_t = trigger_interval_range_s
        # Delay = random fraction of one trigger interval, so env stabilises before first push
        delay_steps = ((torch.rand(len(env_ids), device=env_ids.device) * (hi_t - lo_t) + lo_t) / step_dt).long()
        self.next_trigger_step[env_ids] = current_step + delay_steps


# Module-level singletons — one per disturbance type
_impulse_state = _PushState()
_sustained_state = _PushState()

# ---------------------------------------------------------------------------
# Shared core logic
# ---------------------------------------------------------------------------


def _apply_timed_push(
    state: _PushState,
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_force: float,
    duration_range_s: tuple[float, float],
    trigger_interval_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply / expire a time-limited random XY force for each env in env_ids.

    Called by the interval EventTerm on a short period (~duration_range_s).
    Each call:
      1. Zeroes forces for envs past their duration window.
      2. Triggers new forces for envs whose trigger interval has elapsed.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    step_dt = env.step_dt
    step = env.common_step_counter
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1

    if not state.initialized:
        state.init(env.num_envs, device, step_dt, trigger_interval_range_s)

    # -- 1. Zero forces for envs past their duration window --------------------
    expired = state.active[env_ids] & (step >= state.end_step[env_ids])
    if expired.any():
        exp_ids = env_ids[expired]
        z = torch.zeros(len(exp_ids), num_bodies, 3, device=device)
        asset.set_external_force_and_torque(z, z.clone(), env_ids=exp_ids, body_ids=asset_cfg.body_ids)
        state.active[exp_ids] = False

    # -- 2. Apply new force to envs whose trigger timer has elapsed ------------
    should_trigger = (~state.active[env_ids]) & (step >= state.next_trigger_step[env_ids])
    if should_trigger.any():
        trig_ids = env_ids[should_trigger]
        n = len(trig_ids)

        angles = 2.0 * math.pi * torch.rand(n, device=device)
        magnitudes = max_force * torch.rand(n, device=device)

        forces = torch.zeros(n, num_bodies, 3, device=device)
        forces[:, 0, 0] = magnitudes * torch.cos(angles)
        forces[:, 0, 1] = magnitudes * torch.sin(angles)
        torques = torch.zeros_like(forces)
        asset.set_external_force_and_torque(forces, torques, env_ids=trig_ids, body_ids=asset_cfg.body_ids)

        lo_d, hi_d = duration_range_s
        dur_steps = ((lo_d + (hi_d - lo_d) * torch.rand(n, device=device)) / step_dt).long()
        state.end_step[trig_ids] = step + dur_steps
        state.active[trig_ids] = True
        state.trigger_count[trig_ids] += 1

        lo_t, hi_t = trigger_interval_range_s
        interval_steps = ((lo_t + (hi_t - lo_t) * torch.rand(n, device=device)) / step_dt).long()
        state.next_trigger_step[trig_ids] = step + interval_steps

# ---------------------------------------------------------------------------
# Public event functions
# ---------------------------------------------------------------------------


def apply_push_impulse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_force: float,
    duration_range_s: tuple[float, float] = (0.15, 0.25),
    trigger_interval_range_s: tuple[float, float] = (6.0, 10.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Brief (0.15–0.25 s) random horizontal force push.

    Interval EventTerm should use interval_range_s ≈ duration_range_s so the
    event fires frequently enough to zero expired forces within ~0.2 s.

    Args:
        max_force: Maximum force magnitude in N (curriculum-controlled).
        duration_range_s: (min, max) duration of the force application.
        trigger_interval_range_s: (min, max) time between consecutive triggers per env.
        asset_cfg: Robot body to push (must resolve to a single body).
    """
    _apply_timed_push(
        _impulse_state, env, env_ids,
        max_force, duration_range_s, trigger_interval_range_s, asset_cfg,
    )


def apply_sustained_load(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_force: float,
    duration_range_s: tuple[float, float] = (8.0, 12.0),
    trigger_interval_range_s: tuple[float, float] = (25.0, 40.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Longer (8–12 s) sustained lateral force — simulates wind / surface lean.

    Interval EventTerm should use a short-ish interval (1–2 s) to catch expiry
    within ~1–2 s of the intended duration.

    Args:
        max_force: Maximum force magnitude in N (curriculum-controlled).
        duration_range_s: (min, max) duration of the sustained load.
        trigger_interval_range_s: (min, max) time between consecutive triggers per env.
        asset_cfg: Robot body to push (must resolve to a single body).
    """
    _apply_timed_push(
        _sustained_state, env, env_ids,
        max_force, duration_range_s, trigger_interval_range_s, asset_cfg,
    )


def reset_push_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    impulse_trigger_interval_range_s: tuple[float, float] = (6.0, 10.0),
    sustained_trigger_interval_range_s: tuple[float, float] = (25.0, 40.0),
) -> None:
    """Clear per-env push state on episode reset.

    Register this as a ``mode="reset"`` EventTerm so it runs alongside the
    inherited base_external_force_torque event (which zeroes physical forces).
    """
    step = env.common_step_counter
    step_dt = env.step_dt
    if _impulse_state.initialized:
        _impulse_state.reset_envs(env_ids, step, step_dt, impulse_trigger_interval_range_s)
    if _sustained_state.initialized:
        _sustained_state.reset_envs(env_ids, step, step_dt, sustained_trigger_interval_range_s)
