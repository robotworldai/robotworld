# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-place from conveyor task environment.

Phase 0-1: RL controls approach and lift.
Phase 2-3: Traditional control (IK) overrides actions for transport and release.
Phase 4:   Task complete (object in box).

IK override is disabled by default (safe for RL training).
Set ``env.enable_ik_override = True`` for data collection.

By default, interval events (conveyor friction, phase checks) run once per
control step after the decimation loop - matching the Isaac Lab base class.
Set ``env.interval_events_per_substep = True`` to run them every physics
sub-step instead (useful for eval.py latency simulation).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import get_ik_action_override
from .mdp.terminations import object_in_box


class ConveyorBeltPickAndPlaceEnv(ManagerBasedRLEnv):
    """Pick-place from conveyor environment with hybrid RL / IK control.

    Key behaviour differences from the base class:

    1. **Sub-step camera frames** are buffered so that external consumers
       (e.g. eval.py video recording) can access all frames rendered
       during a single ``step()`` call, not just the last one.
    2. Interval events default to **once per control step** (matching the
       base class).  Set ``interval_events_per_substep = True`` to run
       them every physics sub-step instead.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._substep_video_cam: str | None = None
        self._substep_frames: list[np.ndarray] = []
        self.interval_events_per_substep: bool = False

        # Keep interval events at a fixed 25 Hz (once every 0.04 s).
        self._interval_dt = 1.0 / 25.0
        self._interval_time_accumulator = 0.0

    # ------------------------------------------------------------------ #
    #  Reset override: capture terminal task_phase before reset           #
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Capture the true task_phase at termination before reset clears it.

        Data collection scripts read ``_terminal_task_phase`` to determine
        whether an episode was successful.
        """
        if hasattr(self, "task_phase"):
            if not hasattr(self, "_terminal_task_phase"):
                self._terminal_task_phase = torch.full_like(self.task_phase, -1)
            self._terminal_task_phase[env_ids] = self.task_phase[env_ids]
        super()._reset_idx(env_ids)

    # ------------------------------------------------------------------ #
    #  Sub-step video recording                                            #
    # ------------------------------------------------------------------ #

    def enable_substep_recording(self, cam_name: str) -> None:
        """Start buffering per-physics-step camera frames.

        Call once during setup.  After each :meth:`step`, the buffered
        frames are available in ``self._substep_frames`` as a list of
        ``(num_envs, H, W, 3)`` uint8 numpy arrays - one entry per
        rendered sub-step.
        """
        self._substep_video_cam = cam_name

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # ---- IK override for selected phases (disabled by default for RL safety) ----
        enable_ik_override = getattr(self, "enable_ik_override", False)
        if enable_ik_override and hasattr(self, "task_phase"):
            arm_action, gripper_open = get_ik_action_override(self)
            if arm_action is not None:
                phase_start = getattr(self, "ik_override_phase_start", 2)
                override_mask = self.task_phase >= phase_start
                if override_mask.any():
                    action = action.clone()
                    action[override_mask, :7] = arm_action[override_mask]
                    action[override_mask, 7] = torch.where(
                        gripper_open[override_mask], 1.0, -1.0
                    )

        # ---- Phase 3 gripper override via env var (for testing/video) ----
        if (
            os.environ.get("PHASE3_GRIPPER_OPEN") == "1"
            and hasattr(self, "task_phase")
            and hasattr(self, "ik_step_counter")
        ):
            phase3_mask = self.task_phase == 3
            if not enable_ik_override:
                self.ik_step_counter[phase3_mask] += 1
            phase3_release = phase3_mask & (self.ik_step_counter > 3)
            if phase3_release.any():
                action = action.clone()
                action[phase3_release, -1] = 1.0

        # Expose final action for data collection recording
        self._last_action_for_recording = action.clone()

        # ---- process action (once per control step) ----
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        has_interval_events = "interval" in self.event_manager.available_modes
        record_video = self._substep_video_cam is not None and is_rendering

        self._substep_frames.clear()

        # ---- physics loop (decimation sub-steps) ----
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            did_render = (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            )
            if did_render:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

            # Buffer camera frame at simulation render rate
            if did_render and record_video:
                cam = self.scene[self._substep_video_cam]
                rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._substep_frames.append(rgb)

            # Run interval events at a fixed 25 Hz.
            if has_interval_events:
                self._interval_time_accumulator += self.physics_dt
                if self._interval_time_accumulator >= self._interval_dt - 1e-5:
                    self.event_manager.apply(mode="interval", dt=self._interval_time_accumulator)
                    self._interval_time_accumulator = 0.0

        # ---- post-step bookkeeping (mirrors base class order) ----
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            if reset_env_ids.dim() == 0:
                reset_env_ids = reset_env_ids.unsqueeze(0)
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------ #
    #  Sub-control-step simulation advance                                 #
    # ------------------------------------------------------------------ #

    def advance_sim(self, n_sim_steps: int, action: torch.Tensor) -> None:
        """Advance physics by exactly *n_sim_steps* sub-steps.

        Used when the caller needs to push the simulation forward by a
        duration that does not align with a full control step (e.g. a
        latency period shorter than ``1 / robot_control_freq``).

        Physics, rendering, cameras, and interval events all run at
        ``sim_freq`` - identical to the inner loop of :meth:`step`.
        Episode-level bookkeeping (rewards, terminations, resets,
        episode counters) is **not** performed.  Camera frames are
        appended to ``_substep_frames`` for video capture.
        """
        if n_sim_steps <= 0:
            return

        self.action_manager.process_action(action.to(self.device))

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        has_interval_events = "interval" in self.event_manager.available_modes
        record_video = self._substep_video_cam is not None and is_rendering

        for _ in range(n_sim_steps):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            did_render = (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            )
            if did_render:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

            if did_render and record_video:
                cam = self.scene[self._substep_video_cam]
                rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._substep_frames.append(rgb)

            # Run interval events at a fixed 25 Hz.
            if has_interval_events:
                self._interval_time_accumulator += self.physics_dt
                if self._interval_time_accumulator >= self._interval_dt - 1e-5:
                    self.event_manager.apply(mode="interval", dt=self._interval_time_accumulator)
                    self._interval_time_accumulator = 0.0

        self.obs_buf = self.observation_manager.compute(update_history=True)
