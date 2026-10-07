# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.terminations import ball_in_catch_zone


class BallCatchingEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._substep_video_cam: str | None = None
        self._substep_frames: list[np.ndarray] = []
        self.interval_events_per_substep: bool = False

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if hasattr(self, "task_phase"):
            if not hasattr(self, "_terminal_task_phase"):
                self._terminal_task_phase = torch.full_like(self.task_phase, -1)
            self._terminal_task_phase[env_ids] = self.task_phase[env_ids]
        super()._reset_idx(env_ids)

    def enable_substep_recording(self, cam_name: str) -> None:
        self._substep_video_cam = cam_name

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._last_action_for_recording = action.clone()

        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        has_interval_events = "interval" in self.event_manager.available_modes
        record_video = self._substep_video_cam is not None and is_rendering

        self._substep_frames.clear()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            did_render = (
                self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering
            )
            if did_render:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

            if did_render and record_video:
                cam = self.scene[self._substep_video_cam]
                rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._substep_frames.append(rgb)

            if has_interval_events and self.interval_events_per_substep:
                self._interval_event_dt = self.physics_dt
                self.event_manager.apply(mode="interval", dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if has_interval_events and not self.interval_events_per_substep:
            self._interval_event_dt = self.step_dt
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._last_step_ball_in_catch_zone = ball_in_catch_zone(self)
        if hasattr(self, "task_phase"):
            self._last_step_task_completed = self.task_phase == 4
        else:
            self._last_step_task_completed = self._last_step_ball_in_catch_zone

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

    def advance_sim(self, n_sim_steps: int, action: torch.Tensor) -> None:
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
                self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering
            )
            if did_render:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

            if did_render and record_video:
                cam = self.scene[self._substep_video_cam]
                rgb = cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                self._substep_frames.append(rgb)

            if has_interval_events:
                self._interval_event_dt = self.physics_dt
                self.event_manager.apply(mode="interval", dt=self.physics_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
