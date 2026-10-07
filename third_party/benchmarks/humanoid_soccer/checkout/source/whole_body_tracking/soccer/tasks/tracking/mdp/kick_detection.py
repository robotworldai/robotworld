from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_multi_motion_soccer import MotionCommand


@dataclass
class KickContactEvent:
    """Container for kick contact detection results produced once per step."""

    new_contact: torch.Tensor
    kick_detected: torch.Tensor
    peak_force: torch.Tensor
    force_norm: Optional[torch.Tensor] = None


@dataclass
class ContactFootInfo:
    """Resolved foot metadata for environments with an active kick contact."""

    env_ids: torch.Tensor
    body_indices: torch.Tensor
    sides: torch.Tensor
    expected: torch.Tensor


class KickContactTracker:
    """Shared kick contact detection logic reusable across reward terms."""

    def __init__(self, env: ManagerBasedRLEnv, state_prefix: str):
        self._env = env
        self._state_prefix = state_prefix
        self._device = env.device
        self._num_envs = env.num_envs
        self._cache_valid = False
        self._cached_event: Optional[KickContactEvent] = None
        self._foot_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def begin_step(self, command: MotionCommand):
        """Reset per-step cache and handle envs that just resampled motions."""
        self._cache_valid = False
        self._cached_event = None
        self._handle_resample(command)

    def detect(
        self,
        command: MotionCommand,
        ball_sensor_name: str,
        horizontal_force_threshold: float,
    ) -> KickContactEvent:
        """Detect new kick contacts while ensuring single evaluation per step."""
        if self._cache_valid and self._cached_event is not None:
            return self._cached_event

        ball_sensor = self._get_contact_sensor(ball_sensor_name)
        if ball_sensor is None:
            empty_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            zero_force = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
            event = KickContactEvent(empty_mask, empty_mask, zero_force, zero_force)
            self._cached_event = event
            self._cache_valid = True
            return event

        forces = getattr(ball_sensor.data, "net_forces_w_history", ball_sensor.data.net_forces_w)
        if forces is None or forces.numel() == 0:
            empty_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            zero_force = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
            event = KickContactEvent(empty_mask, empty_mask, zero_force, zero_force)
            self._cached_event = event
            self._cache_valid = True
            return event

        forces = forces.to(device=self._device)
        if forces.ndim > 2:
            forces = forces.amax(dim=1)

        force_norm = torch.linalg.norm(forces, dim=-1)
        peak_force = force_norm.amax(dim=-1) if force_norm.ndim > 1 else force_norm
        kick_detected = peak_force > horizontal_force_threshold

        contact_awarded = self._get_or_init_bool_tensor("target_contact_awarded", default=False)
        new_contact = (~contact_awarded) & kick_detected
        if torch.any(new_contact):
            contact_awarded[new_contact] = True
        self._update_detection_state(new_contact)

        event = KickContactEvent(new_contact, kick_detected, peak_force, force_norm)
        self._cached_event = event
        self._cache_valid = True
        return event

    def record_expected_success(self, mask: torch.Tensor, expected_mask: torch.Tensor):
        """Store whether a detected kick matched the expected leg selection."""
        if expected_mask.dtype != torch.bool:
            expected_mask = expected_mask.to(dtype=torch.bool)
        expected_state = self._get_or_init_bool_tensor("expected_kick_success", default=False)
        expected_state[mask] = expected_mask[mask]

    def get_contact_awarded(self) -> torch.Tensor:
        """Return kick status: False means not kicked yet, True means kicked."""
        return self._get_or_init_bool_tensor("target_contact_awarded", default=False)

    def freeze_proximity_reward(self, env_ids: torch.Tensor, reward_values: torch.Tensor):
        """Freeze proximity reward values at the kick-contact moment."""
        frozen = self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)
        frozen[env_ids] = reward_values

    def get_frozen_proximity_reward(self) -> torch.Tensor:
        """Get frozen proximity reward values."""
        return self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)

    def _get_or_init_float_tensor(self, suffix: str, default: float) -> torch.Tensor:
        name = self._tensor_name(suffix)
        tensor = getattr(self._env, name, None)
        if tensor is None or tensor.shape[0] != self._num_envs:
            tensor = torch.full((self._num_envs,), default, dtype=torch.float32, device=self._device)
            setattr(self._env, name, tensor)
            return tensor
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        setattr(self._env, name, tensor)
        return tensor

    def resolve_contact_foot(
        self,
        command: MotionCommand,
        foot_cfg: SceneEntityCfg,
        mask: torch.Tensor,
    ) -> ContactFootInfo:
        """Determine which foot most likely produced the contact for each env."""
        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self._device)
            zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
            return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

        body_indices, sides = self._get_foot_metadata(command, foot_cfg)
        robot = command.robot

        foot_pos = robot.data.body_pos_w[env_ids][:, body_indices]
        ball_pos = command.soccer_ball_pos[env_ids]
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is not None:
            ball_pos = ball_pos + env_origins[env_ids]

        diff = torch.norm(foot_pos - ball_pos.unsqueeze(1), dim=-1)
        closest_idx = torch.argmin(diff, dim=-1)
        selected_body_indices = body_indices[closest_idx]
        hit_sides = sides[closest_idx]

        expected = command.kick_leg[env_ids].to(torch.int8)
        expected = expected.clamp(min=0)

        return ContactFootInfo(env_ids, selected_body_indices, hit_sides, expected)

    def _handle_resample(self, command: MotionCommand):
        flag_name = self._tensor_name("motion_resampled")
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self._num_envs:
            return

        resample_flags = resample_flags.to(device=self._device, dtype=torch.bool)
        if not torch.any(resample_flags):
            return

        contact_state = self._get_or_init_bool_tensor("target_contact_awarded", default=False)
        kick_success_state = self._get_or_init_bool_tensor("kick_success", default=False)
        expected_state = self._get_or_init_bool_tensor("expected_kick_success", default=False)

        # Skip resamples triggered by imminent episode termination to avoid skewed stats.
        eligible_mask = resample_flags.clone()
        step_buf = getattr(self._env, "episode_length_buf", None)
        if step_buf is not None:
            step_buf = step_buf.to(device=self._device, dtype=torch.long)
            cutoff_value = 1
            # max_episode_length = getattr(self._env, "max_episode_length", None)
            # if isinstance(max_episode_length, torch.Tensor):
            #     cutoff_value = int(max_episode_length.item()) - 1
            # elif isinstance(max_episode_length, (int, float)):
            #     cutoff_value = int(max_episode_length) - 1
            cutoff_value = max(cutoff_value, 0)
            eligible_mask = eligible_mask & (cutoff_value < step_buf)

        num_resampled = int(resample_flags.sum().item())
        num_eligible = int(eligible_mask.sum().item())
        if num_resampled > 0 and hasattr(command, "metrics"):
            if "kick_success_rate" not in command.metrics or command.metrics["kick_success_rate"].shape[0] != self._num_envs:
                command.metrics["kick_success_rate"] = torch.zeros(
                    self._num_envs, device=self._device, dtype=torch.float32
                )
            if "expected_kick_success_rate" not in command.metrics or command.metrics["expected_kick_success_rate"].shape[0] != self._num_envs:
                command.metrics["expected_kick_success_rate"] = torch.zeros(
                    self._num_envs, device=self._device, dtype=torch.float32
                )

            if num_eligible > 0:
                success_rate = kick_success_state[eligible_mask].float().mean()
                expected_rate = expected_state[eligible_mask].float().mean()
                command.metrics["kick_success_rate"].fill_(success_rate.item())
                command.metrics["expected_kick_success_rate"].fill_(expected_rate.item())

        contact_state[resample_flags] = False
        kick_success_state[resample_flags] = False
        expected_state[resample_flags] = False
        
        # Reset frozen proximity reward.
        frozen_proximity = self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)
        frozen_proximity[resample_flags] = 0.0
        
        # Reset reward timers to avoid stale window logic after resampling.
        self._reset_reward_timers(resample_flags)
        
        resample_flags[resample_flags] = False
        setattr(self._env, flag_name, resample_flags)

    def _reset_reward_timers(self, resample_flags: torch.Tensor):
        """Reset reward timer states for environments that have been resampled."""
        timer_suffixes = ["dir_align_timer", "dir_align_prev", "speed_timer", "speed_prev", "z_speed_timer", "z_speed_prev"]
        for suffix in timer_suffixes:
            timer_name = f"_{self._state_prefix}_{suffix}"
            timer = getattr(self._env, timer_name, None)
            if timer is not None and timer.shape[0] == self._num_envs:
                timer[resample_flags] = 0
                setattr(self._env, timer_name, timer)

    def _tensor_name(self, suffix: str) -> str:
        return f"{self._state_prefix}_{suffix}"

    def _get_contact_sensor(self, name: str) -> Optional[ContactSensor]:
        sensors = getattr(self._env.scene, "sensors", None)
        if sensors is None:
            return None
        if isinstance(sensors, dict):
            return sensors.get(name)
        try:
            return sensors[name]
        except (KeyError, TypeError):
            return None

    def _get_or_init_bool_tensor(self, suffix: str, default: bool) -> torch.Tensor:
        name = self._tensor_name(suffix)
        tensor = getattr(self._env, name, None)
        if tensor is None or tensor.shape[0] != self._num_envs:
            tensor = torch.full((self._num_envs,), default, dtype=torch.bool, device=self._device)
            setattr(self._env, name, tensor)
            return tensor

        tensor = tensor.to(device=self._device, dtype=torch.bool)
        setattr(self._env, name, tensor)
        return tensor

    def _update_detection_state(self, new_contact: torch.Tensor):
        if not torch.any(new_contact):
            return
        kick_success_state = self._get_or_init_bool_tensor("kick_success", default=False)
        kick_success_state[new_contact] = True

    def _get_foot_metadata(
        self,
        command: MotionCommand,
        foot_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._foot_cache is not None:
            return self._foot_cache

        robot = self._env.scene[foot_cfg.name]
        indices = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self._device,
        )
        sides = torch.tensor(
            [
                0 if "left" in name.lower() else 1 if "right" in name.lower() else -1
                for name in foot_cfg.body_names
            ],
            dtype=torch.int8,
            device=self._device,
        )
        self._foot_cache = (indices, sides)
        return self._foot_cache
