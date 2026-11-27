from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .kick_detection import KickContactTracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiMotionLoader:
    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu"):
        assert len(motion_files) > 0, "motion_files must not be empty"
        self.num_files = len(motion_files)
        self._body_indexes = body_indexes
        self.device = device

        # Temporarily store data from each file.
        self.motion_name = []
        self.motion_lengths = []

        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        kick_leg_labels = []

        self.fps_list = []

        max_T = 0  # Track maximum frame count.

        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            self.fps_list.append(data["fps"])
            self.motion_name.append(motion_file.split("/")[-1].split(".")[0])  # Store filename without suffix.
            self.motion_lengths.append(data["joint_pos"].shape[0])

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)

            label_value: str | None = None
            if "kick_leg" in data.files:
                raw_label = data["kick_leg"]
                try:
                    label_str = str(raw_label.item()).strip().lower()
                except Exception:
                    label_str = str(raw_label).strip().lower()
                if label_str in {"left", "right"}:
                    label_value = label_str
            kick_leg_labels.append(label_value)

            max_T = max(max_T, jp.shape[0])

        # Pad all files to max_T and stack into tensors.
        def pad_tensor_list(tensor_list, pad_value=0.0):
            padded = []
            for t in tensor_list:
                T, *rest = t.shape
                pad_size = [max_T - T] + rest
                pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device)], dim=0)
                # pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device, dtype=t.dtype)], dim=0)
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)  # shape: (num_files, max_T, ...)

        self.joint_pos = pad_tensor_list(joint_pos_list)
        self.joint_vel = pad_tensor_list(joint_vel_list)
        self._body_pos_w = pad_tensor_list(body_pos_w_list)
        self._body_quat_w = pad_tensor_list(body_quat_w_list)
        self._body_lin_vel_w = pad_tensor_list(body_lin_vel_w_list)
        self._body_ang_vel_w = pad_tensor_list(body_ang_vel_w_list)

        self.time_step_total = max_T  # Maximum frame count.
        self.file_lengths = torch.tensor([jp.shape[0] for jp in joint_pos_list],
                                         dtype=torch.long,
                                         device=self.device)
        self.fps = self.fps_list[0]  # Can be adjusted if needed.
        self._kick_leg_labels = tuple(kick_leg_labels)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

    @property
    def kick_leg_labels(self) -> tuple[str | None, ...]:
        return self._kick_leg_labels
    
    def get_last_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int, motion_length: int) -> torch.Tensor:
        """Get the anchor position at the last frame of the specified motion."""
        last_frame_idx = motion_length - 1
        return self._body_pos_w[motion_idx, last_frame_idx, anchor_body_idx]

    def get_first_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor position at the first frame of the specified motion."""
        return self._body_pos_w[motion_idx, 0, anchor_body_idx]

    def get_first_frame_anchor_quat(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor orientation at the first frame of the specified motion."""
        return self._body_quat_w[motion_idx, 0, anchor_body_idx]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.soccer_ball: RigidObject | None = None
        # Try to get the soccer-ball object.
        if hasattr(env.scene, "__getitem__"):
            try:
                self.soccer_ball = env.scene["soccer_ball"]
            except KeyError:
                self.soccer_ball = None

        # Determine whether the motion sequence has ended.
        term_name = getattr(cfg, "term_name", None)
        if term_name is None:
            term_name = getattr(cfg, "name", None)
        if term_name is None:
            term_name = "motion"
            self._state_prefix = f"_{term_name}"
            self.kick_contact_tracker = KickContactTracker(env, self._state_prefix)

        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MultiMotionLoader(self.cfg.motion_files, self.body_indexes, device=self.device)
        kick_leg_to_id = {"left": 0, "right": 1}
        self._kick_leg_id_to_name = {v: k for k, v in kick_leg_to_id.items()}
        self._kick_leg_id_to_name[-1] = "unknown"
        self.motion_kick_leg = torch.full((self.motion.num_files,), -1, dtype=torch.int8, device=self.device)
        self.motion_kick_leg_names = []
        for idx, label in enumerate(self.motion.kick_leg_labels):
            normalized = label.lower() if isinstance(label, str) else None
            if normalized in kick_leg_to_id:
                self.motion_kick_leg[idx] = kick_leg_to_id[normalized]
                self.motion_kick_leg_names.append(normalized)
            else:
                self.motion_kick_leg_names.append("unknown")

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Randomly assign initial motions.
        if self.motion.num_files > 1:
            self.motion_idx = torch.randint(0, self.motion.num_files, (self.num_envs,), 
                                           dtype=torch.long, device=self.device)
        # Initialize per-environment motion lengths.
        self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling settings.
        # Compute bin count: decimation * dt is one simulation step duration.
        # Thus each bin corresponds to ~1 second and bin_count is the total number of bins.
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(
            (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        # Target-point and soccer-ball generation logic.
        self.target_point_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.soccer_ball_pos = torch.zeros_like(self.target_point_pos)
        self.target_destination_pos = torch.zeros_like(self.target_point_pos)
        # Save initial target position at resample for kick-direction computation.
        self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)
        
        # Blind-zone logic: ball is invisible when robot-ball (x, y) distance is out of range.
        self.blind_distance_min = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.blind_distance_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Target position at last visible frame (robot base frame).
        self.last_visible_target_point_base = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        # Whether currently in blind zone.
        self.is_in_blind_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Height for target_destination.
        self.destination_height = 0.11
        
        # target_destination generation parameters (world-frame based).
        self.destination_center = torch.tensor([0.0, -5.0, self.destination_height], device=self.device)  # Rectangle center (x, y, z).
        self.destination_length = 1.0  # Rectangle length (x-axis).
        self.destination_width = 0.5  # Rectangle width (y-axis).
        
        self.curve_radius_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._radius_offset_min = None
        self._radius_offset_max = None
        curve_cfg = cfg.curve_offset_range or {}
        radius_range = curve_cfg.get("radius")
        if isinstance(radius_range, Sequence) and not isinstance(radius_range, (str, bytes)) and len(radius_range) >= 2:
            self._radius_offset_min = float(radius_range[0])
            self._radius_offset_max = float(radius_range[1])
        elif radius_range is not None:
            value = float(radius_range)
            self._radius_offset_min = value
            self._radius_offset_max = value
        self._target_arc_angle = float(curve_cfg.get("arc_angle", math.pi / 18.0))
        self._target_height = float(curve_cfg.get("height", 0.11))
        marker_cfg = cfg.target_point_marker_cfg
        self.target_point_marker = VisualizationMarkers(marker_cfg) if marker_cfg is not None else None
        dest_marker_cfg = getattr(cfg, "target_destination_marker_cfg", None)
        self.target_destination_marker = VisualizationMarkers(dest_marker_cfg) if dest_marker_cfg is not None else None

        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_soccer_offset(all_env_ids)
        self._compute_soccer_ball_positions(all_env_ids)
        self._update_soccer_ball(all_env_ids)
        self._update_target_points(all_env_ids)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_idx, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_idx, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_idx, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_pelvis_pos_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_pos_w[:, pelvis_index]
    
    @property
    def robot_pelvis_quat_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_quat_w[:, pelvis_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def kick_leg(self) -> torch.Tensor:
        return self.motion_kick_leg[self.motion_idx]

    @property
    def kick_leg_name(self) -> list[str]:
        ids = self.motion_kick_leg[self.motion_idx].tolist()
        return [self._kick_leg_id_to_name.get(i, "unknown") for i in ids]

    def _to_env_id_tensor(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def _sample_soccer_offset(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        if self._radius_offset_min is None or self._radius_offset_max is None:
            self.curve_radius_offset[ids] = 0.0
            return
        if abs(self._radius_offset_max - self._radius_offset_min) < 1e-6:
            self.curve_radius_offset[ids] = self._radius_offset_min
            return

        rand = torch.rand(ids.numel(), device=self.device)
        span = self._radius_offset_max - self._radius_offset_min
        self.curve_radius_offset[ids] = self._radius_offset_min + rand * span

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if isinstance(episode_failed, torch.Tensor):
            episode_failed = episode_failed.to(device=self.device, dtype=torch.bool)
        else:
            episode_failed = torch.tensor(episode_failed, dtype=torch.bool, device=self.device)
        # Clear failure histogram for the current update.
        self._current_bin_failed.zero_()
        # import ipdb; ipdb.set_trace()
        if torch.any(episode_failed):
            # import ipdb; ipdb.set_trace()
            # For failed environments, count the corresponding motion bins.
            failed_env_mask = episode_failed
            failed_motion_idx = self.motion_idx[env_ids][failed_env_mask]                       # [K]
            failed_lengths = self.motion_length[env_ids][failed_env_mask].clamp(min=1).float() # [K]
            failed_steps = self.time_steps[env_ids][failed_env_mask].float()                    # [K]
            # Map time_steps to normalized phase [0, 1], then to bins.
            failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
            failed_bins = torch.clamp((failed_phase * self.bin_count).long(), 0, self.bin_count - 1)  # [K]
            # Accumulate into a 2D histogram via flattened indices.
            flat_idx = failed_motion_idx * self.bin_count + failed_bins                          # [K]
            flat_size = int(self.motion.num_files * self.bin_count)

            # Accumulate safely on GPU to avoid CPU fallback and sync overhead.
            flat_counts = torch.zeros(flat_size, dtype=self._current_bin_failed.dtype, device=self.device)
            if flat_idx.numel() > 0:
                # Ensure indices are on the same device and in long dtype.
                flat_idx = flat_idx.to(self.device).long()
                ones = torch.ones_like(flat_idx, dtype=flat_counts.dtype, device=self.device)
                flat_counts.index_add_(0, flat_idx, ones)

            flat_counts = flat_counts.float()
            # In-place write to keep dtype/device stable.
            self._current_bin_failed[:] = flat_counts.view(self.motion.num_files, self.bin_count)

        # Probability: EMA failure counts plus a uniform prior.
        # Add self.cfg.adaptive_uniform_ratio / (M * B) per element to keep total mass consistent.
        M = max(1, int(self.motion.num_files))
        B = max(1, int(self.bin_count))
        uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
        probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair  # [M, B]
        # Non-causal padding + convolution to smooth along bins per motion.
        probs = torch.nn.functional.pad(
            probs.unsqueeze(1),  # [M, 1, B]
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)         # [M, B]

        # Flatten and sample from joint (motion, bin) distribution.
        probs = probs.view(-1)                                                                    # [M*B]
        probs = probs / (probs.sum() + 1e-12)

        sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)                   # [E]
        sampled_motion = sampled_flat // self.bin_count                                           # [E]
        sampled_bins = sampled_flat % self.bin_count                                              # [E]

        # Map sampled bins to per-motion time_steps with small random offsets.
        self.motion_idx[env_ids] = sampled_motion
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device).float()       # [E]
        sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)              # [E]
        self.time_steps[env_ids] = (sampled_phase * (self.motion_length[env_ids].float() - 1)).long()

        # Metrics for the joint distribution.
        H = -(probs * (probs + 1e-12).log()).sum()
        denom = math.log(self.bin_count * max(1, int(self.motion.num_files)))
        H_norm = H / denom if denom > 1e-12 else torch.tensor(0.0, device=probs.device)
        pmax, imax = probs.max(dim=0)
        top1_motion = (imax // self.bin_count).float()
        top1_bin = (imax % self.bin_count).float() / self.bin_count
        # import ipdb; ipdb.set_trace()

        # Create metric entries only when needed.
        if "sampling_entropy" not in self.metrics or self.metrics["sampling_entropy"].shape[0] != self.num_envs:
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_prob" not in self.metrics or self.metrics["sampling_top1_prob"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_bin" not in self.metrics or self.metrics["sampling_top1_bin"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_motion" not in self.metrics or self.metrics["sampling_top1_motion"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = top1_bin
        self.metrics["sampling_top1_motion"][:] = top1_motion

    def _uniform_sampling(self, env_ids: Sequence[int]):
        # Sample motion and time-step separately to avoid out-of-range issues.
        # First, sample motions.
        motion_indices = torch.randint(0, self.motion.num_files, (len(env_ids),), device=self.device)
        self.motion_idx[env_ids] = motion_indices
        self.motion_length[env_ids] = self.motion.file_lengths[motion_indices]
        
        # Then sample a time-step for each selected motion.
        # time_phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # Start each selected motion from frame 0.
        time_phase = torch.zeros(len(env_ids), device=self.device)

        self.time_steps[env_ids] = (time_phase * (self.motion_length[env_ids].float() - 1)).long()
        

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        if ids.numel() == 0:
            return

        arc_limit = float(self._target_arc_angle)
        base_height = float(self._target_height)

        for env_id in ids:
            motion_idx = int(self.motion_idx[env_id].item())
            motion_len = max(1, int(self.motion_length[env_id].item()))

            first_anchor = self.motion.get_first_frame_anchor_pos(motion_idx, self.motion_anchor_body_index,)
            last_anchor = self.motion.get_last_frame_anchor_pos(motion_idx, self.motion_anchor_body_index, motion_len,)

            radius_vec = last_anchor[:2] - first_anchor[:2]
            radius_sq = torch.dot(radius_vec, radius_vec)
            target_xy = last_anchor[:2]

            radius = torch.sqrt(radius_sq) if float(radius_sq) > 1e-12 else torch.tensor(0.0, device=self.device)
            if float(radius_sq) > 1e-12:
                base_direction = radius_vec / radius
            else:
                base_direction = torch.tensor([1.0, 0.0], device=self.device)

            if arc_limit > 0.0 and float(radius_sq) > 1e-12:
                base_angle = torch.atan2(radius_vec[1], radius_vec[0])
                angle_offset = sample_uniform(-arc_limit, arc_limit, (1,), device=self.device).squeeze(0)
                new_angle = base_angle + angle_offset
                direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)))
            else:
                direction = base_direction

            radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)
            target_xy = first_anchor[:2] + radius * direction

            ball_pos = self.soccer_ball_pos.new_empty(3)
            ball_pos[:2] = target_xy
            ball_pos[2] = base_height
            self.soccer_ball_pos[env_id] = ball_pos

    def _update_target_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        # Also save initial target point for kick-direction computation.
        self.initial_target_point_pos[ids] = self.soccer_ball_pos[ids].clone()

        if self.target_point_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_positions = self.target_point_pos + env_origins
            else:
                world_positions = self.target_point_pos
            self.target_point_marker.visualize(world_positions)

    def _update_target_points_from_sim(self):
        """Read soccer-ball position from simulation each step and update target_point_pos."""
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return
        
        # Read world-space soccer-ball position from simulation.
        ball_world_pos = self.soccer_ball.data.root_pos_w  # [num_envs, 3]
        # Convert to local position relative to env origin.
        self.soccer_ball_pos = ball_world_pos - env_origins
        self.target_point_pos = self.soccer_ball_pos.clone()
        
        # Update visualization marker.
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_world_pos)



    def _update_destination_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        
        # Generate target_destination in world coordinates.
        # Sample destination uniformly within the rectangle.
        rand_x = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_length
        rand_y = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_width
        destination = self.destination_center.expand(ids.numel(), -1) + torch.stack([rand_x, rand_y, torch.zeros_like(rand_x)], dim=1)
        self.target_destination_pos[ids] = destination

        if self.target_destination_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_destination = self.target_destination_pos + env_origins
            else:
                world_destination = self.target_destination_pos
            self.target_destination_marker.visualize(world_destination)
        

    def _update_soccer_ball(self, env_ids: Sequence[int] | torch.Tensor):
        if self.soccer_ball is None or not hasattr(self.soccer_ball, "write_root_state_to_sim"):
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        ball_pos = self.soccer_ball_pos[ids] + env_origins[ids]
        ball_quat = ball_pos.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        
        # Sample initial linear velocity based on config.
        if self.cfg.enable_soccer_ball_init_vel:
            lin_vel_range = self.cfg.soccer_ball_init_lin_vel_range or {}
            lin_vel_ranges = torch.tensor(
                [lin_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
                device=self.device
            )  # [3, 2]
            ball_lin_vel = sample_uniform(
                lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], (ids.numel(), 3), device=self.device
            )
        else:
            ball_lin_vel = ball_pos.new_zeros((ids.numel(), 3))
        
        # Set angular velocity to zero.
        ball_ang_vel = ball_pos.new_zeros((ids.numel(), 3))

        ball_state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = self._to_env_id_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self._sample_soccer_offset(env_ids)
        sampling_strategy = str(self.cfg.sampling_strategy).lower()
        if sampling_strategy == "adaptive":
            self._adaptive_sampling(env_ids)
        elif sampling_strategy == "uniform":
            self._uniform_sampling(env_ids)
        else:
            raise ValueError(f"Unsupported sampling_strategy: {self.cfg.sampling_strategy}")
        self._compute_soccer_ball_positions(env_ids)
        self._update_soccer_ball(env_ids)
        self._update_target_points(env_ids)
        self._update_destination_points(env_ids)
        
        # Sample blind-zone min/max thresholds and reset blind-zone state.
        blind_min_low, blind_min_high = self.cfg.blind_distance_min_range
        blind_max_low, blind_max_high = self.cfg.blind_distance_max_range
        self.blind_distance_min[env_ids] = blind_min_low + torch.rand(env_ids.numel(), device=self.device) * (blind_min_high - blind_min_low)
        self.blind_distance_max[env_ids] = blind_max_low + torch.rand(env_ids.numel(), device=self.device) * (blind_max_high - blind_max_low)
        self.is_in_blind_zone[env_ids] = False
        self.last_visible_target_point_base[env_ids] = 0.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        # Set resample flag so env can refresh observations on next step.
        flag_name = f"{self._state_prefix}_motion_resampled"
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self.num_envs:
            resample_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            resample_flags = resample_flags.to(device=self.device, dtype=torch.bool)
        resample_flags[env_ids] = True
        setattr(self._env, flag_name, resample_flags)

    # Called every step in the IsaacLab main loop.
    def _update_command(self):
        self.kick_contact_tracker.begin_step(self)
        # Increment time_steps; if a sequence ends, resample based on failure statistics.
        self.time_steps += 1
        # env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        env_ids = torch.where(self.time_steps >= self.motion_length)[0]
        self._resample_command(env_ids)
        
        # Update target point each step using current ball position.
        self._update_target_points_from_sim()

        # Continuously refresh pre-kick target until contact occurs; then keep it frozen.
        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    #motion_file: str = MISSING
    motion_files: list[str] = MISSING

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_strategy: str = "uniform"

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.1
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.4

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Target-point marker config; typically overridden in subclasses.
    target_point_marker_cfg: VisualizationMarkersCfg | None = None
    target_destination_marker_cfg: VisualizationMarkersCfg | None = None
    # Offset configuration for arc distribution and destination height.
    curve_offset_range: dict[str, float | tuple[float, float]] | None = None
    
    # Initial soccer-ball velocity configuration.
    enable_soccer_ball_init_vel: bool = False
    soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None
    
    # Blind-zone config: ball is invisible when robot-ball (x, y) distance is outside [min, max].
    blind_distance_min_range: tuple[float, float] = (0.3, 0.5)  # Minimum distance sampling range.
    blind_distance_max_range: tuple[float, float] = (1.5, 2.0)  # Maximum distance sampling range.
