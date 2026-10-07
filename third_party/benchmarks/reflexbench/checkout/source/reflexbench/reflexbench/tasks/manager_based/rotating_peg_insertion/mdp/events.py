# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if env_ids.dim() == 0:
        return env_ids.unsqueeze(0)
    return env_ids


_DISC_DEFAULT_POS = (0.5, 0.0, 0.40)
_DISC_HEIGHT = 0.01
_HOLE_OFFSET_R = 0.07
_HOLE_RADIUS = 0.022
_PEG_RADIUS = 0.005
_HOLE_CLEAR_RADIUS = _HOLE_RADIUS
_PEG_LENGTH = 0.10
_PEG_Z_OFFSET = -0.045
_PEG_DISC_CLEARANCE = 0.002
_INSERTION_DEPTH_THRESHOLD = 0.02
_INSERTION_HOLD_STEPS = 2


def _predicted_hole_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    lead_time: float = 0.0,
) -> torch.Tensor:
    theta = (
        env.disc_angle[env_ids]
        + env.hole_angle_offset[env_ids]
        + env.disc_angular_velocity[env_ids] * lead_time
    )
    hole_pos = env.disc_center_local[env_ids].clone()
    hole_pos[:, 0] = hole_pos[:, 0] + _HOLE_OFFSET_R * torch.cos(theta)
    hole_pos[:, 1] = hole_pos[:, 1] + _HOLE_OFFSET_R * torch.sin(theta)
    hole_pos[:, 2] = hole_pos[:, 2] + _DISC_HEIGHT * 0.5
    return hole_pos


def _update_current_hole_position(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    env.current_hole_world_pos[env_ids] = _predicted_hole_position(env, env_ids)


def _ensure_disc_insertion_buffers(env: ManagerBasedEnv) -> None:
    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "disc_angle"):
        env.disc_angle = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "disc_angular_velocity"):
        env.disc_angular_velocity = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "disc_center_local"):
        env.disc_center_local = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    if not hasattr(env, "hole_angle_offset"):
        env.hole_angle_offset = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "insertion_counter"):
        env.insertion_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "predicted_next_alignment_time"):
        env.predicted_next_alignment_time = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "current_hole_world_pos"):
        env.current_hole_world_pos = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    if not hasattr(env, "face_collision_count"):
        env.face_collision_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    env.task_phase[env_ids] = 0
    env.insertion_counter[env_ids] = 0
    env.predicted_next_alignment_time[env_ids] = 0.0
    env.face_collision_count[env_ids] = 0


def reset_disc_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    disc_pos: tuple[float, float, float] = _DISC_DEFAULT_POS,
    angular_velocity_range: tuple[float, float] = (
        math.radians(90.0),
        math.radians(160.0),
    ),
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    disc: RigidObject = env.scene[asset_cfg.name]
    disc_pos_t = torch.tensor(disc_pos, dtype=torch.float32, device=env.device)

    root_state = disc.data.default_root_state.clone()[env_ids]
    root_state[:, :3] = env.scene.env_origins[env_ids] + disc_pos_t
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:] = 0.0
    disc.write_root_state_to_sim(root_state, env_ids=env_ids)

    env.disc_center_local[env_ids] = disc_pos_t.unsqueeze(0).expand(len(env_ids), -1)

    low, high = angular_velocity_range
    sign = torch.where(torch.rand(len(env_ids), device=env.device) > 0.5, 1.0, -1.0)
    env.disc_angular_velocity[env_ids] = sign * (
        low + (high - low) * torch.rand(len(env_ids), device=env.device)
    )
    env.disc_angle[env_ids] = 2.0 * math.pi * torch.rand(len(env_ids), device=env.device)
    env.hole_angle_offset[env_ids] = 0.0
    _update_current_hole_position(env, env_ids)


def reset_peg_to_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    z_offset: float = _PEG_Z_OFFSET,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)

    peg: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]

    root_state = peg.data.default_root_state.clone()[env_ids]
    root_state[:, 0] = ee_pos_w[:, 0]
    root_state[:, 1] = ee_pos_w[:, 1]
    root_state[:, 2] = ee_pos_w[:, 2] + z_offset
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:] = 0.0
    peg.write_root_state_to_sim(root_state, env_ids=env_ids)


def update_disc_rotation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("disc"),
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    dt = getattr(env, "_interval_event_dt", env.step_dt)
    env.disc_angle[env_ids] += env.disc_angular_velocity[env_ids] * dt
    env.disc_angle[env_ids] = env.disc_angle[env_ids] % (2.0 * math.pi)

    disc: RigidObject = env.scene[asset_cfg.name]
    half_angle = env.disc_angle[env_ids] * 0.5
    qw = torch.cos(half_angle)
    qx = torch.zeros_like(half_angle)
    qy = torch.zeros_like(half_angle)
    qz = torch.sin(half_angle)

    root_state = disc.data.default_root_state.clone()[env_ids]
    root_state[:, :3] = env.scene.env_origins[env_ids] + env.disc_center_local[env_ids]
    root_state[:, 3] = qw
    root_state[:, 4] = qx
    root_state[:, 5] = qy
    root_state[:, 6] = qz
    root_state[:, 7:] = 0.0
    disc.write_root_state_to_sim(root_state, env_ids=env_ids)

    _update_current_hole_position(env, env_ids)


def update_peg_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    z_offset: float = _PEG_Z_OFFSET,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]

    root_state = peg.data.default_root_state.clone()[env_ids]
    root_state[:, 0] = ee_pos_w[:, 0]
    root_state[:, 1] = ee_pos_w[:, 1]
    root_state[:, 2] = ee_pos_w[:, 2] + z_offset
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:] = 0.0

    desired_local = root_state[:, :3] - env.scene.env_origins[env_ids]
    xy_dist = torch.norm(
        desired_local[:, :2] - env.current_hole_world_pos[env_ids, :2],
        dim=1,
    )
    disc_top_z = env.disc_center_local[env_ids, 2] + _DISC_HEIGHT * 0.5
    min_center_z_local = disc_top_z + _PEG_DISC_CLEARANCE + _PEG_LENGTH * 0.5
    min_center_z_world = env.scene.env_origins[env_ids, 2] + min_center_z_local
    would_penetrate_disc = root_state[:, 2] < min_center_z_world
    outside_clear_hole = xy_dist > _HOLE_CLEAR_RADIUS
    face_contact = would_penetrate_disc & outside_clear_hole
    if face_contact.any():
        root_state[face_contact, 2] = min_center_z_world[face_contact]
        env.face_collision_count[env_ids[face_contact]] += 1

    peg.write_root_state_to_sim(root_state, env_ids=env_ids)


def update_hole_marker_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    marker_cfg: SceneEntityCfg = SceneEntityCfg("hole_marker"),
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    marker: RigidObject = env.scene[marker_cfg.name]
    root_state = marker.data.default_root_state.clone()[env_ids]

    root_state[:, 0] = env.scene.env_origins[env_ids, 0] + env.current_hole_world_pos[env_ids, 0]
    root_state[:, 1] = env.scene.env_origins[env_ids, 1] + env.current_hole_world_pos[env_ids, 1]
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + env.current_hole_world_pos[env_ids, 2] + 0.001
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:] = 0.0
    marker.write_root_state_to_sim(root_state, env_ids=env_ids)


def check_phase_transitions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ready_height_above_disc: float = 0.24,
    xy_alignment_threshold: float = _HOLE_RADIUS,
    phase_3_lead_time: float = 0.15,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "task_phase"):
        return

    _update_predicted_alignment(env, env_ids)

    phase = env.task_phase[env_ids].clone()

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_pos_local = ee_pos_w - env.scene.env_origins[env_ids]

    hole_world = env.current_hole_world_pos[env_ids]
    xy_dist = torch.norm(ee_pos_local[:, :2] - hole_world[:, :2], dim=1)
    predicted_hole = _predicted_hole_position(env, env_ids, phase_3_lead_time)
    predicted_xy_dist = torch.norm(ee_pos_local[:, :2] - predicted_hole[:, :2], dim=1)

    disc_top_z = env.disc_center_local[env_ids, 2] + _DISC_HEIGHT * 0.5
    ee_height_above_disc = ee_pos_local[:, 2] - disc_top_z

    disc_center_xy = env.disc_center_local[env_ids, :2]
    ee_dist_to_disc = torch.norm(ee_pos_local[:, :2] - disc_center_xy, dim=1)
    to_phase_1 = (phase == 0) & (ee_dist_to_disc < 0.15) & (ee_height_above_disc < 0.25)
    if to_phase_1.any():
        env.task_phase[env_ids[to_phase_1]] = 1

    to_phase_3 = (
        (phase <= 1)
        & (torch.minimum(xy_dist, predicted_xy_dist) < xy_alignment_threshold * 2.0)
        & (ee_height_above_disc < ready_height_above_disc)
    )
    if to_phase_3.any():
        env.task_phase[env_ids[to_phase_3]] = 3

    to_phase_4 = (phase == 3) & (env.insertion_counter[env_ids] >= _INSERTION_HOLD_STEPS)
    if to_phase_4.any():
        env.task_phase[env_ids[to_phase_4]] = 4


def _update_predicted_alignment(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    _ensure_disc_insertion_buffers(env)

    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_pos_local = ee_pos_w - env.scene.env_origins[env_ids]

    disc_cx = env.disc_center_local[env_ids, 0]
    disc_cy = env.disc_center_local[env_ids, 1]
    ee_angle = torch.atan2(ee_pos_local[:, 1] - disc_cy, ee_pos_local[:, 0] - disc_cx)

    current_hole_angle = env.disc_angle[env_ids] + env.hole_angle_offset[env_ids]
    current_hole_angle = current_hole_angle % (2.0 * math.pi)
    ee_angle_norm = ee_angle % (2.0 * math.pi)

    omega = env.disc_angular_velocity[env_ids]

    angle_diff = (ee_angle_norm - current_hole_angle) % (2.0 * math.pi)

    positive_mask = omega > 1e-6
    negative_mask = omega < -1e-6
    static_mask = ~positive_mask & ~negative_mask

    predicted_time = torch.full((len(env_ids),), float("inf"), device=env.device)
    if positive_mask.any():
        predicted_time[positive_mask] = angle_diff[positive_mask] / omega[positive_mask].abs()
    if negative_mask.any():
        reverse_diff = (2.0 * math.pi - angle_diff[negative_mask]) % (2.0 * math.pi)
        predicted_time[negative_mask] = reverse_diff / omega[negative_mask].abs()
    if static_mask.any():
        predicted_time[static_mask] = float("inf")

    predicted_time = torch.clamp(predicted_time, min=0.0, max=30.0)
    env.predicted_next_alignment_time[env_ids] = predicted_time


def check_insertion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hole_radius: float = _HOLE_CLEAR_RADIUS,
    insertion_depth: float = _INSERTION_DEPTH_THRESHOLD,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_disc_insertion_buffers(env)

    peg: RigidObject = env.scene["peg"]
    peg_pos_w = peg.data.root_pos_w[env_ids, :3]
    peg_pos_local = peg_pos_w - env.scene.env_origins[env_ids]
    peg_tip = peg_pos_local.clone()
    peg_tip[:, 2] = peg_pos_local[:, 2] - _PEG_LENGTH * 0.5

    hole_world = env.current_hole_world_pos[env_ids]
    xy_dist = torch.norm(peg_tip[:, :2] - hole_world[:, :2], dim=1)
    in_radius = xy_dist <= hole_radius

    disc_top_z = env.disc_center_local[env_ids, 2] + _DISC_HEIGHT * 0.5
    peg_depth_below_disc = disc_top_z - peg_tip[:, 2]
    deep_enough = peg_depth_below_disc >= insertion_depth

    in_hole = in_radius & deep_enough & (env.task_phase[env_ids] >= 3)

    if in_hole.any():
        env.insertion_counter[env_ids[in_hole]] += 1
        completed = env.insertion_counter[env_ids[in_hole]] >= _INSERTION_HOLD_STEPS
        if completed.any():
            completed_ids = env_ids[in_hole][completed]
            env.task_phase[completed_ids] = 4

    out_hole = env_ids[~in_hole]
    if len(out_hole) > 0:
        env.insertion_counter[out_hole] = 0
