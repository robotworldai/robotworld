# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the toss-into-box task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Match check_phase_transitions' release detection thresholds.
_GRIPPER_OPEN_THRESHOLD: float = 0.06
_RELEASE_DIST: float = 0.08


def _ball_released_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-env boolean: has the ball *physically* left the gripper?

    Evaluated every termination step (no 40 ms interval lag).  Mirrors the
    logic in ``mdp.events.check_phase_transitions`` so success / miss
    judgement and the phase machine agree on what "released" means.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_width = robot.data.joint_pos[:, finger_joint_ids].sum(dim=1)
    is_gripper_open = finger_width >= _GRIPPER_OPEN_THRESHOLD
    return is_gripper_open | (ee_object_distance > _RELEASE_DIST)


def object_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """True when the ball lies inside the target box volume (env-local coords).

    Catch volume includes a small margin beyond the nominal KLT opening so a
    ball resting against the inner wall/rim is still counted as inside.
    """
    if not hasattr(env, "box_target_position"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_center = env.box_target_position

    half_x = _BOX_HALF_X
    half_y = _BOX_HALF_Y
    in_x = (pos_local[:, 0] >= box_center[:, 0] - half_x) & (
        pos_local[:, 0] <= box_center[:, 0] + half_x
    )
    in_y = (pos_local[:, 1] >= box_center[:, 1] - half_y) & (
        pos_local[:, 1] <= box_center[:, 1] + half_y
    )
    in_z = (pos_local[:, 2] >= _BOX_Z_MIN) & (pos_local[:, 2] <= _BOX_Z_MAX)
    return in_x & in_y & in_z


# Nominal KLT opening is about 0.30 m x 0.20 m.  The tracked pose is the ball
# center, so add roughly one ball radius plus a small tolerance to avoid false
# negatives when the ball is visibly inside but leaning against a side wall.
_BOX_HALF_X: float = 0.15
_BOX_HALF_Y: float = 0.095
_BOX_Z_MIN: float = 0.015
_BOX_Z_MAX: float = 0.18
_BOX_DWELL_STEPS: int = 8
_BOX_SETTLED_LIN_SPEED_MAX: float = 0.35


def _ball_in_box_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-env boolean: is the ball currently inside the catch volume?"""
    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_center = env.box_target_position

    in_x = (pos_local[:, 0] >= box_center[:, 0] - _BOX_HALF_X) & (
        pos_local[:, 0] <= box_center[:, 0] + _BOX_HALF_X
    )
    in_y = (pos_local[:, 1] >= box_center[:, 1] - _BOX_HALF_Y) & (
        pos_local[:, 1] <= box_center[:, 1] + _BOX_HALF_Y
    )
    in_z = (pos_local[:, 2] >= _BOX_Z_MIN) & (pos_local[:, 2] <= _BOX_Z_MAX)
    return in_x & in_y & in_z


def _ball_settled_in_box_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """True when the ball is inside the catch volume and moving slowly."""
    obj: RigidObject = env.scene[object_cfg.name]
    in_box = _ball_in_box_mask(env, object_cfg)
    lin_speed = torch.norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    return in_box & (lin_speed <= _BOX_SETTLED_LIN_SPEED_MAX)


def task_completed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sticky success flag, evaluated every termination step.

    Why this is *not* simply ``task_phase == 2``: success should mean the ball
    actually stayed in the box, not that its center touched the catch volume
    for one frame while skimming the rim.

    Two countermeasures:

    1. We evaluate the in-box test directly on every termination step.
    2. The ball must remain in the catch volume while moving slowly for
       ``_BOX_DWELL_STEPS`` consecutive termination steps before success is
       latched.
    3. The release check is done inline using gripper width + ball-EE
       distance, identical to ``check_phase_transitions`` but with no
       interval lag.

    Result is latched onto ``env.task_succeeded`` only after the dwell
    requirement is met.  The latch and dwell counter are cleared at reset by
    ``reset_task_phase``.
    """
    if not hasattr(env, "box_target_position"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(env, "task_succeeded"):
        env.task_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "box_dwell_steps"):
        env.box_dwell_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    released = _ball_released_mask(env, object_cfg, robot_cfg, ee_frame_cfg)
    settled_in_box_now = _ball_settled_in_box_mask(env, object_cfg) & released
    env.box_dwell_steps = torch.where(
        settled_in_box_now,
        torch.clamp(env.box_dwell_steps + 1, max=_BOX_DWELL_STEPS),
        torch.zeros_like(env.box_dwell_steps),
    )
    env.task_succeeded = env.task_succeeded | (env.box_dwell_steps >= _BOX_DWELL_STEPS)

    # Keep the legacy task_phase in sync so rewards / observers / policy
    # code that still reads it observes the success transition.
    if hasattr(env, "task_phase"):
        promote = env.task_succeeded & (env.task_phase < 2)
        if promote.any():
            env.task_phase[promote] = 2

    return env.task_succeeded


def object_missed_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    floor_z: float = 0.02,
) -> torch.Tensor:
    """True when the released ball has landed on the ground outside the box.

    Conditions (all must hold):
    - ball has *physically* left the gripper (inline release check)
    - ball z is at or below ``floor_z`` (reached the ground)
    - the ball's XY position is outside the catch rectangle
    - the episode has not already been latched as a success

    The last clause prevents a false-miss if the ball briefly enters the
    box (success latched) and then bounces out onto the floor.
    """
    if not hasattr(env, "box_target_position"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    pos_local = obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_center = env.box_target_position

    released = _ball_released_mask(env, object_cfg, robot_cfg, ee_frame_cfg)
    on_floor = pos_local[:, 2] <= floor_z

    outside_x = (pos_local[:, 0] < box_center[:, 0] - _BOX_HALF_X) | (
        pos_local[:, 0] > box_center[:, 0] + _BOX_HALF_X
    )
    outside_y = (pos_local[:, 1] < box_center[:, 1] - _BOX_HALF_Y) | (
        pos_local[:, 1] > box_center[:, 1] + _BOX_HALF_Y
    )
    outside_xy = outside_x | outside_y

    not_yet_succeeded = (
        ~env.task_succeeded
        if hasattr(env, "task_succeeded")
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )

    return released & on_floor & outside_xy & not_yet_succeeded
