# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

from ..constants import (
    BALL_START_POS,
    RAMP_EXIT_LINE_Y_MAX,
    RAMP_EXIT_LINE_Y_MIN,
    RAMP_EXIT_POS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if env_ids.dim() == 0:
        return env_ids.unsqueeze(0)
    return env_ids


# ---------------------------------------------------------------------------
# Ramp / ball constants (must match rolling_ball_interception_env_cfg.py)
# ---------------------------------------------------------------------------
_BALL_START_POS = BALL_START_POS
_BALL_START_POS_NOISE = (0.0, 0.30, 0.0)
# Ramp exit is the minimum-X edge line of the white platform below the ramp.
_RAMP_EXIT_X = RAMP_EXIT_POS[0]
_RAMP_EXIT_Z = RAMP_EXIT_POS[2]
_CATCHER_REL_POS_TO_EE = (
    0.009599369764327999,
    0.011815236881375313,
    0.12011093292236328,
)
_CATCHER_REL_QUAT_TO_EE = (
    0.3191247582435608,
    -0.6629222631454468,
    0.6026695966720581,
    -0.30900222063064575,
)
_INTERCEPT_EE_QUAT = (0.0, 1.0, 0.0, 0.0)
_CATCHER_MOUTH_CENTER_REL_POS_TO_CATCHER = (
    0.0,
    0.0,
    0.05,
)
# Hardcoded mouth intercept X / Z in robot base frame (env-origin-relative).
_HARDCODED_MOUTH_X = 0.55
_HARDCODED_MOUTH_Z = 0.2

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternion (w, x, y, z)."""
    quat_xyz = quat[..., 1:]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(quat_xyz, t, dim=-1)

# Keep _RAMP_EXIT_POS for observation compatibility (Y=0 line midpoint).
_RAMP_EXIT_POS = RAMP_EXIT_POS


def _catcher_mouth_center_w(
    catcher_pos_w: torch.Tensor,
    catcher_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Cup mouth center in world frame from the current catcher pose."""
    mouth_rel_pos = torch.tensor(
        _CATCHER_MOUTH_CENTER_REL_POS_TO_CATCHER,
        dtype=torch.float32,
        device=catcher_pos_w.device,
    ).expand_as(catcher_pos_w)
    return catcher_pos_w + _quat_apply(catcher_quat_w, mouth_rel_pos)


def _ensure_interception_buffers(env: ManagerBasedEnv) -> None:
    """Lazily create task-specific buffers on the env instance."""
    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "rolling_detected"):
        env.rolling_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "gate_timer"):
        env.gate_timer = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "gate_delay"):
        env.gate_delay = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "ball_in_catcher_counter"):
        env.ball_in_catcher_counter = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
    if not hasattr(env, "ramp_exit_pos"):
        env.ramp_exit_pos = (
            torch.tensor(_RAMP_EXIT_POS, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .expand(env.num_envs, -1)
            .clone()
        )
    if not hasattr(env, "predicted_intercept_pos"):
        env.predicted_intercept_pos = torch.zeros(
            env.num_envs, 3, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "predicted_intercept_mouth_pos"):
        env.predicted_intercept_mouth_pos = torch.zeros(
            env.num_envs, 3, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "predicted_intercept_time"):
        env.predicted_intercept_time = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "ball_lane_y"):
        env.ball_lane_y = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )


# -- reset events ----------------------------------------------------------


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset all task-specific state buffers."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_interception_buffers(env)

    env.task_phase[env_ids] = 1  # start in rolling phase immediately
    env.rolling_detected[env_ids] = True  # ball released from the start
    env.ball_in_catcher_counter[env_ids] = 0
    env.predicted_intercept_pos[env_ids] = 0.0
    env.predicted_intercept_mouth_pos[env_ids] = 0.0
    env.predicted_intercept_time[env_ids] = 0.0

    env.gate_delay[env_ids] = 0.0
    env.gate_timer[env_ids] = 0.0


def reset_ball_to_ramp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    start_pos: tuple[float, float, float] = _BALL_START_POS,
    start_pos_noise: tuple[float, float, float] = _BALL_START_POS_NOISE,
) -> None:
    """Place the ball at the top of the ramp with zero velocity."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_interception_buffers(env)

    ball: RigidObject = env.scene[asset_cfg.name]

    root_state = ball.data.default_root_state.clone()[env_ids]
    pos_tensor = torch.tensor(start_pos, dtype=torch.float32, device=env.device)
    noise_tensor = torch.tensor(start_pos_noise, dtype=torch.float32, device=env.device)
    noise = (torch.rand((len(env_ids), 3), device=env.device) * 2.0 - 1.0) * noise_tensor

    root_state[:, :3] = env.scene.env_origins[env_ids] + pos_tensor + noise
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    root_state[:, 7:] = 0.0  # zero velocity

    env.ball_lane_y[env_ids] = root_state[:, 1]

    ball.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_catcher_in_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    catcher_cfg: SceneEntityCfg = SceneEntityCfg("catcher"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> None:
    """Place the catcher (mug) in the gripper and close fingers at episode start.

    Positions the mug relative to the current EE pose (cup opening up) and sets
    the finger joints to a closed position to physically grasp the handle.
    """
    env_ids = _resolve_env_ids(env, env_ids)

    catcher: RigidObject = env.scene[catcher_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]
    rel_pos_to_ee = torch.tensor(_CATCHER_REL_POS_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )
    rel_quat_to_ee = torch.tensor(_CATCHER_REL_QUAT_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )
    catcher_pos_w = ee_pos_w + _quat_apply(ee_quat_w, rel_pos_to_ee)
    catcher_quat_w = _quat_multiply(ee_quat_w, rel_quat_to_ee)

    # Place mug with the same fixed mount transform used in ball catching.
    root_state = catcher.data.default_root_state.clone()[env_ids]
    root_state[:, 0:3] = catcher_pos_w
    root_state[:, 3:7] = catcher_quat_w
    root_state[:, 7:] = 0.0
    catcher.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Close gripper fingers to grasp the mug handle
    finger_ids = robot.find_joints(["panda_finger_joint.*"])[0]
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_pos[:, finger_ids] = 0.0  # fully closed
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# -- interval events -------------------------------------------------------


def release_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Gate mechanism: hold ball stationary until gate_timer expires, then let gravity roll it."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_interception_buffers(env)

    # Only process environments still waiting for release (phase 0)
    waiting_mask = (env.task_phase[env_ids] == 0) & (~env.rolling_detected[env_ids])
    waiting_env_ids = env_ids[waiting_mask]
    if len(waiting_env_ids) == 0:
        return

    dt = getattr(env, "_interval_event_dt", env.step_dt)
    env.gate_timer[waiting_env_ids] -= dt

    still_waiting = waiting_env_ids[env.gate_timer[waiting_env_ids] > 0.0]
    if len(still_waiting) > 0:
        ball: RigidObject = env.scene[asset_cfg.name]
        start_pos = torch.tensor(_BALL_START_POS, dtype=torch.float32, device=env.device)
        root_state = ball.data.default_root_state.clone()[still_waiting]
        root_state[:, :3] = env.scene.env_origins[still_waiting] + start_pos
        root_state[:, 1] = env.ball_lane_y[still_waiting]
        root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
        root_state[:, 7:] = 0.0
        ball.write_root_state_to_sim(root_state, env_ids=still_waiting)

    # Environments where gate just opened: release ball
    released = waiting_env_ids[env.gate_timer[waiting_env_ids] <= 0.0]
    if len(released) > 0:
        env.rolling_detected[released] = True
        env.gate_timer[released] = 0.0
        env.task_phase[released] = 1


def update_ball_tracking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Predict where the ball will cross the ramp exit edge.

    The physical target is the mounted catcher/mug mouth center, not the EE origin:
      - Mouth X fixed slightly behind the ramp exit line
      - Mouth Z fixed on the white platform exit line
      - Mouth Y fixed to the ball's sampled initial lane Y
      - Catcher root and EE target are back-computed from fixed transforms
    """
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_interception_buffers(env)

    tracked_mask = env.rolling_detected[env_ids] & (env.task_phase[env_ids] >= 1)
    tracked_env_ids = env_ids[tracked_mask]

    if len(tracked_env_ids) == 0:
        env.predicted_intercept_pos[env_ids] = 0.0
        env.predicted_intercept_mouth_pos[env_ids] = 0.0
        env.predicted_intercept_time[env_ids] = 0.0
        return

    ball: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    ball_pos_w = ball.data.root_pos_w[tracked_env_ids, :3]
    ball_vel_w = ball.data.root_lin_vel_w[tracked_env_ids, :3]

    # Time for ball to reach the white platform's minimum-X exit line
    exit_x_w = env.scene.env_origins[tracked_env_ids, 0] + _RAMP_EXIT_X
    remaining_dist_x = (ball_pos_w[:, 0] - exit_x_w).clamp(min=0.0)
    speed_toward_exit = (-ball_vel_w[:, 0]).clamp(min=0.01)
    t_intercept = (remaining_dist_x / speed_toward_exit).clamp(min=0.0, max=5.0)
    env.predicted_intercept_time[tracked_env_ids] = t_intercept

    # Predict the ball's lateral position at the exit line from its current
    # state instead of freezing the initial sampled lane. This fixes edge cases
    # where rail contacts or drift make the real exit Y diverge from start Y.
    lane_y_w = ball_pos_w[:, 1] + ball_vel_w[:, 1] * t_intercept
    lane_y_w = lane_y_w.clamp(
        min=RAMP_EXIT_LINE_Y_MIN + env.scene.env_origins[tracked_env_ids, 1],
        max=RAMP_EXIT_LINE_Y_MAX + env.scene.env_origins[tracked_env_ids, 1],
    )

    # Target the current physical cup mouth center at the exit line.
    intercept_mouth_pos_w = torch.zeros(len(tracked_env_ids), 3, device=env.device)
    intercept_mouth_pos_w[:, 0] = env.scene.env_origins[tracked_env_ids, 0] + _HARDCODED_MOUTH_X
    intercept_mouth_pos_w[:, 1] = lane_y_w
    intercept_mouth_pos_w[:, 2] = (
        env.scene.env_origins[tracked_env_ids, 2] + _HARDCODED_MOUTH_Z
    )

    catcher = env.scene["catcher"]
    ee_frame = env.scene["ee_frame"]

    ee_pos_w = ee_frame.data.target_pos_w[tracked_env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[tracked_env_ids, 0, :]
    catcher_pos_w = catcher.data.root_pos_w[tracked_env_ids, :3]
    catcher_quat_w = catcher.data.root_quat_w[tracked_env_ids, :]
    mouth_pos_w = _catcher_mouth_center_w(catcher_pos_w, catcher_quat_w)

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w[tracked_env_ids],
        robot.data.root_quat_w[tracked_env_ids],
        ee_pos_w,
        ee_quat_w,
    )
    mouth_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[tracked_env_ids],
        robot.data.root_quat_w[tracked_env_ids],
        mouth_pos_w,
    )
    intercept_mouth_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[tracked_env_ids],
        robot.data.root_quat_w[tracked_env_ids],
        intercept_mouth_pos_w,
    )
    mouth_rel_to_ee, _ = subtract_frame_transforms(
        ee_pos_b,
        ee_quat_b,
        mouth_pos_b,
    )
    intercept_quat_b = torch.tensor(
        _INTERCEPT_EE_QUAT,
        dtype=torch.float32,
        device=env.device,
    ).expand(len(tracked_env_ids), -1)
    intercept_pos_b = intercept_mouth_pos_b - _quat_apply(intercept_quat_b, mouth_rel_to_ee)

    env.predicted_intercept_pos[tracked_env_ids] = intercept_pos_b
    env.predicted_intercept_mouth_pos[tracked_env_ids] = intercept_mouth_pos_b

    untracked_env_ids = env_ids[~tracked_mask]
    if len(untracked_env_ids) > 0:
        env.predicted_intercept_pos[untracked_env_ids] = 0.0
        env.predicted_intercept_mouth_pos[untracked_env_ids] = 0.0
        env.predicted_intercept_time[untracked_env_ids] = 0.0


def check_phase_transitions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    catch_hold_steps: int = 3,
    intercept_position_threshold: float = 0.001,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "task_phase"):
        return

    phase = env.task_phase[env_ids].clone()

    to_phase_2 = (phase == 1) & (
        torch.norm(env.predicted_intercept_pos[env_ids], dim=1) > 1e-4
    )
    if to_phase_2.any():
        env.task_phase[env_ids[to_phase_2]] = 2

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene["robot"]
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_ids],
        robot.data.root_quat_w[env_ids],
        ee_pos_w,
    )
    dist_to_intercept = torch.norm(
        ee_pos_b - env.predicted_intercept_pos[env_ids], dim=1
    )

    to_phase_3 = (
        (phase == 2)
        & (torch.norm(env.predicted_intercept_pos[env_ids], dim=1) > 1e-4)
        & (dist_to_intercept < intercept_position_threshold)
    )
    if to_phase_3.any():
        env.task_phase[env_ids[to_phase_3]] = 3

    to_phase_4 = (phase == 3) & (env.ball_in_catcher_counter[env_ids] >= catch_hold_steps)
    if to_phase_4.any():
        env.task_phase[env_ids[to_phase_4]] = 4


def check_catch_containment(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    catcher_cfg: SceneEntityCfg = SceneEntityCfg("catcher"),
    catch_radius: float = 0.08,
    catch_depth: float = 0.12,
    catch_hold_steps: int = 3,
) -> None:
    """Check if the ball is contained inside the virtual catcher zone below the mouth."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_interception_buffers(env)

    ball: RigidObject = env.scene[ball_cfg.name]
    catcher: RigidObject = env.scene[catcher_cfg.name]

    ball_pos_w = ball.data.root_pos_w[env_ids, :3]
    catcher_pos_w = catcher.data.root_pos_w[env_ids, :3]
    catcher_quat_w = catcher.data.root_quat_w[env_ids, :]
    mouth_rel_pos_to_catcher = torch.tensor(
        _CATCHER_MOUTH_CENTER_REL_POS_TO_CATCHER,
        dtype=torch.float32,
        device=env.device,
    ).expand(len(env_ids), -1)
    mouth_pos_w = catcher_pos_w + _quat_apply(catcher_quat_w, mouth_rel_pos_to_catcher)

    xy_distance = torch.norm(ball_pos_w[:, :2] - mouth_pos_w[:, :2], dim=1)
    in_radius = xy_distance <= catch_radius
    in_depth = (ball_pos_w[:, 2] <= mouth_pos_w[:, 2]) & (
        ball_pos_w[:, 2] >= mouth_pos_w[:, 2] - catch_depth
    )
    in_zone = in_radius & in_depth & env.rolling_detected[env_ids]

    if in_zone.any():
        zone_env_ids = env_ids[in_zone]
        env.ball_in_catcher_counter[zone_env_ids] += 1

        # Damp the ball velocity to simulate capture absorption
        damped_velocity = torch.cat(
            (
                ball.data.root_lin_vel_w[zone_env_ids] * 0.7,
                ball.data.root_ang_vel_w[zone_env_ids] * 0.7,
            ),
            dim=-1,
        )
        ball.write_root_velocity_to_sim(damped_velocity, env_ids=zone_env_ids)

        completed = (
            (env.ball_in_catcher_counter[zone_env_ids] >= catch_hold_steps)
            & (env.task_phase[zone_env_ids] >= 1)
        )
        if completed.any():
            env.task_phase[zone_env_ids[completed]] = 4

    out_zone = env_ids[~in_zone]
    if len(out_zone) > 0:
        env.ball_in_catcher_counter[out_zone] = 0


def update_catcher_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    catcher_cfg: SceneEntityCfg = SceneEntityCfg("catcher"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> None:
    """Teleport the catcher to follow the EE with fixed relative mount pose."""
    env_ids = _resolve_env_ids(env, env_ids)

    catcher: RigidObject = env.scene[catcher_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]
    rel_pos_to_ee = torch.tensor(_CATCHER_REL_POS_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )
    rel_quat_to_ee = torch.tensor(_CATCHER_REL_QUAT_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )
    catcher_pos_w = ee_pos_w + _quat_apply(ee_quat_w, rel_pos_to_ee)
    catcher_quat_w = _quat_multiply(ee_quat_w, rel_quat_to_ee)

    root_state = catcher.data.default_root_state.clone()[env_ids]
    root_state[:, 0:3] = catcher_pos_w
    root_state[:, 3:7] = catcher_quat_w
    root_state[:, 7:] = 0.0

    catcher.write_root_state_to_sim(root_state, env_ids=env_ids)
