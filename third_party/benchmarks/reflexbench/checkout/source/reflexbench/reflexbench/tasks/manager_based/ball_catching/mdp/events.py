# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_BUCKET_REL_POS_TO_EE = (
    0.009599369764327999,
    0.011815236881375313,
    0.12011093292236328,
)
_BUCKET_REL_QUAT_TO_EE = (
    0.3191247582435608,
    -0.6629222631454468,
    0.6026695966720581,
    -0.30900222063064575,
)


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if env_ids.dim() == 0:
        return env_ids.unsqueeze(0)
    return env_ids


def _ensure_ball_catching_buffers(env: ManagerBasedEnv, history_length: int = 5) -> None:
    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "launch_detected"):
        env.launch_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "launch_timer"):
        env.launch_timer = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "launch_delay"):
        env.launch_delay = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "launch_speed"):
        env.launch_speed = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "launch_pitch"):
        env.launch_pitch = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "launch_yaw"):
        env.launch_yaw = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "ball_in_catch_counter"):
        env.ball_in_catch_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "predicted_intercept_pos"):
        env.predicted_intercept_pos = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    if not hasattr(env, "predicted_intercept_time"):
        env.predicted_intercept_time = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if (
        not hasattr(env, "ball_position_history")
        or env.ball_position_history.shape[1] != history_length
    ):
        env.ball_position_history = torch.zeros(
            env.num_envs, history_length, 3, dtype=torch.float32, device=env.device
        )


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


def _bucket_catch_point_w(
    bucket_pos_w: torch.Tensor,
    bucket_quat_w: torch.Tensor,
    catch_point_z_offset: float,
) -> torch.Tensor:
    """Catch point located along the bucket's local +Z axis."""
    local_offset = torch.zeros_like(bucket_pos_w)
    local_offset[..., 2] = catch_point_z_offset
    return bucket_pos_w + _quat_apply(bucket_quat_w, local_offset)


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    history_length: int = 5,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env, history_length=history_length)

    env.task_phase[env_ids] = 0
    env.launch_detected[env_ids] = False
    env.ball_in_catch_counter[env_ids] = 0
    env.predicted_intercept_pos[env_ids] = 0.0
    env.predicted_intercept_time[env_ids] = 0.0
    env.ball_position_history[env_ids] = 0.0
    # Bucket-to-EE mounting transform is fixed in code and reused unchanged.

    if hasattr(env, "launch_delay"):
        env.launch_timer[env_ids] = env.launch_delay[env_ids]
    else:
        env.launch_timer[env_ids] = 0.0


def reset_ball_to_launcher(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    launcher_pos: tuple[float, float, float] = (1.5, 0.0, 0.45),
    launcher_pos_noise: tuple[float, float, float] = (0.02, 0.02, 0.01),
    launch_speed: float = 3.4,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env)

    ball: RigidObject = env.scene[asset_cfg.name]

    root_state = ball.data.default_root_state.clone()[env_ids]
    launcher_tensor = torch.tensor(launcher_pos, dtype=torch.float32, device=env.device)
    noise_tensor = torch.tensor(launcher_pos_noise, dtype=torch.float32, device=env.device)
    noise = (torch.rand((len(env_ids), 3), device=env.device) * 2.0 - 1.0) * noise_tensor

    root_state[:, :3] = env.scene.env_origins[env_ids] + launcher_tensor + noise
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    root_state[:, 7:] = 0.0

    ball.write_root_state_to_sim(root_state, env_ids=env_ids)

    env.launch_delay[env_ids] = 0.3 + 0.5 * torch.rand(len(env_ids), device=env.device)
    env.launch_timer[env_ids] = env.launch_delay[env_ids]
    env.launch_speed[env_ids] = launch_speed
    env.launch_pitch[env_ids] = torch.deg2rad(50.0 + 15.0 * torch.rand(len(env_ids), device=env.device))
    env.launch_yaw[env_ids] = torch.deg2rad(-15.0 + 30.0 * torch.rand(len(env_ids), device=env.device))


def launch_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    workspace_target_xy: tuple[float, float] = (0.2, 0.0),
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env)

    active_mask = (env.task_phase[env_ids] == 0) & (~env.launch_detected[env_ids])
    active_env_ids = env_ids[active_mask]
    if len(active_env_ids) == 0:
        return

    dt = getattr(env, "_interval_event_dt", env.step_dt)
    env.launch_timer[active_env_ids] -= dt

    launch_env_ids = active_env_ids[env.launch_timer[active_env_ids] <= 0.0]
    if len(launch_env_ids) == 0:
        return

    ball: RigidObject = env.scene[asset_cfg.name]
    ball_pos_local = ball.data.root_pos_w[launch_env_ids, :3] - env.scene.env_origins[launch_env_ids]

    target_xy = torch.tensor(workspace_target_xy, dtype=torch.float32, device=env.device)
    base_yaw = torch.atan2(
        target_xy[1] - ball_pos_local[:, 1],
        target_xy[0] - ball_pos_local[:, 0],
    )
    yaw = base_yaw + env.launch_yaw[launch_env_ids]
    speed = env.launch_speed[launch_env_ids]
    pitch = env.launch_pitch[launch_env_ids]
    horizontal_speed = speed * torch.cos(pitch)

    velocity = torch.zeros((len(launch_env_ids), 6), dtype=torch.float32, device=env.device)
    velocity[:, 0] = horizontal_speed * torch.cos(yaw)
    velocity[:, 1] = horizontal_speed * torch.sin(yaw)
    velocity[:, 2] = speed * torch.sin(pitch)
    ball.write_root_velocity_to_sim(velocity, env_ids=launch_env_ids)

    env.launch_detected[launch_env_ids] = True
    env.launch_timer[launch_env_ids] = 0.0
    env.task_phase[launch_env_ids] = 1


def update_ball_tracking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    catch_height: float = 0.35,
    gravity_magnitude: float = 9.81,
    min_intercept_time: float = 1.0e-3,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env)

    ball: RigidObject = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    ball_pos_w = ball.data.root_pos_w[env_ids, :3]
    ball_vel_w = ball.data.root_lin_vel_w[env_ids, :3]
    ball_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_ids],
        robot.data.root_quat_w[env_ids],
        ball_pos_w,
    )

    env.ball_position_history[env_ids, :-1] = env.ball_position_history[env_ids, 1:].clone()
    env.ball_position_history[env_ids, -1] = ball_pos_b

    tracked_mask = env.launch_detected[env_ids] & (env.task_phase[env_ids] >= 1)
    tracked_env_ids = env_ids[tracked_mask]
    if len(tracked_env_ids) == 0:
        env.predicted_intercept_pos[env_ids] = 0.0
        env.predicted_intercept_time[env_ids] = 0.0
        return

    tracked_ball_pos_w = ball.data.root_pos_w[tracked_env_ids, :3]
    tracked_ball_vel_w = ball.data.root_lin_vel_w[tracked_env_ids, :3]
    z0 = tracked_ball_pos_w[:, 2]
    vz = tracked_ball_vel_w[:, 2]

    discriminant = vz.square() + 2.0 * gravity_magnitude * (z0 - catch_height)
    valid = discriminant >= 0.0

    predicted_time = torch.zeros(len(tracked_env_ids), dtype=torch.float32, device=env.device)
    predicted_pos_b = torch.zeros(len(tracked_env_ids), 3, dtype=torch.float32, device=env.device)

    if valid.any():
        sqrt_disc = torch.sqrt(torch.clamp(discriminant[valid], min=0.0))
        vz_valid = vz[valid]
        t_candidates = torch.stack(
            (
                (vz_valid - sqrt_disc) / gravity_magnitude,
                (vz_valid + sqrt_disc) / gravity_magnitude,
            ),
            dim=1,
        )
        inf = torch.full_like(t_candidates, float("inf"))
        t_candidates = torch.where(t_candidates > min_intercept_time, t_candidates, inf)
        t = torch.min(t_candidates, dim=1).values
        finite_mask = torch.isfinite(t)

        if finite_mask.any():
            valid_env_ids = tracked_env_ids[valid][finite_mask]
            valid_ball_pos_w = tracked_ball_pos_w[valid][finite_mask]
            valid_ball_vel_w = tracked_ball_vel_w[valid][finite_mask]
            valid_t = t[finite_mask]

            intercept_pos_w = valid_ball_pos_w + valid_ball_vel_w * valid_t.unsqueeze(1)
            intercept_pos_w[:, 2] = catch_height
            intercept_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[valid_env_ids],
                robot.data.root_quat_w[valid_env_ids],
                intercept_pos_w,
            )

            predicted_time[valid.nonzero(as_tuple=False).squeeze(-1)[finite_mask]] = valid_t
            predicted_pos_b[valid.nonzero(as_tuple=False).squeeze(-1)[finite_mask]] = intercept_pos_b

    env.predicted_intercept_pos[tracked_env_ids] = predicted_pos_b
    env.predicted_intercept_time[tracked_env_ids] = predicted_time

    invalid_tracked = tracked_env_ids[predicted_time <= 0.0]
    if len(invalid_tracked) > 0:
        env.predicted_intercept_pos[invalid_tracked] = 0.0
        env.predicted_intercept_time[invalid_tracked] = 0.0

    untracked_env_ids = env_ids[~tracked_mask]
    if len(untracked_env_ids) > 0:
        env.predicted_intercept_pos[untracked_env_ids] = 0.0
        env.predicted_intercept_time[untracked_env_ids] = 0.0


def check_phase_transitions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bucket_cfg: SceneEntityCfg = SceneEntityCfg("bucket"),
    catch_hold_steps: int = 3,
    intercept_position_threshold: float = 0.01,
    catch_point_z_offset: float = 0.05,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "task_phase"):
        return

    phase = env.task_phase[env_ids].clone()

    to_phase_1 = (phase == 0) & env.launch_detected[env_ids]
    if to_phase_1.any():
        env.task_phase[env_ids[to_phase_1]] = 1

    to_phase_2 = (phase == 1) & (env.predicted_intercept_time[env_ids] > 0.0)
    if to_phase_2.any():
        env.task_phase[env_ids[to_phase_2]] = 2

    bucket: RigidObject = env.scene[bucket_cfg.name]
    robot = env.scene["robot"]
    bucket_pos_w = bucket.data.root_pos_w[env_ids, :3]
    bucket_quat_w = bucket.data.root_quat_w[env_ids]
    bucket_catch_point_w = _bucket_catch_point_w(bucket_pos_w, bucket_quat_w, catch_point_z_offset)
    bucket_catch_point_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_ids],
        robot.data.root_quat_w[env_ids],
        bucket_catch_point_w,
    )
    dist_to_intercept = torch.norm(
        bucket_catch_point_b - env.predicted_intercept_pos[env_ids], dim=1
    )

    # Debug: print the prediction vs current ee and bucket position
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    # We need to access ball if we want to print its debug info
    if "ball" in env.scene.keys():
        ball = env.scene["ball"]
        if len(env_ids) > 0 and env_ids[0].item() == 0:
            ee_w = ee_frame.data.target_pos_w[0, 0, :]
            ee_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[0:1],
                robot.data.root_quat_w[0:1],
                ee_w.unsqueeze(0),
            )
            bucket_pos_w = bucket.data.root_pos_w[0:1, :3]
            bucket_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[0:1],
                robot.data.root_quat_w[0:1],
                bucket_pos_w,
            )

            pred_b = env.predicted_intercept_pos[0]
            ball_w = ball.data.root_pos_w[0, :3]
            ball_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[0:1],
                robot.data.root_quat_w[0:1],
                ball_w.unsqueeze(0),
            )

            # print(f"  [DEBUG phase {phase[0].item()}] ball_b=({ball_b[0, 0]:.3f}, {ball_b[0, 1]:.3f}, {ball_b[0, 2]:.3f}) "
            #       f"| pred_b=({pred_b[0]:.3f}, {pred_b[1]:.3f}, {pred_b[2]:.3f}) "
            #       f"| bucket_b=({bucket_pos_b[0, 0]:.3f}, {bucket_pos_b[0, 1]:.3f}, {bucket_pos_b[0, 2]:.3f}) "
            #       f"| ee_b=({ee_b[0, 0]:.3f}, {ee_b[0, 1]:.3f}, {ee_b[0, 2]:.3f})")

    to_phase_3 = (
        (phase == 2)
        & (env.predicted_intercept_time[env_ids] > 0.0)
        & (dist_to_intercept < intercept_position_threshold)
    )
    if to_phase_3.any():
        env.task_phase[env_ids[to_phase_3]] = 3

    to_phase_4 = (phase == 3) & (env.ball_in_catch_counter[env_ids] >= catch_hold_steps)
    if to_phase_4.any():
        env.task_phase[env_ids[to_phase_4]] = 4


def check_catch_containment(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    bucket_cfg: SceneEntityCfg = SceneEntityCfg("bucket"),
    catch_radius: float = 0.065,
    catch_depth: float = 0.13,
    catch_hold_steps: int = 3,
    catch_point_z_offset: float = 0.05,
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env)

    ball: RigidObject = env.scene[ball_cfg.name]
    bucket: RigidObject = env.scene[bucket_cfg.name]

    ball_pos_w = ball.data.root_pos_w[env_ids, :3]
    bucket_pos_w = bucket.data.root_pos_w[env_ids, :3]
    bucket_quat_w = bucket.data.root_quat_w[env_ids]
    bucket_catch_point_w = _bucket_catch_point_w(bucket_pos_w, bucket_quat_w, catch_point_z_offset)
    ball_in_catch_frame, _ = subtract_frame_transforms(
        bucket_catch_point_w,
        bucket_quat_w,
        ball_pos_w,
    )

    xy_distance = torch.norm(ball_in_catch_frame[:, :2], dim=1)
    in_radius = xy_distance <= catch_radius
    in_depth = (ball_in_catch_frame[:, 2] <= 0.0) & (ball_in_catch_frame[:, 2] >= -catch_depth)
    in_zone = in_radius & in_depth & env.launch_detected[env_ids]

    if len(env_ids) > 0 and env_ids[0].item() == 0:
        r_xy = xy_distance[0].item()
        dz = ball_in_catch_frame[0, 2].item()

        # print(f"  [CHECK] env 0: r_xy={r_xy:.3f} (max {catch_radius}), dz={dz:.3f} (must be in [-{catch_depth}, 0]) "
        #       f"| in_radius={in_radius[0].item()}, in_depth={in_depth[0].item()}")

    if in_zone.any():
        zone_env_ids = env_ids[in_zone]

        if 0 in zone_env_ids:
            # print(f"  [CATCH] YES! env 0: ball is INSIDE zone (radius<={catch_radius}, depth<={catch_depth})")
            pass

        env.ball_in_catch_counter[zone_env_ids] += 1

        damped_velocity = torch.cat(
            (
                ball.data.root_lin_vel_w[zone_env_ids] * 0.7,
                ball.data.root_ang_vel_w[zone_env_ids] * 0.7,
            ),
            dim=-1,
        )
        ball.write_root_velocity_to_sim(damped_velocity, env_ids=zone_env_ids)

        completed = env.ball_in_catch_counter[zone_env_ids] >= catch_hold_steps
        if completed.any():
            env.task_phase[zone_env_ids[completed]] = 4

    out_zone = env_ids[~in_zone]
    if len(out_zone) > 0:
        env.ball_in_catch_counter[out_zone] = 0


def update_bucket_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bucket_cfg: SceneEntityCfg = SceneEntityCfg("bucket"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_ball_catching_buffers(env)

    bucket: RigidObject = env.scene[bucket_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]
    rel_pos_to_ee = torch.tensor(_BUCKET_REL_POS_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )
    rel_quat_to_ee = torch.tensor(_BUCKET_REL_QUAT_TO_EE, dtype=torch.float32, device=env.device).expand(
        len(env_ids), -1
    )

    bucket_pos_w = ee_pos_w + _quat_apply(ee_quat_w, rel_pos_to_ee)
    bucket_quat_w = _quat_multiply(ee_quat_w, rel_quat_to_ee)

    root_state = bucket.data.default_root_state.clone()[env_ids]
    root_state[:, 0:3] = bucket_pos_w
    root_state[:, 3:7] = bucket_quat_w
    root_state[:, 7:] = 0.0

    bucket.write_root_state_to_sim(root_state, env_ids=env_ids)
