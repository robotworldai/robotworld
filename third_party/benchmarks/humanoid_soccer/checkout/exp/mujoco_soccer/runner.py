from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from .constants import (
    ACTION_SCALE_ARRAY,
    BALL_RADIUS,
    DEFAULT_JOINT_POS_ARRAY,
    ISAACLAB_TO_MUJOCO_REINDEX,
    JOINT_DAMPING_ARRAY,
    JOINT_EFFORT_LIMIT_ARRAY,
    JOINT_STIFFNESS_ARRAY,
)
from .math_utils import normalize_quat, quat_apply, sample_disk
from .metrics import goal_crossed
from .motion import select_motion
from .observations import build_observation
from .policy import PolicyBank, parse_metadata_array
from .scene import MujocoSoccer
from .schemas import ExperimentConfig, MotionClip, SpawnState, TrialResult


def sample_ball_local_xy(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(cfg.ball_forward_range[0], cfg.ball_forward_range[1]),
            rng.uniform(cfg.ball_lateral_range[0], cfg.ball_lateral_range[1]),
        ],
        dtype=np.float64,
    )


def _finite_difference(values: np.ndarray, delta: float) -> np.ndarray:
    return (values[1:] - values[:-1]) * delta


def _mean_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.linalg.norm(values, axis=-1).mean())


def _seq_jerk(global_pos: np.ndarray, delta: float) -> float:
    if global_pos.shape[0] < 4:
        return math.nan
    vel = _finite_difference(global_pos, delta)
    acc = _finite_difference(vel, delta)
    jerk = _finite_difference(acc, delta)
    return float(np.abs(jerk).sum(axis=-1).max(axis=-1).mean())


def eval_baseline_metrics(
    pol_global_pos: np.ndarray,
    ref_global_pos: np.ndarray,
    pol_dof_pos: np.ndarray,
    ref_dof_pos: np.ndarray,
    *,
    delta_per_frame: bool = True,
    fps: float = 1.0,
    success_threshold: float = 0.3,
) -> dict[str, float]:
    delta = 1.0 if delta_per_frame else float(fps)

    pol_zero = pol_global_pos - pol_global_pos[0, 0, :]
    ref_zero = ref_global_pos - ref_global_pos[0, 0, :]
    diff_pos = pol_zero - ref_zero

    frame_global_error = np.linalg.norm(diff_pos, axis=-1).mean(axis=-1)
    root_relative_position = pol_zero - pol_zero[:, 0:1, :]
    root_relative_position_ref = ref_zero - ref_zero[:, 0:1, :]
    relative_diff_pos = root_relative_position - root_relative_position_ref
    frame_relative_error = np.linalg.norm(relative_diff_pos, axis=-1).mean(axis=-1)

    pol_vel = _finite_difference(pol_zero, delta)
    ref_vel = _finite_difference(ref_global_pos, delta)
    pol_acc = _finite_difference(pol_vel, delta)
    ref_acc = _finite_difference(ref_vel, delta)

    return {
        "is_succ_phc": float(frame_global_error.max() <= success_threshold),
        "is_succ_phc_rel": float(frame_relative_error.max() <= success_threshold),
        "E_gmpjpe": float(frame_global_error.mean() * 1000.0),
        "E_mpjpe": float(frame_relative_error.mean() * 1000.0),
        "E_dof_mpjpe": _mean_norm(pol_dof_pos - ref_dof_pos) * 1000.0,
        "E_vel": _mean_norm(pol_vel[:, 0:1, :] - ref_vel[:, 0:1, :]) * 1000.0,
        "E_acc": _mean_norm(pol_acc[:, 0:1, :] - ref_acc[:, 0:1, :]) * 1000.0,
        "E_contact_acc": math.nan,
        "seq_jerk": _seq_jerk(pol_global_pos, delta),
        "seq_ref_jerk": _seq_jerk(ref_global_pos, delta),
    }


def resolve_goal(
    cfg: ExperimentConfig,
    robot_xy: np.ndarray,
    initial_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.goal_frame == "robot":
        goal_offset_world = quat_apply(initial_quat, np.array([cfg.goal[0], cfg.goal[1], 0.0], dtype=np.float64))
        goal_world = np.array([robot_xy[0] + goal_offset_world[0], robot_xy[1] + goal_offset_world[1], BALL_RADIUS])
        goal_normal = quat_apply(initial_quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))[:2]
    else:
        goal_world = np.array([cfg.goal[0], cfg.goal[1], BALL_RADIUS], dtype=np.float64)
        goal_normal = goal_world[:2] - robot_xy

    goal_normal_norm = np.linalg.norm(goal_normal)
    if goal_normal_norm < 1.0e-9:
        goal_normal = np.array([1.0, 0.0], dtype=np.float64)
    else:
        goal_normal = goal_normal / goal_normal_norm
    return goal_world, goal_normal


def sample_ball_velocity(cfg: ExperimentConfig, rng: np.random.Generator, initial_quat: np.ndarray) -> np.ndarray:
    if not cfg.enable_ball_vel:
        return np.zeros(2, dtype=np.float64)
    speed = rng.uniform(cfg.ball_speed_range[0], cfg.ball_speed_range[1])
    direction = rng.uniform(0.0, 2.0 * math.pi)
    ball_vel_local = np.array(
        [
            speed * math.cos(direction),
            speed * math.sin(direction),
            0.0,
        ],
        dtype=np.float64,
    )
    return quat_apply(initial_quat, ball_vel_local)[:2]


def sample_non_overlapping_spawn(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    env: MujocoSoccer,
    clips: list[MotionClip],
) -> SpawnState:
    """Sample a spawn state and reject any initial ball-robot overlap."""
    last_state: tuple[np.ndarray, np.ndarray, str] | None = None
    for attempt in range(1, cfg.ball_spawn_max_attempts + 1):
        robot_xy = sample_disk(rng, cfg.birth_radius)
        ball_local_xy = sample_ball_local_xy(cfg, rng)
        motion, motion_error = select_motion(clips, ball_local_xy)
        initial_quat = normalize_quat(motion.body_quat_w[0, 0].astype(np.float64))
        ball_world_xy = robot_xy + quat_apply(initial_quat, np.array([ball_local_xy[0], ball_local_xy[1], 0.0]))[:2]
        ball_world = np.array([ball_world_xy[0], ball_world_xy[1], BALL_RADIUS], dtype=np.float64)
        goal_world, goal_normal = resolve_goal(cfg, robot_xy, initial_quat)
        ball_vel_world = sample_ball_velocity(cfg, rng, initial_quat)
        translation = env.reset(motion, robot_xy, ball_world, ball_vel_world)
        env.set_goal_marker(goal_world, goal_normal)

        if not env.has_ball_robot_contact():
            return SpawnState(
                robot_xy=robot_xy,
                ball_local_xy=ball_local_xy,
                ball_world=ball_world,
                ball_vel_world=ball_vel_world,
                goal_world=goal_world,
                goal_normal=goal_normal,
                motion=motion,
                motion_error=motion_error,
                translation=translation,
                attempts=attempt,
            )
        last_state = (robot_xy, ball_local_xy, motion.name)

    robot_xy, ball_local_xy, motion_name = last_state if last_state is not None else (np.zeros(2), np.zeros(2), "unknown")
    raise RuntimeError(
        "Failed to sample a non-overlapping ball spawn after "
        f"{cfg.ball_spawn_max_attempts} attempts. Last sample: robot_xy={robot_xy.tolist()}, "
        f"ball_local_xy={ball_local_xy.tolist()}, motion={motion_name}. "
        "Try moving --ball-forward-range farther forward, narrowing --ball-lateral-range, "
        "or increasing --ball-spawn-max-attempts."
    )


def run_trial(
    trial_idx: int,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    env: MujocoSoccer,
    clips: list[MotionClip],
    policy_bank: PolicyBank,
    viewer=None,
) -> TrialResult:
    spawn = sample_non_overlapping_spawn(cfg, rng, env, clips)
    motion = spawn.motion
    policy = policy_bank.find_for_motion(motion)
    policy.reset()
    metadata = getattr(policy, "metadata", {})
    default_joint_pos = parse_metadata_array(metadata, "default_joint_pos", DEFAULT_JOINT_POS_ARRAY)
    action_scale = parse_metadata_array(metadata, "action_scale", ACTION_SCALE_ARRAY)
    stiffness = parse_metadata_array(metadata, "joint_stiffness", JOINT_STIFFNESS_ARRAY)
    damping = parse_metadata_array(metadata, "joint_damping", JOINT_DAMPING_ARRAY)
    effort_limit = JOINT_EFFORT_LIMIT_ARRAY

    control_steps = int(round(cfg.sim_time / cfg.control_dt))
    sim_steps_per_control = max(1, int(round(cfg.control_dt / env.dt)))
    last_action = np.zeros(len(default_joint_pos), dtype=np.float32)
    pol_body_pos_values: list[np.ndarray] = []
    ref_body_pos_values: list[np.ndarray] = []
    pol_dof_pos_values: list[np.ndarray] = []
    ref_dof_pos_values: list[np.ndarray] = []
    success = False
    first_contact_time = math.nan
    max_ball_speed = 0.0
    max_contact_ball_speed = 0.0
    best_alignment_angle = 1.0
    had_ball_contact = False
    prev_ball = env.ball_pos.copy()

    owns_viewer = False
    if cfg.render and viewer is None:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(env.model, env.data)
        owns_viewer = True

    try:
        for control_step in range(control_steps):
            wall_step_start = time.perf_counter()
            time_step = min(control_step, motion.length - 1)
            obs = build_observation(
                env,
                motion,
                time_step,
                last_action,
                policy.ref_joint_pos,
                policy.ref_joint_vel,
                policy.ref_ang_vel,
                env.ball_pos,
                spawn.goal_world,
                default_joint_pos,
                policy.obs_dim,
            )
            policy_step = policy.step(obs, time_step, motion)
            last_action = policy_step.action.astype(np.float32)

            for _ in range(sim_steps_per_control):
                env.set_pd_action(policy_step.action, default_joint_pos, action_scale, stiffness, damping, effort_limit)
                mujoco.mj_step(env.model, env.data)
                curr_ball = env.ball_pos.copy()
                if not success and goal_crossed(prev_ball, curr_ball, spawn.goal_world, spawn.goal_normal, cfg.goal_width):
                    success = True
                if env.has_ball_foot_contact():
                    had_ball_contact = True
                    if math.isnan(first_contact_time):
                        first_contact_time = float(env.data.time)
                prev_ball = curr_ball

            if viewer is not None:
                viewer.sync()
                elapsed = time.perf_counter() - wall_step_start
                if elapsed < cfg.control_dt:
                    time.sleep(cfg.control_dt - elapsed)

            t = min(control_step, motion.length - 1)
            ref_body_pos = motion.body_pos_w[t, env.tracked_body_motion_ids] + spawn.translation
            sim_body_pos = env.data.xpos[env.tracked_body_ids].copy()
            pol_body_pos_values.append(sim_body_pos)
            ref_body_pos_values.append(ref_body_pos)
            pol_dof_pos_values.append(env.joint_pos.copy())
            ref_dof_pos_values.append(motion.joint_pos[t][ISAACLAB_TO_MUJOCO_REINDEX].astype(np.float64))

            ball_vel = env.ball_vel
            ball_speed = float(np.linalg.norm(ball_vel[:2]))
            if ball_speed > max_ball_speed:
                max_ball_speed = ball_speed
            if had_ball_contact and ball_speed > max_contact_ball_speed:
                max_contact_ball_speed = ball_speed
                desired = spawn.goal_world[:2] - env.ball_pos[:2]
                desired_norm = np.linalg.norm(desired)
                if ball_speed > cfg.ball_speed_threshold and desired_norm > 1.0e-9:
                    cos_angle = float(np.dot(ball_vel[:2], desired) / (ball_speed * desired_norm))
                    best_alignment_angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
    finally:
        if owns_viewer and viewer is not None:
            viewer.close()

    final_ball = env.ball_pos
    baseline_metrics = eval_baseline_metrics(
        np.stack(pol_body_pos_values, axis=0),
        np.stack(ref_body_pos_values, axis=0),
        np.stack(pol_dof_pos_values, axis=0),
        np.stack(ref_dof_pos_values, axis=0),
        delta_per_frame=True,
        fps=motion.fps,
    )
    return TrialResult(
        trial=trial_idx,
        motion=motion.name,
        kick_leg=motion.kick_leg,
        robot_x=float(spawn.robot_xy[0]),
        robot_y=float(spawn.robot_xy[1]),
        ball_local_x=float(spawn.ball_local_xy[0]),
        ball_local_y=float(spawn.ball_local_xy[1]),
        ball_x=float(spawn.ball_world[0]),
        ball_y=float(spawn.ball_world[1]),
        goal_x=float(spawn.goal_world[0]),
        goal_y=float(spawn.goal_world[1]),
        selected_motion_error=float(spawn.motion_error),
        spawn_attempts=spawn.attempts,
        success=success,
        is_succ_phc=baseline_metrics["is_succ_phc"],
        is_succ_phc_rel=baseline_metrics["is_succ_phc_rel"],
        E_gmpjpe=baseline_metrics["E_gmpjpe"],
        E_mpjpe=baseline_metrics["E_mpjpe"],
        E_dof_mpjpe=baseline_metrics["E_dof_mpjpe"],
        E_vel=baseline_metrics["E_vel"],
        E_acc=baseline_metrics["E_acc"],
        E_contact_acc=baseline_metrics["E_contact_acc"],
        seq_jerk=baseline_metrics["seq_jerk"],
        seq_ref_jerk=baseline_metrics["seq_ref_jerk"],
        global_mpjpe=baseline_metrics["E_gmpjpe"],
        mpjpe=baseline_metrics["E_mpjpe"],
        vel_error=baseline_metrics["E_vel"],
        ball_vel_dir_alignment_angle=float(best_alignment_angle),
        max_ball_speed=float(max_ball_speed),
        first_contact_time=float(first_contact_time),
        final_ball_x=float(final_ball[0]),
        final_ball_y=float(final_ball[1]),
    )
