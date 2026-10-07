from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class ExperimentConfig:
    experiment_name: str | None
    num_trials: int
    birth_radius: float
    ball_forward_range: tuple[float, float]
    ball_lateral_range: tuple[float, float]
    goal: tuple[float, float]
    goal_width: float
    enable_ball_vel: bool
    ball_speed_range: tuple[float, float]
    motion_path: Path
    mjcf: Path
    policy: Path | None
    policy_dir: Path | None
    output_dir: Path
    seed: int
    sim_time: float
    control_dt: float
    goal_frame: str
    ball_speed_threshold: float
    render: bool
    preload_policies: bool
    reference_only: bool
    rnn_num_layers: int
    rnn_hidden_dim: int
    ball_spawn_max_attempts: int


@dataclasses.dataclass
class MotionClip:
    path: Path
    name: str
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    local_displacement_xy: np.ndarray
    length: int
    kick_leg: str | None


@dataclasses.dataclass
class SpawnState:
    robot_xy: np.ndarray
    ball_local_xy: np.ndarray
    ball_world: np.ndarray
    ball_vel_world: np.ndarray
    goal_world: np.ndarray
    goal_normal: np.ndarray
    motion: MotionClip
    motion_error: float
    translation: np.ndarray
    attempts: int


@dataclasses.dataclass
class TrialResult:
    trial: int
    motion: str
    kick_leg: str | None
    robot_x: float
    robot_y: float
    ball_local_x: float
    ball_local_y: float
    ball_x: float
    ball_y: float
    goal_x: float
    goal_y: float
    selected_motion_error: float
    spawn_attempts: int
    success: bool
    is_succ_phc: float
    is_succ_phc_rel: float
    E_gmpjpe: float
    E_mpjpe: float
    E_dof_mpjpe: float
    E_vel: float
    E_acc: float
    E_contact_acc: float
    seq_jerk: float
    seq_ref_jerk: float
    global_mpjpe: float
    mpjpe: float
    vel_error: float
    ball_vel_dir_alignment_angle: float
    max_ball_speed: float
    first_contact_time: float
    final_ball_x: float
    final_ball_y: float
