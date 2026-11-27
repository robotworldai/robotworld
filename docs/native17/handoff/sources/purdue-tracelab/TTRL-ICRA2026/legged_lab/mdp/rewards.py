# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from typing import Optional

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.base.legged_env import LeggedEnv
    from legged_lab.envs.base.tt_env import TTEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def lin_vel_x_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])

def lin_vel_y_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])

def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_l2(env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: TTEnv, threshold: float, vel_ref: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    # reward *= (
    #     torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    # ) > 0.1
    reward *= (
        torch.norm(env.robot_future_vel[:, :2], dim=1)
    ) > vel_ref
    return reward

def feet_air_time_negative_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward-threshold, min=0.0)
    # no reward for zero command
    # reward *= (
    #     torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    # ) > 0.1
    return reward

def late_serve_unstable_support(
    env: TTEnv, sensor_cfg: SceneEntityCfg, min_fraction: float, max_fraction: float, force_threshold: float = 0.1
) -> torch.Tensor:
    """Counts unstable support (single-stance or both-feet-in-air) during a specified
    progress window of the ball episode.

    Returns a positive count (0 or 1) per env when the following hold:
    - The normalized ball episode progress `p = steps / max_steps` satisfies
      `min_fraction < p < max_fraction`.
    - Exactly one foot is in contact (single stance) OR both feet are in air (no contact).

    Use with a negative weight in the reward config to penalize such events.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine foot contacts from forces within recent history: max force over history > threshold
    net_forces_hist = contact_sensor.data.net_forces_w_history  # [N, T, B, 3]
    in_contact = (
        net_forces_hist[:, :, sensor_cfg.body_ids, :]  # [N, T, B, 3]
        .norm(dim=-1)                                 # [N, T, B]
        .max(dim=1)[0] > force_threshold              # [N, B]
    )
    n_contacts = torch.sum(in_contact.int(), dim=1)  # [N]

    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air  # [N]

    # Window mask based on ball episode progress
    progress = env.ball_episode_length_buf.float() / float(env.max_ball_episode_length)
    late_mask = (progress > min_fraction) & (progress < max_fraction)

    # Positive count (0 or 1) to be used with a negative weight
    reward = (unstable & late_mask).float()
    return reward


def hit_unstable_support(
    env: TTEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1
) -> torch.Tensor:
    """Counts unstable support exactly at hit moments.

    Returns 1 when, at the current step:
    - contact forces indicate feet state is unstable (single stance or both feet in air), and
    - the ball is in contact region with the paddle ("during hit"), approximated by ``env.ball_contact > 0``.

    Use with a negative weight in the reward config to penalize such events.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine foot contacts from forces within recent history: max force over history > threshold
    net_forces_hist = contact_sensor.data.net_forces_w_history  # [N, T, B, 3]
    in_contact = (
        net_forces_hist[:, :, sensor_cfg.body_ids, :]  # [N, T, B, 3]
        .norm(dim=-1)                                 # [N, T, B]
        .max(dim=1)[0] > force_threshold              # [N, B]
    )
    n_contacts = torch.sum(in_contact.int(), dim=1)  # [N]

    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air  # [N] bool

    # "During hit" mask: ball currently in the paddle contact region
    # during_hit = env.ball_contact > 0.0  # [N] bool
    reward = unstable.float()*env.ball_contact_rew
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def penalty_robot_table_proximity_x(
    env: TTEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.10,
    std: float = 0.1,
) -> torch.Tensor:

    table_half_length = -1.37 - min_distance # half of table length in x-direction
    robot_pos_x = env.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    denom=std * std + 1e-12
    penalty = torch.exp(-torch.clamp(torch.abs(robot_pos_x - table_half_length ), min=1e-6) / denom)

    return penalty

def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def paddel_too_near_humanoid(
    env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 1
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(link_pos[:, 0] - env.paddle_touch_point, dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_too_high(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess_height = torch.clamp(feet_z - threshold, min=0.0)
    penalty = torch.sum(excess_height, dim=1) 
    return penalty

def reward_ball_z_nearnet(env: TTEnv, z_threshold: float = 0.94, x_tolerance: float = 0.03) -> torch.Tensor:
    """Reward when the ball's height exceeds ``z_threshold`` at the net plane after a paddle hit.

    Conditions (all must hold):
    - ``env.has_touch_paddle`` is True (ball has been hit).
    - Ball longitudinal position is near the net plane: ``abs(ball_x) < x_tolerance``.
    - Ball is traveling towards the opponent: ``vx > 0``.

    Returns a binary tensor of shape [N], 1.0 when conditions are met and ``z > z_threshold``, else 0.0.
    """
    # Ball state in env-local/table frame
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]

    at_net_plane = torch.abs(x) < x_tolerance
    moving_forward = vx > 0.0
    hit = env.has_touch_paddle
    high_enough = z > z_threshold

    reward = (hit & at_net_plane & moving_forward & high_enough).float()
    return reward

@torch.jit.script
def create_stance_mask(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a stance mask based on the gait phase.
    """
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(-1).repeat(1, 2)
    stance_mask = torch.where(sin_pos >= 0, 1, 0)
    stance_mask[:, 1] = 1 - stance_mask[:, 1]
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    return stance_mask, mask_2


@torch.jit.script
def compute_reward_reward_feet_contact_number(
    contacts: torch.Tensor,
    phase: torch.Tensor,
    pos_rw: float,
    neg_rw: float,
    command: torch.Tensor,
):

    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    reward = torch.mean(reward, dim=1)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def reward_feet_contact_number(
    env: LeggedEnv,
    sensor_cfg: SceneEntityCfg,
    pos_rw: float,
    neg_rw: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    command = env.command_generator.command[:, :2]

    return compute_reward_reward_feet_contact_number(
        contacts, phase, pos_rw, neg_rw, command
    )


@torch.jit.script
def compute_reward_foot_clearance_reward(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    current_foot_z: torch.Tensor,
    target_height: float,
    std: float,
    tanh_mult: float,
    body_lin_vel_w: torch.Tensor,
    command: torch.Tensor,
):
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = (
        0.0626  # recorded from the default position, 0.1 compensation for walking
    )
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)
    foot_z_target_error = torch.square(
        (current_foot_z - (target_height + offset).repeat(1, 2)).clip(max=0.0)
    )
    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(body_lin_vel_w, dim=2))
    reward = foot_velocity_tanh * foot_z_target_error
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def foot_clearance_reward(
    env: LeggedEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    com_z = env.robot.data.root_pos_w[:, 2]
    current_foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # this the default standing position of the robot on the ground 
    standing_position_com_z = env.robot.data.default_root_state[:, 2]
    body_lin_vel_w = env.robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    command = env.command_generator.command[:, :2]

    return compute_reward_foot_clearance_reward(
        com_z,
        standing_position_com_z,
        current_foot_z,
        target_height,
        std,
        tanh_mult,
        body_lin_vel_w,
        command,
    )

@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0

@torch.jit.script
def compute_reward_track_foot_height(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    phase: torch.Tensor,
    foot_z: torch.Tensor,
    standing_position_toe_roll_z: float,
    std: float,
    command: torch.Tensor,
):

    standing_height = com_z - standing_position_com_z

    offset = standing_height + standing_position_toe_roll_z

    stance_mask, mask_2 = create_stance_mask(phase)

    swing_mask = 1 - stance_mask

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def track_foot_height(
    env: LeggedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """"""

    foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    command = env.command_generator.command[:, :2]
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    # contacts = (
    #     contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
    #     .norm(dim=-1)
    #     .max(dim=1)[0]
    #     > 1.0
    # )
    com_z = env.robot.data.root_pos_w[:, 2]
    standing_position_com_z =env.robot.data.default_root_state[:, 2]
    phase = env.get_phase()

    return compute_reward_track_foot_height(
        com_z, standing_position_com_z, phase, foot_z, 0.0626, std, command
    )

@torch.compile
def bezier_curve(control_points, t):
    """
    Computes Bézier curve for given control points and parameter t.
    """
    n = len(control_points) - 1  # Degree of the Bézier curve
    dim = control_points.shape[1]  # Dimension of control points
    curve_points = torch.zeros(
        (t.shape[0], dim), dtype=control_points.dtype, device=t.device
    )

    # Calculate the Bézier curve points
    for k in range(n + 1):
        binomial_coeff = math.comb(n, k)
        bernstein_poly = binomial_coeff * (t**k) * ((1 - t) ** (n - k))
        curve_points += bernstein_poly.unsqueeze(1) * control_points[k]

    return curve_points


@torch.compile
def desired_height(phase, starting_foot):
    """
    Computes the desired heights for both legs for each environment.

    Args:
        phase (torch.Tensor): Tensor of shape (n_envs,) representing current phase values.
        starting_foot (torch.Tensor): Tensor of shape (n_envs,) with values 0 or 1 indicating which foot starts swinging first.

    Returns:
        torch.Tensor: Tensor of shape (n_envs, 2) containing desired heights for both legs.
    """
    n_envs = phase.shape[0]
    desired_heights = torch.zeros((n_envs, 2), dtype=phase.dtype, device=phase.device)

    # Step length (L) and max height (H) for the swing phase
    L = 0.5  # Step length
    H = 0.15  # Maximum height in the swing phase

    # Define control points for the swing phase Bézier curve
    control_points_swing = torch.tensor(
        [
            [0.0, 0.0],  # Start of swing phase
            [0.3 * L, 0.1 * H],  # Lift-off point
            [0.6 * L, H],  # Peak of swing
            [L, 0.0],  # Landing point
        ],
        dtype=torch.float32,
        device=phase.device,
    )

    # Loop over legs (0: left leg, 1: right leg)
    for leg in [0, 1]:
        # Determine which environments have this leg as the starting foot
        is_starting_leg = starting_foot == leg
        is_other_leg = ~is_starting_leg

        # Swing phase masks for the starting leg
        swing_mask_starting_leg = is_starting_leg & (phase >= 0.02) & (phase < 0.5)
        t_swing_starting = (phase[swing_mask_starting_leg] - 0.02) / 0.48

        # Swing phase masks for the other leg
        swing_mask_other_leg = is_other_leg & (phase >= 0.52) & (phase < 1.0)
        t_swing_other = (phase[swing_mask_other_leg] - 0.52) / 0.48

        # Combine swing masks
        swing_mask_leg = swing_mask_starting_leg | swing_mask_other_leg

        # Initialize t_swing for all environments
        t_swing = torch.zeros(n_envs, dtype=phase.dtype, device=phase.device)
        t_swing[swing_mask_starting_leg] = t_swing_starting
        t_swing[swing_mask_other_leg] = t_swing_other

        # Compute desired heights for swing phase
        if swing_mask_leg.any():
            t_swing_leg = t_swing[swing_mask_leg]
            swing_heights = bezier_curve(control_points_swing, t_swing_leg)
            desired_heights[swing_mask_leg, leg] = swing_heights[
                :, 1
            ]  # Only the y-coordinate (height)

        # Stance phase (excluding double stance): foot is on the ground (height = 0)
        # No action needed since desired_heights is initialized to zero

    # For double stance phases, both legs are already set to height = 0
    return desired_heights

def reward_contact(env: TTEnv) -> torch.Tensor:
    return env.ball_contact_rew.float()

# #(2) penalize the ball for going below a certain height.
# def penalty_ball_to_floor(env: TTEnv) -> torch.Tensor:
#     env.penalty_ball_to_floor = (env.ball_pos[:, 2] < 0.60).float()
#     return env.penalty_ball_to_floor

# # (3) encourage ball return faster x-velocity 
# def reward_vel(env: TTEnv) -> torch.Tensor:
    
#     reward_vel = (
#             env.ball_linvel[:, 0]
#             * env.has_touch_paddle.float()
#             * (env.ball_contact == 0).float()
#             * torch.logical_not(env.reward_vel_prev)
#         )
#     still_false = env.reward_vel_prev == 0    
#     env.reward_vel_prev[still_false] = reward_vel[still_false]

#     return reward_vel

# (4) reward when the ball first bounces on opponent side.
def reward_table_success(env: TTEnv) -> torch.Tensor:
    rew_table_success = (env.has_touch_paddle.float() * env.has_touch_opponent_table_just_now.float())
    return rew_table_success

# # (5) Encourage forward ball position.
# def reward_ball_pos(env: TTEnv) -> torch.Tensor:

#     rew_table_success = reward_table_success(env)
#     rew_ball_pos = (rew_table_success * env.ball_pos[:, 0])

#     return rew_ball_pos

# # (6) Penalize ball hitting own table before opponent touch
# def penalty_table_fail(env: TTEnv) -> torch.Tensor:

#     penalty_table_fail = (env.has_first_bounce_prev * env.has_touch_own_table.float())
#     ball_x = env.ball_pos[:, 0]  # (N,)
#     mask_fail = penalty_table_fail != 0  # (N,) boolean
#     penalty_table_fail[mask_fail] += -ball_x[mask_fail] + 0.1

#     return penalty_table_fail

# def reward_paddle_ball_yz_distance_exp(env: TTEnv, std: float) -> torch.Tensor:
#     distance = torch.norm(env.ball_global_pos[:, 1:3] - env.paddle_touch_point[:, 1:3], dim=1)
#     return torch.exp(-distance / std**2) * env.has_touch_own_table

# def reward_paddle_ball_y_distance_exp(env: TTEnv, std: float) -> torch.Tensor:
#     distance = torch.abs(env.ball_global_pos[:, 1] - env.paddle_touch_point[:, 1])
#     return torch.exp(-distance / std**2)

def reward_paddle_distance_terminal(env: TTEnv, coeff: float=100.0) -> torch.Tensor:
    distance = torch.norm(env.ball_global_pos - env.paddle_touch_point, dim=1) - 0.02
    # distance = torch.norm(env.ball_global_pos[:, 1:3] - env.paddle_touch_point[:, 1:3], dim=1) - 0.02
    reward = 1 / (1 + coeff * distance**2)**2 # Tail ends at 0.25, gradient rises under 0.1
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_paddle_distance_terminal_weighted(env: TTEnv, coeff_x: float=100.0, coeff_y: float=100.0, coeff_z: float=100.0, weight_x: float=1.0, weight_y: float=1.0, weight_z: float=1.0) -> torch.Tensor:
    d_x = torch.abs(env.ball_global_pos[:, 0] - env.paddle_touch_point[:, 0])
    reward_x = weight_x / (1 + coeff_x * d_x**2)**2
    d_y = torch.abs(env.ball_global_pos[:, 1] - env.paddle_touch_point[:, 1])
    reward_y = weight_y / (1 + coeff_y * d_y**2)**2
    d_z = torch.abs(env.ball_global_pos[:, 2] - env.paddle_touch_point[:, 2])
    reward_z = weight_z / (1 + coeff_z * d_z**2)**2
    reward = reward_x + reward_y + reward_z
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_future_ee_target(
    env: TTEnv,
    std_ee: float = 0.4,
    threshold: float = 0.01,
) -> torch.Tensor:

    # dist_ee_before = torch.linalg.norm(env.pos_pred_before - env.paddle_pos, dim=1)
    # dist_ee_after = torch.linalg.norm(env.pos_pred_after - env.paddle_pos, dim=1)

    # dist_ee_before = torch.where(env.valid_before, dist_ee_before, torch.full_like(dist_ee_before, float("inf")))
    # dist_ee_after = torch.where(env.mask_after, dist_ee_after, torch.full_like(dist_ee_after, float("inf")))

    # denom_ee = std_ee * std_ee + 1e-12
    # rew_ee_before = torch.exp(-torch.clamp(dist_ee_before, min=threshold) / denom_ee)
    # rew_ee_after = torch.exp(-torch.clamp(dist_ee_after, min=threshold) / denom_ee)

    # reward_ee = torch.zeros_like(rew_ee_before)
    # reward_ee = torch.where(env.mask_before, rew_ee_before, reward_ee)
    # reward_ee = torch.where(env.mask_after, rew_ee_after, reward_ee)

    dist_ee = torch.linalg.norm(env.ball_future_pose - env.paddle_pos, dim=1)
    denom_ee = std_ee * std_ee + 1e-12
    reward_ee = torch.exp(-torch.clamp(dist_ee, min=threshold) / denom_ee)
    # mask_invalid_ee = (ball_pos[:, 0] < -1.6) | (vx > 0) | (z < 0.7) 
    reward_ee = torch.where(env.mask_invalid, torch.zeros_like(reward_ee), reward_ee)
    reward = torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)
    return reward

def paddel_ball_distance(
    env: TTEnv,
    std_ee: float = 0.3,
    threshold: float = 0.01,
) -> torch.Tensor:

    denom_ee = std_ee * std_ee + 1e-12
    rew_ee = torch.exp(-torch.clamp(env.paddel_ball_distance, min=threshold) / denom_ee)
    mask_invalid_ee = (self.ball_pos[:, 0] < -1.35) | (self.ball_pos[:, 2] < 0.75)  | env.has_touch_paddle
    reward_ee = torch.where(mask_invalid_ee, torch.zeros_like(reward_ee), reward_ee)
    reward = torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)
    return reward



def reward_future_body_target(
    env: TTEnv,
    std_ro: float = 0.5,
    threshold: float = 0.05,
) -> torch.Tensor:

    dist_ro = torch.linalg.norm(env.robot_future_pos[:, 0:2] - env.robot_pos[:, 0:2], dim=1)
    denom_ro = std_ro * std_ro + 1e-12
    rew_ro = torch.exp(-torch.clamp(dist_ro, min=threshold) / denom_ro)
    reward = torch.where(env.mask_invalid, torch.zeros_like(rew_ro), rew_ro)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return reward



def reward_future_vel_target(
    env: TTEnv,
    threshold: float = 0.03,
    vel_std: float= 1.41,
) -> torch.Tensor:
    # vdiff_ro_before = torch.linalg.norm(env.vel_ro_before[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    # vdiff_ro_after = torch.linalg.norm(env.vel_ro_after[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    
    # vdiff_ro_before = torch.where(env.valid_before, vdiff_ro_before, torch.full_like(vdiff_ro_before, float(0.0)))
    # vdiff_ro_after = torch.where(env.mask_after, vdiff_ro_after, torch.full_like(vdiff_ro_after, float(0.0)))

    denom_vel=vel_std * vel_std + 1e-12
    # rew_ro_before = torch.exp(-torch.clamp(vdiff_ro_before, min=threshold) / denom_vel)
    # rew_ro_after = torch.exp(-torch.clamp(vdiff_ro_after, min=threshold) / denom_vel)
    # reward_ro = torch.zeros_like(rew_ro_before)
    # reward_ro = torch.where(env.mask_before, rew_ro_before, reward_ro)
    # reward_ro = torch.where(env.mask_after, rew_ro_after, reward_ro)

    vdiff_ro = torch.linalg.norm(env.robot_future_vel[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    reward_ro = torch.exp(-torch.clamp(vdiff_ro, min=threshold) / denom_vel)

    reward_ro = torch.where(env.mask_invalid, torch.zeros_like(reward_ro), reward_ro)
    reward = torch.nan_to_num(reward_ro, nan=0.0, posinf=0.0, neginf=0.0)
    return reward

def reward_future_landing_dis(
    env: TTEnv,
    threshold: float= 2.0,
) -> torch.Tensor:
    target_x=1.15
    target_y=0
    # Stack into 2D points
    pred_land = torch.stack([env.predict_x_land, env.predict_y_land], dim=1)
    target_land = torch.tensor([target_x, target_y], device=pred_land.device)
    # Compute 2D distance (batch-wise)
    dist_ball_land = torch.linalg.norm(pred_land - target_land, dim=1)
    # Reward = negative distance
    reward = (threshold - dist_ball_land )
    
    # reward = torch.where(env.touched_paddel_no_bounce_table, reward, torch.zeros_like(reward)) # continuous
    mask = env.ball_landing_dis_rew
    reward = torch.where(mask, reward, torch.zeros_like(reward)) # sparse
    return reward

def reward_future_pass_net(
    env: TTEnv,
    std_h : float = 0.06,
    z_target : float = 0.76 + 0.24,  # target height at net (table height + ball height above table
) -> torch.Tensor:


    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vz = env.ball_linvel[:, 2]

    # Net plane assumed at x = 0. Consider only balls moving forward (towards +x).
    moving_forward = vx > 0.0
    dx_to_net = torch.clamp(0.0 - x, min=0.0)

    # Simple flight model without drag
    # Time to net plane (avoid division by zero)
    g = 9.81
    eps = torch.tensor(1e-6, device=env.device, dtype=vx.dtype)
    vx_safe = torch.clamp(vx, min=eps)
    t_net = dx_to_net / vx_safe
    t_net = torch.clamp(t_net, min=0.0)

    # Vertical position at t_net: z + vz*t - 0.5*g*t^2
    z_at_net = z + vz * t_net - 0.5 * g * (t_net * t_net)

    # Positive reward: closer to target height is better (bell-shaped)
    height_err = torch.abs(z_at_net - z_target)
    reward = torch.exp(-height_err / (std_h + 1e-12))

    # Valid only when still moving forward to the net
    reward = torch.where(moving_forward, reward, torch.zeros_like(reward)) 
    # Apply provided sparse mask to trigger the reward once after hitting
    mask = env.ball_landing_dis_rew
    reward = torch.where(mask, reward, torch.zeros_like(reward))  # sparse
    return reward

def robot_px_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - asset.data.default_root_state[:, 0])

def robot_heading_quad(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.heading_w - target_heading)

def body_heading_quad(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_heading = yaw
    return torch.square(body_heading - target_heading)

def body_heading_exp(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_heading = yaw
    return torch.exp(-torch.abs(body_heading - target_heading) / std)

def body_pitch_exp(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_pitch = pitch
    return torch.exp(-torch.abs(body_pitch - target) / std)

def body_pitch_contact_exp(env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_pitch = pitch
    reward = torch.exp(-torch.abs(body_pitch - target) / std)
    return (env.ball_contact_rew > 0).float() * reward


def penalty_stand_still(
    env: TTEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1, move_threshold: float = 0.1
):  
    """
    For TTEnv only.
    Penalize the robot for standing still when both feet are in contact
    and the predicted movement is below a given threshold.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Recent contact forces, only for specified body IDs
    contacts = (
        net_contact_forces[:, :, sensor_cfg.body_ids, :]  # (N, T, B, 3)
        .norm(dim=-1)                                     # (N, T, B)
        .max(dim=1)[0] > force_threshold                        # (N, B)
    )

    # True if all specified bodies are in contact
    both_feet_in_contact = torch.all(contacts, dim=1)     # (N,)

    # Compute movement magnitude (L2 norm between predicted and current position)
    position_diff = torch.norm(env.robot_future_pos - env.robot_pos, dim=-1)  # (N,)

    # Apply penalty only if robot is standing still (contact) AND not moving enough
    penalty = both_feet_in_contact & (position_diff > move_threshold)
    penalty = penalty.float()
    return penalty
