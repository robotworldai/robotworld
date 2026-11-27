from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Global tensor for tracking previous foot velocity (for smooth velocity reward)
_foot_vel_z_prev: torch.Tensor | None = None  # Stores previous timestep velocity for each foot


"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_deviation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """Reward joint positions being close to their default positions using exponential kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        lambda_exp: Lambda parameter for the exponential kernel. Higher values make the
                   reward more sensitive to joint deviations. Default is 1.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1], where 1 is
        all joints at default positions and lower values indicate deviation.

    Note:
        The reward is computed as: exp(-lambda_exp * sum(|angle_i - default_i|))
        Only the joints configured in :attr:`asset_cfg.joint_ids` contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute joint deviations from default positions
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)
    return torch.exp(-lambda_exp * deviation)


"""
Feet rewards.
"""


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_smooth_velocity_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    lambda_exp: float = 1.0,
) -> torch.Tensor:
    """Reward for smooth foot vertical velocity changes across the entire period using exponential kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset. Should specify feet bodies.
        lambda_exp: Lambda parameter for the exponential kernel. Higher values make the
                   reward more sensitive to velocity changes.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1], where 1 is
        minimal velocity changes (smooth movement) and lower values indicate jerky or
        sudden movements.

    Note:
        This function maintains previous velocity in the global tensor _foot_vel_z_prev.
    """
    global _foot_vel_z_prev

    robot: Articulation = env.scene[asset_cfg.name]

    # Current foot vertical velocity (world z-axis)
    foot_vel_z = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # Shape: [num_envs, num_feet]

    # Initialize previous velocity tensor on first call or if environment count changed
    if _foot_vel_z_prev is None or _foot_vel_z_prev.shape[0] != env.num_envs:
        # Initialize with current velocity (no change on first step)
        _foot_vel_z_prev = foot_vel_z.clone()
        return torch.ones(env.num_envs, device=env.device)  # Return max reward on first step

    # Compute velocity change from previous timestep using L2 (squared) for smooth gradients
    # Shape: [num_envs, num_feet]
    delta_v_z_sq = torch.square(foot_vel_z - _foot_vel_z_prev)

    # Sum over all feet to get total squared velocity change per environment
    # Shape: [num_envs]
    total_dv = torch.sum(delta_v_z_sq, dim=1)

    # Update previous velocity for next timestep
    _foot_vel_z_prev = foot_vel_z.clone()

    # Apply exponential kernel to convert to positive reward
    return torch.exp(-lambda_exp * total_dv)


"""
Body-specific velocity and height rewards.
"""

def body_lin_vel_z_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 2.0
) -> torch.Tensor:
    """Reward for keeping z-axis body linear velocity near zero using exponential kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset. Should specify a single body.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 2.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].

    Note:
        This function uses link linear velocity (more robust to CoM changes).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))

def body_ang_vel_xy_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """Reward for keeping xy-axis body angular velocity near zero using exponential kernel.

    Penalizes rotation around X and Y axes (roll and pitch).

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset. Should specify a single body.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 1.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].

    Note:
        This function uses link angular velocity (more robust to CoM changes).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    return torch.exp(-lambda_exp * ang_vel_xy_norm_sq)


def body_upright_bonus_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lambda_exp: float = 4.0
) -> torch.Tensor:
    """Reward for keeping the robot body upright using an exponential kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset. Should specify a single body.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 4.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range [0, 2], where 2 is
        perfectly upright and lower values indicate tilting.

    Note:
        This function only accepts a single body name in asset_cfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute projected gravity for the specified body
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    gravity_vec = asset.data.GRAVITY_VEC_W
    projected_gravity = math_utils.quat_apply_inverse(body_quat, gravity_vec)

    # Reward for keeping xy components of projected gravity near zero (upright orientation)
    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)


def body_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    lambda_exp: float = 10.0
) -> torch.Tensor:
    """Reward for maintaining body at target height using exponential kernel.

    Args:
        env: The environment instance.
        target_height: The target height to maintain the body at.
        asset_cfg: Configuration for the robot asset. Should specify a single body.
        sensor_cfg: Optional sensor configuration for terrain-aware height adjustment.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 10.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height

    # Get current height
    current_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_error = current_height - adjusted_target_height

    # Compute exponential reward
    return torch.exp(-lambda_exp * torch.abs(height_error))


def track_lin_vel_xy_yaw_body_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned body frame using exponential kernel.

    Unlike track_lin_vel_xy_yaw_frame_exp which uses root link, this uses a specific body (e.g., torso_link).
    Uses link frame velocity (robust to CoM changes) transformed to yaw-aligned frame.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # Get body quaternion and link frame velocity (more robust to CoM randomization)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], :3]

    # Transform velocity to yaw-aligned frame
    vel_yaw = quat_apply_inverse(yaw_quat(body_quat_w), body_lin_vel_w)

    # Compute error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_body_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in body frame using exponential kernel.

    Unlike track_ang_vel_z_exp which uses root link, this uses a specific body (e.g., torso_link).
    Uses link frame angular velocity in body frame (robust to CoM changes).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get body quaternion and link frame angular velocity (more robust to CoM randomization)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :]

    # Transform angular velocity from world frame to body frame
    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)

    # Compute error for z-axis (yaw) in body frame
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


"""
Action penalty
"""

def action_rate_l2_clipped(env: ManagerBasedRLEnv, max_penalty: float = 1.0) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel with clipping to avoid reward explosion.

    Args:
        env: The environment instance.
        max_penalty: Maximum penalty value to clip the reward at. Default is 1.0.

    Returns:
        torch.Tensor: Clipped penalty values for each environment.
    """
    action_rate_penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(action_rate_penalty, max=max_penalty)


"""
Object rewards.
"""


def object_upright_bonus_exp(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, lambda_exp: float = 4.0) -> torch.Tensor:
    """Reward for keeping the cup upright using an exponential kernel."""
    object: RigidObject = env.scene[object_cfg.name]

    # Compute the projected gravity for the object
    projected_gravity = object.data.projected_gravity_b
    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)


def object_ang_vel_xy_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """Reward for keeping xy-axis object angular velocity near zero using exponential kernel.

    Penalizes rotation around X and Y axes (roll and pitch),
    helping to keep the object stable and prevent it from tumbling.

    Args:
        env: The environment instance.
        object_cfg: Configuration for the object asset.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 1.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].
    """
    object: RigidObject = env.scene[object_cfg.name]
    ang_vel_xy = object.data.body_ang_vel_w[:, 0, :2]
    ang_vel_xy_norm_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    return torch.exp(-lambda_exp * ang_vel_xy_norm_sq)


def object_lin_vel_z_exp(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lambda_exp: float = 2.0
) -> torch.Tensor:
    """Reward for keeping z-axis object linear velocity near zero using exponential kernel.

    Args:
        env: The environment instance.
        object_cfg: Configuration for the object asset.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 2.0.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].
    """
    object: RigidObject = env.scene[object_cfg.name]
    vel_z = object.data.body_lin_vel_w[:, 0, 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))


def entity_quat_l1(env: ManagerBasedRLEnv, entity1_cfg: SceneEntityCfg, entity2_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for difference between two entities' quaternion orientations.

    This function supports both RigidObject and Articulation entities.

    Args:
        env: The environment instance.
        entity1_cfg: Configuration for the first entity.
        entity2_cfg: Configuration for the second entity.

    Returns:
        torch.Tensor: Angular distance in radians. Range: [0, pi].
    """
    entity1 = env.scene[entity1_cfg.name]
    entity2 = env.scene[entity2_cfg.name]

    # Get current orientations in world frame
    if isinstance(entity1, RigidObject):
        entity1_quat = entity1.data.body_quat_w[:, 0, :]
    else:  # Articulation
        entity1_quat = entity1.data.body_quat_w[:, entity1_cfg.body_ids[0], :]

    if isinstance(entity2, RigidObject):
        entity2_quat = entity2.data.body_quat_w[:, 0, :]
    else:  # Articulation
        entity2_quat = entity2.data.body_quat_w[:, entity2_cfg.body_ids[0], :]

    # Compute quaternion difference using dot product
    quat_dot = torch.sum(entity1_quat * entity2_quat, dim=-1)
    quat_dot = torch.clamp(torch.abs(quat_dot), max=1.0)
    orientation_difference = 2.0 * torch.acos(quat_dot)

    return orientation_difference


def entity_quat_exp(
    env: ManagerBasedRLEnv,
    entity1_cfg: SceneEntityCfg,
    entity2_cfg: SceneEntityCfg,
    lambda_exp: float = 1.0
) -> torch.Tensor:
    """Reward for two entities maintaining similar orientations using exponential kernel.

    This function supports both RigidObject and Articulation entities.

    Args:
        env: The environment instance.
        entity1_cfg: Configuration for the first entity.
        entity2_cfg: Configuration for the second entity.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 1.0.

    Returns:
        torch.Tensor: Reward values in range (0, 1] for each environment.
    """
    # Compute the angular distance using the L1 penalty function
    orientation_error_l1 = entity_quat_l1(
        env=env,
        entity1_cfg=entity1_cfg,
        entity2_cfg=entity2_cfg,
    )

    # Apply exponential kernel to convert penalty to positive reward
    return torch.exp(-lambda_exp * orientation_error_l1)


"""
Contact rewards.
"""


def desired_contacts_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Reward proportional to the total number of active contact frames across history.

    This variant counts all active contacts across the entire history window, providing
    a continuous measure of contact quality. Higher values indicate more sustained contact.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Force threshold (in Newtons) above which a contact is considered active. Default is 0.5.

    Returns:
        Total count of active contacts across all filtered bodies and all history frames.

    Note:
        This uses filtered contact force history (force_matrix_w_history).
        The sensor must be configured with filter_prim_paths_expr.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Use filtered force matrix
    contacts = (
        contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :].norm(dim=-1) > threshold
    )
    # Sum across history, bodies, and filtered_bodies dimensions
    total_active_contacts = contacts.sum(dim=(1, 2, 3)).float()

    return total_active_contacts


def contact_force_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    lambda_exp: float = 0.01
) -> torch.Tensor:
    """Exponential reward for minimizing contact forces (encourages gentle contact).

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        lambda_exp: Lambda parameter for the exponential kernel. Default is 0.01.

    Returns:
        Reward tensor of shape (num_envs,) with values in range (0, 1].

    Note:
        This uses the most recent filtered contact forces (not history).
        The sensor must be configured with filter_prim_paths_expr.
    """
    # Get contact sensor data
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Use filtered force matrix
    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]

    # Compute the L2 norm of forces
    force_magnitudes = torch.norm(contact_forces, dim=-1)

    # Sum forces across all bodies and filtered bodies
    total_force = force_magnitudes.sum(dim=(1, 2))

    # Apply exponential kernel: lower forces -> higher reward
    return torch.exp(-lambda_exp * total_force)
