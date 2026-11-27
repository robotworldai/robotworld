from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import NoiseCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rigid_body_projected_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rigid_object")
) -> torch.Tensor:
    """Get projected gravity for specified rigid body in world frame."""
    rigid_object: RigidObject = env.scene[asset_cfg.name]

    # Compute the projected gravity for the object
    projected_gravity = rigid_object.data.projected_gravity_b

    return projected_gravity


def object_rel_pos(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object"
) -> torch.Tensor:
    """Get object position relative to reference frame using transform sensor.

    Args:
        env: The environment instance.
        sensor_cfg: The transform sensor configuration.
        target_frame_name: Name of the target frame in the transform sensor.

    Returns:
        torch.Tensor: Relative position [num_envs, 3].
    """
    # Get the frame transformer sensor
    frame_transformer = env.scene[sensor_cfg.name]

    # Get relative position from transform sensor
    relative_pos = frame_transformer.data.target_pos_source[:, frame_transformer.data.target_frame_names.index(target_frame_name), :]

    return relative_pos


def object_rel_quat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object"
) -> torch.Tensor:
    """Get object orientation relative to reference frame using transform sensor.

    Args:
        env: The environment instance.
        sensor_cfg: The transform sensor configuration.
        target_frame_name: Name of the target frame in the transform sensor.

    Returns:
        torch.Tensor: Relative orientation as quaternion [num_envs, 4].
    """
    # Get the frame transformer sensor
    frame_transformer = env.scene[sensor_cfg.name]

    # Get relative orientation from transform sensor
    relative_quat = frame_transformer.data.target_quat_source[:, frame_transformer.data.target_frame_names.index(target_frame_name), :]

    return relative_quat


def object_rel_quat_with_noise(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object",
    noise_std: float = 0.05
) -> torch.Tensor:
    """Get object orientation relative to reference frame with multiplicative noise.

    Adds small random rotation in SO(3) by composing the measured quaternion with a
    random quaternion generated from small angle-axis perturbations.

    Args:
        env: The environment instance.
        sensor_cfg: The transform sensor configuration.
        target_frame_name: Name of the target frame in the transform sensor.
        noise_std: Standard deviation of the random angle-axis noise (in radians).

    Returns:
        torch.Tensor: Relative orientation as quaternion with noise [num_envs, 4].
    """
    # Get the clean relative quaternion
    relative_quat = object_rel_quat(env, sensor_cfg=sensor_cfg, target_frame_name=target_frame_name)

    # Generate small random angle-axis perturbations
    random_axis = torch.randn(env.num_envs, 3, device=env.device)
    random_axis = random_axis / (torch.norm(random_axis, dim=-1, keepdim=True) + 1e-8)

    # Sample random angles from normal distribution
    random_angles = torch.randn(env.num_envs, device=env.device) * noise_std

    # Convert angle-axis to quaternion
    half_angles = random_angles / 2.0
    noise_quat = torch.zeros(env.num_envs, 4, device=env.device)
    noise_quat[:, 0] = torch.cos(half_angles)  # w component
    noise_quat[:, 1:] = torch.sin(half_angles).unsqueeze(-1) * random_axis  # xyz components

    # Compose quaternions: q_noisy = q_noise * q_measured
    noisy_quat = math_utils.quat_mul(noise_quat, relative_quat)

    return noisy_quat


def object_rel_lin_vel(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis")
) -> torch.Tensor:
    """Get target asset linear velocity relative to reference asset.

    Works with both RigidObject and Articulation assets.

    Args:
        env: The environment instance.
        target_asset_cfg: The target asset configuration (RigidObject or Articulation).
        reference_asset_cfg: The reference asset configuration (RigidObject or Articulation).

    Returns:
        torch.Tensor: Relative linear velocity [num_envs, 3].
    """
    # Get target and reference assets
    target_asset = env.scene[target_asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]

    # Get target asset linear velocity in world frame
    if isinstance(target_asset, RigidObject):
        target_lin_vel_w = target_asset.data.root_link_lin_vel_w
    elif isinstance(target_asset, Articulation):
        if isinstance(target_asset_cfg.body_ids, slice):
            raise ValueError("target_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = target_asset_cfg.body_ids[0] if isinstance(target_asset_cfg.body_ids, list) else target_asset_cfg.body_ids
        target_lin_vel_w = target_asset.data.body_link_lin_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported target asset type: {type(target_asset)}")

    # Get reference asset linear velocity in world frame
    if isinstance(reference_asset, RigidObject):
        reference_lin_vel_w = reference_asset.data.root_link_lin_vel_w
    elif isinstance(reference_asset, Articulation):
        if isinstance(reference_asset_cfg.body_ids, slice):
            raise ValueError("reference_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = reference_asset_cfg.body_ids[0] if isinstance(reference_asset_cfg.body_ids, list) else reference_asset_cfg.body_ids
        reference_lin_vel_w = reference_asset.data.body_link_lin_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported reference asset type: {type(reference_asset)}")

    # Compute relative linear velocity (target velocity relative to reference)
    rel_lin_vel = target_lin_vel_w - reference_lin_vel_w

    return rel_lin_vel


def object_rel_ang_vel(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis")
) -> torch.Tensor:
    """Get target asset angular velocity relative to reference asset.

    Works with both RigidObject and Articulation assets.

    Args:
        env: The environment instance.
        target_asset_cfg: The target asset configuration (RigidObject or Articulation).
        reference_asset_cfg: The reference asset configuration (RigidObject or Articulation).

    Returns:
        torch.Tensor: Relative angular velocity [num_envs, 3].
    """
    # Get target and reference assets
    target_asset = env.scene[target_asset_cfg.name]
    reference_asset = env.scene[reference_asset_cfg.name]

    # Get target asset angular velocity in world frame
    if isinstance(target_asset, RigidObject):
        target_ang_vel_w = target_asset.data.root_link_ang_vel_w
    elif isinstance(target_asset, Articulation):
        if isinstance(target_asset_cfg.body_ids, slice):
            raise ValueError("target_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = target_asset_cfg.body_ids[0] if isinstance(target_asset_cfg.body_ids, list) else target_asset_cfg.body_ids
        target_ang_vel_w = target_asset.data.body_link_ang_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported target asset type: {type(target_asset)}")

    # Get reference asset angular velocity in world frame
    if isinstance(reference_asset, RigidObject):
        reference_ang_vel_w = reference_asset.data.root_link_ang_vel_w
    elif isinstance(reference_asset, Articulation):
        if isinstance(reference_asset_cfg.body_ids, slice):
            raise ValueError("reference_asset_cfg.body_ids cannot be a slice for Articulation.")
        else:
            body_idx = reference_asset_cfg.body_ids[0] if isinstance(reference_asset_cfg.body_ids, list) else reference_asset_cfg.body_ids
        reference_ang_vel_w = reference_asset.data.body_link_ang_vel_w[:, body_idx, :]
    else:
        raise ValueError(f"Unsupported reference asset type: {type(reference_asset)}")

    # Compute relative angular velocity (target velocity relative to reference)
    rel_ang_vel = target_ang_vel_w - reference_ang_vel_w

    return rel_ang_vel


def object_rel_pos_top(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
    target_frame_name: str = "object",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale_event_term_name: str = "random_object_scale"
) -> torch.Tensor:
    """Get object top surface position relative to reference frame using transform sensor.

    This function returns the position of the top of the object instead of its center,
    by transforming the offset from object frame to camera frame using the relative quaternion.

    Args:
        env: The environment instance.
        sensor_cfg: The transform sensor configuration.
        target_frame_name: Name of the target frame in the transform sensor.
        object_cfg: The object asset configuration to get base height.
        scale_event_term_name: Name of the event term that stores object scales.

    Returns:
        torch.Tensor: Relative position of object top [num_envs, 3].
    """
    # Get the frame transformer sensor
    frame_transformer = env.scene[sensor_cfg.name]
    target_idx = frame_transformer.data.target_frame_names.index(target_frame_name)

    # Get relative position from transform sensor (this is the center position in camera frame)
    relative_pos = frame_transformer.data.target_pos_source[:, target_idx, :]

    # Get relative quaternion (object orientation in camera frame)
    relative_quat = frame_transformer.data.target_quat_source[:, target_idx, :]

    # Get the object to retrieve base height
    target_object: RigidObject = env.scene[object_cfg.name]
    object_base_height = getattr(target_object.cfg.spawn, 'height', 0.1)

    # Try to get the scale from event manager
    try:
        scales = env.event_manager.get_term_return_value(scale_event_term_name)
        if scales is not None and scales.numel() > 0:
            height_scales = scales[:, 1]
        else:
            height_scales = torch.ones(env.num_envs, device=env.device)
    except (ValueError, AttributeError, IndexError):
        height_scales = torch.ones(env.num_envs, device=env.device)

    # Calculate half of the scaled height
    half_scaled_height = (object_base_height * height_scales) / 2.0

    # Create offset in object frame [num_envs, 3] - vertical offset only
    offset_object_frame = torch.zeros(env.num_envs, 3, device=env.device)
    offset_object_frame[:, 2] = half_scaled_height

    # Transform offset from object frame to camera frame using relative quaternion
    offset_camera_frame = math_utils.quat_apply(relative_quat, offset_object_frame)

    # Add transformed offset to get top position in camera frame
    relative_pos_top = relative_pos + offset_camera_frame

    return relative_pos_top

def tray_holder_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tray_contact_sensor"),
) -> torch.Tensor:
    """Get contact force vectors between tray and tray holders.

    Returns the 3D NORMAL contact force vectors for each holder (left and right).

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the tray contact sensor.

    Returns:
        Tensor of shape [num_envs, 6] containing normal force vectors:
        - Columns 0-2: Left holder contact force vector (Fx, Fy, Fz) in world frame
        - Columns 3-5: Right holder contact force vector (Fx, Fy, Fz) in world frame

    Note:
        Uses force_matrix_w which contains filtered contact forces only.
        Shape: force_matrix_w is (N, B=1, M=2, 3) where M=2 are the two holders.
    """
    # Get filtered contact forces from the sensor
    force_matrix = env.scene.sensors[sensor_cfg.name].data.force_matrix_w

    # Squeeze out the sensor body dimension (B=1, only the tray)
    holder_forces = force_matrix.squeeze(1)

    # Flatten to get force vectors: shape (num_envs, 6)
    force_vectors = holder_forces.view(holder_forces.shape[0], -1)

    return force_vectors


class CombinedCameraObjectObservations(ManagerTermBase):
    """Combined camera-based object observations with synchronized delays.

    This class combines object position (top surface) and quaternion observations
    from a camera reference frame with synchronized delays for distillation.
    """

    def __init__(
        self,
        cfg: SceneEntityCfg,
        env: ManagerBasedRLEnv,
        camera_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_camera_transform"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        target_frame_name: str = "object",
        scale_event_term_name: str = "random_object_scale",
        # Enable/disable individual components
        include_pos: bool = True,
        include_quat: bool = True,
        # Individual scaling factors
        pos_scale: float = 1.0,
        quat_scale: float = 1.0,
        # Individual clipping ranges (None = no clipping)
        pos_clip: tuple[float, float] | None = None,
        quat_clip: tuple[float, float] | None = None,
        # Individual noise configurations
        pos_noise: NoiseCfg | None = None,
        quat_noise_std: float = 0.0,
    ):
        # Call parent constructor
        super().__init__(cfg, env)

        # Resolve SceneEntityCfg objects
        camera_sensor_cfg.resolve(env.scene)
        object_asset_cfg.resolve(env.scene)

        self.camera_sensor_cfg = camera_sensor_cfg
        self.object_asset_cfg = object_asset_cfg
        self.target_frame_name = target_frame_name
        self.scale_event_term_name = scale_event_term_name

        self.include_pos = include_pos
        self.include_quat = include_quat

        self.pos_scale = pos_scale
        self.quat_scale = quat_scale

        self.pos_clip = pos_clip
        self.quat_clip = quat_clip

        self.pos_noise = pos_noise
        self.quat_noise_std = quat_noise_std

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        # These parameters must match __init__ for IsaacLab's parameter validation
        camera_sensor_cfg: SceneEntityCfg | None = None,
        object_asset_cfg: SceneEntityCfg | None = None,
        target_frame_name: str | None = None,
        scale_event_term_name: str | None = None,
        include_pos: bool | None = None,
        include_quat: bool | None = None,
        pos_scale: float | None = None,
        quat_scale: float | None = None,
        pos_clip: tuple[float, float] | None = None,
        quat_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        quat_noise_std: float | None = None,
    ) -> torch.Tensor:
        observations = []

        # 1. Object relative position (top surface) in camera frame (3D)
        if self.include_pos:
            pos_rel = object_rel_pos_top(
                env,
                sensor_cfg=self.camera_sensor_cfg,
                target_frame_name=self.target_frame_name,
                object_cfg=self.object_asset_cfg,
                scale_event_term_name=self.scale_event_term_name
            )
            # Apply noise first (if configured)
            if self.pos_noise is not None:
                pos_rel = self.pos_noise.func(pos_rel, self.pos_noise)
            # Then clip
            if self.pos_clip is not None:
                pos_rel = torch.clamp(pos_rel, self.pos_clip[0], self.pos_clip[1])
            # Then scale
            pos_rel = pos_rel * self.pos_scale
            observations.append(pos_rel)

        # 2. Object relative quaternion in camera frame (4D)
        if self.include_quat:
            quat_rel = object_rel_quat_with_noise(
                env,
                sensor_cfg=self.camera_sensor_cfg,
                target_frame_name=self.target_frame_name,
                noise_std=self.quat_noise_std
            )
            # Clip (if configured)
            if self.quat_clip is not None:
                quat_rel = torch.clamp(quat_rel, self.quat_clip[0], self.quat_clip[1])
            # Then scale
            quat_rel = quat_rel * self.quat_scale
            observations.append(quat_rel)

        # Concatenate all enabled observations
        if observations:
            combined = torch.cat(observations, dim=-1)
        else:
            combined = torch.zeros(env.num_envs, 0, device=env.device)

        return combined


class CombinedObjectObservationsDict(ManagerTermBase):
    """Combined object observations with synchronized delays and flexible configuration.

    This class combines object position, angular velocity, linear velocity, and
    projected gravity observations with per-component noise, scaling, and clipping.
    All enabled observations share the same delay buffer for synchronized delays.
    """

    def __init__(
        self,
        cfg: SceneEntityCfg,
        env: ManagerBasedRLEnv,
        object_sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_tray_transform"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
        target_frame_name: str = "object",
        # Enable/disable individual components
        include_pos: bool = True,
        include_ang_vel: bool = True,
        include_lin_vel: bool = True,
        include_gravity: bool = True,
        # Individual scaling factors
        pos_scale: float = 1.0,
        ang_vel_scale: float = 1.0,
        lin_vel_scale: float = 1.0,
        gravity_scale: float = 1.0,
        # Individual clipping ranges (None = no clipping)
        pos_clip: tuple[float, float] | None = None,
        ang_vel_clip: tuple[float, float] | None = None,
        lin_vel_clip: tuple[float, float] | None = None,
        gravity_clip: tuple[float, float] | None = None,
        # Individual noise configurations (None = no noise)
        pos_noise: NoiseCfg | None = None,
        ang_vel_noise: NoiseCfg | None = None,
        lin_vel_noise: NoiseCfg | None = None,
        gravity_noise: NoiseCfg | None = None,
    ):
        # Call parent constructor
        super().__init__(cfg, env)

        # Resolve SceneEntityCfg objects to convert body_names to body_ids
        object_sensor_cfg.resolve(env.scene)
        object_asset_cfg.resolve(env.scene)
        robot_torso_cfg.resolve(env.scene)

        self.object_sensor_cfg = object_sensor_cfg
        self.object_asset_cfg = object_asset_cfg
        self.robot_torso_cfg = robot_torso_cfg
        self.target_frame_name = target_frame_name

        self.include_pos = include_pos
        self.include_ang_vel = include_ang_vel
        self.include_lin_vel = include_lin_vel
        self.include_gravity = include_gravity

        self.pos_scale = pos_scale
        self.ang_vel_scale = ang_vel_scale
        self.lin_vel_scale = lin_vel_scale
        self.gravity_scale = gravity_scale

        self.pos_clip = pos_clip
        self.ang_vel_clip = ang_vel_clip
        self.lin_vel_clip = lin_vel_clip
        self.gravity_clip = gravity_clip

        # Store noise configurations
        self.pos_noise = pos_noise
        self.ang_vel_noise = ang_vel_noise
        self.lin_vel_noise = lin_vel_noise
        self.gravity_noise = gravity_noise

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        # These parameters must match __init__ for IsaacLab's parameter validation
        object_sensor_cfg: SceneEntityCfg | None = None,
        object_asset_cfg: SceneEntityCfg | None = None,
        robot_torso_cfg: SceneEntityCfg | None = None,
        target_frame_name: str | None = None,
        include_pos: bool | None = None,
        include_ang_vel: bool | None = None,
        include_lin_vel: bool | None = None,
        include_gravity: bool | None = None,
        pos_scale: float | None = None,
        ang_vel_scale: float | None = None,
        lin_vel_scale: float | None = None,
        gravity_scale: float | None = None,
        pos_clip: tuple[float, float] | None = None,
        ang_vel_clip: tuple[float, float] | None = None,
        lin_vel_clip: tuple[float, float] | None = None,
        gravity_clip: tuple[float, float] | None = None,
        pos_noise: NoiseCfg | None = None,
        ang_vel_noise: NoiseCfg | None = None,
        lin_vel_noise: NoiseCfg | None = None,
        gravity_noise: NoiseCfg | None = None,
    ) -> torch.Tensor:
        observations = []

        # 1. Object relative position (3D)
        if self.include_pos:
            pos_rel = object_rel_pos(
                env,
                sensor_cfg=self.object_sensor_cfg,
                target_frame_name=self.target_frame_name
            )
            if self.pos_noise is not None:
                pos_rel = self.pos_noise.func(pos_rel, self.pos_noise)
            if self.pos_clip is not None:
                pos_rel = torch.clamp(pos_rel, self.pos_clip[0], self.pos_clip[1])
            pos_rel = pos_rel * self.pos_scale
            observations.append(pos_rel)

        # 2. Object relative angular velocity (3D)
        if self.include_ang_vel:
            ang_vel_rel = object_rel_ang_vel(
                env,
                target_asset_cfg=self.object_asset_cfg,
                reference_asset_cfg=self.robot_torso_cfg
            )
            if self.ang_vel_noise is not None:
                ang_vel_rel = self.ang_vel_noise.func(ang_vel_rel, self.ang_vel_noise)
            if self.ang_vel_clip is not None:
                ang_vel_rel = torch.clamp(ang_vel_rel, self.ang_vel_clip[0], self.ang_vel_clip[1])
            ang_vel_rel = ang_vel_rel * self.ang_vel_scale
            observations.append(ang_vel_rel)

        # 3. Object relative linear velocity (3D)
        if self.include_lin_vel:
            lin_vel_rel = object_rel_lin_vel(
                env,
                target_asset_cfg=self.object_asset_cfg,
                reference_asset_cfg=self.robot_torso_cfg
            )
            if self.lin_vel_noise is not None:
                lin_vel_rel = self.lin_vel_noise.func(lin_vel_rel, self.lin_vel_noise)
            if self.lin_vel_clip is not None:
                lin_vel_rel = torch.clamp(lin_vel_rel, self.lin_vel_clip[0], self.lin_vel_clip[1])
            lin_vel_rel = lin_vel_rel * self.lin_vel_scale
            observations.append(lin_vel_rel)

        # 4. Object projected gravity (3D)
        if self.include_gravity:
            proj_grav = rigid_body_projected_gravity(
                env,
                asset_cfg=self.object_asset_cfg
            )
            if self.gravity_noise is not None:
                proj_grav = self.gravity_noise.func(proj_grav, self.gravity_noise)
            if self.gravity_clip is not None:
                proj_grav = torch.clamp(proj_grav, self.gravity_clip[0], self.gravity_clip[1])
            proj_grav = proj_grav * self.gravity_scale
            observations.append(proj_grav)

        # Concatenate all enabled observations
        if observations:
            combined = torch.cat(observations, dim=-1)
        else:
            combined = torch.zeros(env.num_envs, 0, device=env.device)

        return combined
