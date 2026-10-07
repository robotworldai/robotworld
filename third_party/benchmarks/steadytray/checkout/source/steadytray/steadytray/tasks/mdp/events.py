"""Event functions for the locomotion environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, Sdf, UsdGeom, Vt
import omni.usd
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_all_rel_pos = torch.tensor([])  # Global variable to store relative positions across resets
_all_scales = torch.tensor([])  # Global variable to store object body scales (radius, height) across resets

def _compute_and_set_object_state(
    env: ManagerBasedRLEnv,
    qualified_env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]],
    relative_velocity: dict[str, float],
    z_offset: float | None = None,
) -> torch.Tensor:
    """Compute and set the target object state relative to base asset.
    
    Args:
        env: The environment instance.
        qualified_env_ids: The environment IDs that qualify for object placement.
        base_asset_cfg: The base asset configuration (can be Articulation or RigidObject).
        target_asset_cfg: The target rigid object asset configuration.
        relative_pose: The relative pose for the object positioning. 
                      Note: If z_offset is None, "z" is the absolute height.
                            If z_offset is provided, it overrides "z" and represents the gap.
        relative_velocity: The relative velocity for the object.
        z_offset: Optional gap between object bottom and base_asset top. When provided,
                 the object position is calculated as: gap + (object_height * height_scale) / 2.
        
    Returns:
        torch.Tensor: The relative position in base asset's local frame for qualified environments.
    """
    global _all_scales
    
    # Extract base and target assets
    base_asset: Articulation | RigidObject = env.scene[base_asset_cfg.name]
    target_object: RigidObject = env.scene[target_asset_cfg.name]
    
    # Get the base height of the object from its spawn configuration (only needed if z_offset is used)
    object_base_height = getattr(target_object.cfg.spawn, 'height', 0.1)
    
    # Get base asset state (position and orientation)
    if isinstance(base_asset, Articulation):
        # For articulation, get body link pose
        if isinstance(base_asset_cfg.body_ids, slice):
            raise ValueError("Body IDs for articulation must be specified as a list or int, not slice.")
        else:
            body_idx = base_asset_cfg.body_ids[0]
        base_pos = base_asset.data.body_link_pos_w[qualified_env_ids, body_idx]
        base_quat = base_asset.data.body_link_quat_w[qualified_env_ids, body_idx]
    else:
        # For rigid object, get root pose
        base_pos = base_asset.data.root_pos_w[qualified_env_ids]
        base_quat = base_asset.data.root_quat_w[qualified_env_ids]

    # Use fixed relative position (no randomization)
    num_resets = len(qualified_env_ids)

    # Create relative position in robot local frame
    # If relative_pose["x"] and "y" are tuples, randomize within the range
    # Otherwise, use the fixed value
    if isinstance(relative_pose["x"], tuple):
        rel_pos_x = torch.rand(num_resets, device=env.device) * (relative_pose["x"][1] - relative_pose["x"][0]) + relative_pose["x"][0]
    else:
        rel_pos_x = torch.full((num_resets,), float(relative_pose["x"]), device=env.device)
        
    if isinstance(relative_pose["y"], tuple):
        rel_pos_y = torch.rand(num_resets, device=env.device) * (relative_pose["y"][1] - relative_pose["y"][0]) + relative_pose["y"][0]
    else:
        rel_pos_y = torch.full((num_resets,), float(relative_pose["y"]), device=env.device)

    # Compute z position based on whether z_offset parameter is provided
    if z_offset is not None:
        # z_offset mode: gap between object bottom and base_asset top
        # Object position = z_offset + (object_height * height_scale) / 2
        # The root link is in the middle of the cylinder, so we add half the scaled height
        z_gap = torch.full((num_resets,), float(z_offset), device=env.device)
        
        # Get the height scale for these environments from the global variable
        # _all_scales shape: [num_envs, 2] where column 1 is height_scale
        if _all_scales.numel() > 0 and _all_scales.shape[0] >= env.num_envs:
            height_scales = _all_scales[qualified_env_ids, 1]  # Get height scale for qualified envs
        else:
            # If scales not available, use default scale of 1.0 (no scaling)
            height_scales = torch.ones(num_resets, device=env.device)
        
        # Calculate the z position: gap + half of scaled object height
        scaled_half_height = (object_base_height * height_scales) / 2.0
        rel_pos_z = z_gap + scaled_half_height
    else:
        # Original mode: use z from relative_pose as absolute height
        if isinstance(relative_pose["z"], tuple):
            rel_pos_z = torch.rand(num_resets, device=env.device) * (relative_pose["z"][1] - relative_pose["z"][0]) + relative_pose["z"][0]
        else:
            rel_pos_z = torch.full((num_resets,), float(relative_pose["z"]), device=env.device)
    
    rel_pos_local = torch.stack([rel_pos_x, rel_pos_y, rel_pos_z], dim=1)
    
    # Transform relative position from base local frame to world frame
    rel_pos_world = math_utils.quat_apply(base_quat, rel_pos_local)
    object_pos = base_pos + rel_pos_world

    # Create relative orientation (roll, pitch, yaw)
    # Support randomization for roll, pitch, and yaw angles
    roll_spec = relative_pose.get("roll", 0.0)
    if isinstance(roll_spec, tuple):
        rel_roll = torch.rand(num_resets, device=env.device) * (roll_spec[1] - roll_spec[0]) + roll_spec[0]
    else:
        rel_roll = torch.full((num_resets,), float(roll_spec), device=env.device)
    
    pitch_spec = relative_pose.get("pitch", 0.0)
    if isinstance(pitch_spec, tuple):
        rel_pitch = torch.rand(num_resets, device=env.device) * (pitch_spec[1] - pitch_spec[0]) + pitch_spec[0]
    else:
        rel_pitch = torch.full((num_resets,), float(pitch_spec), device=env.device)
    
    yaw_spec = relative_pose.get("yaw", 0.0)
    if isinstance(yaw_spec, tuple):
        rel_yaw = torch.rand(num_resets, device=env.device) * (yaw_spec[1] - yaw_spec[0]) + yaw_spec[0]
    else:
        rel_yaw = torch.full((num_resets,), float(yaw_spec), device=env.device)
    
    # Convert euler angles (roll, pitch, yaw) to quaternion
    # Note: This creates a quaternion in the base asset's local frame
    rel_quat = math_utils.quat_from_euler_xyz(rel_roll, rel_pitch, rel_yaw)
    
    # Combine base orientation with relative orientation
    # object_quat = base_quat * rel_quat (quaternion multiplication)
    object_quat = math_utils.quat_mul(base_quat, rel_quat)

    # Transform relative velocities from base local frame to world frame
    rel_lin_vel_local = torch.stack([
        torch.full((num_resets,), float(relative_velocity["x"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["y"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["z"]), device=env.device)
    ], dim=1)
    
    rel_ang_vel_local = torch.stack([
        torch.full((num_resets,), float(relative_velocity["roll"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["pitch"]), device=env.device),
        torch.full((num_resets,), float(relative_velocity["yaw"]), device=env.device)
    ], dim=1)
    
    # Transform linear and angular velocities from base local frame to world frame
    object_lin_vel = math_utils.quat_apply(base_quat, rel_lin_vel_local)
    object_ang_vel = math_utils.quat_apply(base_quat, rel_ang_vel_local)
    
    # Set the target object root state
    object_root_state = target_object.data.default_root_state[qualified_env_ids].clone()
    object_root_state[:, :3] = object_pos
    object_root_state[:, 3:7] = object_quat
    object_root_state[:, 7:10] = object_lin_vel
    object_root_state[:, 10:13] = object_ang_vel
    
    # Apply the new root state
    target_object.write_root_state_to_sim(object_root_state, qualified_env_ids)

    # Return the relative position in local frame (before transformation to world)
    return rel_pos_local

def set_rigid_object_relative_to_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]] = {
        "x": (-0.05, 0.05),  # Randomize x and y position within -5cm to +5cm
        "y": (-0.05, 0.05),
        "z": 0.08,           # Height above plate (or gap if z_offset is provided)
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0, 
    },
    relative_velocity: dict[str, float] = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
    z_offset: float | None = None,
) -> torch.Tensor:
    """Set target object position relative to base asset (articulation or rigid object).

    This function places the target object at the desired relative position to the base asset's
    current position and orientation for all provided environment IDs. The base asset can be
    either an Articulation (e.g., robot with a specific body) or a RigidObject.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to set.
        base_asset_cfg: The base asset configuration (can be Articulation body or RigidObject).
        target_asset_cfg: The target rigid object asset configuration.
        relative_pose: The fixed or randomized range of relative pose for the object positioning.
                      Note: "z" is absolute height if z_offset is None, otherwise ignored.
        relative_velocity: The fixed relative velocity for the object.
        z_offset: Optional gap between object bottom and base_asset top. When provided,
                 it overrides relative_pose["z"] and the object position is calculated as:
                 gap + (object_height * height_scale) / 2.
        
    Returns:
        torch.Tensor: The relative position in base asset's local frame for the specified environments.
                     Shape: [len(env_ids), 3].
    """
    global _all_rel_pos

    # Initialize _all_rel_pos if not already done or resize if needed
    if _all_rel_pos.numel() == 0 or _all_rel_pos.shape[0] < env.num_envs:
        _all_rel_pos = torch.zeros((env.num_envs, 3), device=env.device)

    # Ensure _all_rel_pos is on the correct device
    if _all_rel_pos.device != env.device:
        _all_rel_pos = _all_rel_pos.to(env.device)

    # Resolve the asset configurations
    base_asset_cfg.resolve(env.scene)
    target_asset_cfg.resolve(env.scene)
    
    # Simply call the compute and set function for all provided env_ids
    rel_pos_local = _compute_and_set_object_state(
        env,
        env_ids,
        base_asset_cfg,
        target_asset_cfg,
        relative_pose,
        relative_velocity,
        z_offset,
    )

    # Store the relative position in local frame for all environments
    _all_rel_pos[env_ids] = rel_pos_local

    return _all_rel_pos


def randomize_cylinder_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    radius_scale_range: tuple[float, float],
    height_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
) -> torch.Tensor:
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
        
    Returns:
        torch.Tensor: The sampled scales for all environments. 
                     Shape: [num_envs, 2] where columns are [radius_scale, height_scale].
    """
    global _all_scales
    
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # Initialize _all_scales if not already done or resize if needed
    if _all_scales.numel() == 0 or _all_scales.shape[0] < env.num_envs:
        _all_scales = torch.ones((env.num_envs, 2), device=env.device)
    
    # Ensure _all_scales is on the correct device
    if _all_scales.device != env.device:
        _all_scales = _all_scales.to(env.device)

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)


    # sample scale values
    radius_samples = math_utils.sample_uniform(
        radius_scale_range[0], radius_scale_range[1], (len(env_ids),), device="cpu"
    )
    height_samples = math_utils.sample_uniform(
        height_scale_range[0], height_scale_range[1], (len(env_ids),), device="cpu"
    )
    
    # Store the sampled scales in the global variable (before converting to list)
    sampled_scales = torch.stack([radius_samples, height_samples], dim=1).to(env.device)
    _all_scales[env_ids] = sampled_scales
    
    # convert to list for the for loop
    rand_samples = torch.stack([radius_samples, radius_samples, height_samples], dim=1)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
    
    # Return the stored scales for all environments
    return _all_scales

def randomize_rigid_body_com_fixed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.
    .. note::z
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()
    # Randomize the com in range
    coms[:,:3] += rand_samples
    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)