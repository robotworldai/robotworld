# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the toss-into-box task.

Phase machine (3 phases, ball pre-grasped):
    0: Hold / pre-release  – ball in gripper, robot is swinging the arm
    1: Released            – gripper opened (or ball drifted from EE)
    2: Done                – ball inside the target box (success)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.sim.utils import get_all_matching_child_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Nominal target-box centre in env-local coordinates.  Matches the
# AssetBaseCfg position in ``BallThrowingSceneCfg`` (KLT bin 0.8 m in front
# of the robot along +X).  Reset randomisation jitters the catch target
# around this point.
_NOMINAL_BOX_POS: tuple[float, float, float] = (0.8, 0.0, 0.075)

# Gravitational acceleration magnitude used when the post-release event
# re-applies gravity to the ball as an external force in body local frame.
_GRAVITY_MAGNITUDE: float = 9.81


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if env_ids.dim() == 0:
        return env_ids.unsqueeze(0)
    return env_ids


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternion (w, x, y, z)."""
    quat_xyz = quat[..., 1:]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(quat_xyz, t, dim=-1)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of quaternion (w, x, y, z)."""
    q_w = quat[..., 0:1]
    q_vec = quat[..., 1:]
    a = torch.cross(q_vec, vec, dim=-1)
    b = torch.cross(q_vec, a, dim=-1)
    return vec + 2 * (-q_w * a + b)


def _nominal_box_position(env: ManagerBasedEnv) -> tuple[float, float, float]:
    """Return the configured env-local KLT-box centre.

    The visible KLT mesh is a static ``AssetBaseCfg``.  If the environment
    cfg randomizes ``scene.target_box.init_state.pos`` before scene creation,
    all task-side target tensors must use that same configured position.
    """
    try:
        pos = env.cfg.scene.target_box.init_state.pos
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except Exception:
        return _NOMINAL_BOX_POS


def _get_prim_at_path(prim_path: str):
    """Return a USD prim using the Isaac Sim import path available at runtime."""
    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        from omni.isaac.core.utils.prims import get_prim_at_path

    return get_prim_at_path(prim_path)


def _set_box_prim_position(env: ManagerBasedEnv, env_id: int, pos_local: torch.Tensor) -> bool:
    """Move the static KLT box prim to ``pos_local`` inside one env namespace."""
    box_prim = _get_prim_at_path(f"/World/envs/env_{env_id}/TargetBox")
    if box_prim is None or not box_prim.IsValid():
        return False

    xformable = UsdGeom.Xformable(box_prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()

    value_cls = (
        Gf.Vec3f
        if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
        else Gf.Vec3d
    )
    translate_op.Set(
        value_cls(
            float(pos_local[0].item()),
            float(pos_local[1].item()),
            float(pos_local[2].item()),
        )
    )
    return True


# ------------------------------------------------------------------ #
#  Startup events                                                     #
# ------------------------------------------------------------------ #


def fix_box_collision(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Force convex decomposition on box meshes so the bin opening stays open.

    Same trick used in conveyor_belt_pick_and_place and the legacy ball_throwing: Isaac
    Lab would otherwise simplify the concave KLT bin into a single convex hull,
    sealing the opening.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for env_id in env_ids:
        box_path = f"/World/envs/env_{env_id.item()}/TargetBox"
        meshes = get_all_matching_child_prims(box_path, lambda p: p.GetTypeName() == "Mesh")
        for mesh_prim in meshes:
            if not UsdPhysics.MeshCollisionAPI(mesh_prim):
                UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
            mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            if not PhysxSchema.PhysxConvexDecompositionCollisionAPI(mesh_prim):
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)




# ------------------------------------------------------------------ #
#  Reset events                                                       #
# ------------------------------------------------------------------ #


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Initialise / reset task_phase and the env-local box target position."""
    env_ids = _resolve_env_ids(env, env_ids)

    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        env.task_phase[env_ids] = 0

    # Sticky success flag and in-box dwell counter.  Cleared here at every
    # episode reset so each new attempt starts un-succeeded.
    if not hasattr(env, "task_succeeded"):
        env.task_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        env.task_succeeded[env_ids] = False
    if not hasattr(env, "box_dwell_steps"):
        env.box_dwell_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        env.box_dwell_steps[env_ids] = 0

    if not hasattr(env, "box_target_position"):
        nominal_box_pos = _nominal_box_position(env)
        env.box_target_position = torch.zeros(env.num_envs, 3, device=env.device)
        env.box_target_position[:, 0] = nominal_box_pos[0]
        env.box_target_position[:, 1] = nominal_box_pos[1]
        env.box_target_position[:, 2] = nominal_box_pos[2]


def reset_box_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """Randomise the visible KLT box and its env-local catch target."""
    env_ids = _resolve_env_ids(env, env_ids)
    if not hasattr(env, "box_target_position"):
        nominal_box_pos = _nominal_box_position(env)
        env.box_target_position = torch.zeros(env.num_envs, 3, device=env.device)
        env.box_target_position[:, 0] = nominal_box_pos[0]
        env.box_target_position[:, 1] = nominal_box_pos[1]
        env.box_target_position[:, 2] = nominal_box_pos[2]

    n = len(env_ids)
    dx = torch.rand(n, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
    dy = torch.rand(n, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
    nominal_box_pos = _nominal_box_position(env)

    # 1) Env-local catch-region centre (used by judges).
    env.box_target_position[env_ids, 0] = nominal_box_pos[0] + dx
    env.box_target_position[env_ids, 1] = nominal_box_pos[1] + dy
    env.box_target_position[env_ids, 2] = nominal_box_pos[2]

    moved_any = False
    for env_id in env_ids.tolist():
        moved_any = _set_box_prim_position(
            env,
            int(env_id),
            env.box_target_position[env_id],
        ) or moved_any

    if moved_any:
        env.sim.forward()


def reset_ball_in_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    finger_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Spawn the ball at the end-effector frame so the gripper is holding it.

    The end-effector frame target (configured in joint_pos_env_cfg) already
    points to the geometric centre between the two fingers (panda_hand + 0.1034
    m along its +Z), so by default we simply place the ball there.  Pass a
    non-zero ``finger_offset`` if a different pickup point is desired (offset
    is expressed in the EE local frame).
    """
    env_ids = _resolve_env_ids(env, env_ids)

    # Force FK to update so we read the post-reset EE pose rather than the
    # stale value from the previous episode.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)

    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]

    offset_local = torch.tensor(finger_offset, dtype=torch.float32, device=env.device)
    offset_local = offset_local.unsqueeze(0).expand(len(env_ids), -1)
    ball_pos_w = ee_pos_w + _quat_apply(ee_quat_w, offset_local)

    root_state = obj.data.default_root_state.clone()[env_ids]
    root_state[:, 0:3] = ball_pos_w
    root_state[:, 3:7] = ee_quat_w
    root_state[:, 7:] = 0.0

    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


# ------------------------------------------------------------------ #
#  Interval event: phase transitions                                  #
# ------------------------------------------------------------------ #


def check_phase_transitions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    release_dist: float = 0.08,
    gripper_open_threshold: float = 0.06,
    box_half_x: float = 0.15,
    box_half_y: float = 0.095,
    box_z_min: float = 0.015,
    box_z_max: float = 0.18,
) -> None:
    """Phase machine for the simplified toss-into-box task.

    Transitions:
        0 -> 1 : gripper opens OR ball drifts ``release_dist`` away from EE
        1 -> 2 : handled by ``task_completed`` after in-box dwell

    The catch volume is a rectangular box expanded by about one ball radius
    beyond the nominal KLT opening, so edge contacts still count as success.
    """
    if not hasattr(env, "task_phase"):
        return

    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    is_gripper_open = finger_width >= gripper_open_threshold
    is_released = is_gripper_open | (ee_object_distance > release_dist)

    # ----- Phase 0 -> 1: release detected ----- #
    phase_0_mask = env.task_phase == 0
    to_phase_1 = phase_0_mask & is_released
    if to_phase_1.any():
        env.task_phase[to_phase_1] = 1

    # Phase 1 -> 2 is intentionally not handled here.  ``task_completed`` runs
    # every termination step and requires several consecutive in-box frames
    # before latching success, so grazing the catch volume cannot mark success.


# ------------------------------------------------------------------ #
#  Interval event: re-apply gravity after release                     #
# ------------------------------------------------------------------ #


def apply_post_release_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ball_mass: float = 0.04,
) -> None:
    """Re-apply gravity to the ball once the task transitions to phase >= 1.

    The ball is spawned with ``disable_gravity=True`` to guarantee a stable
    grasp at episode start (the gripper only needs to keep the ball from
    drifting laterally; there is no downward weight to support).  This
    interval event compensates for that by applying ``mass * g`` as an
    external force on phase-1+ envs.

    The force is expressed in the body local frame (PhysX convention), so we
    project the world -Z direction into the current ball orientation each
    call.  For phase-0 envs the external force is held at zero.
    """
    if not hasattr(env, "task_phase"):
        return

    env_ids = _resolve_env_ids(env, env_ids)
    obj: RigidObject = env.scene[object_cfg.name]

    # World -Z expressed in each ball's local frame.
    quat_w = obj.data.root_quat_w[env_ids]
    world_down = torch.zeros((len(env_ids), 3), device=env.device)
    world_down[:, 2] = -1.0
    local_down = _quat_rotate_inverse(quat_w, world_down)

    released = env.task_phase[env_ids] >= 1
    force_magnitude = ball_mass * _GRAVITY_MAGNITUDE
    forces = torch.zeros((len(env_ids), 1, 3), device=env.device)
    forces[released, 0, :] = local_down[released] * force_magnitude
    torques = torch.zeros_like(forces)

    obj.set_external_force_and_torque(forces, torques, env_ids=env_ids)
