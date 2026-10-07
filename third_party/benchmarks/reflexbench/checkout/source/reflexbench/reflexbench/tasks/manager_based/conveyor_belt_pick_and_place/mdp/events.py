# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for pick-place from conveyor: reset, conveyor friction, phase transitions, IK override."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import PhysxSchema, UsdPhysics

from isaaclab.assets import RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.sim.utils import get_all_matching_child_prims
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def fix_box_collision(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None) -> None:
    """Force convex decomposition on box meshes for correct collision."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for env_id in env_ids[:1]:
        box_path = f"/World/envs/env_{env_id.item()}/Box"
        meshes = get_all_matching_child_prims(box_path, lambda p: p.GetTypeName() == "Mesh")
        for mesh_prim in meshes:
            if not UsdPhysics.MeshCollisionAPI(mesh_prim):
                UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
            mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            if not PhysxSchema.PhysxConvexDecompositionCollisionAPI(mesh_prim):
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)


def reset_conveyor_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (-0.4, -0.2),
) -> None:
    """Set per-env random conveyor Y velocity at reset."""
    if not hasattr(env, "conveyor_random_velocity"):
        env.conveyor_random_velocity = torch.zeros(env.num_envs, 3, device=env.device)
    min_vel, max_vel = velocity_range
    random_y_vel = torch.rand(len(env_ids), device=env.device) * (max_vel - min_vel) + min_vel
    env.conveyor_random_velocity[env_ids, 1] = random_y_vel


def conveyor_belt_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    conveyor_height_range: tuple[float, float] = (0.10, 0.20),
    conveyor_x_range: tuple[float, float] = (0.3, 0.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Apply conveyor velocity to object when in height and X range.

    Only replaces Y component of velocity, preserving X/Z physics.
    """
    if not hasattr(env, "conveyor_random_velocity"):
        return
    obj: RigidObject = env.scene[asset_cfg.name]
    object_pos = obj.data.root_pos_w
    object_height = object_pos[:, 2]
    env_origins_x = env.scene.env_origins[:, 0]
    object_x_local = object_pos[:, 0] - env_origins_x

    min_z, max_z = conveyor_height_range
    min_x, max_x = conveyor_x_range
    is_on_conveyor = (
        (object_height >= min_z) & (object_height <= max_z)
        & (object_x_local >= min_x) & (object_x_local <= max_x)
    )
    if not is_on_conveyor.any():
        return
    conveyor_vel = env.conveyor_random_velocity
    current_lin_vel = obj.data.root_lin_vel_w.clone()
    current_ang_vel = obj.data.root_ang_vel_w.clone()
    new_lin_vel = current_lin_vel.clone()
    new_lin_vel[is_on_conveyor, 1] = conveyor_vel[is_on_conveyor, 1]
    obj.write_root_velocity_to_sim(
        torch.cat([new_lin_vel, current_ang_vel], dim=-1)
    )


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset task phase, one-shot reward flags, and init IK controller / buffers.

    Task phases:
        0: Approach  - RL controls approach and grasp
        1: Lift      - RL controls lifting object
        2: Transport - IK auto-transport to box
        3: Release   - Open gripper to release object
        4: Done      - Object in box, success
    """
    if not hasattr(env, "task_phase"):
        env.task_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        env.task_phase[env_ids] = 0

    # Reset all one-shot reward flags
    for flag_name in [
        "_grasp_success_given",
        "_lift_milestone_given",
        "_arrive_box_vertical_given",
        "_object_in_box_given",
        "_completion_reward_given",
    ]:
        if hasattr(env, flag_name):
            getattr(env, flag_name)[env_ids] = False

    if not hasattr(env, "box_pos_tensor"):
        env.box_pos_tensor = torch.tensor((0.0, -0.4, 0.3), device=env.device).unsqueeze(0)
    if not hasattr(env, "box_target_pose"):
        env.box_target_pose = torch.tensor(
            [0.0, -0.4, 0.3, 0.0, 1.0, 0.0, 0.0], device=env.device
        ).unsqueeze(0)

    if not hasattr(env, "ik_controller"):
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        )
        env.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=env.num_envs, device=env.device
        )
        robot = env.scene[robot_cfg.name]
        env.ee_body_idx = robot.find_bodies("panda_hand")[0][0]
        env.ee_jacobi_idx = env.ee_body_idx - 1
        env.arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    env.ik_controller.reset(env_ids)

    if not hasattr(env, "ik_step_counter"):
        env.ik_step_counter = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
    else:
        env.ik_step_counter[env_ids] = 0

    if not hasattr(env, "grasp_offset_b"):
        env.grasp_offset_b = torch.zeros(env.num_envs, 3, device=env.device)
    else:
        env.grasp_offset_b[env_ids] = 0.0


def check_phase_transitions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_threshold: float = 0.03,
    grasp_y_range: tuple[float, float] = (-0.25, 0.25),
    lift_height: float = 0.25,
    position_threshold: float = 0.05,
) -> None:
    """Update task phase: 0->1 (grasp), 1->2 (lift), 2->3 (at box), 3->4 (in box).

    Includes gripper closure verification and drop fallback logic.
    """
    if not hasattr(env, "task_phase"):
        return
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    ee_object_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    # Gripper closure check for robust grasp detection
    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    finger_width = finger_pos.sum(dim=1)
    is_gripper_closed = finger_width < 0.05

    is_grasping = (ee_object_distance < grasp_threshold) & is_gripper_closed
    object_height = object_pos_w[:, 2]

    # Phase 0 -> 1: Grasp success
    phase_0_mask = env.task_phase == 0
    can_transition_0_to_1 = phase_0_mask & is_grasping
    if can_transition_0_to_1.any():
        obj_quat_w = obj.data.root_quat_w
        rel_pos, _ = subtract_frame_transforms(
            ee_pos_w, ee_quat_w, object_pos_w, obj_quat_w
        )
        env.grasp_offset_b[can_transition_0_to_1] = rel_pos[can_transition_0_to_1]
        env.task_phase[can_transition_0_to_1] = 1

    # Phase 1 -> 2: Lift complete
    phase_1_mask = env.task_phase == 1
    can_transition_1_to_2 = phase_1_mask & is_grasping & (object_height >= lift_height)
    env.task_phase[can_transition_1_to_2] = 2
    if can_transition_1_to_2.any():
        env.ik_step_counter[can_transition_1_to_2] = 0

    # Phase 1 drop fallback -> Phase 0
    phase_1_dropped = phase_1_mask & (~is_grasping)
    if phase_1_dropped.any():
        env.task_phase[phase_1_dropped] = 0
        if hasattr(env, "_grasp_success_given"):
            env._grasp_success_given[phase_1_dropped] = False

    # Phase 2 -> 3: Arrived above box (local coords + grasp check)
    phase_2_mask = env.task_phase == 2
    obj_pos_local = object_pos_w - env.scene.env_origins[:, :3]
    dist_to_box = torch.norm(obj_pos_local - env.box_target_pose[:, :3], dim=1)
    can_transition_2_to_3 = phase_2_mask & (dist_to_box < position_threshold) & is_grasping
    env.task_phase[can_transition_2_to_3] = 3
    env.ik_step_counter[can_transition_2_to_3] = 0

    # Phase 2 drop fallback -> Phase 0
    phase_2_dropped = phase_2_mask & (~is_grasping)
    if phase_2_dropped.any():
        env.task_phase[phase_2_dropped] = 0
        if hasattr(env, "_grasp_success_given"):
            env._grasp_success_given[phase_2_dropped] = False

    # Phase 3 -> 4: Object in box (spatial detection)
    phase_3_mask = env.task_phase == 3
    obj_pos_local = object_pos_w - env.scene.env_origins[:, :3]
    in_box = (
        (obj_pos_local[:, 0] >= -0.15) & (obj_pos_local[:, 0] <= 0.15)
        & (obj_pos_local[:, 1] >= -0.6) & (obj_pos_local[:, 1] <= -0.3)
        & (obj_pos_local[:, 2] >= 0.02) & (obj_pos_local[:, 2] <= 0.05)
    )
    can_transition_3_to_4 = phase_3_mask & in_box
    env.task_phase[can_transition_3_to_4] = 4


def get_ik_action_override(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_scale: float = 0.5,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (arm_action, gripper_open) for phases 2-3; (None, None) otherwise.

    Phase 2: IK transport to box target.
    Phase 3: Delayed gripper release (hold closed for 3 steps, then open).
    """
    if not hasattr(env, "task_phase") or not hasattr(env, "ik_controller"):
        return None, None

    robot = env.scene[robot_cfg.name]
    arm_action = torch.zeros(env.num_envs, 7, device=env.device)
    gripper_open_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    default_joint_pos = robot.data.default_joint_pos[:, :7]
    current_joint_pos = robot.data.joint_pos[:, env.arm_joint_ids]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pos_w, ee_quat_w
    )
    jacobian = robot.root_physx_view.get_jacobians()[
        :, env.ee_jacobi_idx, :, env.arm_joint_ids
    ]
    target_pose = torch.zeros(env.num_envs, 7, device=env.device)

    phase_23_mask = (env.task_phase == 2) | (env.task_phase == 3)
    if phase_23_mask.any():
        target_pose[phase_23_mask] = env.box_target_pose
        env.ik_controller.set_command(target_pose)
        joint_pos_des = env.ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, current_joint_pos
        )
        arm_action[phase_23_mask] = (
            joint_pos_des[phase_23_mask] - default_joint_pos[phase_23_mask]
        ) / action_scale

    # Phase 3: delayed gripper release (hold 3 steps then open)
    phase_3_mask = env.task_phase == 3
    env.ik_step_counter[phase_3_mask] += 1
    phase_3_release = phase_3_mask & (env.ik_step_counter > 3)
    gripper_open_mask[phase_3_release] = True

    return arm_action, gripper_open_mask


def print_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distractor_cfg: SceneEntityCfg = SceneEntityCfg("distractor"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_threshold: float = 0.03,
) -> None:
    """Debug: print object/EE positions, phase, gripper width, and reward for env 0."""
    obj: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_w = obj.data.root_pos_w[0, :3]
    ee_pos_w = ee_frame.data.target_pos_w[0, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[0, 0, :]

    distractor_pos_str = ""
    if distractor_cfg.name in env.scene.keys():
        distractor_asset: RigidObject = env.scene[distractor_cfg.name]
        dist_pos_w = distractor_asset.data.root_pos_w[0, :3]
        distractor_pos_str = (
            f" | Distractor: [{dist_pos_w[0]:.3f}, {dist_pos_w[1]:.3f}, {dist_pos_w[2]:.3f}]"
        )

    root_pos_w = env.scene["robot"].data.root_pos_w[0, :3]
    root_quat_w = env.scene["robot"].data.root_quat_w[0, :]
    pos_b, _ = subtract_frame_transforms(
        root_pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
        pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
    )
    ee_pos_b, _ = subtract_frame_transforms(
        root_pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
        ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0),
    )
    pos = pos_b[0]

    robot = env.scene["robot"]
    ee_obj_dist = torch.norm(pos_w - ee_pos_w).item()
    phase = env.task_phase[0].item() if hasattr(env, "task_phase") else -1
    joints = robot.data.joint_pos[0, :7].tolist()
    joints_str = ", ".join([f"{j:.2f}" for j in joints])

    finger_joint_ids = robot.find_joints(["panda_finger.*"])[0]
    gripper_width = robot.data.joint_pos[0, finger_joint_ids].sum().item()

    box_dist = -1.0
    error_str = ""
    if hasattr(env, "box_target_pose"):
        target_xyz = env.box_target_pose[0, :3]
        errors = pos - target_xyz
        box_dist = torch.norm(pos - target_xyz).item()
        error_str = f"ErrXYZ: [{errors[0]:.3f}, {errors[1]:.3f}, {errors[2]:.3f}]"

    reward = env.reward_buf[0].item() if hasattr(env, "reward_buf") else 0.0

    print(
        f"[Debug] Step: {env.common_step_counter:4d} | "
        f"Phase: {phase} | "
        f"Reward: {reward:.2f} | "
        f"EEDist: {ee_obj_dist:.3f} | "
        f"GripWidth: {gripper_width:.3f} | "
        f"ObjLocal: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        f"{distractor_pos_str} | "
        f"{error_str} | "
        f"Dist: {box_dist:.3f}"
    )
    print(f"        Joints: [{joints_str}]")
