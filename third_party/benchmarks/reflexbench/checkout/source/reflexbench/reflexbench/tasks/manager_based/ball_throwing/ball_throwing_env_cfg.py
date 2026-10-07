# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base config for the toss-into-box task (ball pre-grasped in gripper)."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class BallThrowingSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    # Box is placed 0.8 m in front of the robot along +X.  The reset event
    # randomizes both this visible KLT USD prim and the task-side catch target
    # around this nominal pose.
    target_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBox",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.8, 0.0, 0.075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd",
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = (
        MISSING
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        object_velocity = ObsTerm(func=mdp.object_velocity_in_robot_root_frame)
        gripper_width = ObsTerm(func=mdp.gripper_width_obs)
        task_phase = ObsTerm(func=mdp.task_phase_obs)
        box_target = ObsTerm(func=mdp.box_target_position_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    fix_box_collision = EventTerm(
        func=mdp.fix_box_collision,
        mode="startup",
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    reset_box_position = EventTerm(
        func=mdp.reset_box_position,
        mode="reset",
        params={
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
        },
    )

    # Must run AFTER reset_robot_joints so the EE pose used to place the ball
    # reflects the (slightly randomised) reset arm posture.
    # ``finger_offset`` is in the EE local frame; the EE target sits at
    # panda_hand+0.1034 m (Franka TCP) which is past the finger tips, so we
    # pull the ball back 2.2 cm along -Z to land it between the two fingers.
    reset_ball_in_gripper = EventTerm(
        func=mdp.reset_ball_in_gripper,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "finger_offset": (0.0, 0.0, -0.022),
        },
    )

    check_phase = EventTerm(
        func=mdp.check_phase_transitions,
        mode="interval",
        interval_range_s=(0.04, 0.04),
        params={
            "release_dist": 0.08,
            "gripper_open_threshold": 0.06,
            # KLT bin is ~0.30 m x ~0.20 m at the opening.  Expand by about
            # one ball radius so balls resting against the inner wall/rim
            # still count as inside.
            "box_half_x": 0.15,
            "box_half_y": 0.095,
            "box_z_min": 0.015,
            "box_z_max": 0.18,
        },
    )

    # The ball spawns with ``disable_gravity=True`` so it cannot fall out
    # of the gripper before the policy stabilises.  This event re-applies
    # gravity as an external force as soon as phase >= 1 (released).
    apply_post_release_gravity = EventTerm(
        func=mdp.apply_post_release_gravity,
        mode="interval",
        interval_range_s=(0.04, 0.04),
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ball_mass": 0.04,
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Hard safety net: ball fell through the floor (should never happen but
    # keeps the env from running indefinitely in a numerical edge case).
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Ball has been released but landed on the ground outside the box -- end
    # the episode immediately so the env resets.
    object_missed_box = DoneTerm(
        func=mdp.object_missed_box,
        params={"object_cfg": SceneEntityCfg("object"), "floor_z": 0.02},
    )

    task_completed = DoneTerm(
        func=mdp.task_completed,
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class BallThrowingEnvCfg(ManagerBasedRLEnvCfg):
    sim_freq: float = 100.0
    robot_control_freq: float = 25.0

    scene: BallThrowingSceneCfg = BallThrowingSceneCfg(num_envs=4096, env_spacing=5.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Per-episode reset perturbation of the KLT box and catch target.
    box_reset_x_range: tuple[float, float] = (-0.15, 0.15)
    box_reset_y_range: tuple[float, float] = (-0.15, 0.15)

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = max(1, round(self.sim_freq / self.robot_control_freq))
        self.episode_length_s = 2.0

        self.sim.render_interval = self.decimation

        self.events.reset_box_position.params["x_range"] = self.box_reset_x_range
        self.events.reset_box_position.params["y_range"] = self.box_reset_y_range

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**19
        self.sim.physx.gpu_max_rigid_contact_count = 2**19
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physx.gpu_heap_capacity = 2**25
        self.sim.physx.gpu_temp_buffer_capacity = 2**24
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.gpu_collision_stack_size = 2**25
