# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class BallCatchingSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    ball: RigidObjectCfg = MISSING
    bucket: RigidObjectCfg = MISSING

    # -- static visual elements (defaults, no physics) --
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.5, 0.0, 0.2]),
    )

    # -- launcher visual (static, no collision) --
    launcher_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LauncherBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.30, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.5, 0.0, 0.42]),
    )

    launcher_cradle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LauncherCradle",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.40)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.5, 0.0, 0.455]),
    )


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ball_position = ObsTerm(func=mdp.ball_position_in_robot_root_frame)
        ball_velocity = ObsTerm(func=mdp.ball_velocity_in_robot_root_frame)
        gripper_width = ObsTerm(func=mdp.gripper_width_obs)
        task_phase = ObsTerm(func=mdp.task_phase_obs)
        launch_detected = ObsTerm(func=mdp.launch_detected_obs)
        predicted_intercept_pos = ObsTerm(func=mdp.predicted_intercept_position_obs)
        predicted_intercept_time = ObsTerm(func=mdp.predicted_intercept_time_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_ball_state = EventTerm(
        func=mdp.reset_ball_to_launcher,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("ball")},
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    launch_ball = EventTerm(
        func=mdp.launch_ball,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("ball")},
    )

    update_ball_tracking = EventTerm(
        func=mdp.update_ball_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("ball")},
    )

    update_bucket = EventTerm(
        func=mdp.update_bucket_position,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "bucket_cfg": SceneEntityCfg("bucket"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )

    check_phase = EventTerm(
        func=mdp.check_phase_transitions,
        mode="interval",
        interval_range_s=(0.04, 0.04),
        params={
            "bucket_cfg": SceneEntityCfg("bucket"),
            "intercept_position_threshold": 0.001,
        },
    )

    check_catch = EventTerm(
        func=mdp.check_catch_containment,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "bucket_cfg": SceneEntityCfg("bucket"),
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ball_on_ground = DoneTerm(func=mdp.ball_on_ground)
    task_completed = DoneTerm(func=mdp.task_completed)


@configclass
class CurriculumCfg:
    pass


@configclass
class BallCatchingEnvCfg(ManagerBasedRLEnvCfg):
    sim_freq: float = 100.0
    robot_control_freq: float = 25.0

    scene: BallCatchingSceneCfg = BallCatchingSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    episode_length_s = 4.0

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = max(1, round(self.sim_freq / self.robot_control_freq))
        self.episode_length_s = 4.0

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**19
        self.sim.physx.gpu_max_rigid_contact_count = 2**19
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physx.gpu_heap_capacity = 2**25
        self.sim.physx.gpu_temp_buffer_capacity = 2**24
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.gpu_collision_stack_size = 2**25
