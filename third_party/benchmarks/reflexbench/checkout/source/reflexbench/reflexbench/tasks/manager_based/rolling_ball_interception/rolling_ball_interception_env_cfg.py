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
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from .constants import (
    RAMP_CENTER_POS,
    RAMP_EXIT_ZONE_CENTER_POS,
    RAMP_EXIT_ZONE_SIZE,
    RAMP_QUAT,
    RAMP_RAIL_LEFT_POS,
    RAMP_RAIL_RIGHT_POS,
    RAMP_RAIL_SIZE,
    RAMP_SIZE,
)
from . import mdp

# ---------------------------------------------------------------------------
# Ramp geometry constants
# ---------------------------------------------------------------------------
# The ramp exit is defined as the minimum-X edge line of the white platform
# below the ramp. `RAMP_EXIT_POS` is the midpoint of that line at Y=0.


@configclass
class RollingBallInterceptionSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    ball: RigidObjectCfg = MISSING
    catcher: RigidObjectCfg = MISSING

    # -- ground plane --
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- ramp surface (static collider) --
    ramp_surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RampSurface",
        spawn=sim_utils.CuboidCfg(
            size=RAMP_SIZE,
            collision_props=CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.4,
                restitution=0.1,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(RAMP_CENTER_POS),
            rot=list(RAMP_QUAT),
        ),
    )

    # -- ramp guide rail left (static collider) --
    ramp_rail_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RampRailLeft",
        spawn=sim_utils.CuboidCfg(
            size=RAMP_RAIL_SIZE,
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.40, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(RAMP_RAIL_LEFT_POS),
            rot=list(RAMP_QUAT),
        ),
    )

    # -- ramp guide rail right (static collider) --
    ramp_rail_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RampRailRight",
        spawn=sim_utils.CuboidCfg(
            size=RAMP_RAIL_SIZE,
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.40, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(RAMP_RAIL_RIGHT_POS),
            rot=list(RAMP_QUAT),
        ),
    )

    # -- ramp exit flat zone (static collider) --
    ramp_exit_zone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RampExitZone",
        spawn=sim_utils.CuboidCfg(
            size=RAMP_EXIT_ZONE_SIZE,
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.40, 0.40)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(RAMP_EXIT_ZONE_CENTER_POS),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
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
        rolling_detected = ObsTerm(func=mdp.rolling_detected_obs)
        ramp_exit_position = ObsTerm(func=mdp.ramp_exit_position_obs)
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
        func=mdp.reset_ball_to_ramp,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("ball")},
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    release_ball = EventTerm(
        func=mdp.release_ball,
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

    check_phase = EventTerm(
        func=mdp.check_phase_transitions,
        mode="interval",
        interval_range_s=(0.04, 0.04),
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    check_catch = EventTerm(
        func=mdp.check_catch_containment,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "catcher_cfg": SceneEntityCfg("catcher"),
        },
    )

    update_catcher = EventTerm(
        func=mdp.update_catcher_position,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "catcher_cfg": SceneEntityCfg("catcher"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
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
class RollingBallInterceptionEnvCfg(ManagerBasedRLEnvCfg):
    sim_freq: float = 100.0
    robot_control_freq: float = 25.0

    scene: RollingBallInterceptionSceneCfg = RollingBallInterceptionSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    episode_length_s = 3.0

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = max(1, round(self.sim_freq / self.robot_control_freq))
        self.episode_length_s = 3.0

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
