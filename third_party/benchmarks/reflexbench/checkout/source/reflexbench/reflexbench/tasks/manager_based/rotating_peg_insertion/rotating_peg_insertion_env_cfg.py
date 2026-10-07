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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class RotatingPegInsertionSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    disc: RigidObjectCfg = MISSING
    peg: RigidObjectCfg = MISSING
    hole_marker: RigidObjectCfg = MISSING

    # -- static visual elements --
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # -- disc mount pedestal (static visual, no collision) --
    disc_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DiscPedestal",
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.35,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.40, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.175]),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        peg_tip_position = ObsTerm(func=mdp.peg_tip_position_in_robot_root_frame)
        disc_angle = ObsTerm(func=mdp.disc_angle_obs)
        disc_angular_velocity = ObsTerm(func=mdp.disc_angular_velocity_obs)
        hole_position = ObsTerm(func=mdp.hole_position_in_robot_root_frame)
        predicted_alignment_time = ObsTerm(func=mdp.predicted_next_alignment_time_obs)
        gripper_width = ObsTerm(func=mdp.gripper_width_obs)
        task_phase = ObsTerm(func=mdp.task_phase_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )

    reset_disc_state = EventTerm(
        func=mdp.reset_disc_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("disc")},
    )

    reset_peg_to_gripper = EventTerm(
        func=mdp.reset_peg_to_gripper,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("peg")},
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    update_disc_rotation = EventTerm(
        func=mdp.update_disc_rotation,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("disc")},
    )

    update_peg_position = EventTerm(
        func=mdp.update_peg_position,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )

    update_hole_marker = EventTerm(
        func=mdp.update_hole_marker_position,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"marker_cfg": SceneEntityCfg("hole_marker")},
    )

    check_phase = EventTerm(
        func=mdp.check_phase_transitions,
        mode="interval",
        interval_range_s=(0.04, 0.04),
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    check_insertion = EventTerm(
        func=mdp.check_insertion,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    pass


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_completed = DoneTerm(func=mdp.task_completed)
    peg_dropped = DoneTerm(func=mdp.peg_dropped)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    pass


# ---------------------------------------------------------------------------
# Environment Config
# ---------------------------------------------------------------------------
@configclass
class RotatingPegInsertionEnvCfg(ManagerBasedRLEnvCfg):
    sim_freq: float = 100.0
    robot_control_freq: float = 25.0

    scene: RotatingPegInsertionSceneCfg = RotatingPegInsertionSceneCfg(
        num_envs=4096, env_spacing=2.5
    )

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
