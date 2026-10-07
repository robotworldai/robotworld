# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-place from conveyor: base environment and scene configuration.

Task: pick object from moving conveyor and lift to target height (e.g. 60 cm).
- Conveyor moves objects; lifted objects are no longer driven by conveyor.
- MDP: observations, actions, rewards, terminations, events defined below.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class ConveyorBeltPickAndPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: robot, conveyor, box, object, optional distractor, plane, light."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    distractor: RigidObjectCfg | DeformableObjectCfg | None = None

    conveyor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Conveyor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, -0.45, -1.3], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A06.usd",
            scale=(0.6, 0.4, 0.8),
            collision_props=CollisionPropertiesCfg(),
        ),
    )

    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0, -0.4, 0.074],
            rot=[0.7071, 0.0, 0.0, 0.7071],
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
    """MDP commands (no dynamic target)."""

    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """MDP actions."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = (
        MISSING
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """MDP observations (policy group with optional corruption)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

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
    """MDP events: reset, conveyor friction, phase checks, optional debug."""

    fix_box_collision = EventTerm(
        func=mdp.fix_box_collision,
        mode="startup",
    )

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

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (0.0, 0.4),
                "z": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_distractor_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (0.0, 0.2),
                "z": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("distractor"),
        },
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    reset_conveyor_velocity = EventTerm(
        func=mdp.reset_conveyor_velocity,
        mode="reset",
        params={
            "velocity_range": (-0.3, -0.3),
        },
    )

    conveyor_friction = EventTerm(
        func=mdp.conveyor_belt_friction,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "conveyor_height_range": (0.10, 0.25),
            "conveyor_x_range": (0.3, 0.7),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    conveyor_friction_distractor = EventTerm(
        func=mdp.conveyor_belt_friction,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "conveyor_height_range": (0.135, 0.160),
            "conveyor_x_range": (0.3, 0.7),
            "asset_cfg": SceneEntityCfg("distractor"),
        },
    )

    check_phase = EventTerm(
        func=mdp.check_phase_transitions,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        params={
            "grasp_threshold": 0.03,
            "grasp_y_range": (-0.25, 0.25),
            "lift_height": 0.25,
            "position_threshold": 0.08,
        },
    )

    print_pos = EventTerm(
        func=mdp.print_object_position,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "distractor_cfg": SceneEntityCfg("distractor"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "grasp_threshold": 0.03,
        },
    )


@configclass
class RewardsCfg:
    """MDP rewards: phase-based reward shaping.

    Phase 0: reaching_object (tanh dense) + grasp_success (one-shot)
    Phase 1: stable_grasp + lifting_object (linear) + lift_milestone (one-shot)
    Phase 2: object_goal_tracking (linear) + fine (tanh) + arrive_box_vertical (one-shot)
    Phase 3: gripper_open_release + object_near_target + object_in_box (one-shot)
    Global: orientation_alignment, action_rate, joint_vel, joint_limits, joint_posture
    """

    # Phase 0: approach
    reaching_object = RewTerm(
        func=mdp.reaching_object_reward,
        params={"std": 0.1},
        weight=1000.0,
    )
    grasp_success = RewTerm(
        func=mdp.grasp_success_reward,
        params={"bonus": 1000.0, "grasp_threshold": 0.03},
        weight=1.0,
    )

    # Phase 1/2: stable grasp + lift
    stable_grasp = RewTerm(
        func=mdp.stable_grasp_reward,
        params={"grasp_threshold": 0.03},
        weight=500.0,
    )
    lifting_object = RewTerm(
        func=mdp.lifting_object_reward,
        params={"minimal_height": 0.15, "max_height": 0.3},
        weight=5000.0,
    )
    lift_milestone = RewTerm(
        func=mdp.lift_milestone_reward,
        params={"threshold": 0.35, "bonus": 1000.0},
        weight=1.0,
    )

    # Phase 2: goal tracking
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_tracking_reward,
        params={
            "max_distance": 1.2,
            "target_pos": (0.0, -0.4, 0.3),
            "grasp_threshold": 0.03,
        },
        weight=50000.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_tracking_fine_reward,
        params={
            "std": 0.05,
            "target_pos": (0.0, -0.4, 0.3),
            "grasp_threshold": 0.03,
        },
        weight=50000.0,
    )
    arrive_box_vertical = RewTerm(
        func=mdp.arrive_box_vertical_reward,
        params={
            "target_pos": (0.0, -0.4, 0.3),
            "position_threshold": 0.05,
            "vertical_threshold": 0.15,
            "bonus": 1.0,
            "grasp_threshold": 0.03,
        },
        weight=200000.0,
    )

    # Phase 3: release into box
    gripper_open_release = RewTerm(
        func=mdp.gripper_open_release_reward,
        params={
            "target_pos": (0.0, -0.4, 0.3),
            "release_radius": 0.10,
        },
        weight=200000.0,
    )
    object_near_target_phase3 = RewTerm(
        func=mdp.object_near_target_phase3_reward,
        params={
            "target_pos": (0.0, -0.4, 0.3),
            "proximity_threshold": 0.05,
            "vertical_threshold": 0.1,
        },
        weight=1000000.0,
    )
    object_in_box = RewTerm(
        func=mdp.object_in_box_reward,
        params={"bonus": 1.0},
        weight=200000.0,
    )

    # Global penalties
    orientation_alignment = RewTerm(
        func=mdp.orientation_alignment_penalty,
        params={"penalty_scale": -1.0},
        weight=100.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_limits = RewTerm(
        func=mdp.joint_limit_penalty,
        params={"margin": 0.1, "penalty_scale": -1.0},
        weight=100.0,
    )
    joint_posture = RewTerm(
        func=mdp.joint_posture_penalty,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-0.5,
    )


@configclass
class TerminationsCfg:
    """MDP terminations: timeout, object drop, task completed."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.027, "asset_cfg": SceneEntityCfg("object")},
    )

    task_completed = DoneTerm(
        func=mdp.task_completed,
    )


@configclass
class CurriculumCfg:
    """Curriculum: gradually increase action/velocity penalties.

    First 10000 steps: free exploration with low penalties.
    Then ramp action_rate and joint_vel weights to -1e-1.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class ConveyorBeltPickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment config for pick-place from conveyor.

    Simulation and robot control frequencies are decoupled so that
    changing the robot control rate does not affect other objects'
    motion (e.g. conveyor belt).

    - ``sim_freq``:  physics simulation frequency (Hz).
    - ``robot_control_freq``:  robot arm control frequency (Hz).
    - ``decimation`` is derived as ``round(sim_freq / robot_control_freq)``.
    """

    sim_freq: float = 100.0
    """Physics simulation frequency in Hz (all objects, conveyor, etc.)."""

    robot_control_freq: float = 25.0
    """Robot arm control frequency in Hz (how often new actions are applied)."""

    scene: ConveyorBeltPickAndPlaceSceneCfg = ConveyorBeltPickAndPlaceSceneCfg(
        num_envs=4096, env_spacing=2.5
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = max(1, round(self.sim_freq / self.robot_control_freq))
        self.episode_length_s = 15

        # Render every physics step so cameras run at sim_freq,
        # completely independent of robot_control_freq / decimation.
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
