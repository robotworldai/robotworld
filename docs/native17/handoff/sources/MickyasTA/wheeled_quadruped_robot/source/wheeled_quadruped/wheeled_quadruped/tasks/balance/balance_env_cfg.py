# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance task for the wheeled quadruped.

The robot balances on its two rear wheels (segway-style) at a base height of
0.828 m. It uses an asymmetric actor-critic: the policy observes only quantities
obtainable from onboard sensors, while the critic additionally observes
privileged state (base linear velocity and base height).
"""

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wheeled_quadruped.assets import WHEELED_QUADRUPED_CFG

# Only four joints are actuated. The front wheels are fixed.
THIGH_JOINTS = ["robot1_front_left_thigh_joint", "robot1_front_right_thigh_joint"]
WHEEL_JOINTS = ["robot1_rl_wheel_joint", "robot1_rr_wheel_joint"]


##
# Scene definition
##


@configclass
class WheeledQuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for the wheeled quadruped scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # wheeled quadruped robot
    robot: ArticulationCfg = WHEELED_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    thigh_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=THIGH_JOINTS, scale=0.5, use_default_offset=True
    )
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINTS, scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (asymmetric actor-critic)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Onboard-obtainable observations for the policy (actor)."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        thigh_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=THIGH_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the critic (value function only)."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_pos_z)
        thigh_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=THIGH_JOINTS)},
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # The USD export merged fixed joints: the trunk rigid body is
            # robot1_base_footprint (~17.94 kg), not robot1_base_link.
            "asset_cfg": SceneEntityCfg("robot", body_names="robot1_base_footprint"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # keep the robot alive and penalize failure
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # primary task: hold the balancing base height (squared-error penalty -> negative weight)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.828, "asset_cfg": SceneEntityCfg("robot")},
    )
    # keep the base upright and still
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # effort / smoothness penalties
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # discourage the robot from driving away while balancing
    wheel_spin = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 3})
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})


##
# Environment configuration
##


@configclass
class WheeledQuadrupedBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the wheeled quadruped balance environment."""

    # Scene settings
    scene: WheeledQuadrupedSceneCfg = WheeledQuadrupedSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)


@configclass
class WheeledQuadrupedBalanceEnvCfg_PLAY(WheeledQuadrupedBalanceEnvCfg):
    """Play/eval configuration: fewer envs, no observation noise, no pushes."""

    def __post_init__(self) -> None:
        # start from the parent configuration
        super().__post_init__()
        # make a smaller scene for playing
        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove the interval push event
        self.events.push_robot = None
