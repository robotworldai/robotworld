# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking task for the wheeled quadruped.

Built on top of the balance task: the robot keeps balancing on its two rear
wheels (segway-style) while driving to track a commanded base velocity. Because
it stands on two rear wheels it behaves like a differential drive -- it can move
forward/backward and yaw, but it cannot translate laterally, so the commanded
lateral velocity (lin_vel_y) is pinned to zero.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from wheeled_quadruped.tasks.balance.balance_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    WheeledQuadrupedBalanceEnvCfg,
)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the velocity task."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        # lin_vel_y is zero: driving on two rear wheels cannot translate laterally.
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class VelocityObservationsCfg(ObservationsCfg):
    """Balance observations plus the commanded velocity (asymmetric actor-critic)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Onboard-obtainable observations plus the commanded velocity."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Privileged observations plus the commanded velocity."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class VelocityRewardsCfg(RewardsCfg):
    """Balance rewards plus velocity-tracking terms."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        # Yaw tracking previously plateaued below linear tracking. Raise its weight to
        # match linear (0.5 -> 1.0) and tighten the exp kernel (std sqrt(0.25) -> sqrt(0.15))
        # so the policy is rewarded for more precise turning.
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.15),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Environment configuration
##


@configclass
class WheeledQuadrupedVelocityEnvCfg(WheeledQuadrupedBalanceEnvCfg):
    """Velocity-tracking environment: drive while balancing on the rear wheels."""

    # new command group; balance env has none
    commands: CommandsCfg = CommandsCfg()
    # observations/rewards extend the balance groups with the command terms
    observations: VelocityObservationsCfg = VelocityObservationsCfg()
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Relax the balance-hold shaping so the policy is free to drive: the height
        # and orientation penalties are softened and the alive bonus is reduced so
        # the velocity-tracking terms dominate.
        self.rewards.alive.weight = 0.25
        self.rewards.base_height.weight = -10.0
        self.rewards.flat_orientation.weight = -2.0
        # The wheels are supposed to spin while driving -> drop the spin penalty.
        self.rewards.wheel_spin = None
        # Command extremes need ~12 rad/s at the wheels (1.0 m/s / 0.1008 m radius
        # plus the yaw differential over the 0.44 m track); the balance task's
        # scale of 5.0 would push half the command range outside the unit-action
        # envelope.
        self.actions.wheel_vel.scale = 12.0


@configclass
class WheeledQuadrupedVelocityEnvCfg_PLAY(WheeledQuadrupedVelocityEnvCfg):
    """Play/eval configuration: fewer envs, no noise, no pushes, fixed forward command."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # make a smaller scene for playing
        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove the interval push event
        self.events.push_robot = None
        # fixed forward command for a clean demo (every env drives forward)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
