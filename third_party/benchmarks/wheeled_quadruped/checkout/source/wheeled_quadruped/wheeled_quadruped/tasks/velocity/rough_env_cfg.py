# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain variant of the velocity (drive-while-balancing) task.

Combines the velocity-tracking task with the generated rough terrain and adds a
terrain-level curriculum: robots that track their commanded velocity well enough
to travel far are promoted to rougher terrain, while robots that fall behind are
demoted. This is the standard Isaac Lab locomotion recipe, applied to a wheeled
balancer.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

# terrain_levels_vel is a locomotion-task curriculum, not part of core isaaclab.envs.mdp.
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel

from wheeled_quadruped.tasks.balance.rough_env_cfg import WheeledQuadrupedRoughSceneCfg

from .velocity_env_cfg import WheeledQuadrupedVelocityEnvCfg


@configclass
class CurriculumCfg:
    """Terrain-level curriculum driven by how far each robot tracks its command."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


@configclass
class WheeledQuadrupedVelocityRoughEnvCfg(WheeledQuadrupedVelocityEnvCfg):
    """Drive while balancing on rough terrain, with a terrain-difficulty curriculum."""

    scene: WheeledQuadrupedRoughSceneCfg = WheeledQuadrupedRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # enable progressive-difficulty terrain generation to match the curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        # The stronger yaw-tracking reward (set in velocity_env_cfg) is a clean win on
        # flat ground, but on rough terrain it trades off forward-speed tracking. Keep
        # the balanced weights here so uneven-ground driving stays well-rounded.
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.track_ang_vel_z.params["std"] = math.sqrt(0.25)


@configclass
class WheeledQuadrupedVelocityRoughEnvCfg_PLAY(WheeledQuadrupedVelocityRoughEnvCfg):
    """Play/eval configuration: fewer envs, smaller terrain, fixed forward command."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # make a smaller scene for playing
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # fixed forward command so every robot drives across the terrain for the demo
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
