# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain variant of the balance task.

Swaps the flat ground plane for a generated rough terrain (random roughness +
gentle slopes, see :mod:`wheeled_quadruped.terrains`) so the policy learns to
balance on uneven ground. Everything else — observations, actions, rewards,
terminations, domain randomization — is inherited from the flat balance task.
"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from wheeled_quadruped.terrains import WHEELED_ROUGH_TERRAINS_CFG

from .balance_env_cfg import WheeledQuadrupedBalanceEnvCfg, WheeledQuadrupedSceneCfg


@configclass
class WheeledQuadrupedRoughSceneCfg(WheeledQuadrupedSceneCfg):
    """Scene with the flat ground plane replaced by a generated rough terrain."""

    # remove the flat ground plane inherited from the base scene
    ground = None
    # generated rough terrain at the same prim path
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WHEELED_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


@configclass
class WheeledQuadrupedBalanceRoughEnvCfg(WheeledQuadrupedBalanceEnvCfg):
    """Balance on rough terrain. Robots are spread across all terrain difficulties."""

    scene: WheeledQuadrupedRoughSceneCfg = WheeledQuadrupedRoughSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class WheeledQuadrupedBalanceRoughEnvCfg_PLAY(WheeledQuadrupedBalanceRoughEnvCfg):
    """Play/eval configuration: fewer envs, smaller terrain, no noise, no pushes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # make a smaller scene for playing
        self.scene.num_envs = 32
        # a small, fixed terrain grid is enough to look at
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
