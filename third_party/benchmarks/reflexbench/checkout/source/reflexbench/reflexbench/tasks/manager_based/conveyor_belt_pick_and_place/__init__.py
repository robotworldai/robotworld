# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-place from conveyor: pick object from moving conveyor and lift to target height.

Registered envs:
- ConveyorBeltPickAndPlace-Franka-v0: joint position control
- ConveyorBeltPickAndPlace-Franka-Play-v0: joint position (play)
- ConveyorBeltPickAndPlace-Franka-IK-Abs-v0: IK absolute pose
- ConveyorBeltPickAndPlace-Franka-IK-Rel-v0: IK relative pose
- ConveyorBeltPickAndPlace-Franka-DataCollection-v0: with cameras
"""

import gymnasium as gym

from .conveyor_belt_pick_and_place_env_cfg import ConveyorBeltPickAndPlaceEnvCfg

_AGENTS_PKG = f"{__name__}.config.franka.agents"

gym.register(
    id="ConveyorBeltPickAndPlace-Franka-v0",
    entry_point=f"{__name__}.conveyor_belt_pick_and_place_env:ConveyorBeltPickAndPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaConveyorBeltPickAndPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_cfg:ConveyorBeltPickAndPlacePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_PKG}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="ConveyorBeltPickAndPlace-Franka-Play-v0",
    entry_point=f"{__name__}.conveyor_belt_pick_and_place_env:ConveyorBeltPickAndPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaConveyorBeltPickAndPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_cfg:ConveyorBeltPickAndPlacePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_PKG}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="ConveyorBeltPickAndPlace-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.conveyor_belt_pick_and_place_env:ConveyorBeltPickAndPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaConveyorBeltPickAndPlaceEnvCfg_IK_Abs",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="ConveyorBeltPickAndPlace-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.conveyor_belt_pick_and_place_env:ConveyorBeltPickAndPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaConveyorBeltPickAndPlaceEnvCfg_IK_Rel",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="ConveyorBeltPickAndPlace-Franka-DataCollection-v0",
    entry_point=f"{__name__}.conveyor_belt_pick_and_place_env:ConveyorBeltPickAndPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaConveyorBeltPickAndPlaceDataCollectionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_cfg:ConveyorBeltPickAndPlacePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)
