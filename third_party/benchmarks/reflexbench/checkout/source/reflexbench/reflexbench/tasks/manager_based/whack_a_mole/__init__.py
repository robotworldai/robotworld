# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .whack_a_mole_env_cfg import WhackAMoleEnvCfg

__all__ = ["WhackAMoleEnvCfg"]


gym.register(
    id="WhackAMole-Franka-v0",
    entry_point=f"{__name__}.whack_a_mole_env:WhackAMoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaWhackAMoleEnvCfg",
    },
)

gym.register(
    id="WhackAMole-Franka-Play-v0",
    entry_point=f"{__name__}.whack_a_mole_env:WhackAMoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaWhackAMoleEnvCfg_PLAY",
    },
)

gym.register(
    id="WhackAMole-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.whack_a_mole_env:WhackAMoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaWhackAMoleEnvCfg_IK_Abs",
    },
)

gym.register(
    id="WhackAMole-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.whack_a_mole_env:WhackAMoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaWhackAMoleEnvCfg_IK_Rel",
    },
)

gym.register(
    id="WhackAMole-Franka-DataCollection-v0",
    entry_point=f"{__name__}.whack_a_mole_env:WhackAMoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaWhackAMoleDataCollectionEnvCfg",
    },
)
