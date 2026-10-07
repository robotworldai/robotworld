# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rotating_peg_insertion_env_cfg import RotatingPegInsertionEnvCfg

__all__ = ["RotatingPegInsertionEnvCfg"]


gym.register(
    id="RotatingPegInsertion-Franka-v0",
    entry_point=f"{__name__}.rotating_peg_insertion_env:RotatingPegInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRotatingPegInsertionEnvCfg",
    },
)

gym.register(
    id="RotatingPegInsertion-Franka-Play-v0",
    entry_point=f"{__name__}.rotating_peg_insertion_env:RotatingPegInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRotatingPegInsertionEnvCfg_PLAY",
    },
)

gym.register(
    id="RotatingPegInsertion-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.rotating_peg_insertion_env:RotatingPegInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRotatingPegInsertionEnvCfg_IK_Abs",
    },
)

gym.register(
    id="RotatingPegInsertion-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.rotating_peg_insertion_env:RotatingPegInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRotatingPegInsertionEnvCfg_IK_Rel",
    },
)

gym.register(
    id="RotatingPegInsertion-Franka-DataCollection-v0",
    entry_point=f"{__name__}.rotating_peg_insertion_env:RotatingPegInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRotatingPegInsertionDataCollectionEnvCfg",
    },
)
