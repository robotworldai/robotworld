# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .ball_catching_env_cfg import BallCatchingEnvCfg

__all__ = ["BallCatchingEnvCfg"]


gym.register(
    id="BallCatching-Franka-v0",
    entry_point=f"{__name__}.ball_catching_env:BallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallCatchingEnvCfg",
    },
)

gym.register(
    id="BallCatching-Franka-Play-v0",
    entry_point=f"{__name__}.ball_catching_env:BallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallCatchingEnvCfg_PLAY",
    },
)

gym.register(
    id="BallCatching-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.ball_catching_env:BallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallCatchingEnvCfg_IK_Abs",
    },
)

gym.register(
    id="BallCatching-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.ball_catching_env:BallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallCatchingEnvCfg_IK_Rel",
    },
)

gym.register(
    id="BallCatching-Franka-DataCollection-v0",
    entry_point=f"{__name__}.ball_catching_env:BallCatchingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallCatchingDataCollectionEnvCfg",
    },
)
