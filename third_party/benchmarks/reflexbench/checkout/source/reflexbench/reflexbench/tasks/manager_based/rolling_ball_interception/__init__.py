# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .rolling_ball_interception_env_cfg import RollingBallInterceptionEnvCfg

__all__ = ["RollingBallInterceptionEnvCfg"]


gym.register(
    id="RollingBallInterception-Franka-v0",
    entry_point=f"{__name__}.rolling_ball_interception_env:RollingBallInterceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRollingBallInterceptionEnvCfg",
    },
)

gym.register(
    id="RollingBallInterception-Franka-Play-v0",
    entry_point=f"{__name__}.rolling_ball_interception_env:RollingBallInterceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRollingBallInterceptionEnvCfg_PLAY",
    },
)

gym.register(
    id="RollingBallInterception-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.rolling_ball_interception_env:RollingBallInterceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRollingBallInterceptionEnvCfg_IK_Abs",
    },
)

gym.register(
    id="RollingBallInterception-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.rolling_ball_interception_env:RollingBallInterceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRollingBallInterceptionEnvCfg_IK_Rel",
    },
)

gym.register(
    id="RollingBallInterception-Franka-DataCollection-v0",
    entry_point=f"{__name__}.rolling_ball_interception_env:RollingBallInterceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaRollingBallInterceptionDataCollectionEnvCfg",
    },
)
