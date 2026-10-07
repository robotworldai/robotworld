# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ball throwing: throw a pre-grasped ball into a target box.

Registered envs:
- BallThrowing-Franka-v0: joint position control
- BallThrowing-Franka-Play-v0: joint position (play / eval)
- BallThrowing-Franka-IK-Abs-v0: IK absolute pose
- BallThrowing-Franka-IK-Rel-v0: IK relative pose
- BallThrowing-Franka-DataCollection-v0: with fixed + wrist cameras
"""

import gymnasium as gym

from .ball_throwing_env_cfg import BallThrowingEnvCfg

__all__ = ["BallThrowingEnvCfg"]


_AGENTS_PKG = f"{__name__}.config.franka.agents"


gym.register(
    id="BallThrowing-Franka-v0",
    entry_point=f"{__name__}.ball_throwing_env:BallThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallThrowingEnvCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="BallThrowing-Franka-Play-v0",
    entry_point=f"{__name__}.ball_throwing_env:BallThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallThrowingEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="BallThrowing-Franka-IK-Abs-v0",
    entry_point=f"{__name__}.ball_throwing_env:BallThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallThrowingEnvCfg_IK_Abs",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="BallThrowing-Franka-IK-Rel-v0",
    entry_point=f"{__name__}.ball_throwing_env:BallThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallThrowingEnvCfg_IK_Rel",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="BallThrowing-Franka-DataCollection-v0",
    entry_point=f"{__name__}.ball_throwing_env:BallThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.franka:FrankaBallThrowingDataCollectionEnvCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)
