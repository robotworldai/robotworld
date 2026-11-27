"""isaaclab_go2_pushrecovery — Go2 push-recovery locomotion task.

Importing this module registers two Gym environments:

    Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0       (training)
    Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-Play-v0  (evaluation)
"""

import gymnasium as gym

from .env_cfg import (
    UnitreeGo2PushRecoveryEnvCfg,
    UnitreeGo2PushRecoveryEnvCfg_PLAY,
    UnitreeGo2PushRecoveryPPORunnerCfg,  # noqa: F401 — re-exported for gym entry_point
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2PushRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2PushRecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2PushRecoveryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:UnitreeGo2PushRecoveryPPORunnerCfg",
    },
)
