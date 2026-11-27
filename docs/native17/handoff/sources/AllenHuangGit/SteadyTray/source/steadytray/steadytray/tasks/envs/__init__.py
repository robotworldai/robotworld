import gymnasium as gym

gym.register(
    id="G1-Steady-Tray-Pre-Locomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"steadytray.tasks.agents.rsl_rl_ppo_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="G1-Steady-Tray-Locomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"steadytray.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="G1-Steady-Tray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.steady_tray_env_cfg:SteadyTrayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.steady_tray_env_cfg:SteadyTrayPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"steadytray.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="G1-Steady-Object",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.steady_object_env_cfg:SteadyObjectEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.steady_object_env_cfg:SteadyObjectPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"steadytray.tasks.agents.rsl_rl_ppo_cfg:G1AdapterSteadyTrayRunnerCfg",
    },
)

gym.register(
    id="G1-Steady-Object-Distillation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.steady_object_distill_env_cfg:SteadyObjectDistillEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.steady_object_distill_env_cfg:SteadyObjectDistillPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"steadytray.tasks.agents.rsl_rl_ppo_cfg:G1AdapterDistillationRunnerCfg",
    },
)