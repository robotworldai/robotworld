#!/usr/bin/env python3
"""Train an RL baseline for Aerial-Balance-Bench with skrl."""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from omni.isaac.lab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


DEFAULT_RUN_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "rl_target_position_rpo_train.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "zero_action.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "rl_rpo.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(description="Train an RL baseline for Aerial-Balance-Bench.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_RUN_CONFIG), help="Path to an RL train run YAML.")
    parser.add_argument("--env_config", type=str, default=None, help="Override environment YAML config path.")
    parser.add_argument("--policy_config", type=str, default=None, help="Override RL policy YAML config path.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed. Use -1 for a random seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override training iterations.")
    parser.add_argument("--run_name", type=str, default=None, help="Override log run name.")
    parser.add_argument("--video", action="store_true", default=False, help="Record training video.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.video:
        args_cli.enable_cameras = True
    return args_cli


def _cli_option_present(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def _read_yaml_safely(path: str | Path) -> dict[str, Any]:
    try:
        with open(Path(path).expanduser().resolve(), encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    except FileNotFoundError:
        return {}


def _resolve_config_path(path: str | Path | None, base_dir: Path, default_path: Path) -> Path:
    if path is None:
        return default_path.resolve()
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    for root in (PROJECT_ROOT, base_dir):
        candidate = (root / raw_path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / raw_path).resolve()


args_cli = _parse_args()
_run_config_path = Path(args_cli.config).expanduser().resolve()
_launch_run_config = _read_yaml_safely(_run_config_path)
_launch_env_config_path = _resolve_config_path(
    args_cli.env_config or _launch_run_config.get("env_config"),
    _run_config_path.parent,
    DEFAULT_ENV_CONFIG,
)
_launch_env_config = _read_yaml_safely(_launch_env_config_path)
_yaml_device = _launch_env_config.get("env", {}).get("device")
if _yaml_device is not None and not _cli_option_present("--device"):
    args_cli.device = _yaml_device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from aerial_balance_bench.baselines.rl_env_wrapper import NormalizedRLTrainingWrapper
from aerial_balance_bench.baselines.rl_models import make_agent_class_and_cfg, make_models, require_skrl
from aerial_balance_bench.baselines.rl_observation_adapter import RLObservationAdapterCfg
from aerial_balance_bench.baselines.rl_policy import RLPolicyCfg
from aerial_balance_bench.environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
from aerial_balance_bench.utils.io import ensure_dir, save_yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _maybe_set_attrs(target: Any, values: dict[str, Any]):
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def _resolve_seed(run_config: dict[str, Any], env_config: dict[str, Any]) -> int:
    if args_cli.seed is not None:
        seed = args_cli.seed
    else:
        seed = int(run_config.get("seed", env_config.get("env", {}).get("seed", 666)))
    if seed == -1:
        seed = random.randint(0, 10000)
    return seed


def _build_env_cfg(config: dict[str, Any], seed: int) -> AerialBalanceEnvCfg:
    env_cfg = AerialBalanceEnvCfg()
    env_values = config.get("env", {})
    env_cfg.seed = seed
    env_cfg.task_name = config.get("task_name", env_values.get("task_name", env_cfg.task_name))
    env_cfg.interface_name = config.get("interface_name", env_values.get("interface_name", env_cfg.interface_name))
    env_cfg.episode_length_s = float(env_values.get("episode_length_s", env_cfg.episode_length_s))
    env_cfg.scene.num_envs = int(
        args_cli.num_envs if args_cli.num_envs is not None else env_values.get("num_envs", env_cfg.scene.num_envs)
    )

    if _cli_option_present("--device"):
        env_cfg.sim.device = args_cli.device
    else:
        env_cfg.sim.device = env_values.get("device", env_cfg.sim.device)

    sim_values = config.get("sim", {})
    if "dt" in sim_values:
        env_cfg.sim.dt = float(sim_values["dt"])
    if "decimation" in sim_values:
        env_cfg.decimation = int(sim_values["decimation"])
        env_cfg.sim.render_interval = env_cfg.decimation
    if "rope_length" in env_values:
        env_cfg.rope_length = float(env_values["rope_length"])

    _maybe_set_attrs(env_cfg.target_position_task, config.get("target_position_task", config.get("task", {})))
    _maybe_set_attrs(env_cfg.trajectory_tracking_task, config.get("trajectory_tracking_task", {}))
    _maybe_set_attrs(env_cfg.velocity_interface, config.get("velocity_interface", {}))
    _maybe_set_attrs(env_cfg.position_interface, config.get("position_interface", {}))
    _maybe_set_attrs(env_cfg.thrust_interface, config.get("thrust_interface", {}))
    _maybe_set_attrs(env_cfg.robustness, config.get("robustness", {}))
    _maybe_set_attrs(env_cfg.target_position_evaluator, config.get("target_position_evaluator", config.get("evaluator", {})))
    _maybe_set_attrs(env_cfg.trajectory_tracking_evaluator, config.get("trajectory_tracking_evaluator", {}))
    env_cfg.target_position_evaluator.episode_length_s = env_cfg.episode_length_s
    env_cfg.trajectory_tracking_evaluator.episode_length_s = env_cfg.episode_length_s
    return env_cfg


def _validate_env_cfg(env_cfg: AerialBalanceEnvCfg):
    if env_cfg.task_name != "target_position":
        raise ValueError("RL training currently supports only task_name='target_position'.")
    if env_cfg.interface_name != "velocity":
        raise ValueError("RL training currently supports only interface_name='velocity'.")


def _deep_update(target: dict, values: dict):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _make_output_dir(run_config: dict[str, Any], seed: int) -> Path:
    logging_cfg = run_config.get("logging", {})
    root_dir = ensure_dir(PROJECT_ROOT / logging_cfg.get("root_dir", "logs/rl"))
    if args_cli.run_name is not None:
        run_name = args_cli.run_name
    else:
        run_name = logging_cfg.get("run_name", f"rl_train_seed_{seed}")
    return ensure_dir(root_dir / run_name)


def main():
    require_skrl()
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    run_config_path = Path(args_cli.config).expanduser().resolve()
    run_config = _load_yaml(run_config_path)
    env_config_path = _resolve_config_path(
        args_cli.env_config or run_config.get("env_config"),
        run_config_path.parent,
        DEFAULT_ENV_CONFIG,
    )
    policy_config_path = _resolve_config_path(
        args_cli.policy_config or run_config.get("policy_config"),
        run_config_path.parent,
        DEFAULT_POLICY_CONFIG,
    )
    env_config = _load_yaml(env_config_path)
    policy_config = _load_yaml(policy_config_path)
    effective_policy_config = dict(policy_config)
    _deep_update(effective_policy_config, run_config.get("policy_overrides", {}))
    policy_cfg = RLPolicyCfg.from_dict(effective_policy_config)

    seed = _resolve_seed(run_config, env_config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed, deterministic=True)

    env_cfg = _build_env_cfg(env_config, seed)
    _validate_env_cfg(env_cfg)
    train_cfg = run_config.get("training", {})
    max_iterations = int(args_cli.max_iterations if args_cli.max_iterations is not None else train_cfg.get("max_iterations", 200))

    output_dir = _make_output_dir(run_config, seed)
    save_yaml(run_config, output_dir / "input_run_config.yaml")
    save_yaml(env_config, output_dir / "input_env_config.yaml")
    save_yaml(policy_config, output_dir / "input_policy_config.yaml")

    env = None
    try:
        env = AerialBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_interval = int(max_iterations * env.max_episode_length // 5)
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step % video_interval == 0,
                "video_length": env.max_episode_length - 1,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        base_env = env.unwrapped
        physical_action_limit = float(base_env.action_space.high[0])
        adapter_cfg = RLObservationAdapterCfg(observation_mode=policy_cfg.observation_mode)
        train_env = NormalizedRLTrainingWrapper(env, adapter_cfg, physical_action_limit)
        skrl_env = wrap_env(env=train_env, wrapper="isaaclab", verbose=True)

        rollouts_cfg = train_cfg.get("rollouts", "auto")
        if rollouts_cfg in (None, "auto"):
            memory_size = base_env.max_episode_length
        else:
            memory_size = int(rollouts_cfg)
        memory = RandomMemory(memory_size=memory_size, num_envs=skrl_env.num_envs, device=skrl_env.device)

        agent_overrides = dict(policy_cfg.agent)
        _deep_update(agent_overrides, run_config.get("agent", {}))
        batch_size = int(train_cfg.get("batch_size", 512))
        mini_batches = train_cfg.get("mini_batches", "auto")
        if mini_batches == "auto":
            mini_batches = max(1, memory_size * skrl_env.num_envs // batch_size)
        checkpoint_interval = train_cfg.get("checkpoint_interval")
        if checkpoint_interval is None:
            checkpoint_interval = max(1, max_iterations * base_env.max_episode_length // 5)
        wandb_kwargs = dict(train_cfg.get("wandb_kwargs", {}))
        wandb_kwargs["project"] = "aerial_balance_bench"
        default_overrides = {
            "rollouts": memory_size,
            "learning_epochs": int(train_cfg.get("learning_epochs", 4)),
            "mini_batches": int(mini_batches),
            "random_timesteps": int(train_cfg.get("random_timesteps", 0)),
            "learning_starts": int(train_cfg.get("learning_starts", 0)),
            "experiment": {
                "directory": str(output_dir.parent),
                "experiment_name": output_dir.name,
                "write_interval": int(train_cfg.get("write_interval", 10)),
                "checkpoint_interval": int(checkpoint_interval),
                "wandb": bool(train_cfg.get("wandb", False)),
                "wandb_kwargs": wandb_kwargs,
            },
        }
        if policy_cfg.algorithm == "rpo":
            default_overrides["alpha"] = float(train_cfg.get("rpo_alpha", 0.5))
        _deep_update(default_overrides, agent_overrides)
        default_overrides.setdefault("experiment", {}).setdefault("wandb_kwargs", {})
        default_overrides["experiment"]["wandb_kwargs"]["project"] = "aerial_balance_bench"

        agent_cls, agent_cfg = make_agent_class_and_cfg(
            policy_cfg.algorithm,
            default_overrides,
            skrl_env.observation_space,
            skrl_env.device,
        )
        models = make_models(
            policy_cfg.algorithm,
            policy_cfg.network,
            skrl_env.observation_space,
            skrl_env.action_space,
            skrl_env.device,
        )
        agent = agent_cls(
            models=models,
            memory=memory,
            cfg=agent_cfg,
            observation_space=skrl_env.observation_space,
            action_space=skrl_env.action_space,
            device=skrl_env.device,
        )
        trainer_cfg = {
            "timesteps": int(train_cfg.get("timesteps", max_iterations * base_env.max_episode_length)),
            "headless": bool(args_cli.headless),
        }
        trainer = SequentialTrainer(cfg=trainer_cfg, env=skrl_env, agents=agent)
        save_yaml(
            {
                "seed": seed,
                "num_envs": base_env.num_envs,
                "episode_length_s": env_cfg.episode_length_s,
                "task_name": env_cfg.task_name,
                "interface_name": env_cfg.interface_name,
                "algorithm": policy_cfg.algorithm,
                "observation_mode": policy_cfg.observation_mode,
                "policy_input_dim": train_env.adapter.input_dim,
                "physical_action_limit": physical_action_limit,
                "max_iterations": max_iterations,
                "trainer_timesteps": trainer_cfg["timesteps"],
                "wandb_project": default_overrides["experiment"]["wandb_kwargs"]["project"],
                "run_config_path": str(run_config_path),
                "env_config_path": str(env_config_path),
                "policy_config_path": str(policy_config_path),
            },
            output_dir / "resolved_run.yaml",
        )

        start_time = time.perf_counter()
        trainer.train()
        elapsed_s = time.perf_counter() - start_time
        save_yaml({"training_elapsed_s": elapsed_s}, output_dir / "training_time.yaml")
        print(f"[INFO] RL training finished. Logs saved to: {output_dir}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
