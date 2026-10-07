#!/usr/bin/env python3
"""Evaluate the NMPC baseline against AerialBalanceEnv and log rollout data."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omni.isaac.lab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


DEFAULT_RUN_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "nmpc_target_position_run.yaml"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "environments" / "configs" / "zero_action.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "baselines" / "configs" / "nmpc.yaml"


def _parse_args():
    parser = argparse.ArgumentParser(description="NMPC baseline runner for Aerial-Balance-Bench.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_RUN_CONFIG), help="Path to an NMPC run YAML.")
    parser.add_argument("--env_config", type=str, default=None, help="Override environment YAML config path.")
    parser.add_argument("--policy_config", type=str, default=None, help="Override NMPC policy YAML config path.")
    parser.add_argument("--episodes", type=int, default=None, help="Override target completed episodes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
    parser.add_argument("--n_horizon", type=int, default=None, help="Override NMPC solver horizon.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed. Use -1 for a random seed.")
    parser.add_argument("--run_name", type=str, default=None, help="Override log run name.")
    parser.add_argument("--video", action="store_true", default=False, help="Record rollout video.")
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

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from aerial_balance_bench.baselines import NMPCPolicy, NMPCPolicyCfg
from aerial_balance_bench.environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
from aerial_balance_bench.utils.io import append_csv_row, ensure_dir, save_yaml


OBSERVATION_FIELDS = [
    "pb",
    "vb",
    "ab",
    "theta",
    "omega",
    "alpha",
    "drz",
    "vrz",
    "arz",
    "pg",
    "a_prev",
]

STEP_EXTRA_FIELDS = (
    "pb",
    "pg",
    "vg",
    "command_z",
    "executed_command_z",
    "vrz_cmd",
    "executed_vrz_cmd",
    "ball_mass",
    "low_level_controller_gain",
    "velocity_gain",
    "action_delay_enabled",
    "delay_step",
    "delayed_command_z",
    "external_disturbance_enabled",
    "external_disturbance_vel_z",
    "last_action",
)


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
        raise ValueError("NMPCPolicy runner currently supports only task_name='target_position'.")
    if env_cfg.interface_name != "velocity":
        raise ValueError("NMPCPolicy runner currently supports only interface_name='velocity'.")


def _resolve_policy_cfg_from_env(policy_cfg: NMPCPolicyCfg, env_cfg: AerialBalanceEnvCfg, step_dt: float):
    if _is_auto(policy_cfg.t_step):
        policy_cfg.t_step = float(step_dt)
    if _is_auto(policy_cfg.max_acc):
        policy_cfg.max_acc = float(env_cfg.velocity_interface.max_acc)
    if _is_auto(policy_cfg.max_velocity):
        policy_cfg.max_velocity = float(env_cfg.velocity_interface.max_velocity)
    if _is_auto(policy_cfg.gravity):
        policy_cfg.gravity = float(abs(env_cfg.sim.gravity[-1]))
    if _is_auto(policy_cfg.plank_length):
        policy_cfg.plank_length = float(env_cfg.plank_length)
    if _is_auto(policy_cfg.rope_length):
        policy_cfg.rope_length = float(env_cfg.rope_length)
    if _is_auto(policy_cfg.ball_radius):
        policy_cfg.ball_radius = float(env_cfg.ball_cfg.spawn.radius)
    if _is_auto(policy_cfg.ball_mass):
        policy_cfg.ball_mass = float(env_cfg.ball_cfg.spawn.mass_props.mass)

    predictor_cfg = policy_cfg.state_predictor
    if _is_auto(predictor_cfg.delay_step):
        if env_cfg.robustness.enabled and env_cfg.robustness.action_delay_enabled:
            predictor_cfg.delay_step = int(env_cfg.robustness.delay_step)
        else:
            predictor_cfg.delay_step = 0
    if _is_auto(predictor_cfg.step_dt) or float(predictor_cfg.step_dt) <= 0.0:
        predictor_cfg.step_dt = float(step_dt)
    if _is_auto(predictor_cfg.max_acc):
        predictor_cfg.max_acc = float(env_cfg.velocity_interface.max_acc)
    if _is_auto(predictor_cfg.max_velocity):
        predictor_cfg.max_velocity = float(env_cfg.velocity_interface.max_velocity)
    if _is_auto(predictor_cfg.plank_length):
        predictor_cfg.plank_length = float(env_cfg.plank_length)
    if _is_auto(predictor_cfg.rope_length):
        predictor_cfg.rope_length = float(env_cfg.rope_length)
    if _is_auto(predictor_cfg.gravity):
        predictor_cfg.gravity = float(abs(env_cfg.sim.gravity[-1]))
    if _is_auto(predictor_cfg.ball_radius):
        predictor_cfg.ball_radius = float(env_cfg.ball_cfg.spawn.radius)
    if _is_auto(predictor_cfg.ball_mass):
        predictor_cfg.ball_mass = float(env_cfg.ball_cfg.spawn.mass_props.mass)


def _is_auto(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _deep_update(target: dict, values: dict):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _make_output_dir(run_config: dict[str, Any], seed: int, target_episodes: int, root_dir: Path) -> Path:
    logging_cfg = run_config.get("logging", {})
    root_dir = ensure_dir(root_dir)
    if args_cli.run_name is not None:
        run_name = args_cli.run_name
    else:
        run_name = logging_cfg.get("run_name")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"nmpc_ep_{target_episodes}_seed_{seed}_{timestamp}"
    return ensure_dir(root_dir / run_name)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _nmpc_logging_root(n_horizon: int) -> Path:
    return PROJECT_ROOT / f"logs/nmpc/nmpc_{int(n_horizon)}"


def _scalar_metric(value: Any) -> float | bool:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return bool(value.detach().cpu().item())
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_tensor_fields(source: dict[str, Any], field_names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    collected = {}
    items = source.items() if field_names is None else ((key, source.get(key)) for key in field_names)
    for key, value in items:
        if isinstance(value, torch.Tensor):
            collected[key] = _tensor_to_numpy(value)
    return collected


def _collect_step_extras(infos: dict[str, Any]) -> dict[str, np.ndarray]:
    return _collect_tensor_fields(infos.get("step", {}), STEP_EXTRA_FIELDS)


def _collect_benchmark_metrics(infos: dict[str, Any]) -> dict[str, float | bool]:
    benchmark = infos.get("benchmark", {})
    return {key: _scalar_metric(value) for key, value in benchmark.items()}


def _stack_or_empty(records: list[np.ndarray], shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if records:
        return np.stack(records, axis=0)
    return np.empty(shape, dtype=dtype)


def main():
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
    policy_cfg = NMPCPolicyCfg.from_dict(effective_policy_config)
    if args_cli.n_horizon is not None:
        if args_cli.n_horizon <= 0:
            raise ValueError(f"--n_horizon must be a positive integer, got {args_cli.n_horizon}.")
        policy_cfg.solver.n_horizon = int(args_cli.n_horizon)

    seed = _resolve_seed(run_config, env_config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = _build_env_cfg(env_config, seed)
    _validate_env_cfg(env_cfg)
    runner_cfg = run_config.get("runner", {})
    target_episodes = int(args_cli.episodes if args_cli.episodes is not None else runner_cfg.get("target_episodes", 1))
    target_episodes = max(target_episodes, 1)
    env_cfg.target_position_evaluator.num_eval_episodes = target_episodes
    env_cfg.trajectory_tracking_evaluator.num_eval_episodes = target_episodes

    logging_root_dir = _nmpc_logging_root(policy_cfg.solver.n_horizon)
    output_dir = _make_output_dir(run_config, seed, target_episodes, logging_root_dir)
    save_yaml(run_config, output_dir / "input_run_config.yaml")
    save_yaml(env_config, output_dir / "input_env_config.yaml")
    save_yaml(policy_config, output_dir / "input_policy_config.yaml")

    env = None
    policy = None
    try:
        env = AerialBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": env.max_episode_length - 1,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        base_env = env.unwrapped
        _resolve_policy_cfg_from_env(policy_cfg, env_cfg, base_env.step_dt)
        policy = NMPCPolicy(policy_cfg, base_env.num_envs, base_env.device, base_env.step_dt)
        num_envs = base_env.num_envs

        save_yaml(
            {
                "seed": seed,
                "num_envs": num_envs,
                "episode_length_s": env_cfg.episode_length_s,
                "task_name": env_cfg.task_name,
                "interface_name": env_cfg.interface_name,
                "policy_name": policy_cfg.name,
                "policy_step_dt": base_env.step_dt,
                "n_horizon": policy_cfg.solver.n_horizon,
                "nmpc_t_step": policy_cfg.solver.t_step,
                "nmpc_max_acc": policy_cfg.max_acc,
                "use_multiprocessing": policy_cfg.use_multiprocessing,
                "state_predictor_enabled": policy_cfg.state_predictor.enabled,
                "state_predictor_delay_step": policy_cfg.state_predictor.delay_step,
                "run_config_path": str(run_config_path),
                "env_config_path": str(env_config_path),
                "policy_config_path": str(policy_config_path),
                "logging_root_dir": str(logging_root_dir),
                "observation_fields": OBSERVATION_FIELDS,
            },
            output_dir / "resolved_run.yaml",
        )

        observations, infos = env.reset()
        policy.reset()
        configured_max_steps = runner_cfg.get("max_steps")
        if configured_max_steps is None:
            max_steps = math.ceil(target_episodes / num_envs) * base_env.max_episode_length
        else:
            max_steps = int(configured_max_steps)

        obs_records: list[np.ndarray] = []
        action_records: list[np.ndarray] = []
        reward_records: list[np.ndarray] = []
        terminated_records: list[np.ndarray] = []
        truncated_records: list[np.ndarray] = []
        benchmark_records: list[dict[str, float | bool]] = []
        solver_status_records: list[np.ndarray] = []
        step_extra_records: dict[str, list[np.ndarray]] = {key: [] for key in STEP_EXTRA_FIELDS}
        policy_extra_records: dict[str, list[np.ndarray]] = {}

        iterator = range(max_steps)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="NMPC rollout")

        final_metrics = _collect_benchmark_metrics(infos)
        for _ in iterator:
            obs_records.append(_tensor_to_numpy(observations["policy"]))
            with torch.no_grad():
                actions = policy.act(observations, infos)
            policy_extras = _collect_tensor_fields(policy.get_state())
            solver_status_records.append(np.asarray(policy.last_solver_status))

            next_observations, reward, terminated, truncated, infos = env.step(actions)
            action_records.append(_tensor_to_numpy(actions))
            reward_records.append(_tensor_to_numpy(reward))
            terminated_records.append(_tensor_to_numpy(terminated))
            truncated_records.append(_tensor_to_numpy(truncated))

            for key, value in _collect_step_extras(infos).items():
                step_extra_records[key].append(value)
            for key, value in policy_extras.items():
                policy_extra_records.setdefault(key, []).append(value)

            final_metrics = _collect_benchmark_metrics(infos)
            benchmark_records.append(final_metrics)
            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                policy.reset(done_env_ids)
            observations = next_observations

            if bool(runner_cfg.get("render", True)) and not args_cli.headless:
                env.render()

            if int(final_metrics.get("completed_episodes", 0)) >= target_episodes:
                break

        rollout_steps = len(obs_records)
        observations_np = _stack_or_empty(obs_records, (0, num_envs, len(OBSERVATION_FIELDS)))
        rewards_np = _stack_or_empty(reward_records, (0, num_envs))
        if bool(runner_cfg.get("save_rollout", True)):
            rollout_payload = {
                "observations": observations_np,
                "actions": _stack_or_empty(action_records, (0, num_envs, 1)),
                "rewards": rewards_np,
                "terminated": _stack_or_empty(terminated_records, (0, num_envs), dtype=bool),
                "truncated": _stack_or_empty(truncated_records, (0, num_envs), dtype=bool),
                "nmpc_solver_status": _stack_or_empty(solver_status_records, (0, num_envs), dtype="<U64"),
                "observation_fields": np.asarray(OBSERVATION_FIELDS),
            }
            for key, records in step_extra_records.items():
                if records:
                    rollout_payload[f"step_{key}"] = _stack_or_empty(records, (0, num_envs))
                else:
                    rollout_payload[f"step_{key}"] = np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
            for key, records in policy_extra_records.items():
                rollout_payload[key] = _stack_or_empty(records, (0, *records[0].shape))
            if benchmark_records:
                for key in benchmark_records[-1].keys():
                    rollout_payload[f"benchmark_{key}"] = np.asarray(
                        [record.get(key, np.nan) for record in benchmark_records]
                    )
            np.savez_compressed(output_dir / "rollout.npz", **rollout_payload)

        completed_episodes = int(final_metrics.get("completed_episodes", 0))
        compute_time_records = policy_extra_records.get("nmpc_compute_time", [])
        if compute_time_records:
            nmpc_compute_time_mean = float(np.mean(np.stack(compute_time_records, axis=0)))
        else:
            nmpc_compute_time_mean = float("nan")
        summary_row = {
            "run_name": output_dir.name,
            "run_config_path": str(run_config_path),
            "env_config_path": str(env_config_path),
            "policy_config_path": str(policy_config_path),
            "seed": seed,
            "num_envs": num_envs,
            "target_episodes": target_episodes,
            "completed_episodes": completed_episodes,
            "rollout_steps": rollout_steps,
            "episode_length_s": env_cfg.episode_length_s,
            "task_name": env_cfg.task_name,
            "interface_name": env_cfg.interface_name,
            "policy_name": policy_cfg.name,
            "n_horizon": policy_cfg.solver.n_horizon,
            "mean_reward": float(np.mean(rewards_np)) if rewards_np.size else float("nan"),
            "nmpc_compute_time_mean": nmpc_compute_time_mean,
        }
        for key, value in final_metrics.items():
            summary_row[f"benchmark_{key}"] = value
        append_csv_row(output_dir / "summary.csv", summary_row)
        append_csv_row(logging_root_dir / "summary.csv", summary_row)

        print(f"[INFO] NMPC rollout finished. Logs saved to: {output_dir}")
        print(f"[INFO] Summary: {summary_row}")
    finally:
        if policy is not None:
            policy.close()
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
