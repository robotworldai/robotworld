#!/usr/bin/env python3
"""Run a zero-action policy against AerialBalanceEnv and log rollout data."""

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


def _parse_args():
    parser = argparse.ArgumentParser(description="Zero-action policy runner for Aerial-Balance-Bench.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "environments" / "configs" / "template.yaml"),
        help="Path to a YAML environment/deployment config.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override target completed episodes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
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


args_cli = _parse_args()
_launch_config = _read_yaml_safely(args_cli.config)
_yaml_device = _launch_config.get("env", {}).get("device")
if _yaml_device is not None and not _cli_option_present("--device"):
    args_cli.device = _yaml_device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this smoke runner.
    tqdm = None

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
    "drz_cmd",
    "executed_drz_cmd",
    "target_position_z",
    "frz_cmd",
    "executed_frz_cmd",
    "delta_frz_cmd",
    "hover_force",
    "target_pitch",
    "target_height_acc",
    "beta",
    "ball_mass",
    "low_level_controller_gain",
    "position_gain",
    "velocity_gain",
    "attitude_gain",
    "action_delay_enabled",
    "delay_step",
    "delayed_command_z",
    "external_disturbance_enabled",
    "external_disturbance_vel_z",
    "external_disturbance_ou_theta",
    "external_disturbance_ou_sigma",
    "last_action",
    "trajectory_type_id",
    "trajectory_amplitude",
    "trajectory_period",
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _maybe_set_attrs(target: Any, values: dict[str, Any]):
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def _format_float_for_name(value: float) -> str:
    return f"{value:.6g}".replace("-", "neg").replace(".", "p")


def _resolve_seed(config: dict[str, Any]) -> int:
    if args_cli.seed is not None:
        seed = args_cli.seed
    else:
        seed = int(config.get("env", {}).get("seed", 666))
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

    # AppLauncher owns the actual simulation device. Keep YAML as the default,
    # but let Isaac Lab CLI args override it when provided.
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

    target_position_task_cfg = config.get("target_position_task", config.get("task", {}))
    _maybe_set_attrs(env_cfg.target_position_task, target_position_task_cfg)
    _maybe_set_attrs(env_cfg.trajectory_tracking_task, config.get("trajectory_tracking_task", {}))
    _maybe_set_attrs(env_cfg.velocity_interface, config.get("velocity_interface", {}))
    _maybe_set_attrs(env_cfg.position_interface, config.get("position_interface", {}))
    _maybe_set_attrs(env_cfg.thrust_interface, config.get("thrust_interface", {}))
    _maybe_set_attrs(env_cfg.robustness, config.get("robustness", {}))

    target_position_evaluator_cfg = config.get("target_position_evaluator", config.get("evaluator", {}))
    _maybe_set_attrs(env_cfg.target_position_evaluator, target_position_evaluator_cfg)
    _maybe_set_attrs(env_cfg.trajectory_tracking_evaluator, config.get("trajectory_tracking_evaluator", {}))
    env_cfg.target_position_evaluator.episode_length_s = env_cfg.episode_length_s
    env_cfg.trajectory_tracking_evaluator.episode_length_s = env_cfg.episode_length_s

    return env_cfg


def _make_output_dir(config: dict[str, Any], seed: int, target_episodes: int) -> Path:
    logging_cfg = config.get("logging", {})
    root_dir = ensure_dir(PROJECT_ROOT / logging_cfg.get("root_dir", "logs/zero_action"))

    if args_cli.run_name is not None:
        run_name = args_cli.run_name
    else:
        run_name = logging_cfg.get("run_name")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"zero_action_ep_{target_episodes}_seed_{seed}_{timestamp}"

    return ensure_dir(root_dir / run_name)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _scalar_metric(value: Any) -> float | bool:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return bool(value.detach().cpu().item())
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_step_extras(infos: dict[str, Any]) -> dict[str, np.ndarray]:
    step_info = infos.get("step", {})
    collected = {}
    for key in STEP_EXTRA_FIELDS:
        value = step_info.get(key)
        if isinstance(value, torch.Tensor):
            collected[key] = _tensor_to_numpy(value)
    return collected


def _collect_benchmark_metrics(infos: dict[str, Any]) -> dict[str, float | bool]:
    benchmark = infos.get("benchmark", {})
    return {key: _scalar_metric(value) for key, value in benchmark.items()}


def _stack_or_empty(records: list[np.ndarray], shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if records:
        return np.stack(records, axis=0)
    return np.empty(shape, dtype=dtype)


def main():
    config_path = Path(args_cli.config).expanduser().resolve()
    config = _load_yaml(config_path)
    seed = _resolve_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = _build_env_cfg(config, seed)
    runner_cfg = config.get("runner", {})
    target_episodes = int(args_cli.episodes if args_cli.episodes is not None else runner_cfg.get("target_episodes", 1))
    target_episodes = max(target_episodes, 1)
    env_cfg.target_position_evaluator.num_eval_episodes = target_episodes
    env_cfg.trajectory_tracking_evaluator.num_eval_episodes = target_episodes

    output_dir = _make_output_dir(config, seed, target_episodes)
    save_yaml(config, output_dir / "input_config.yaml")
    save_yaml(
        {
            "seed": seed,
            "num_envs": env_cfg.scene.num_envs,
            "episode_length_s": env_cfg.episode_length_s,
            "task_name": env_cfg.task_name,
            "interface_name": env_cfg.interface_name,
            "target_episodes": target_episodes,
            "sim_device": env_cfg.sim.device,
            "config_path": str(config_path),
            "observation_fields": OBSERVATION_FIELDS,
        },
        output_dir / "resolved_run.yaml",
    )

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
    num_envs = base_env.num_envs
    observations, infos = env.reset()
    observations = observations["policy"]

    configured_max_steps = runner_cfg.get("max_steps")
    if configured_max_steps is None:
        max_steps = math.ceil(target_episodes / num_envs) * base_env.max_episode_length + 1
    else:
        max_steps = int(configured_max_steps)

    obs_records: list[np.ndarray] = []
    action_records: list[np.ndarray] = []
    reward_records: list[np.ndarray] = []
    terminated_records: list[np.ndarray] = []
    truncated_records: list[np.ndarray] = []
    policy_compute_time_records: list[float] = []
    benchmark_records: list[dict[str, float | bool]] = []
    step_extra_records: dict[str, list[np.ndarray]] = {key: [] for key in STEP_EXTRA_FIELDS}

    iterator = range(max_steps)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Zero-action rollout")

    final_metrics = _collect_benchmark_metrics(infos)
    for _ in iterator:
        obs_records.append(_tensor_to_numpy(observations))

        start_time = time.perf_counter()
        actions = torch.zeros((num_envs, 1), device=base_env.device, dtype=observations.dtype)
        policy_compute_time = time.perf_counter() - start_time

        next_observations, reward, terminated, truncated, infos = env.step(actions)

        action_records.append(_tensor_to_numpy(actions))
        reward_records.append(_tensor_to_numpy(reward))
        terminated_records.append(_tensor_to_numpy(terminated))
        truncated_records.append(_tensor_to_numpy(truncated))
        policy_compute_time_records.append(policy_compute_time)

        step_extras = _collect_step_extras(infos)
        for key, value in step_extras.items():
            step_extra_records[key].append(value)

        final_metrics = _collect_benchmark_metrics(infos)
        benchmark_records.append(final_metrics)
        observations = next_observations["policy"]

        if bool(runner_cfg.get("render", True)) and not args_cli.headless:
            env.render()

        completed_episodes = int(final_metrics.get("completed_episodes", 0))
        if completed_episodes >= target_episodes:
            break

    rollout_steps = len(obs_records)
    env.close()

    observations_np = _stack_or_empty(obs_records, (0, num_envs, len(OBSERVATION_FIELDS)))
    rewards_np = _stack_or_empty(reward_records, (0, num_envs))

    if bool(runner_cfg.get("save_rollout", True)):
        rollout_payload = {
            "observations": observations_np,
            "actions": _stack_or_empty(action_records, (0, num_envs, 1)),
            "rewards": rewards_np,
            "terminated": _stack_or_empty(terminated_records, (0, num_envs), dtype=bool),
            "truncated": _stack_or_empty(truncated_records, (0, num_envs), dtype=bool),
            "policy_compute_time": np.asarray(policy_compute_time_records, dtype=np.float64),
            "observation_fields": np.asarray(OBSERVATION_FIELDS),
        }
        for key, records in step_extra_records.items():
            if records:
                rollout_payload[f"step_{key}"] = _stack_or_empty(records, (0, num_envs))
            else:
                rollout_payload[f"step_{key}"] = np.full((rollout_steps, num_envs), np.nan, dtype=np.float32)
        if benchmark_records:
            for key in benchmark_records[-1].keys():
                rollout_payload[f"benchmark_{key}"] = np.asarray(
                    [record.get(key, np.nan) for record in benchmark_records]
                )
        np.savez_compressed(output_dir / "rollout.npz", **rollout_payload)

    completed_episodes = int(final_metrics.get("completed_episodes", 0))
    mean_reward = float(np.mean(rewards_np)) if rewards_np.size else float("nan")
    summary_row = {
        "run_name": output_dir.name,
        "config_path": str(config_path),
        "seed": seed,
        "num_envs": num_envs,
        "target_episodes": target_episodes,
        "completed_episodes": completed_episodes,
        "rollout_steps": rollout_steps,
        "episode_length_s": env_cfg.episode_length_s,
        "interface_name": env_cfg.interface_name,
        "mean_reward": mean_reward,
        "success_rate": final_metrics.get("success_rate", float("nan")),
        "steady_state_error": final_metrics.get("steady_state_error", float("nan")),
        "steady_state_error_std": final_metrics.get("steady_state_error_std", float("nan")),
        "convergence_time": final_metrics.get("convergence_time", float("nan")),
        "convergence_time_std": final_metrics.get("convergence_time_std", float("nan")),
        "climbing_time": final_metrics.get("climbing_time", float("nan")),
        "climbing_time_std": final_metrics.get("climbing_time_std", float("nan")),
        "policy_compute_time_mean": float(np.mean(policy_compute_time_records))
        if policy_compute_time_records
        else float("nan"),
        "evaluation_complete": final_metrics.get("evaluation_complete", False),
    }
    for key, value in final_metrics.items():
        summary_row[f"benchmark_{key}"] = value
    append_csv_row(output_dir / "summary.csv", summary_row)
    append_csv_row(PROJECT_ROOT / config.get("logging", {}).get("root_dir", "logs/zero_action") / "summary.csv", summary_row)

    print(f"[INFO] Zero-action rollout finished. Logs saved to: {output_dir}")
    print(f"[INFO] Summary: {summary_row}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
