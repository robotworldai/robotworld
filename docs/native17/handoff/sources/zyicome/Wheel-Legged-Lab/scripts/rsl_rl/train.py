# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--no_debug_metrics",
    action="store_true",
    default=False,
    help="Disable the compact wheel-legged debug line in terminal output.",
)
parser.add_argument(
    "--load_weights_only",
    action="store_true",
    default=False,
    help="Load actor/critic weights but start with a fresh optimizer and iteration counter.",
)
parser.add_argument(
    "--load_checkpoint_path",
    type=str,
    default=None,
    help=(
        "Load an explicit checkpoint path. Unlike --load_run/--checkpoint, "
        "this may come from a different experiment while logs use the new task's experiment."
    ),
)
parser.add_argument("--min_action_std", type=float, default=None, help="Optional lower bound for learned action std.")
parser.add_argument("--max_action_std", type=float, default=None, help="Optional upper bound for learned action std.")
parser.add_argument(
    "--moving_jump_initial_level",
    type=int,
    default=None,
    choices=range(5),
    metavar="{0,1,2,3,4}",
    help=(
        "Initial moving-jump curriculum level: "
        "0/1/2/3/4 maps to 0.2/0.4/0.6/0.8/1.0 m/s."
    ),
)
parser.add_argument(
    "--obstacle_initial_level",
    type=int,
    default=None,
    choices=range(7),
    metavar="{0,1,2,3,4,5,6}",
    help=(
        "Initial obstacle geometry curriculum level. The seven levels end at "
        "0.08 m height and 0.08 m width."
    ),
)
parser.add_argument(
    "--early_stop_config",
    type=str,
    default=None,
    help="JSON metric-gate configuration used by the staged training pipeline.",
)
parser.add_argument(
    "--early_stop_status_file",
    type=str,
    default=None,
    help="Write the metric-gate result as JSON when training exits.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# RTX cameras are only needed for video. The perceptive task uses the scalable
# ray-cast depth backend and therefore remains available in normal headless mode.
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path


def _expand_observation_checkpoint(checkpoint_path: str, runner, output_dir: str) -> str:
    """Pad actor/critic observation tensors while preserving all learned columns."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    groups = (
        ("actor_state_dict", runner.alg.actor.state_dict()),
        ("critic_state_dict", runner.alg.critic.state_dict()),
    )
    changed = False
    for group_name, target_state in groups:
        source_state = checkpoint.get(group_name)
        if source_state is None:
            continue
        for key, target in target_state.items():
            source = source_state.get(key)
            if source is None or not torch.is_tensor(source) or source.shape == target.shape:
                continue
            # Observation expansion changes only vector length or the input
            # dimension of the first MLP layer. Other mismatches remain errors.
            compatible = (
                source.ndim == target.ndim
                and source.ndim in (1, 2)
                and all(
                    source.shape[dim] == target.shape[dim]
                    for dim in range(source.ndim - 1)
                )
                and source.shape[-1] < target.shape[-1]
            )
            if not compatible:
                continue
            expanded = target.detach().cpu().clone()
            expanded[..., : source.shape[-1]] = source.detach().cpu()
            source_state[key] = expanded
            changed = True
            print(
                f"[INFO] Expanded {group_name}.{key}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )
    if not changed:
        return checkpoint_path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "expanded_observation_checkpoint.pt")
    torch.save(checkpoint, output_path)
    return output_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import wheel_legged_robot.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class RollingMetricGate:
    """Stop a training stage after a stable window satisfies every metric rule."""

    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as stream:
            config = json.load(stream)
        self.name = str(config.get("name", "unnamed"))
        self.min_iterations = int(config.get("min_iterations", 0))
        self.window = int(config.get("window", 20))
        self.consecutive = int(config.get("consecutive", 5))
        self.rules = config.get("rules", {})
        if self.window <= 0 or self.consecutive <= 0 or not self.rules:
            raise ValueError("Early-stop config requires positive window/consecutive and non-empty rules.")
        self.history = {
            name: deque(maxlen=self.window) for name in self.rules
        }
        self.consecutive_passes = 0
        self.passed = False
        self.iteration = -1
        self.averages: dict[str, float] = {}

    @staticmethod
    def _rule_passes(value: float, rule: dict) -> bool:
        if "min" in rule and value < float(rule["min"]):
            return False
        if "max" in rule and value > float(rule["max"]):
            return False
        if "abs_max" in rule and abs(value) > float(rule["abs_max"]):
            return False
        return True

    def __call__(self, metrics: dict[str, float], iteration) -> None:
        from rl_utils import TrainingEarlyStop

        self.iteration = int(iteration) if iteration is not None else self.iteration + 1
        for name, values in self.history.items():
            if name in metrics:
                values.append(float(metrics[name]))

        ready = (
            self.iteration >= self.min_iterations
            and all(len(values) == self.window for values in self.history.values())
        )
        self.averages = {
            name: sum(values) / len(values)
            for name, values in self.history.items()
            if values
        }
        if not ready:
            self.consecutive_passes = 0
            return

        if all(
            self._rule_passes(self.averages[name], rule)
            for name, rule in self.rules.items()
        ):
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        if self.consecutive_passes >= self.consecutive:
            self.passed = True
            summary = ", ".join(
                f"{name}={value:.4f}" for name, value in self.averages.items()
            )
            raise TrainingEarlyStop(
                f"stage {self.name!r} passed at iteration {self.iteration}: {summary}"
            )

    def status(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "iteration": self.iteration,
            "window": self.window,
            "consecutive_passes": self.consecutive_passes,
            "metrics": self.averages,
        }


def _write_early_stop_status(path: str | None, gate: RollingMetricGate | None) -> None:
    if path is None or gate is None:
        return
    status_path = Path(path).expanduser().resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(status_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(gate.status(), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary_path, status_path)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.moving_jump_initial_level is not None:
        curriculum_cfg = getattr(env_cfg, "curriculum", None)
        speed_term_cfg = (
            getattr(curriculum_cfg, "speed_levels", None)
            if curriculum_cfg is not None
            else None
        )
        if speed_term_cfg is None:
            raise ValueError(
                "--moving_jump_initial_level requires a task with the "
                "moving-jump speed curriculum."
            )
        speed_term_cfg.params["initial_level"] = args_cli.moving_jump_initial_level
        print(
            "[INFO] Moving-jump curriculum starts at level "
            f"{args_cli.moving_jump_initial_level}."
        )
    if args_cli.obstacle_initial_level is not None:
        curriculum_cfg = getattr(env_cfg, "curriculum", None)
        geometry_term_cfg = (
            getattr(curriculum_cfg, "geometry_levels", None)
            if curriculum_cfg is not None
            else None
        )
        if geometry_term_cfg is None:
            raise ValueError(
                "--obstacle_initial_level requires the Oracle obstacle "
                "geometry curriculum task."
            )
        geometry_term_cfg.params["initial_level"] = args_cli.obstacle_initial_level
        print(
            "[INFO] Obstacle geometry curriculum starts at level "
            f"{args_cli.obstacle_initial_level}."
        )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.load_checkpoint_path is not None:
        resume_path = os.path.abspath(os.path.expanduser(args_cli.load_checkpoint_path))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Explicit checkpoint does not exist: {resume_path}")
        agent_cfg.resume = True
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Keep the official logger, TensorBoard scalars and checkpoint behavior, but
    # display the wheel-legged environment diagnostics as one compact line.
    metric_gate = (
        RollingMetricGate(args_cli.early_stop_config)
        if args_cli.early_stop_config is not None
        else None
    )
    if metric_gate is not None and args_cli.no_debug_metrics:
        raise ValueError("--early_stop_config cannot be combined with --no_debug_metrics.")
    if not args_cli.no_debug_metrics:
        from rl_utils import install_rsl_rl_debug_logger

        install_rsl_rl_debug_logger(runner, metric_callback=metric_gate)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.load_weights_only:
            resume_path = _expand_observation_checkpoint(
                resume_path, runner, log_dir
            )
            runner.load(
                resume_path,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
        else:
            runner.load(resume_path)

    min_action_std = args_cli.min_action_std
    max_action_std = args_cli.max_action_std
    if args_cli.task in (
        "Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0",
        "Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0",
    ):
        min_action_std = 0.10 if min_action_std is None else min_action_std
        max_action_std = 0.50 if max_action_std is None else max_action_std
    elif args_cli.task in (
        "Wheel-Legged-Recovery-Flat-v0",
        "Wheel-Legged-Terrain-Reactive-v0",
        "Wheel-Legged-Terrain-Perceptive-v0",
    ):
        min_action_std = 0.08 if min_action_std is None else min_action_std
        max_action_std = 0.50 if max_action_std is None else max_action_std
    elif args_cli.task and args_cli.task.startswith("Wheel-Legged-Jump-"):
        min_action_std = 0.05 if min_action_std is None else min_action_std
        max_action_std = 0.50 if max_action_std is None else max_action_std
    if min_action_std is not None or max_action_std is not None:
        if min_action_std is None or max_action_std is None:
            raise ValueError("--min_action_std and --max_action_std must be specified together.")
        if not 0.0 < min_action_std <= max_action_std:
            raise ValueError("Action std bounds must satisfy 0 < min <= max.")
        from rl_utils import install_rsl_rl_action_std_bounds

        install_rsl_rl_action_std_bounds(
            runner, min_std=min_action_std, max_std=max_action_std
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except Exception as exc:
        from rl_utils import TrainingEarlyStop

        if not isinstance(exc, TrainingEarlyStop):
            _write_early_stop_status(args_cli.early_stop_status_file, metric_gate)
            env.close()
            raise
        print(f"[AUTO-STAGE] {exc}")
        checkpoint_path = os.path.join(
            log_dir, f"model_{runner.current_learning_iteration}.pt"
        )
        runner.save(checkpoint_path)
        if runner.logger.writer is not None:
            runner.logger.stop_logging_writer()
        print(f"[AUTO-STAGE] Saved passing checkpoint: {checkpoint_path}")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    _write_early_stop_status(args_cli.early_stop_status_file, metric_gate)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
