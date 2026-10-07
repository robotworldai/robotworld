# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-install smoke test for the wheeled-quadruped environments.

This replaces the legacy ``create_wheeled_quadruped_env.py``. It builds one of the
registered environments, prints the articulation's joint/body names and the shape of
each observation group, then drives the environment with random actions for a fixed
number of steps while checking every observation and reward tensor for NaN/Inf. On
completion it prints a PASS/FAIL summary.

Example:
    python scripts/verify_env.py --task Wheeled-Quadruped-Balance-v0 --num_envs 8 --steps 200 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Smoke test for the wheeled-quadruped environments.")
parser.add_argument("--task", type=str, default="Wheeled-Quadruped-Balance-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=200, help="Number of random-action steps to run.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import gymnasium as gym
import torch

import wheeled_quadruped.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _find_bad_tensors(named_tensors):
    """Return a list of names whose tensor contains a NaN or Inf value.

    Args:
        named_tensors: Iterable of ``(name, value)`` pairs. Values that are not
            floating-point tensors are skipped; nested dicts are inspected recursively.
    """
    bad = []
    for name, value in named_tensors:
        if isinstance(value, dict):
            bad.extend(_find_bad_tensors((f"{name}.{k}", v) for k, v in value.items()))
        elif isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            if not torch.isfinite(value).all():
                bad.append(name)
    return bad


def main() -> int:
    """Build the environment, inspect it, and run a random-action rollout."""
    env = None
    try:
        # parse the environment configuration (uses the config's default fabric setting)
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

        # create the environment
        env = gym.make(args_cli.task, cfg=env_cfg)
        device = env.unwrapped.device

        print("=" * 80)
        print(f"[verify_env] Task            : {args_cli.task}")
        print(f"[verify_env] Num envs        : {env.unwrapped.num_envs}")
        print(f"[verify_env] Device          : {device}")
        print(f"[verify_env] Gym obs space   : {env.observation_space}")
        print(f"[verify_env] Gym action space: {env.action_space}")

        # inspect the articulation
        robot = env.unwrapped.scene["robot"]
        print("-" * 80)
        print(f"[verify_env] Articulation joints ({robot.num_joints}): {list(robot.joint_names)}")
        print(f"[verify_env] Articulation bodies ({robot.num_bodies}): {list(robot.body_names)}")

        # reset and report the observation group shapes
        obs, _ = env.reset()
        print("-" * 80)
        print("[verify_env] Observation groups:")
        for group_name, group_value in obs.items():
            if isinstance(group_value, torch.Tensor):
                print(f"  - {group_name}: shape={tuple(group_value.shape)} dtype={group_value.dtype}")
            elif isinstance(group_value, dict):
                for term_name, term_value in group_value.items():
                    shape = tuple(term_value.shape) if isinstance(term_value, torch.Tensor) else "?"
                    print(f"  - {group_name}.{term_name}: shape={shape}")
            else:
                print(f"  - {group_name}: {type(group_value)}")

        # check the reset observation for NaN/Inf
        bad_names = _find_bad_tensors(obs.items())

        # run the random-action rollout
        print("-" * 80)
        print(f"[verify_env] Running {args_cli.steps} random-action steps ...")
        total_resets = 0
        reward_sum = 0.0
        reward_count = 0
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                print("[verify_env] Simulation app stopped early; ending rollout.")
                break
            with torch.inference_mode():
                actions = 2.0 * torch.rand(env.action_space.shape, device=device) - 1.0
                obs, rew, terminated, truncated, info = env.step(actions)

            # scan observations and reward for NaN/Inf
            step_bad = _find_bad_tensors(obs.items())
            step_bad += _find_bad_tensors([("reward", rew)])
            if step_bad:
                for name in step_bad:
                    tagged = f"step{step}:{name}"
                    if tagged not in bad_names:
                        bad_names.append(tagged)

            # accounting
            total_resets += int((terminated | truncated).sum().item())
            reward_sum += float(rew.float().mean().item())
            reward_count += 1

        mean_reward = reward_sum / reward_count if reward_count > 0 else float("nan")

        # summary
        passed = len(bad_names) == 0
        print("=" * 80)
        print("[verify_env] SUMMARY")
        print(f"  steps run          : {reward_count}")
        print(f"  episode resets seen: {total_resets}")
        print(f"  mean reward        : {mean_reward:.6f}")
        print(f"  NaN/Inf detected   : {'YES -> ' + ', '.join(bad_names) if bad_names else 'no'}")
        print(f"[verify_env] RESULT: {'PASS' if passed else 'FAIL'}")
        print("=" * 80)
        return 0 if passed else 1

    except Exception:
        traceback.print_exc()
        print("[verify_env] RESULT: FAIL (exception raised)")
        return 1
    finally:
        # clean shutdown of the environment
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        # close sim app even on exception
        simulation_app.close()
    raise SystemExit(exit_code)
