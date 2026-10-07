# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how accurately a velocity policy tracks its commanded velocity.

Unlike the exp-kernel reward term (whose scale depends on its weight and std),
this reports the *physical* mean-absolute error between the commanded and the
achieved base velocity, so two policies trained with different reward shaping can
be compared apples-to-apples. Error is accumulated only over upright timesteps
(so a fall does not pollute the average).

Example:
    python scripts/eval_tracking.py --task Wheeled-Quadruped-Velocity-v0 \
        --checkpoint logs/rsl_rl/wheeled_quadruped_velocity/<run>/model_2999.pt \
        --num_envs 256 --steps 1000 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports (scripts/rsl_rl is on the path via the launcher call)
sys.path.append(__file__.rsplit("scripts", 1)[0] + "scripts/rsl_rl")
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Evaluate velocity-tracking error of an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--task", type=str, default="Wheeled-Quadruped-Velocity-v0", help="Task id.")
parser.add_argument("--steps", type=int, default=1000, help="Measurement steps (after a short settle).")
parser.add_argument("--settle", type=int, default=50, help="Warm-up steps to skip before measuring.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after the app is up."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import wheeled_quadruped.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[eval] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]

    # accumulators (masked to upright timesteps)
    dev = env.unwrapped.device
    sum_abs_vx = torch.zeros(1, device=dev)
    sum_abs_yaw = torch.zeros(1, device=dev)
    sum_sq_vx = torch.zeros(1, device=dev)
    sum_sq_yaw = torch.zeros(1, device=dev)
    count = torch.zeros(1, device=dev)
    # focus subset: envs whose current command actually asks for turning
    sum_abs_yaw_turn = torch.zeros(1, device=dev)
    count_turn = torch.zeros(1, device=dev)

    obs = env.get_observations()
    total = args_cli.settle + args_cli.steps
    for t in range(total):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if t < args_cli.settle:
                continue
            cmd = env.unwrapped.command_manager.get_command("base_velocity")  # (N,3): vx, vy, yaw
            vx_cmd, yaw_cmd = cmd[:, 0], cmd[:, 2]
            vx_act = robot.data.root_lin_vel_b[:, 0]
            yaw_act = robot.data.root_ang_vel_b[:, 2]
            # upright mask: projected-gravity z near -1 means the torso is vertical
            upright = robot.data.projected_gravity_b[:, 2] < -0.8
            m = upright.float()
            e_vx = (vx_cmd - vx_act).abs()
            e_yaw = (yaw_cmd - yaw_act).abs()
            sum_abs_vx += (e_vx * m).sum()
            sum_abs_yaw += (e_yaw * m).sum()
            sum_sq_vx += ((vx_cmd - vx_act) ** 2 * m).sum()
            sum_sq_yaw += ((yaw_cmd - yaw_act) ** 2 * m).sum()
            count += m.sum()
            turn = (yaw_cmd.abs() > 0.2) & upright
            sum_abs_yaw_turn += (e_yaw * turn.float()).sum()
            count_turn += turn.float().sum()

    n = count.clamp(min=1)
    nt = count_turn.clamp(min=1)
    print("\n================ VELOCITY TRACKING ERROR ================")
    print(f"task            : {args_cli.task}")
    print(f"checkpoint      : {resume_path}")
    print(f"upright samples : {int(count.item())}  (of {args_cli.num_envs} envs x {args_cli.steps} steps)")
    print(f"lin-vel x  MAE  : {(sum_abs_vx / n).item():.4f} m/s   RMSE {(sum_sq_vx / n).sqrt().item():.4f}")
    print(f"yaw-rate   MAE  : {(sum_abs_yaw / n).item():.4f} rad/s RMSE {(sum_sq_yaw / n).sqrt().item():.4f}")
    print(f"yaw MAE (|cmd|>0.2 only): {(sum_abs_yaw_turn / nt).item():.4f} rad/s over {int(count_turn.item())} samples")
    print("========================================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
