# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import argparse
import os
import numpy as np

import torch
from isaaclab.app import AppLauncher
# Use the Isaac Lab adapter for RSL-RL to match the env API
from rsl_rl.rsl_rl.runners import OnPolicyRunner
# from isaaclab_rl.rsl_rl import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train/Play an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--predictor", action="store_true", help="Use predictor-augmented runner for loading predictor weights and inference")
# recording / debugging flags
parser.add_argument(
    "--record_action",
    action="store_true",
    help="Record observations and actions during play; also plots obs[48:69].",
)
# parser.add_argument("--check", type=str, default='model_.*.pt', help="checkpoint, defaul the lastest")
# parser.add_argument("--run", type=str, default='.*', help="experiment run name, defaul the latest run")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # agent_cfg.load_run = "2025-08-23_00-53-24"
    # agent_cfg.load_checkpoint = "model_12000.pt"

    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.events.push_robot = None
    #env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 5
    # env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    # env_cfg.commands.ranges.lin_vel_y = (0.6, 0.6)
    # env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # agent_cfg.load_run=args_cli.run
    # agent_cfg.load_checkpoint=args_cli.checkpoint
    print(f"\n[INFO] Loading experiment from directory: {log_root_path}")
    print(f'agent_cfg.load_run:{agent_cfg.load_run}')
    print(f'agent_cfg.load_checkpoint:{agent_cfg.load_checkpoint}\n')
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Choose runner implementation
    if args_cli.predictor:
        from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner
        runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    # Export predictor weights only when predictor-augmented runner is used
    if args_cli.predictor:
        # Save predictor as generic TorchScript module (policy-specific exporter not applicable)
        # Move to CPU before scripting/saving to avoid CUDA dependency during load.
        try:
            orig_device = next(runner._predictor.parameters()).device
        except StopIteration:
            orig_device = torch.device("cpu")
        runner._predictor.to("cpu").eval()
        ts_predictor = torch.jit.script(runner._predictor)
        ts_predictor.save(os.path.join(export_model_dir, "predictor.pt"))
        # Restore original device for runtime
        runner._predictor.to(orig_device)
    # Export policy in both JIT and ONNX formats
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    # Prepare paths for periodic eval result saving
    result_dir = os.path.join(os.path.dirname(resume_path), "eval_result")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "eval_results.csv")

    # Prepare record buffers if requested
    record_action = bool(getattr(args_cli, "record_action", False))
    if record_action:
        obs_records = []        # list[np.ndarray] of shape (obs_dim,)
        action_records = []     # list[np.ndarray] of shape (act_dim,)
        obs_slice_records = []  # list[np.ndarray] of shape (21,) for obs[48:69]
    # Success / serve counters for play session
    succ_total, hit_total, serve_total = 0, 0, 0
    try:
        # per-env flag: whether current serve achieved success
        serve_success_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # per-env flag: whether current serve had any paddle hit
        serve_hit_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    except Exception:
        serve_success_flag = None
        serve_hit_flag = None
    step_count = 0

    # --- Evaluation tracking: per-env label + position, and final records ---
    # Label: 0=missed (default), 1=hit, 2=success
    try:
        current_label = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        current_pos = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)
        # Initialize default position guess as future pose (env-local)
        if hasattr(env, "ball_future_pose"):
            current_pos.copy_(env.ball_future_pose)
    except Exception:
        current_label = None
        current_pos = None
    # Collected rows: (env_id, label_str, x, y, z)
    eval_rows = []
    last_saved_len = 0

    # Skip first 2 * num_envs serves from stats/records (allow robot to stabilize)
    warmup_remaining = int(2 * env.num_envs)

    def _save_eval_rows():
        nonlocal last_saved_len
        try:
            with open(csv_path, "w") as f:
                f.write("env_id,label,x,y,z\n")
                for row in eval_rows:
                    f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")
            last_saved_len = len(eval_rows)
            print(f"[INFO] Saved evaluation results to: {csv_path} ({last_saved_len} records)")
        except Exception as e:
            print(f"[WARN] Failed to save evaluation results: {e}")

    try:
        while simulation_app.is_running():

            with torch.inference_mode():
                actions = policy(obs)
                # Record inputs/outputs of policy inference for env 0
                if record_action:
                    try:
                        obs0 = obs[0].detach().to("cpu").numpy()
                        act0 = actions[0].detach().to("cpu").numpy()
                        obs_records.append(obs0)
                        action_records.append(act0)
                        # Guard slice if obs is shorter than expected
                        if obs0.shape[0] >= 69:
                            obs_slice_records.append(obs0[48:69])
                        else:
                            # Pad or slice safely
                            start = 48
                            end = min(69, obs0.shape[0])
                            pad_len = 69 - end
                            safe_slice = obs0[start:end]
                            if pad_len > 0:
                                safe_slice = np.pad(safe_slice, (0, pad_len), mode="constant")
                            obs_slice_records.append(safe_slice)
                    except Exception:
                        pass
                obs, _, _, _ = env.step(actions)
                # If predictor runner is used, update learned prediction each step for visualization/observations
                if args_cli.predictor:
                    try:
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        pass
                # Track success/serve and print periodically
                try:
                    # Update per-env success/hit flags during ongoing serve
                    if (
                        hasattr(env, "has_touch_opponent_table_just_now")
                        and hasattr(env, "has_touch_paddle")
                        and serve_success_flag is not None
                    ):
                        event_mask = (env.has_touch_opponent_table_just_now & env.has_touch_paddle)
                        serve_success_flag |= event_mask.to(serve_success_flag.device)
                    if (
                        hasattr(env, "ball_contact_rew")
                        and serve_hit_flag is not None
                    ):
                        hit_mask = (env.ball_contact_rew > 0.0)
                        serve_hit_flag |= hit_mask.to(serve_hit_flag.device)

                    # Update per-env HIT position/label (first contact only)
                    if (
                        hasattr(env, "ball_contact_rew")
                        and current_label is not None
                        and current_pos is not None
                    ):
                        hit_now = (env.ball_contact_rew > 0.0) & (current_label < 1)
                        if hit_now.any():
                            ids_hit = torch.nonzero(hit_now, as_tuple=False).squeeze(-1)
                            # record hit position at time of contact (env-local), keep for success as well
                            try:
                                current_pos[ids_hit] = env.ball_pos[ids_hit]
                            except Exception:
                                pass
                            current_label[ids_hit] = 1  # hit

                    # Update per-env SUCCESS label (keep pos set at hit)
                    if (
                        hasattr(env, "has_touch_opponent_table_just_now")
                        and hasattr(env, "has_touch_paddle")
                        and current_label is not None
                    ):
                        succ_now = (env.has_touch_opponent_table_just_now & env.has_touch_paddle) & (current_label >= 1) & (current_label < 2)
                        if succ_now.any():
                            ids_succ = torch.nonzero(succ_now, as_tuple=False).squeeze(-1)
                            current_label[ids_succ] = 2  # success

                    # On serve boundary, aggregate session stats and finalize records
                    if hasattr(env, "ball_reset_ids") and env.ball_reset_ids is not None:
                        ids = env.ball_reset_ids
                        if isinstance(ids, torch.Tensor) and ids.numel() > 0 and serve_success_flag is not None:
                            ids_dev = ids.to(serve_success_flag.device)
                            # 1) Handle warmup serves: clear flags and reinitialize, but don't count or record
                            take_start = 0
                            if warmup_remaining > 0:
                                skip_n = int(min(warmup_remaining, ids_dev.numel()))
                                warmup_remaining -= skip_n
                                ids_skip = ids_dev[:skip_n]
                                # clear flags
                                serve_success_flag[ids_skip] = False
                                serve_hit_flag[ids_skip] = False
                                # re-init per-env tracking
                                if current_label is not None and current_pos is not None:
                                    try:
                                        current_label[ids_skip] = 0
                                        if hasattr(env, "ball_future_pose"):
                                            current_pos[ids_skip] = env.ball_future_pose[ids_skip]
                                    except Exception:
                                        pass
                                take_start = skip_n

                            # 2) Remaining serves: count and record
                            if take_start < ids_dev.numel():
                                ids_take = ids_dev[take_start:]
                                serve_total += int(ids_take.numel())
                                succ_total += int(serve_success_flag[ids_take].sum().item())
                                hit_total += int(serve_hit_flag[ids_take].sum().item())
                                serve_success_flag[ids_take] = False
                                serve_hit_flag[ids_take] = False
                                # Finalize rows for completed serves
                                if current_label is not None and current_pos is not None:
                                    try:
                                        record_pos = current_pos[ids_take]
                                        labels = current_label[ids_take]
                                        for k in range(ids_take.numel()):
                                            eid = int(ids_take[k].item())
                                            lab = int(labels[k].item())
                                            lab_str = "missed" if lab == 0 else ("hit" if lab == 1 else "success")
                                            px, py, pz = [float(v) for v in record_pos[k].tolist()]
                                            eval_rows.append((eid, lab_str, px, py, pz))
                                    except Exception:
                                        pass
                                    # Re-initialize for next serves: default to missed with future pose
                                    current_label[ids_take] = 0
                                    try:
                                        if hasattr(env, "ball_future_pose"):
                                            current_pos[ids_take] = env.ball_future_pose[ids_take]
                                    except Exception:
                                        pass
                    # print(f"envheading_w {env.robot.data.heading_w}")
                except Exception:
                    pass
                step_count += 1
                if step_count % 50 == 0:
                    succ_rate = (succ_total / serve_total) if serve_total > 0 else 0.0
                    hit_rate = (hit_total / serve_total) if serve_total > 0 else 0.0
                    print(f"[Play] Success {succ_total}/{serve_total} ({succ_rate:.3f}) | Hits {hit_total}/{serve_total} ({hit_rate:.3f})")
                # Periodic autosave every 1000 steps to be robust to interruptions
                if step_count % 1000 == 0 and len(eval_rows) > last_saved_len:
                    _save_eval_rows()
                # print("predicted future ball pose:", env.ball_future_pose[0].cpu().numpy())
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Finalizing and saving recordings...")
    finally:
        # On exit: save recordings if requested (plotting handled by separate script)
        if record_action:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Save arrays
                obs_arr = np.asarray(obs_records) if len(obs_records) > 0 else np.empty((0,))
                act_arr = np.asarray(action_records) if len(action_records) > 0 else np.empty((0,))
                slice_arr = np.asarray(obs_slice_records) if len(obs_slice_records) > 0 else np.empty((0,))
                npz_path = os.path.join(log_dir, "play_obs_actions.npz")
                np.savez(npz_path, obs=obs_arr, actions=act_arr, obs_slice=slice_arr)
                print(f"[INFO] Saved obs/action records to: {npz_path}")
            except Exception as e:
                print(f"[WARN] Failed to save recordings: {e}")

        # Save evaluation rows as CSV under checkpoint path (on exit)
        _save_eval_rows()


if __name__ == "__main__":
    play()
    simulation_app.close()
