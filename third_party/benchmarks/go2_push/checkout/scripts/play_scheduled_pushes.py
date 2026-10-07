"""Play a single Go2 push-recovery policy with scheduled pushes.

This script plays back one robot (num_envs=1) under deterministic push timing
and records a single MP4 at the requested path.

Usage (run from project root with isaaclab_env activated):

    python scripts/play_scheduled_pushes.py \
        --checkpoint /abs/path/to/model_XXXX.pt \
        --video_path logs/videos/scheduled_pushes.mp4 \
        --seed 42 \
        --push_magnitude 120
"""

# ── 1. Pre-launch ─────────────────────────────────────────────────────────────

import argparse
import os
import sys

_IL_RSL_RL_DIR = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, _IL_RSL_RL_DIR)

from isaaclab.app import AppLauncher
import cli_args  # noqa: F401  (from _IL_RSL_RL_DIR)

parser = argparse.ArgumentParser(description="Play a Go2 push-recovery checkpoint with scheduled pushes.")
parser.add_argument("--video_path", type=str, required=True, help="Output MP4 path.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--push_magnitude", type=float, default=120.0, help="Scheduled push magnitude (N).")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0",
)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
if args_cli.checkpoint is None:
    raise ValueError("--checkpoint is required for scheduled-push playback.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Post-launch imports ────────────────────────────────────────────────────

import gymnasium as gym  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402


from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401, E402

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))
import isaaclab_go2_pushrecovery  # noqa: F401, E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


# ── 3. Play loop ──────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load checkpoint and run policy with scheduled pushes."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable random push event terms (impulse + sustained) for this replay only.
    for term_name in ("push_robot_impulse", "push_robot_sustained", "reset_push_event_states"):
        if hasattr(env_cfg.events, term_name):
            setattr(env_cfg.events, term_name, None)
    for term_name in (
        "push_impulse_force",
        "push_sustained_force",
        "push_impulse_count",
        "push_sustained_count",
    ):
        if hasattr(env_cfg.curriculum, term_name):
            setattr(env_cfg.curriculum, term_name, None)

    # Constant command: v_x=0.5 m/s, v_y=0, omega_z=0.
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # Initial viewer setup; will be updated each step to follow the robot.
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = (-2.5, -2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
        # Make the viewer track the robot's base body.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"

    video_path = os.path.abspath(args_cli.video_path)
    video_dir = os.path.dirname(video_path)
    os.makedirs(video_dir, exist_ok=True)
    name_prefix = os.path.splitext(os.path.basename(video_path))[0]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    step_dt = env.unwrapped.step_dt
    total_steps = 1250

    video_kwargs = {
        "video_folder": video_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": total_steps,
        "disable_logger": True,
        "name_prefix": name_prefix,
    }
    print("[INFO] Recording scheduled-push video.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    robot = env.unwrapped.scene["robot"]
    body_ids = robot.find_bodies("base")
    if not (isinstance(body_ids, (list, tuple)) and all(isinstance(x, int) for x in body_ids)):
        if isinstance(body_ids, str) and hasattr(robot, "body_names"):
            body_ids = [robot.body_names.index(body_ids)]
        elif hasattr(robot, "body_names"):
            body_ids = [robot.body_names.index("base")]
    env_ids = torch.tensor([0], device=robot.device, dtype=torch.long)

    push_times_s = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    push_duration_steps = int(round(0.3 / step_dt))
    push_dirs = [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    ]

    push_windows: list[tuple[int, int, tuple[float, float]]] = []
    for t_s, direction in zip(push_times_s, push_dirs):
        start_step = int(round(t_s / step_dt))
        end_step = start_step + push_duration_steps
        push_windows.append((start_step, end_step, direction))

    obs, _ = env.get_observations()
    timestep = 0
    active_force = torch.zeros(1, 1, 3, device=robot.device)

    while simulation_app.is_running() and timestep < total_steps:
        with torch.inference_mode():
            actions = policy(obs)

        # Apply scheduled push if within a window; otherwise zero the force.
        force = active_force.zero_()
        for start_step, end_step, direction in push_windows:
            if start_step <= timestep < end_step:
                force = active_force
                force[:, 0, 0] = args_cli.push_magnitude * direction[0]
                force[:, 0, 1] = args_cli.push_magnitude * direction[1]
                break

        robot.set_external_force_and_torque(force, torch.zeros_like(force), env_ids=env_ids, body_ids=body_ids)

        obs, _, _, _ = env.step(actions)


        if args_cli.real_time:
            time.sleep(step_dt)
        timestep += 1

    env.close()

    recorded_path = os.path.join(video_dir, f"{name_prefix}-step-0.mp4")
    if recorded_path != video_path and os.path.exists(recorded_path):
        os.replace(recorded_path, video_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
