# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Control vx/wz/height; J triggers jumps and K toggles actuator power.",
)
parser.add_argument("--power_ramp_time", type=float, default=0.60)
parser.add_argument("--power_extend_time", type=float, default=0.80)
parser.add_argument("--power_stabilize_time", type=float, default=0.60)
parser.add_argument("--power_handoff_time", type=float, default=1.00)
parser.add_argument("--power_target_length", type=float, default=0.21)
parser.add_argument(
    "--power_restart_max_tilt",
    type=float,
    default=0.60,
    help="Refuse K-key restart above this tilt in radians; this is not self-righting.",
)
parser.add_argument(
    "--eval_pushes",
    action="store_true",
    default=False,
    help=(
        "Keep the task's periodic push event enabled during Play. Useful for "
        "Recovery/Terrain evaluation; disabled by default for deterministic Play."
    ),
)
parser.add_argument("--keyboard_height", type=float, default=0.21, help="Initial body height in keyboard mode.")
parser.add_argument(
    "--keyboard_height_step", type=float, default=0.01, help="Body-height increment for each R/F key press."
)
parser.add_argument(
    "--keyboard_terrain",
    choices=("task", "flat", "slope", "stairs", "mixed"),
    default="task",
    help=(
        "Terrain preset for keyboard testing: keep the task terrain, or replace "
        "it with flat, slope, stairs, or mixed. Non-task presets require --keyboard."
    ),
)
parser.add_argument(
    "--keyboard_terrain_difficulty",
    type=float,
    default=0.5,
    help=(
        "Difficulty in [0, 1] for slope/stairs terrain. It controls slope angle "
        "or stair height; default is a moderate setting."
    ),
)
parser.add_argument(
    "--command_range",
    type=float,
    default=0.1,
    help="Wheel-legged vx limit. Use the vx_lim reached by the loaded checkpoint.",
)
parser.add_argument(
    "--yaw_command_range",
    type=float,
    default=0.5,
    help="Wheel-legged wz limit. Use the wz_lim reached by the loaded checkpoint.",
)
parser.add_argument(
    "--jump_height",
    type=float,
    default=None,
    help="Fix jump target height in meters (for example 0.12 for clearance evaluation).",
)
parser.add_argument(
    "--jump_distance",
    type=float,
    default=None,
    help="Fix the forward landing-target distance in meters.",
)
parser.add_argument(
    "--obstacle_height",
    type=float,
    default=None,
    help=(
        "Use one exact exposed obstacle height in meters during Oracle obstacle "
        "Play (supported range: 0.0 to 0.08)."
    ),
)
parser.add_argument(
    "--obstacle_width",
    type=float,
    default=None,
    help=(
        "Use one exact obstacle width in meters during Oracle obstacle Play "
        "(must be positive)."
    ),
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=100,
    help="Print compact environment diagnostics every N play steps; <=0 disables it.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
for duration_name in (
    "power_ramp_time",
    "power_extend_time",
    "power_stabilize_time",
    "power_handoff_time",
):
    if getattr(args_cli, duration_name) <= 0.0:
        parser.error(f"--{duration_name} must be positive.")
if args_cli.power_restart_max_tilt <= 0.0:
    parser.error("--power_restart_max_tilt must be positive.")
# RTX cameras are only needed for video. The perceptive task uses the scalable
# ray-cast depth backend and therefore remains available in normal headless mode.
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_legged_robot.tasks  # noqa: F401
from wheel_legged_robot.tasks.manager_based.wheel_legged_robot.mdp.power_on import (
    PowerOnStandCfg,
    PowerOnStandController,
)



def _configure_keyboard_terrain(env_cfg: ManagerBasedRLEnvCfg, terrain_name: str, difficulty: float) -> None:
    """Replace the flat test plane with a small deterministic terrain island.

    The training tasks remain unchanged: this helper is called only by Play and
    is intended for one-robot interactive testing.  The center of every
    generated terrain is a flat platform, so the robot starts safely and the
    keyboard can drive it onto the slope or stairs.
    """
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("--keyboard_terrain_difficulty must be in [0, 1].")
    if terrain_name == "task":
        print("[INFO] Keyboard test terrain: keep the task's configured terrain.")
        return
    if terrain_name == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        print("[INFO] Keyboard test terrain: flat plane.")
        return

    # Keep the interactive test compact: one terrain island, a central safe
    # platform, and enough room to approach the features from multiple sides.
    slope_angle = 0.08 + 0.16 * difficulty
    step_height = 0.025 + 0.035 * difficulty
    if terrain_name == "slope":
        sub_terrains = {
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(slope_angle, slope_angle),
                platform_width=2.5,
                border_width=0.35,
            )
        }
    elif terrain_name == "stairs":
        sub_terrains = {
            "stairs": terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(step_height, step_height),
                step_width=0.35,
                platform_width=2.5,
                border_width=0.35,
            )
        }
    else:  # mixed
        sub_terrains = {
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.35,
                slope_range=(slope_angle, slope_angle),
                platform_width=2.5,
                border_width=0.35,
            ),
            "stairs": terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.35,
                step_height_range=(step_height, step_height),
                step_width=0.35,
                platform_width=2.5,
                border_width=0.35,
            ),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.30,
                noise_range=(0.01, 0.02 + 0.04 * difficulty),
                noise_step=0.02,
                border_width=0.35,
            ),
        }
    generator = TerrainGeneratorCfg(
        size=(12.0, 12.0),
        border_width=2.0,
        border_height=1.0,
        num_rows=1,
        num_cols=1,
        curriculum=False,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        color_scheme="height",
        use_cache=False,
        sub_terrains=sub_terrains,
    )
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = generator
    print(
        "[INFO] Keyboard test terrain: "
        f"{terrain_name} (difficulty={difficulty:.2f}, "
        f"slope={slope_angle:.3f} rad, step_height={step_height:.3f} m)."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # Use a deterministic, evaluation-friendly environment configuration.
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    if args_cli.keyboard_terrain != "task" and not args_cli.keyboard:
        raise ValueError(
            "--keyboard_terrain other than 'task' requires --keyboard so the "
            "single-robot terrain test is not accidentally used for batch Play."
        )
    if args_cli.keyboard:
        _configure_keyboard_terrain(
            env_cfg,
            args_cli.keyboard_terrain,
            args_cli.keyboard_terrain_difficulty,
        )

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
        env_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(env_cfg.events, "push_robot") and not args_cli.eval_pushes:
        env_cfg.events.push_robot = None
    elif args_cli.eval_pushes and hasattr(env_cfg.events, "push_robot"):
        push_cfg = env_cfg.events.push_robot
        if push_cfg is None:
            raise ValueError(
                f"Task {task_name!r} does not configure a periodic push event."
            )
        print(
            "[INFO] Periodic evaluation pushes enabled: "
            f"interval={push_cfg.interval_range_s} s, "
            f"max_push_vel_xy={push_cfg.params.get('max_push_vel_xy')}"
        )

    if hasattr(env_cfg.commands, "wheel_legged_commands"):
        command_cfg = env_cfg.commands.wheel_legged_commands
        if (
            task_name in (
                "Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0",
                "Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0",
            )
            and not args_cli.keyboard
        ):
            command_cfg.ranges.lin_vel_x = (
                0.45,
                max(0.45, args_cli.command_range),
            )
        else:
            command_cfg.ranges.lin_vel_x = (
                -args_cli.command_range,
                args_cli.command_range,
            )
        command_cfg.ranges.ang_vel_yaw = (
            -args_cli.yaw_command_range,
            args_cli.yaw_command_range,
        )
        print(
            "[INFO] Wheel-legged play command ranges: "
            f"vx={command_cfg.ranges.lin_vel_x} m/s, "
            f"wz={command_cfg.ranges.ang_vel_yaw} rad/s"
        )
    if args_cli.jump_height is not None and hasattr(env_cfg.commands, "jump_command"):
        jump_height = max(0.0, args_cli.jump_height)
        env_cfg.commands.jump_command.target_height_range = (jump_height, jump_height)
        print(f"[INFO] Fixed jump target height: {jump_height:.3f} m")
    if args_cli.jump_distance is not None and hasattr(env_cfg.commands, "jump_command"):
        jump_distance = float(args_cli.jump_distance)
        env_cfg.commands.jump_command.target_distance_range = (
            jump_distance,
            jump_distance,
        )
        # An explicit evaluation target takes precedence over the training
        # stage's automatic velocity-to-distance coupling.
        env_cfg.commands.jump_command.couple_distance_to_velocity = False
        print(f"[INFO] Fixed jump target distance: {jump_distance:.3f} m")
    if args_cli.obstacle_height is not None or args_cli.obstacle_width is not None:
        obstacle_cfg = getattr(env_cfg, "obstacle_oracle", None)
        if obstacle_cfg is None:
            raise ValueError(
                "--obstacle_height/--obstacle_width require "
                "an Oracle or Perceptive obstacle task."
            )
        if args_cli.obstacle_height is not None:
            obstacle_height = float(args_cli.obstacle_height)
            if not 0.0 <= obstacle_height <= obstacle_cfg.obstacle_max_height:
                raise ValueError(
                    "--obstacle_height must be in [0.0, "
                    f"{obstacle_cfg.obstacle_max_height:.3f}] m."
                )
            obstacle_cfg.fixed_height = obstacle_height
        if args_cli.obstacle_width is not None:
            obstacle_width = float(args_cli.obstacle_width)
            if obstacle_width <= 0.0:
                raise ValueError("--obstacle_width must be positive.")
            obstacle_cfg.fixed_width = obstacle_width
        shown_height = (
            f"{obstacle_cfg.fixed_height:.3f} m"
            if obstacle_cfg.fixed_height is not None
            else "curriculum level 0"
        )
        shown_width = (
            f"{obstacle_cfg.fixed_width:.3f} m"
            if obstacle_cfg.fixed_width is not None
            else "curriculum level 0"
        )
        print(
            "[INFO] Oracle obstacle Play geometry: "
            f"height={shown_height}, width={shown_width}."
        )
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "command_levels"):
        env_cfg.curriculum.command_levels = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "speed_levels"):
        # Evaluation uses the explicit --command_range selected by the user;
        # do not let a freshly initialized curriculum force it back to 0.2.
        env_cfg.curriculum.speed_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        # Interactive power loss must not be hidden by an automatic upright
        # reset as the chassis settles. The user can still restart Play if the
        # pose is outside the controller's recoverable range.
        for termination_name in (
            "time_out",
            "bad_orientation",
            "joint_limits",
            "minimum_height",
        ):
            if hasattr(env_cfg.terminations, termination_name):
                setattr(env_cfg.terminations, termination_name, None)
        command_cfg = env_cfg.commands.wheel_legged_commands
        command_cfg.heading_command = False
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.debug_vis = False
        if hasattr(env_cfg.commands, "jump_command"):
            # The keyboard controller supplies the trigger; suppress random
            # autonomous jumps before CommandManager is initialized.
            env_cfg.commands.jump_command.jump_probability = 0.0
            env_cfg.commands.jump_command.resampling_time_range = (1.0e9, 1.0e9)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    keyboard_controller = None
    if args_cli.keyboard:
        from rl_utils import WheelLeggedKeyboardController

        keyboard_controller = WheelLeggedKeyboardController(
            env,
            vx_limit=args_cli.command_range,
            wz_limit=args_cli.yaw_command_range,
            default_height=args_cli.keyboard_height,
            height_step=args_cli.keyboard_height_step,
            jump_height=args_cli.jump_height,
            jump_distance=args_cli.jump_distance,
        )

    command_term = env.unwrapped.command_manager._terms.get("wheel_legged_commands")
    # Debug markers are useful in an interactive viewport, but headless
    # recording has no viewport and may emit callbacks during environment
    # teardown after ``scene`` has already been released.
    if command_term is not None and not args_cli.keyboard and not args_cli.headless:
        command_term.set_debug_vis(True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    vmc_term = env.unwrapped.action_manager.get_term("vmc")
    power_mode = "on"
    power_controller = None
    power_contact_sensor = None
    power_wheel_body_ids = None
    if keyboard_controller is not None:
        power_contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        power_wheel_body_ids, _ = power_contact_sensor.find_bodies(".*wheel.*")
        if len(power_wheel_body_ids) != 2:
            raise RuntimeError("Keyboard power cycle requires exactly two wheel bodies.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    from rl_utils import camera_follow, format_debug_metrics

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if keyboard_controller is not None:
            keyboard_controller.update()
            if keyboard_controller.consume_power_cycle_request():
                if power_mode == "off":
                    gravity_b = env.unwrapped.scene["robot"].data.projected_gravity_b
                    restart_tilt = torch.acos(
                        torch.clamp(-gravity_b[:, 2], -1.0, 1.0)
                    )
                    wheel_forces = power_contact_sensor.data.net_forces_w[
                        :, power_wheel_body_ids
                    ]
                    wheels_grounded = (
                        torch.linalg.vector_norm(wheel_forces, dim=-1) > 2.0
                    ).all(dim=1)
                    if (
                        bool(
                            torch.any(
                                restart_tilt > args_cli.power_restart_max_tilt
                            ).item()
                        )
                        or not bool(torch.all(wheels_grounded).item())
                    ):
                        print(
                            "[POWER] Restart refused: wait for both wheels to contact "
                            "the ground and keep tilt below "
                            f"{args_cli.power_restart_max_tilt:.2f} rad. "
                            "A fully fallen robot requires Self-righting."
                        )
                        keyboard_controller.stop_commands(cancel_jump=True)
                        obs = env.get_observations()
                        continue
                    power_controller = PowerOnStandController(
                        env.unwrapped.num_envs,
                        env.unwrapped.device,
                        vmc_term.cfg,
                        # One control frame captures the final passive geometry
                        # before the actuator-output ramp begins.
                        passive_durations=dt,
                        cfg=PowerOnStandCfg(
                            torque_ramp_time=args_cli.power_ramp_time,
                            extend_time=args_cli.power_extend_time,
                            stabilize_time=args_cli.power_stabilize_time,
                            handoff_time=args_cli.power_handoff_time,
                            target_leg_length=args_cli.power_target_length,
                        ),
                    )
                    power_mode = "starting"
                    print("[POWER] Restart requested: beginning safe power-on sequence.")
                else:
                    power_mode = "off"
                    power_controller = None
                    keyboard_controller.cancel_active_jump()
                    vmc_term.reset()
                    vmc_term.set_motor_enable_scale(0.0)
                    reset_mask = torch.ones(
                        env.unwrapped.num_envs,
                        dtype=torch.bool,
                        device=env.unwrapped.device,
                    )
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(reset_mask)
                    else:
                        policy_nn.reset(reset_mask)
                    print(
                        "[POWER] Actuator output disabled. Press K again to restart."
                    )
            if power_mode != "on":
                keyboard_controller.stop_commands(cancel_jump=True)
            # Make the new command visible to the policy in the same frame.
            obs = env.get_observations()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            policy_actions = policy(obs)
            if power_mode == "off":
                actions = torch.zeros_like(policy_actions)
                vmc_term.set_motor_enable_scale(0.0)
            elif power_mode == "starting":
                gravity_b = env.unwrapped.scene["robot"].data.projected_gravity_b
                base_pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
                base_pitch_rate = env.unwrapped.scene["robot"].data.root_ang_vel_b[:, 1]
                actions, motor_scale, phase, handoff_started = power_controller.compute(
                    env.unwrapped.L0,
                    base_pitch=base_pitch,
                    base_pitch_rate=base_pitch_rate,
                    policy_actions=policy_actions,
                )
                if handoff_started.any():
                    handoff_ids = handoff_started.nonzero(as_tuple=False).squeeze(-1)
                    vmc_term.reset(handoff_ids)
                vmc_term.set_motor_enable_scale(motor_scale)
                power_controller.advance(dt)
                if torch.all(phase == PowerOnStandController.COMPLETE):
                    power_mode = "on"
                    power_controller = None
                    vmc_term.set_motor_enable_scale(1.0)
                    print("[POWER] Policy handoff complete; keyboard control restored.")
            else:
                actions = policy_actions
                vmc_term.set_motor_enable_scale(1.0)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.print_interval > 0 and timestep % args_cli.print_interval == 0:
            debug_line = format_debug_metrics(extras.get("log", {}))
            if debug_line:
                print(debug_line)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
