from __future__ import annotations

import argparse
import datetime as dt
import re
import secrets
from pathlib import Path

import numpy as np

from .constants import DEFAULT_MJCF, DEFAULT_MOTION_PATH, DEFAULT_OUTPUT_ROOT
from .metrics import summarize, write_outputs
from .motion import load_motion_clips
from .policy import PolicyBank
from .runner import run_trial
from .scene import MujocoSoccer, make_experiment_xml
from .schemas import ExperimentConfig, TrialResult


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stage-2 soccer policy in a MuJoCo experiment.")
    parser.add_argument("--num-trials", "--experiment-times", type=int, default=20, help="Number of randomized trials.")
    parser.add_argument("--birth-radius", type=float, default=0.25, help="Robot spawn disk radius around the origin.")
    parser.add_argument(
        "--ball-forward-range",
        type=float,
        nargs=2,
        default=(0.5, 3.0),
        metavar=("MIN", "MAX"),
        help="Ball robot-local forward/x spawn range in front of the robot.",
    )
    parser.add_argument(
        "--ball-lateral-range",
        type=float,
        nargs=2,
        default=(-1.5, 1.5),
        metavar=("MIN", "MAX"),
        help="Ball robot-local lateral/y spawn range.",
    )
    parser.add_argument(
        "--goal",
        "--goal-xy",
        type=float,
        nargs=2,
        default=(6.0, 0.0),
        metavar=("X", "Y"),
        help="Goal center. Interpreted in robot local frame unless --goal-frame world is used.",
    )
    parser.add_argument("--goal-width", "--goal-length", type=float, default=2.0, help="Goal mouth width.")
    parser.add_argument("--enable-ball-vel", action="store_true", default=False, help="Enable randomized initial ball velocity.")
    parser.add_argument(
        "--ball-speed-range",
        "--ball-vel-speed-range",
        type=float,
        nargs=2,
        default=(0.1, 0.3),
        metavar=("MIN", "MAX"),
        help="Initial ball speed magnitude range. Direction is sampled uniformly in the robot local xy plane.",
    )
    parser.add_argument("--motion-path", type=Path, default=DEFAULT_MOTION_PATH, help="Motion .npz file or directory.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Base G1 actuator MJCF.")
    parser.add_argument("--policy", type=Path, default=None, help="Single exported ONNX policy.")
    parser.add_argument("--policy-dir", type=Path, default=None, help="Directory of per-motion exported ONNX policies.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for CSV and summary JSON.")
    parser.add_argument(
        "--experiment-name",
        "--run-name",
        type=str,
        default=None,
        help="Optional experiment name. Used in config.json and the default output directory name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit to generate a fresh seed from system entropy for each run.",
    )
    parser.add_argument("--sim-time", type=float, default=6.0, help="Seconds per trial.")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Policy/control period in seconds.")
    parser.add_argument(
        "--goal-frame",
        choices=("robot", "world"),
        default="robot",
        help="Coordinate frame for --goal. The ball spawn ranges are always robot-local.",
    )
    parser.add_argument(
        "--ball-speed-threshold",
        type=float,
        default=0.2,
        help="Minimum horizontal speed used for ball direction alignment angle.",
    )
    parser.add_argument(
        "--render",
        "--visualize",
        action="store_true",
        dest="render",
        help="Open a MuJoCo viewer window and pace the rollout in real time.",
    )
    parser.add_argument("--preload-policies", action="store_true", help="Load all ONNX policies before the first trial.")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Ignore ONNX policies and use reference joint trajectories as PD targets.",
    )
    parser.add_argument("--rnn-num-layers", type=int, default=2, help="Fallback recurrent layer count for dynamic ONNX shapes.")
    parser.add_argument("--rnn-hidden-dim", type=int, default=128, help="Fallback recurrent hidden dim for dynamic ONNX shapes.")
    parser.add_argument(
        "--ball-spawn-max-attempts",
        type=int,
        default=128,
        help="Maximum resampling attempts used to avoid initial ball-robot overlap.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    def parse_range(values: tuple[float, float], name: str) -> tuple[float, float]:
        lower = float(values[0])
        upper = float(values[1])
        if lower > upper:
            raise ValueError(f"{name} lower bound must be <= upper bound: got {lower} > {upper}")
        return lower, upper

    def sanitize_experiment_name(name: str) -> str:
        slug = re.sub(r"[^\w.-]+", "_", name.strip())
        return slug.strip("._-")

    experiment_name = args.experiment_name.strip() if args.experiment_name else None
    seed = int(args.seed) if args.seed is not None else secrets.randbits(63)
    ball_speed_range = parse_range(args.ball_speed_range, "--ball-speed-range")
    if ball_speed_range[0] < 0.0:
        raise ValueError(f"--ball-speed-range lower bound must be >= 0: got {ball_speed_range[0]}")
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif experiment_name:
        safe_name = sanitize_experiment_name(experiment_name)
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{safe_name or 'experiment'}"
    else:
        output_dir = DEFAULT_OUTPUT_ROOT / timestamp
    return ExperimentConfig(
        experiment_name=experiment_name,
        num_trials=args.num_trials,
        birth_radius=args.birth_radius,
        ball_forward_range=parse_range(args.ball_forward_range, "--ball-forward-range"),
        ball_lateral_range=parse_range(args.ball_lateral_range, "--ball-lateral-range"),
        goal=(float(args.goal[0]), float(args.goal[1])),
        goal_width=float(args.goal_width),
        enable_ball_vel=bool(args.enable_ball_vel),
        ball_speed_range=ball_speed_range,
        motion_path=args.motion_path.resolve(),
        mjcf=args.mjcf.resolve(),
        policy=args.policy.resolve() if args.policy is not None else None,
        policy_dir=args.policy_dir.resolve() if args.policy_dir is not None else None,
        output_dir=output_dir.resolve(),
        seed=seed,
        sim_time=float(args.sim_time),
        control_dt=float(args.control_dt),
        goal_frame=args.goal_frame,
        ball_speed_threshold=float(args.ball_speed_threshold),
        render=bool(args.render),
        preload_policies=bool(args.preload_policies),
        reference_only=bool(args.reference_only),
        rnn_num_layers=int(args.rnn_num_layers),
        rnn_hidden_dim=int(args.rnn_hidden_dim),
        ball_spawn_max_attempts=int(args.ball_spawn_max_attempts),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    rng = np.random.default_rng(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = make_experiment_xml(cfg.mjcf, cfg.output_dir, cfg.goal_width)
    clips = load_motion_clips(cfg.motion_path)
    env = MujocoSoccer(xml_path)
    policy_bank = PolicyBank(cfg)

    print(f"[INFO] Loaded {len(clips)} motions from {cfg.motion_path}")
    if cfg.reference_only or (cfg.policy is None and cfg.policy_dir is None):
        print("[INFO] Running reference-only PD controller. Pass --policy-dir or --policy to use stage-2 ONNX weights.")
    elif cfg.policy_dir is not None:
        print(f"[INFO] Using ONNX policy directory: {cfg.policy_dir}")
    else:
        print(f"[INFO] Using ONNX policy: {cfg.policy}")
    print(f"[INFO] Writing results to: {cfg.output_dir}")
    if cfg.experiment_name:
        print(f"[INFO] Experiment name: {cfg.experiment_name}")
    print(f"[INFO] Random seed: {cfg.seed}")

    viewer = None
    try:
        if cfg.render:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(env.model, env.data)

        results: list[TrialResult] = []
        for trial in range(cfg.num_trials):
            result = run_trial(trial, cfg, rng, env, clips, policy_bank, viewer=viewer)
            results.append(result)
            print(
                "[TRIAL {trial:04d}] motion={motion} success={success} "
                "spawn_attempts={spawn_attempts} E_gmpjpe={E_gmpjpe:.2f} E_mpjpe={E_mpjpe:.2f} "
                "E_vel={E_vel:.2f} E_acc={E_acc:.2f} "
                "angle_rad={angle:.4f} max_ball_speed={speed:.3f}".format(
                    trial=trial,
                    motion=result.motion,
                    success=int(result.success),
                    spawn_attempts=result.spawn_attempts,
                    E_gmpjpe=result.E_gmpjpe,
                    E_mpjpe=result.E_mpjpe,
                    E_vel=result.E_vel,
                    E_acc=result.E_acc,
                    angle=result.ball_vel_dir_alignment_angle,
                    speed=result.max_ball_speed,
                )
            )

        write_outputs(cfg, xml_path, results)
        summary = summarize(results)
        print("[SUMMARY]")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    finally:
        if viewer is not None:
            viewer.close()
