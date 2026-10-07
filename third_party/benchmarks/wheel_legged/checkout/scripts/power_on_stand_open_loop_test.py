# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop power-on stand and policy-handoff test.

Each vectorized environment starts with zero actuator output, settles under
gravity, ramps VMC torque, extends to a standing leg length and optionally
blends into an exported locomotion policy.  No PPO training is performed.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def _float_list(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated floats, got {value!r}."
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Wheel-Legged-Recovery-Flat-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--passive_times", type=_float_list, default=[0.4, 0.7])
parser.add_argument("--initial_heights", type=_float_list, default=[0.20, 0.25])
parser.add_argument("--initial_roll_degs", type=_float_list, default=[-5.0, 0.0, 5.0])
parser.add_argument("--initial_pitch_degs", type=_float_list, default=[-8.0, 0.0, 8.0])
parser.add_argument(
    "--left_leg_joint_offsets",
    type=_float_list,
    default=[0.0, 0.12],
    help="Asymmetric left-leg joint offsets in radians.",
)
parser.add_argument("--torque_ramp_time", type=float, default=0.60)
parser.add_argument("--extend_time", type=float, default=0.80)
parser.add_argument("--stabilize_time", type=float, default=0.60)
parser.add_argument("--handoff_time", type=float, default=1.00)
parser.add_argument(
    "--post_handoff_time",
    type=float,
    default=1.50,
    help="Evaluation time after the policy blend has completed.",
)
parser.add_argument("--target_length", type=float, default=0.21)
parser.add_argument("--balance_pitch_kp", type=float, default=3.0)
parser.add_argument("--balance_pitch_kd", type=float, default=0.45)
parser.add_argument("--balance_wheel_action_limit", type=float, default=0.65)
parser.add_argument("--balance_leg_angle_gain", type=float, default=-1.0)
parser.add_argument("--balance_leg_action_limit", type=float, default=0.80)
parser.add_argument("--success_tilt", type=float, default=0.20)
parser.add_argument("--success_height_min", type=float, default=0.17)
parser.add_argument("--success_height_max", type=float, default=0.25)
parser.add_argument("--success_ang_vel", type=float, default=0.50)
parser.add_argument("--success_hold_time", type=float, default=0.50)
parser.add_argument("--contact_threshold", type=float, default=2.0)
parser.add_argument("--trace_stride", type=int, default=5)
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Optional exported RSL-RL JIT policy.pt for the HANDOFF phase.",
)
parser.add_argument(
    "--policy_observation_dims",
    type=int,
    default=None,
    help="Optionally pass only the first N policy observations to a legacy JIT policy.",
)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.trace_stride <= 0:
    parser.error("--trace_stride must be positive.")
if args_cli.post_handoff_time <= 0.0:
    parser.error("--post_handoff_time must be positive.")

cases = [
    {
        "case_id": index,
        "passive_time_s": values[0],
        "initial_height_m": values[1],
        "initial_roll_deg": values[2],
        "initial_pitch_deg": values[3],
        "left_leg_joint_offset_rad": values[4],
    }
    for index, values in enumerate(
        itertools.product(
            args_cli.passive_times,
            args_cli.initial_heights,
            args_cli.initial_roll_degs,
            args_cli.initial_pitch_degs,
            args_cli.left_leg_joint_offsets,
        )
    )
]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim must be running before the remaining imports."""

import csv
import json

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import wheel_legged_robot.tasks  # noqa: F401
from wheel_legged_robot.tasks.manager_based.wheel_legged_robot.mdp.power_on import (
    PowerOnStandCfg,
    PowerOnStandController,
)


def _disable_training_features(env_cfg) -> None:
    """Keep physics deterministic and prevent reset during the one-shot test."""
    for name in (
        "randomize_material",
        "randomize_base_mass",
        "randomize_other_mass",
        "randomize_com",
        "randomize_joints_pos",
        "push_robot",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        for ranges in (
            env_cfg.events.reset_base.params["pose_range"],
            env_cfg.events.reset_base.params["velocity_range"],
        ):
            for key in ranges:
                ranges[key] = (0.0, 0.0)
    for name in ("time_out", "bad_orientation", "joint_limits", "minimum_height"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    if hasattr(env_cfg, "curriculum"):
        for name in ("command_levels", "terrain_levels", "speed_levels"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    command_cfg = env_cfg.commands.wheel_legged_commands
    command_cfg.heading_command = False
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.ang_vel_yaw = (0.0, 0.0)
    command_cfg.ranges.heading = (0.0, 0.0)
    command_cfg.ranges.height = (args_cli.target_length, args_cli.target_length)
    command_cfg.debug_vis = False

    env_cfg.actions.vmc.randomize_kp = False
    env_cfg.actions.vmc.randomize_kd = False
    env_cfg.actions.vmc.randomize_torque_scale = False


def _policy_actions(policy, observations, num_envs: int, device) -> torch.Tensor | None:
    if policy is None:
        return None
    policy_observation = observations["policy"] if isinstance(observations, dict) else observations
    if args_cli.policy_observation_dims is not None:
        policy_observation = policy_observation[:, : args_cli.policy_observation_dims]
    actions = policy(policy_observation)
    if isinstance(actions, (tuple, list)):
        actions = actions[0]
    actions = actions.to(device=device, dtype=torch.float32)
    if actions.shape != (num_envs, 6):
        raise RuntimeError(
            f"Policy returned {tuple(actions.shape)}, expected ({num_envs}, 6)."
        )
    return actions


def _write_initial_states(unwrapped, robot, vmc_term, device) -> None:
    """Apply the deterministic case grid before passive settling."""
    num_envs = len(cases)
    root_pose = robot.data.root_pose_w.clone()
    env_origins = unwrapped.scene.env_origins
    root_pose[:, 2] = env_origins[:, 2] + torch.tensor(
        [case["initial_height_m"] for case in cases], device=device
    )
    roll = torch.deg2rad(
        torch.tensor([case["initial_roll_deg"] for case in cases], device=device)
    )
    pitch = torch.deg2rad(
        torch.tensor([case["initial_pitch_deg"] for case in cases], device=device)
    )
    yaw = torch.zeros(num_envs, device=device)
    root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    offsets = torch.tensor(
        [case["left_leg_joint_offset_rad"] for case in cases], device=device
    )
    left_hip_id, left_knee_id = vmc_term.joint_ids[0], vmc_term.joint_ids[1]
    joint_pos[:, left_hip_id] += offsets
    joint_pos[:, left_knee_id] -= offsets
    limits = robot.data.soft_joint_pos_limits
    joint_pos = torch.maximum(joint_pos, limits[..., 0] + 1.0e-3)
    joint_pos = torch.minimum(joint_pos, limits[..., 1] - 1.0e-3)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    vmc_term.reset()
    vmc_term.set_motor_enable_scale(0.0)


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else project_root / "power_on_stand_results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=len(cases),
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    _disable_training_features(env_cfg)
    env_cfg.scene.env_spacing = 2.0
    longest_passive = max(case["passive_time_s"] for case in cases)
    test_after_handoff = args_cli.post_handoff_time
    env_cfg.episode_length_s = (
        longest_passive
        + args_cli.torque_ramp_time
        + args_cli.extend_time
        + args_cli.stabilize_time
        + args_cli.handoff_time
        + test_after_handoff
        + 1.0
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    vmc_term = unwrapped.action_manager.get_term("vmc")

    wheel_ids, wheel_names = contact_sensor.find_bodies(".*wheel.*")
    base_ids, base_names = contact_sensor.find_bodies("base_link")
    if len(wheel_ids) != 2 or len(base_ids) != 1:
        raise RuntimeError(
            "Power-on test requires two wheel contact bodies and one base_link: "
            f"wheels={wheel_names}, base={base_names}."
        )
    leg_local_ids = [0, 1, 3, 4]
    leg_asset_ids = [vmc_term.joint_ids[index] for index in leg_local_ids]

    policy = None
    policy_path = None
    if args_cli.policy:
        policy_path = Path(args_cli.policy).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Exported JIT policy not found: {policy_path}")
        policy = torch.jit.load(str(policy_path), map_location=device).eval()

    observations, _ = env.reset()
    _write_initial_states(unwrapped, robot, vmc_term, device)
    passive_durations = torch.tensor(
        [case["passive_time_s"] for case in cases], device=device
    )
    controller_cfg = PowerOnStandCfg(
        torque_ramp_time=args_cli.torque_ramp_time,
        extend_time=args_cli.extend_time,
        stabilize_time=args_cli.stabilize_time,
        handoff_time=args_cli.handoff_time,
        target_leg_length=args_cli.target_length,
        balance_pitch_kp=args_cli.balance_pitch_kp,
        balance_pitch_kd=args_cli.balance_pitch_kd,
        balance_wheel_action_limit=args_cli.balance_wheel_action_limit,
        balance_leg_angle_gain=args_cli.balance_leg_angle_gain,
        balance_leg_action_limit=args_cli.balance_leg_action_limit,
    )
    controller = PowerOnStandController(
        len(cases), device, vmc_term.cfg, passive_durations, controller_cfg
    )
    step_dt = float(unwrapped.step_dt)
    total_time = float(controller.total_duration.max().item()) + test_after_handoff
    total_steps = max(1, round(total_time / step_dt))
    success_hold_steps = max(1, round(args_cli.success_hold_time / step_dt))

    max_tilt = torch.zeros(len(cases), device=device)
    max_abs_vz = torch.zeros(len(cases), device=device)
    max_ang_vel = torch.zeros(len(cases), device=device)
    max_torque_saturation = torch.zeros(len(cases), device=device)
    min_joint_margin = torch.full((len(cases),), torch.inf, device=device)
    max_base_contact_force = torch.zeros(len(cases), device=device)
    max_wheel_contact_force = torch.zeros(len(cases), device=device)
    minimum_height = torch.full((len(cases),), torch.inf, device=device)
    stable_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    first_success_time = torch.full((len(cases),), torch.nan, device=device)
    handoff_seen = torch.zeros(len(cases), dtype=torch.bool, device=device)
    post_handoff_stable_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    post_handoff_total_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    trace_rows: list[dict[str, object]] = []

    print(f"[INFO] Running {len(cases)} power-on stand cases.")
    print(
        "[INFO] Handoff mode: "
        + (f"RSL-RL JIT policy ({policy_path})" if policy is not None else "deterministic stand hold")
    )
    print(
        "[INFO] Schedule: "
        f"passive={args_cli.passive_times}s, ramp={args_cli.torque_ramp_time:.2f}s, "
        f"extend={args_cli.extend_time:.2f}s, stabilize={args_cli.stabilize_time:.2f}s, "
        f"handoff={args_cli.handoff_time:.2f}s"
    )

    with torch.inference_mode():
        for step in range(total_steps):
            policy_action = _policy_actions(policy, observations, len(cases), device)
            gravity_b = robot.data.projected_gravity_b
            base_pitch = torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])
            base_roll = torch.atan2(-gravity_b[:, 1], -gravity_b[:, 2])
            base_pitch_rate = robot.data.root_ang_vel_b[:, 1]
            actions, motor_scale, phase, handoff_started = controller.compute(
                unwrapped.L0,
                base_pitch=base_pitch,
                base_pitch_rate=base_pitch_rate,
                policy_actions=policy_action,
            )
            if handoff_started.any():
                handoff_ids = handoff_started.nonzero(as_tuple=False).squeeze(-1)
                # Clear wheel PI wind-up and stale action targets before blending.
                vmc_term.reset(handoff_ids)
                handoff_seen[handoff_ids] = True
            vmc_term.set_motor_enable_scale(motor_scale)
            observations, _, _, _, _ = env.step(actions)
            controller.advance(step_dt)

            tilt = torch.acos(
                torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
            )
            base_height = unwrapped.base_height
            vz = robot.data.root_lin_vel_w[:, 2]
            angular_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=1)
            forces = contact_sensor.data.net_forces_w
            wheel_force = torch.linalg.vector_norm(forces[:, wheel_ids], dim=-1)
            base_force = torch.linalg.vector_norm(forces[:, base_ids[0]], dim=-1)
            wheel_contact = wheel_force > args_cli.contact_threshold
            base_contact = base_force > args_cli.contact_threshold
            joint_pos = robot.data.joint_pos[:, leg_asset_ids]
            joint_limits = robot.data.soft_joint_pos_limits[:, leg_asset_ids]
            joint_margin = torch.minimum(
                joint_pos - joint_limits[..., 0],
                joint_limits[..., 1] - joint_pos,
            ).min(dim=1).values

            stable_now = (
                (phase >= PowerOnStandController.STABILIZE)
                & (tilt < args_cli.success_tilt)
                & (base_height >= args_cli.success_height_min)
                & (base_height <= args_cli.success_height_max)
                & wheel_contact.all(dim=1)
                & (~base_contact)
                & (angular_speed < args_cli.success_ang_vel)
            )
            stable_steps = torch.where(
                stable_now, stable_steps + 1, torch.zeros_like(stable_steps)
            )
            new_success = torch.isnan(first_success_time) & (
                stable_steps >= success_hold_steps
            )
            first_success_time[new_success] = controller.elapsed[new_success]
            post_handoff = phase == PowerOnStandController.COMPLETE
            post_handoff_total_steps += post_handoff.long()
            post_handoff_stable_steps += (post_handoff & stable_now).long()

            max_tilt = torch.maximum(max_tilt, tilt)
            max_abs_vz = torch.maximum(max_abs_vz, vz.abs())
            max_ang_vel = torch.maximum(max_ang_vel, angular_speed)
            max_torque_saturation = torch.maximum(
                max_torque_saturation, vmc_term.torque_saturation
            )
            min_joint_margin = torch.minimum(min_joint_margin, joint_margin)
            max_base_contact_force = torch.maximum(max_base_contact_force, base_force)
            max_wheel_contact_force = torch.maximum(
                max_wheel_contact_force, wheel_force.max(dim=1).values
            )
            minimum_height = torch.minimum(minimum_height, base_height)

            if step % args_cli.trace_stride == 0:
                for env_id in range(len(cases)):
                    trace_rows.append(
                        {
                            "time_s": float(controller.elapsed[env_id].item()),
                            "case_id": env_id,
                            "phase": PowerOnStandController.PHASE_NAMES[int(phase[env_id].item())],
                            "motor_enable_scale": float(motor_scale[env_id].item()),
                            "base_height_m": float(base_height[env_id].item()),
                            "base_vz_mps": float(vz[env_id].item()),
                            "tilt_rad": float(tilt[env_id].item()),
                            "roll_rad": float(base_roll[env_id].item()),
                            "pitch_rad": float(base_pitch[env_id].item()),
                            "pitch_rate_radps": float(base_pitch_rate[env_id].item()),
                            "angular_speed_radps": float(angular_speed[env_id].item()),
                            "L0_left_m": float(unwrapped.L0[env_id, 0].item()),
                            "L0_right_m": float(unwrapped.L0[env_id, 1].item()),
                            "wheel_action": float(actions[env_id, 2].item()),
                            "wheel_vel_left_radps": float(
                                robot.data.joint_vel[env_id, vmc_term.joint_ids[2]].item()
                            ),
                            "wheel_vel_right_radps": float(
                                robot.data.joint_vel[env_id, vmc_term.joint_ids[5]].item()
                            ),
                            "wheel_contact_count": int(wheel_contact[env_id].sum().item()),
                            "base_contact": bool(base_contact[env_id].item()),
                            "base_contact_force_N": float(base_force[env_id].item()),
                            "max_wheel_force_N": float(wheel_force[env_id].max().item()),
                            "torque_saturation": float(vmc_term.torque_saturation[env_id].item()),
                            "joint_margin_min_rad": float(joint_margin[env_id].item()),
                            "stable": bool(stable_now[env_id].item()),
                        }
                    )

    final_tilt = torch.acos(
        torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    final_height = unwrapped.base_height.clone()
    final_ang_vel = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=1)
    final_forces = contact_sensor.data.net_forces_w
    final_wheel_contact = (
        torch.linalg.vector_norm(final_forces[:, wheel_ids], dim=-1)
        > args_cli.contact_threshold
    )
    final_base_contact = (
        torch.linalg.vector_norm(final_forces[:, base_ids[0]], dim=-1)
        > args_cli.contact_threshold
    )
    post_handoff_stable_fraction = post_handoff_stable_steps.float() / torch.clamp(
        post_handoff_total_steps, min=1
    )
    success = (
        (post_handoff_stable_fraction >= 0.80)
        & (final_tilt < args_cli.success_tilt)
        & (final_height >= args_cli.success_height_min)
        & (final_height <= args_cli.success_height_max)
        & (final_ang_vel < args_cli.success_ang_vel)
    )

    metrics = {
        "settled_L0_left_m": _to_float_list(controller.start_lengths[:, 0]),
        "settled_L0_right_m": _to_float_list(controller.start_lengths[:, 1]),
        "minimum_height_m": _to_float_list(minimum_height),
        "final_height_m": _to_float_list(final_height),
        "final_tilt_rad": _to_float_list(final_tilt),
        "final_angular_speed_radps": _to_float_list(final_ang_vel),
        "final_wheel_contact_count": final_wheel_contact.sum(dim=1).detach().cpu().tolist(),
        "final_base_contact": final_base_contact.detach().cpu().tolist(),
        "post_handoff_stable_fraction": _to_float_list(post_handoff_stable_fraction),
        "max_tilt_rad": _to_float_list(max_tilt),
        "max_abs_vz_mps": _to_float_list(max_abs_vz),
        "max_angular_speed_radps": _to_float_list(max_ang_vel),
        "max_torque_saturation": _to_float_list(max_torque_saturation),
        "minimum_joint_margin_rad": _to_float_list(min_joint_margin),
        "max_base_contact_force_N": _to_float_list(max_base_contact_force),
        "max_wheel_contact_force_N": _to_float_list(max_wheel_contact_force),
        "first_success_time_s": _to_float_list(first_success_time),
        "handoff_seen": handoff_seen.detach().cpu().tolist(),
        "success": success.detach().cpu().tolist(),
    }
    summary_rows: list[dict[str, object]] = []
    for case in cases:
        env_id = case["case_id"]
        row = dict(case)
        for key, values in metrics.items():
            row[key] = values[env_id]
        summary_rows.append(row)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    trace_path = output_dir / "trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    finite_success_times = first_success_time[torch.isfinite(first_success_time)]
    metadata = {
        "task": args_cli.task,
        "timestamp": timestamp,
        "num_cases": len(cases),
        "policy_path": str(policy_path) if policy_path else None,
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_first_success_time_s": (
            float(finite_success_times.mean().item())
            if finite_success_times.numel() > 0
            else None
        ),
        "schedule": {
            "torque_ramp_time_s": args_cli.torque_ramp_time,
            "extend_time_s": args_cli.extend_time,
            "stabilize_time_s": args_cli.stabilize_time,
            "handoff_time_s": args_cli.handoff_time,
            "post_handoff_time_s": args_cli.post_handoff_time,
            "target_leg_length_m": args_cli.target_length,
            "balance_pitch_kp": args_cli.balance_pitch_kp,
            "balance_pitch_kd": args_cli.balance_pitch_kd,
            "balance_wheel_action_limit": args_cli.balance_wheel_action_limit,
            "balance_leg_angle_gain": args_cli.balance_leg_angle_gain,
            "balance_leg_action_limit": args_cli.balance_leg_action_limit,
        },
        "thresholds": {
            "tilt_rad": args_cli.success_tilt,
            "height_min_m": args_cli.success_height_min,
            "height_max_m": args_cli.success_height_max,
            "angular_speed_radps": args_cli.success_ang_vel,
            "hold_time_s": args_cli.success_hold_time,
            "contact_force_N": args_cli.contact_threshold,
        },
        "summary_csv": str(summary_path),
        "trace_csv": str(trace_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\n[RESULT] Power-on stand success: {int(success.sum().item())}/{len(cases)} "
        f"({100.0 * success.float().mean().item():.1f}%)"
    )
    if finite_success_times.numel() > 0:
        print(
            "[RESULT] Mean first stable time: "
            f"{finite_success_times.mean().item():.3f} s"
        )
    print(
        "[RESULT] Worst safety metrics: "
        f"tilt={max_tilt.max().item():.3f} rad, "
        f"|vz|={max_abs_vz.max().item():.3f} m/s, "
        f"tau_clip={max_torque_saturation.max().item():.3f}, "
        f"joint_margin={min_joint_margin.min().item():.3f} rad"
    )
    print(f"[RESULT] Results saved to: {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
