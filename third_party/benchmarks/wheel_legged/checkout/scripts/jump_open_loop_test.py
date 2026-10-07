# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop crouch/thrust test for the wheel-legged robot.

Each vectorized environment evaluates one combination of crouch length,
crouch duration and extension length.  The script deliberately does not load
an RL checkpoint: it tests whether the VMC/action/actuator/physics chain can
produce a measurable jump before jump rewards are designed.
"""

from __future__ import annotations

import argparse
import itertools
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got '{value}'.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


parser = argparse.ArgumentParser(description="Test open-loop jump capability of the wheel-legged VMC robot.")
parser.add_argument("--task", type=str, default="Wheel-Legged-Flat-v0", help="Registered Isaac Lab task.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic environment seed.")
parser.add_argument("--crouch_lengths", type=_float_list, default=[0.18, 0.20, 0.22])
parser.add_argument("--crouch_durations", type=_float_list, default=[0.15, 0.25, 0.35])
parser.add_argument("--extension_lengths", type=_float_list, default=[0.28, 0.30])
parser.add_argument("--settle_length", type=float, default=0.237)
parser.add_argument("--landing_length", type=float, default=0.22)
parser.add_argument("--settle_time", type=float, default=1.0)
parser.add_argument("--thrust_time", type=float, default=0.18)
parser.add_argument("--observe_time", type=float, default=1.2)
parser.add_argument("--contact_threshold", type=float, default=2.0, help="Wheel contact-force threshold in N.")
parser.add_argument("--min_air_time", type=float, default=0.03, help="Minimum continuous air time for takeoff.")
parser.add_argument("--min_jump_height", type=float, default=0.03, help="Minimum apex rise for a useful jump.")
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help=(
        "Optional exported RSL-RL JIT policy.pt used to stabilize wheel speed and leg angle. "
        "The test still overrides both leg-length actions with the open-loop trajectory."
    ),
)
parser.add_argument(
    "--policy_observation_dims",
    type=int,
    default=None,
    help="Optionally pass only the first N observations to a legacy policy (use 36 on the jump task).",
)
parser.add_argument("--output_dir", type=str, default=None, help="Result directory; defaults under project root.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

cases = [
    {
        "case_id": index,
        "crouch_length": values[0],
        "crouch_duration": values[1],
        "extension_length": values[2],
    }
    for index, values in enumerate(
        itertools.product(args_cli.crouch_lengths, args_cli.crouch_durations, args_cli.extension_lengths)
    )
]
if not cases:
    parser.error("The parameter grid produced no test cases.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim must be running before the remaining imports."""

import csv
import json

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import wheel_legged_robot.tasks  # noqa: F401


def _disable_randomization_and_termination(env_cfg) -> None:
    """Make the parameter sweep deterministic and prevent automatic resets."""
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

    # Keep the normal reset handlers but remove their random pose and velocity.
    if getattr(env_cfg.events, "reset_base", None) is not None:
        pose_range = env_cfg.events.reset_base.params["pose_range"]
        velocity_range = env_cfg.events.reset_base.params["velocity_range"]
        for key in pose_range:
            pose_range[key] = (0.0, 0.0)
        for key in velocity_range:
            velocity_range[key] = (0.0, 0.0)

    for name in ("time_out", "bad_orientation", "joint_limits", "minimum_height"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "command_levels"):
        env_cfg.curriculum.command_levels = None
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    command_cfg = env_cfg.commands.wheel_legged_commands
    command_cfg.heading_command = False
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.ang_vel_yaw = (0.0, 0.0)
    command_cfg.ranges.heading = (0.0, 0.0)
    command_cfg.ranges.height = (args_cli.settle_length, args_cli.settle_length)
    command_cfg.debug_vis = False

    # When testing the registered jump task, align its trigger with the start
    # of the open-loop crouch trajectory so the contact state machine can be
    # validated against the independently measured physical events.
    if hasattr(env_cfg.commands, "jump_command"):
        jump_cfg = env_cfg.commands.jump_command
        jump_cfg.resampling_time_range = (1.0e9, 1.0e9)
        jump_cfg.jump_probability = 1.0
        jump_cfg.trigger_delay_range = (args_cli.settle_time, args_cli.settle_time)
        jump_cfg.trigger_pulse_time = 0.20

    # VMC gain randomization happens inside the action term rather than EventManager.
    env_cfg.actions.vmc.randomize_kp = False
    env_cfg.actions.vmc.randomize_kd = False
    env_cfg.actions.vmc.randomize_torque_scale = False


def _length_action(lengths: torch.Tensor, vmc_cfg) -> torch.Tensor:
    return torch.clamp((lengths - vmc_cfg.l0_offset) / vmc_cfg.action_scale_l0, -1.0, 1.0)


def _make_actions(lengths: torch.Tensor, vmc_cfg, policy, observations) -> torch.Tensor:
    if policy is None:
        actions = torch.zeros((len(cases), 6), dtype=torch.float32, device=lengths.device)
    else:
        policy_observation = observations["policy"] if isinstance(observations, dict) else observations
        if args_cli.policy_observation_dims is not None:
            policy_observation = policy_observation[:, : args_cli.policy_observation_dims]
        actions = policy(policy_observation).clone()
    normalized_length = _length_action(lengths, vmc_cfg)
    actions[:, 1] = normalized_length
    actions[:, 4] = normalized_length
    return torch.clamp(actions, -1.0, 1.0)


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else project_root / "jump_open_loop_results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=len(cases),
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    _disable_randomization_and_termination(env_cfg)
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = args_cli.settle_time + max(
        case["crouch_duration"] for case in cases
    ) + args_cli.thrust_time + args_cli.observe_time + 1.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    vmc_term = unwrapped.action_manager.get_term("vmc")
    vmc_cfg = vmc_term.cfg

    wheel_ids, wheel_names = contact_sensor.find_bodies(".*wheel.*")
    if len(wheel_ids) != 2:
        raise RuntimeError(
            f"Expected two wheel contact bodies, resolved {wheel_names} from {contact_sensor.body_names}."
        )
    robot_wheel_ids, robot_wheel_names = robot.find_bodies(".*wheel.*")
    if len(robot_wheel_ids) != 2:
        raise RuntimeError(
            f"Expected two robot wheel bodies, resolved {robot_wheel_names} from {robot.body_names}."
        )

    leg_local_ids = [0, 1, 3, 4]
    leg_asset_ids = [vmc_term.joint_ids[index] for index in leg_local_ids]
    robot_mass = float(robot.root_physx_view.get_masses()[0].sum().item())
    policy = None
    policy_path = None
    if args_cli.policy:
        policy_path = Path(args_cli.policy).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Exported JIT policy not found: {policy_path}")
        policy = torch.jit.load(str(policy_path), map_location=device).eval()

    print(f"[INFO] Running {len(cases)} open-loop jump cases.")
    print(
        "[INFO] Stabilization mode: "
        + (f"RSL-RL JIT policy ({policy_path})" if policy is not None else "pure open loop")
    )
    print(f"[INFO] Robot mass: {robot_mass:.3f} kg")
    print(f"[INFO] Wheel contact bodies: {wheel_names}")
    print(
        "[INFO] VMC: "
        f"L=[{vmc_cfg.l0_min:.3f}, {vmc_cfg.l0_max:.3f}] m, "
        f"kp_l0={vmc_cfg.kp_l0:.1f}, kd_l0={vmc_cfg.kd_l0:.1f}, "
        f"feedforward={vmc_cfg.feedforward_force:.1f} N"
    )

    observations, _ = env.reset()
    step_dt = float(unwrapped.step_dt)
    settle_steps = max(1, round(args_cli.settle_time / step_dt))
    settle_lengths = torch.full((len(cases),), args_cli.settle_length, device=device)

    with torch.inference_mode():
        for _ in range(settle_steps):
            observations, _, _, _, _ = env.step(
                _make_actions(settle_lengths, vmc_cfg, policy, observations)
            )

    baseline_z = robot.data.root_pos_w[:, 2].clone()
    baseline_wheel_z = robot.data.body_pos_w[:, robot_wheel_ids, 2].amin(dim=1).clone()
    baseline_x = robot.data.root_pos_w[:, 0].clone()
    initial_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))

    max_z = baseline_z.clone()
    max_wheel_z = baseline_wheel_z.clone()
    min_z = baseline_z.clone()
    max_vz = robot.data.root_lin_vel_w[:, 2].clone()
    max_tilt = initial_tilt.clone()
    max_torque_saturation = vmc_term.torque_saturation.clone()
    min_joint_margin = torch.full((len(cases),), torch.inf, device=device)
    max_wheel_force = torch.zeros(len(cases), device=device)
    max_air_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    current_air_steps = torch.zeros_like(max_air_steps)
    takeoff_time = torch.full((len(cases),), torch.nan, device=device)
    landing_time = torch.full_like(takeoff_time, torch.nan)
    landing_vz = torch.full_like(takeoff_time, torch.nan)
    was_airborne = torch.zeros(len(cases), dtype=torch.bool, device=device)
    landing_contact_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    task_phase_seen = torch.zeros(len(cases), 6, dtype=torch.bool, device=device)

    crouch_lengths = torch.tensor([case["crouch_length"] for case in cases], device=device)
    crouch_durations = torch.tensor([case["crouch_duration"] for case in cases], device=device)
    extension_lengths = torch.tensor([case["extension_length"] for case in cases], device=device)
    landing_lengths = torch.full((len(cases),), args_cli.landing_length, device=device)
    total_test_time = float(crouch_durations.max().item()) + args_cli.thrust_time + args_cli.observe_time
    total_steps = max(1, round(total_test_time / step_dt))
    trace_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for step in range(total_steps):
            elapsed = step * step_dt
            crouching = elapsed < crouch_durations
            thrusting = (elapsed >= crouch_durations) & (
                elapsed < crouch_durations + args_cli.thrust_time
            )
            desired_lengths = torch.where(
                crouching,
                crouch_lengths,
                torch.where(thrusting, extension_lengths, landing_lengths),
            )
            observations, _, _, _, _ = env.step(
                _make_actions(desired_lengths, vmc_cfg, policy, observations)
            )
            if hasattr(unwrapped, "jump_phase"):
                task_phase_seen |= torch.nn.functional.one_hot(
                    unwrapped.jump_phase, num_classes=6
                ).bool()

            z = robot.data.root_pos_w[:, 2]
            wheel_z = robot.data.body_pos_w[:, robot_wheel_ids, 2].amin(dim=1)
            vz = robot.data.root_lin_vel_w[:, 2]
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
            forces = contact_sensor.data.net_forces_w[:, wheel_ids]
            wheel_force = torch.linalg.vector_norm(forces, dim=-1)
            wheel_contact = wheel_force > args_cli.contact_threshold
            airborne = ~wheel_contact.any(dim=1)

            current_air_steps = torch.where(airborne, current_air_steps + 1, torch.zeros_like(current_air_steps))
            max_air_steps = torch.maximum(max_air_steps, current_air_steps)
            new_takeoff = (
                torch.isnan(takeoff_time)
                & (current_air_steps >= max(1, round(args_cli.min_air_time / step_dt)))
                & (elapsed >= crouch_durations)
            )
            takeoff_time[new_takeoff] = elapsed - (current_air_steps[new_takeoff] - 1) * step_dt
            was_airborne |= new_takeoff

            contact_after_takeoff = was_airborne & wheel_contact.any(dim=1)
            landing_contact_steps = torch.where(
                contact_after_takeoff,
                landing_contact_steps + 1,
                torch.zeros_like(landing_contact_steps),
            )
            new_landing = torch.isnan(landing_time) & (landing_contact_steps >= 2)
            landing_time[new_landing] = elapsed
            landing_vz[new_landing] = vz[new_landing]

            joint_pos = robot.data.joint_pos[:, leg_asset_ids]
            joint_limits = robot.data.soft_joint_pos_limits[:, leg_asset_ids]
            joint_margin = torch.minimum(
                joint_pos - joint_limits[..., 0],
                joint_limits[..., 1] - joint_pos,
            ).min(dim=1).values

            max_z = torch.maximum(max_z, z)
            max_wheel_z = torch.maximum(max_wheel_z, wheel_z)
            min_z = torch.minimum(min_z, z)
            max_vz = torch.maximum(max_vz, vz)
            max_tilt = torch.maximum(max_tilt, tilt)
            max_torque_saturation = torch.maximum(max_torque_saturation, vmc_term.torque_saturation)
            min_joint_margin = torch.minimum(min_joint_margin, joint_margin)
            max_wheel_force = torch.maximum(max_wheel_force, wheel_force.max(dim=1).values)

            phase = torch.where(
                crouching,
                torch.zeros_like(crouching, dtype=torch.long),
                torch.where(thrusting, torch.ones_like(crouching, dtype=torch.long), 2),
            )
            for env_id in range(len(cases)):
                trace_rows.append(
                    {
                        "time_s": elapsed,
                        "case_id": env_id,
                        "phase": ("crouch", "thrust", "landing")[int(phase[env_id].item())],
                        "desired_length_m": float(desired_lengths[env_id].item()),
                        "base_z_m": float(z[env_id].item()),
                        "base_vz_mps": float(vz[env_id].item()),
                        "min_wheel_center_z_m": float(wheel_z[env_id].item()),
                        "wheel_clearance_m": float(
                            (wheel_z[env_id] - baseline_wheel_z[env_id]).item()
                        ),
                        "L0_left_m": float(unwrapped.L0[env_id, 0].item()),
                        "L0_right_m": float(unwrapped.L0[env_id, 1].item()),
                        "wheel_contact_count": int(wheel_contact[env_id].sum().item()),
                        "max_wheel_force_N": float(wheel_force[env_id].max().item()),
                        "tilt_rad": float(tilt[env_id].item()),
                        "torque_saturation": float(vmc_term.torque_saturation[env_id].item()),
                        "joint_margin_min_rad": float(joint_margin[env_id].item()),
                    }
                )

    final_x = robot.data.root_pos_w[:, 0]
    apex_rise = max_z - baseline_z
    wheel_clearance = max_wheel_z - baseline_wheel_z
    max_air_time = max_air_steps.float() * step_dt
    min_air_steps = max(1, round(args_cli.min_air_time / step_dt))
    useful_jump = (
        (max_air_steps >= min_air_steps)
        & (apex_rise >= args_cli.min_jump_height)
        & torch.isfinite(landing_time)
        & (max_tilt < 0.5)
    )

    scalar_metrics = {
        "baseline_z_m": _to_float_list(baseline_z),
        "apex_z_m": _to_float_list(max_z),
        "apex_rise_m": _to_float_list(apex_rise),
        "wheel_clearance_m": _to_float_list(wheel_clearance),
        "minimum_z_m": _to_float_list(min_z),
        "max_vz_mps": _to_float_list(max_vz),
        "max_air_time_s": _to_float_list(max_air_time),
        "takeoff_time_s": _to_float_list(takeoff_time),
        "landing_time_s": _to_float_list(landing_time),
        "landing_vz_mps": _to_float_list(landing_vz),
        "forward_displacement_m": _to_float_list(final_x - baseline_x),
        "max_tilt_rad": _to_float_list(max_tilt),
        "max_torque_saturation": _to_float_list(max_torque_saturation),
        "minimum_joint_margin_rad": _to_float_list(min_joint_margin),
        "max_wheel_force_N": _to_float_list(max_wheel_force),
        "useful_jump": useful_jump.detach().cpu().tolist(),
    }
    if hasattr(unwrapped, "jump_phase"):
        for phase_index, phase_name in enumerate(
            ("idle", "crouch", "thrust", "flight", "landing", "recovery")
        ):
            scalar_metrics[f"task_phase_seen_{phase_name}"] = (
                task_phase_seen[:, phase_index].detach().cpu().tolist()
            )

    summary_rows: list[dict[str, object]] = []
    for case in cases:
        env_id = case["case_id"]
        row = dict(case)
        for key, values in scalar_metrics.items():
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

    metadata = {
        "task": args_cli.task,
        "timestamp": timestamp,
        "num_cases": len(cases),
        "stabilization_mode": "rsl_rl_jit" if policy is not None else "pure_open_loop",
        "policy_path": str(policy_path) if policy_path is not None else None,
        "robot_mass_kg": robot_mass,
        "wheel_body_names": wheel_names,
        "step_dt_s": step_dt,
        "test_thresholds": {
            "contact_force_N": args_cli.contact_threshold,
            "minimum_air_time_s": args_cli.min_air_time,
            "minimum_jump_height_m": args_cli.min_jump_height,
            "maximum_useful_tilt_rad": 0.5,
        },
        "vmc": {
            "l0_offset_m": vmc_cfg.l0_offset,
            "l0_min_m": vmc_cfg.l0_min,
            "l0_max_m": vmc_cfg.l0_max,
            "kp_l0": vmc_cfg.kp_l0,
            "kd_l0": vmc_cfg.kd_l0,
            "feedforward_force_N": vmc_cfg.feedforward_force,
        },
        "useful_jump_count": int(useful_jump.sum().item()),
        "best_apex_rise_m": float(apex_rise.max().item()),
        "best_takeoff_vz_mps": float(max_vz.max().item()),
        "summary_csv": str(summary_path),
        "trace_csv": str(trace_path),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    order = torch.argsort(apex_rise, descending=True).detach().cpu().tolist()
    print("\n[RESULT] Top cases by apex rise:")
    for rank, env_id in enumerate(order[: min(8, len(order))], start=1):
        row = summary_rows[env_id]
        print(
            f"  {rank:>2}. case={env_id:02d} "
            f"crouch={row['crouch_length']:.3f} m/{row['crouch_duration']:.2f} s "
            f"extend={row['extension_length']:.3f} m "
            f"rise={row['apex_rise_m']:.4f} m "
            f"vz={row['max_vz_mps']:.3f} m/s "
            f"air={row['max_air_time_s']:.3f} s "
            f"land_vz={row['landing_vz_mps']:.3f} m/s "
            f"tilt={row['max_tilt_rad']:.3f} rad "
            f"tau_clip={row['max_torque_saturation']:.3f} "
            f"useful={row['useful_jump']}"
        )
    print(f"\n[RESULT] Useful jumps: {int(useful_jump.sum().item())}/{len(cases)}")
    print(f"[RESULT] Results saved to: {output_dir}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
