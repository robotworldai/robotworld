#!/usr/bin/env python3

"""Run a selectable interval of the flat-to-obstacle RSL-RL pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("staged_training_config.json")
DEFAULT_ISAACLAB = PROJECT_ROOT.parent / "IsaacLab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--isaaclab-path", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--start-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to enter or continue a selected pipeline stage.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        help=(
            "Stage that produced --start-checkpoint. Valid names are read from "
            "staged_training_config.json."
        ),
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default=None,
        help=(
            "Last stage to train and validate (inclusive). Defaults to the final "
            "stage configured in staged_training_config.json."
        ),
    )
    parser.add_argument(
        "--start-mode",
        choices=("continue", "next"),
        default=None,
        help=(
            "'continue' resumes the selected stage including optimizer/iteration; "
            "'next' loads its policy into the following stage with a fresh optimizer."
        ),
    )
    parser.add_argument(
        "--flat-checkpoint",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility alias for --start-checkpoint PATH "
            "--start-stage flat --start-mode next. "
            "A checkpoint already produced by expand_rsl_checkpoint_for_jump.py "
            "is reused directly instead of being expanded again."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands and gates without training."
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("\n[PIPELINE] " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [
        path for path in run_dir.glob("model_*.pt") if checkpoint_iteration(path) >= 0
    ]
    if not checkpoints:
        raise RuntimeError(f"No model checkpoint was saved in {run_dir}")
    return max(checkpoints, key=checkpoint_iteration)


def find_run_dir(experiment: str, run_name: str) -> Path:
    root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment
    matches = sorted(
        root.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime
    )
    if not matches:
        raise RuntimeError(f"Cannot find run '*_{run_name}' below {root}")
    return matches[-1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def is_jump_init_checkpoint(path: Path) -> bool:
    """Return whether a checkpoint was already expanded for jump observations."""
    # This is the canonical output name used by this pipeline. Checking it first
    # keeps the lightweight orchestration script independent of PyTorch.
    if path.name in {"model_flat_jump_init.pt", "model_pre_jump_init.pt"}:
        return True

    try:
        import torch
    except ModuleNotFoundError:
        return False

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    conversion = (checkpoint.get("infos") or {}).get("jump_checkpoint_conversion")
    if not isinstance(conversion, dict):
        return False
    actor_dims = conversion.get("actor_observations")
    critic_dims = conversion.get("critic_observations")
    return (
        isinstance(actor_dims, (list, tuple))
        and len(actor_dims) == 2
        and isinstance(critic_dims, (list, tuple))
        and len(critic_dims) == 2
    )


def resolve_start_request(
    args: argparse.Namespace, stages: list[dict[str, object]]
) -> tuple[Path | None, int, bool, str | None, str | None]:
    """Resolve checkpoint start options.

    Returns checkpoint, first stage index, whether the first load is a full
    same-stage resume, source stage name, and start mode.
    """
    explicit_values = (args.start_checkpoint, args.start_stage, args.start_mode)
    if args.flat_checkpoint is not None:
        if any(value is not None for value in explicit_values):
            raise ValueError(
                "--flat-checkpoint cannot be combined with --start-checkpoint, "
                "--start-stage, or --start-mode."
            )
        checkpoint = args.flat_checkpoint
        source_stage_name = "flat"
        start_mode = "next"
    else:
        supplied = [value is not None for value in explicit_values]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "--start-checkpoint, --start-stage, and --start-mode must be "
                "specified together."
            )
        if not any(supplied):
            return None, 0, False, None, None
        checkpoint = args.start_checkpoint
        source_stage_name = args.start_stage
        start_mode = args.start_mode

    stage_names = [str(stage["name"]) for stage in stages]
    if source_stage_name not in stage_names:
        raise ValueError(
            f"Unknown --start-stage {source_stage_name!r}; choose one of: "
            + ", ".join(stage_names)
        )
    source_index = stage_names.index(source_stage_name)
    first_stage_index = source_index + (1 if start_mode == "next" else 0)
    if first_stage_index >= len(stages):
        raise ValueError(
            f"Stage {source_stage_name!r} is the final stage, so --start-mode next "
            "has no following stage."
        )

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Start checkpoint does not exist: {checkpoint}")
    return (
        checkpoint,
        first_stage_index,
        start_mode == "continue",
        source_stage_name,
        start_mode,
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stages = config["stages"]
    if not stages or stages[0]["name"] != "flat":
        raise ValueError("The first staged-training entry must be 'flat'.")

    isaaclab_sh = args.isaaclab_path.expanduser().resolve() / "isaaclab.sh"
    if not isaaclab_sh.is_file():
        raise FileNotFoundError(f"Cannot find Isaac Lab launcher: {isaaclab_sh}")

    pipeline_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_dir = (
        PROJECT_ROOT / "logs" / "rsl_rl" / "staged_pipeline" / pipeline_id
    )
    state = {
        "pipeline_id": pipeline_id,
        "config": str(config_path),
        "completed": [],
        "passed": False,
    }
    (
        previous_checkpoint,
        first_stage_index,
        resume_first_stage,
        source_stage_name,
        start_mode,
    ) = resolve_start_request(args, stages)
    stage_names = [str(stage["name"]) for stage in stages]
    if args.end_stage is None:
        end_stage_index = len(stages) - 1
    elif args.end_stage not in stage_names:
        raise ValueError(
            f"Unknown --end-stage {args.end_stage!r}; choose one of: "
            + ", ".join(stage_names)
        )
    else:
        end_stage_index = stage_names.index(args.end_stage)
    if end_stage_index < first_stage_index:
        raise ValueError(
            f"--end-stage {stage_names[end_stage_index]!r} is earlier than the "
            f"first stage that would actually train, {stage_names[first_stage_index]!r}."
        )
    end_stage_name = stage_names[end_stage_index]
    state["first_training_stage"] = stage_names[first_stage_index]
    state["end_stage"] = end_stage_name
    if not args.dry_run:
        write_json(state_dir / "state.json", state)

    if previous_checkpoint is not None:
        source_checkpoint = previous_checkpoint
        already_converted = is_jump_init_checkpoint(source_checkpoint)
        source_stage = stages[stage_names.index(str(source_stage_name))]
        entering_jump_from_base = (
            bool(source_stage.get("convert_to_jump", False))
            and start_mode == "next"
        )
        if (
            bool(source_stage.get("convert_to_jump", False))
            and start_mode == "continue"
            and already_converted
        ):
            raise ValueError(
                f"--start-mode continue for stage {source_stage_name!r} requires "
                "that stage's original checkpoint, not a checkpoint already "
                "expanded for jump observations. Use --start-mode next instead."
            )
        if entering_jump_from_base:
            if already_converted:
                print(
                    "\n[PIPELINE] The supplied checkpoint is already expanded for "
                    "jump observations; reusing it directly.",
                    flush=True,
                )
            else:
                converted = state_dir / "model_pre_jump_init.pt"
                conversion_command = [
                    str(isaaclab_sh),
                    "-p",
                    "scripts/expand_rsl_checkpoint_for_jump.py",
                    str(source_checkpoint),
                    "--output",
                    str(converted),
                ]
                run_command(conversion_command, dry_run=args.dry_run)
                previous_checkpoint = (
                    Path("<converted-pre-jump-checkpoint>")
                    if args.dry_run
                    else converted
                )
        if not args.dry_run:
            state["start"] = {
                "checkpoint": str(source_checkpoint),
                "source_stage": source_stage_name,
                "mode": start_mode,
                "first_training_stage": stages[first_stage_index]["name"],
                "full_state_resume": resume_first_stage,
            }
            if entering_jump_from_base:
                state["pre_jump_checkpoint"] = str(previous_checkpoint)
                state["source_checkpoint_already_converted"] = already_converted
            write_json(state_dir / "state.json", state)

    for index, stage in enumerate(stages):
        if index < first_stage_index:
            continue
        if index > end_stage_index:
            break

        stage_name = stage["name"]
        run_name = f"auto_{pipeline_id}_{stage_name}"
        gate_path = state_dir / f"{index:02d}_{stage_name}_gate.json"
        status_path = state_dir / f"{index:02d}_{stage_name}_status.json"
        gate = {
            key: stage[key]
            for key in ("name", "min_iterations", "window", "consecutive", "rules")
        }
        if not args.dry_run:
            write_json(gate_path, gate)

        command = [
            str(isaaclab_sh),
            "-p",
            "scripts/rsl_rl/train.py",
            "--task",
            stage["task"],
            "--headless",
            "--num_envs",
            str(stage.get("num_envs", args.num_envs)),
            "--max_iterations",
            str(stage["max_iterations"]),
            "--run_name",
            run_name,
            "--early_stop_config",
            str(gate_path),
            "--early_stop_status_file",
            str(status_path),
        ]
        if args.device is not None:
            command.extend(("--device", args.device))
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        if previous_checkpoint is not None:
            command.extend(("--load_checkpoint_path", str(previous_checkpoint)))
            if not (resume_first_stage and index == first_stage_index):
                command.append("--load_weights_only")

        run_command(command, dry_run=args.dry_run)
        if args.dry_run:
            previous_checkpoint = Path(f"<checkpoint-from-{stage_name}>")
            if bool(stage.get("convert_to_jump", False)) and index < end_stage_index:
                converted = state_dir / "model_pre_jump_init.pt"
                conversion_command = [
                    str(isaaclab_sh),
                    "-p",
                    "scripts/expand_rsl_checkpoint_for_jump.py",
                    str(previous_checkpoint),
                    "--output",
                    str(converted),
                ]
                run_command(conversion_command, dry_run=True)
                previous_checkpoint = Path("<converted-pre-jump-checkpoint>")
            resume_first_stage = False
            continue

        run_dir = find_run_dir(stage["experiment"], run_name)
        checkpoint = latest_checkpoint(run_dir)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state["active_stage"] = stage_name
        state["last_checkpoint"] = str(checkpoint)
        state["last_status"] = status
        write_json(state_dir / "state.json", state)
        if not status.get("passed", False):
            print(
                f"\n[PIPELINE] Stage {stage_name!r} exhausted its training budget "
                f"without satisfying the gate.\n"
                f"[PIPELINE] Checkpoint retained at: {checkpoint}\n"
                f"[PIPELINE] Status: {status_path}",
                file=sys.stderr,
            )
            return 2

        state["completed"].append(
            {
                "stage": stage_name,
                "task": stage["task"],
                "checkpoint": str(checkpoint),
                "status": status,
            }
        )
        write_json(state_dir / "state.json", state)
        previous_checkpoint = checkpoint
        resume_first_stage = False

        if bool(stage.get("convert_to_jump", False)) and index < end_stage_index:
            converted = state_dir / "model_pre_jump_init.pt"
            conversion_command = [
                str(isaaclab_sh),
                "-p",
                "scripts/expand_rsl_checkpoint_for_jump.py",
                str(checkpoint),
                "--output",
                str(converted),
            ]
            run_command(conversion_command, dry_run=args.dry_run)
            previous_checkpoint = (
                Path("<converted-pre-jump-checkpoint>")
                if args.dry_run
                else converted
            )
            if not args.dry_run:
                state["pre_jump_checkpoint"] = str(converted)
                write_json(state_dir / "state.json", state)

    if args.dry_run:
        print("\n[PIPELINE] Dry run completed.")
        return 0

    state["passed"] = True
    state["final_checkpoint"] = str(previous_checkpoint)
    state.pop("active_stage", None)
    write_json(state_dir / "state.json", state)
    print(
        f"\n[PIPELINE] Requested stages passed through {end_stage_name!r}."
    )
    print(f"[PIPELINE] Final checkpoint: {previous_checkpoint}")
    print(f"[PIPELINE] Pipeline record: {state_dir / 'state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
