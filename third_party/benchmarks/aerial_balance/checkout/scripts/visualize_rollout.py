#!/usr/bin/env python3
"""Visualize saved Aerial-Balance-Bench rollout states and actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from aerial_balance_bench.utils.visualization import (
    available_series_keys,
    build_time_axis,
    load_rollout_data,
    plot_rollout_groups,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize rollout.npz states, actions, and policy signals.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--run-dir", type=Path, help="Run directory containing rollout.npz.")
    input_group.add_argument("--rollout", type=Path, help="Path to a rollout.npz file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs. Defaults to <run_dir>/visualizations.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        choices=["states", "actions", "policy", "all"],
        help="Plot groups to generate.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        help="Comma-separated custom signal keys. When set, writes custom.png and ignores --groups.",
    )
    parser.add_argument(
        "--env-ids",
        type=str,
        default="all",
        help="Environment ids to plot, e.g. 'all' or '0,1,2'. Defaults to all.",
    )
    parser.add_argument("--step-dt", type=float, default=None, help="Override rollout step duration in seconds.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    parser.add_argument("--list-keys", action="store_true", help="Print available signal keys and exit.")
    return parser.parse_args()


def _resolve_rollout_path(run_dir: Path | None, rollout: Path | None) -> Path:
    if rollout is not None:
        rollout_path = rollout.expanduser().resolve()
    else:
        rollout_path = run_dir.expanduser().resolve() / "rollout.npz"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
    return rollout_path


def _resolve_output_dir(output_dir: Path | None, rollout_path: Path) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return rollout_path.parent / "visualizations"


def _parse_keys(raw_keys: str | None) -> list[str] | None:
    if raw_keys is None:
        return None
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    if not keys:
        raise ValueError("--keys was provided but no valid keys were parsed.")
    return keys


def main():
    args = _parse_args()
    rollout_path = _resolve_rollout_path(args.run_dir, args.rollout)
    output_dir = _resolve_output_dir(args.output_dir, rollout_path)

    rollout_data = load_rollout_data(rollout_path)
    if args.list_keys:
        print("\n".join(available_series_keys(rollout_data)))
        return

    time_axis = build_time_axis(rollout_data, rollout_path=rollout_path, step_dt=args.step_dt)
    output_paths = plot_rollout_groups(
        rollout_data,
        output_dir=output_dir,
        time_axis=time_axis,
        groups=args.groups,
        keys=_parse_keys(args.keys),
        env_ids=args.env_ids,
        dpi=args.dpi,
    )

    if not output_paths:
        print("[WARN] No matching plottable signals were found.")
        return

    print(f"[INFO] Visualizations saved to: {output_dir}")
    for path in output_paths:
        print(f"[INFO] {path}")


if __name__ == "__main__":
    main()
