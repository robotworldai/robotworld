#!/usr/bin/env python3
"""Visualize real-world experiment pickle results in one figure."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
import matplotlib.pyplot as plt


PLOT_KEYS = ("e", "target_vel_z", "drone_vel_z")
POSITION_ERROR_LIMIT = 0.7
DEFAULT_MAX_TRAJECTORIES = 9
LABELS = {
    "e": "Pos Error",
    "drone_vel_z": "Real Vel",
    "target_vel_z": "Exp Vel",
}


@dataclass
class TrajectoryRecord:
    """Aligned real-world experiment trajectory."""

    directory: Path
    file: Path
    trajectory_index: int
    time_s: np.ndarray
    values: dict[str, np.ndarray]
    num_samples: int
    duration_s: float
    plotted_samples: int
    plotted_duration_s: float


@dataclass
class TargetPositionMetricRecord:
    """Target-position metrics for one plotted trajectory."""

    directory: Path
    file: Path
    trajectory_index: int
    success: bool
    steady_state_error: float
    convergence_time: float
    climbing_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize real-world Aerial-Balance-Bench experiment results.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing real-world experiment pkl files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to <data_dir>/visualizations.")
    parser.add_argument("--pattern", type=str, default="*.pkl", help="Pickle filename glob pattern.")
    parser.add_argument("--recursive", action="store_true", default=False, help="Search for pickle files recursively.")
    parser.add_argument("--sample_rate", type=float, default=60.0, help="Experiment sample rate in Hz.")
    parser.add_argument("--duration_s", type=float, default=20.0, help="Visualization time range in seconds.")
    parser.add_argument("--error_tolerance", type=float, default=0.03, help="Target-position error tolerance epsilon.")
    parser.add_argument("--steady_window_s", type=float, default=5.0, help="Terminal window W for steady-state error.")
    parser.add_argument("--metric_timeout_s", type=float, default=20.017, help="Fallback CONT/CLIT value when unavailable.")
    parser.add_argument("--max_trajectories", type=int, default=DEFAULT_MAX_TRAJECTORIES, help="Maximum trajectories to draw.")
    parser.add_argument("--selection_seed", type=int, default=0, help="Random seed for trajectory subsampling.")
    parser.add_argument("--output_name", type=str, default="real_experiment_results.png", help="Output image filename.")
    parser.add_argument(
        "--legend_mode",
        choices=("axis", "figure", "none"),
        default="axis",
        help="Legend mode. 'axis' uses one shared legend inside the top axis.",
    )
    parser.add_argument("--legend_cols", type=int, default=3, help="Number of columns in the legend.")
    parser.add_argument("--line_width", type=float, default=2.6, help="Trajectory line width.")
    parser.add_argument("--line_alpha", type=float, default=1.0, help="Trajectory line alpha.")
    parser.add_argument("--show", action="store_true", default=False, help="Show the figure after saving.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist or is not a directory: {data_dir}")
    if args.sample_rate <= 0.0:
        raise ValueError(f"--sample_rate must be positive, got {args.sample_rate}.")
    if args.duration_s <= 0.0:
        raise ValueError(f"--duration_s must be positive, got {args.duration_s}.")
    if args.error_tolerance <= 0.0:
        raise ValueError(f"--error_tolerance must be positive, got {args.error_tolerance}.")
    if args.steady_window_s <= 0.0:
        raise ValueError(f"--steady_window_s must be positive, got {args.steady_window_s}.")
    if args.metric_timeout_s <= 0.0:
        raise ValueError(f"--metric_timeout_s must be positive, got {args.metric_timeout_s}.")
    if args.max_trajectories <= 0:
        raise ValueError(f"--max_trajectories must be positive, got {args.max_trajectories}.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_files = _find_pickle_files(data_dir, args.pattern, args.recursive)
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files matching '{args.pattern}' found under {data_dir}.")

    records: list[TrajectoryRecord] = []
    for path in pickle_files:
        records.extend(_load_trajectory_records(path, args.sample_rate, args.duration_s))

    if not records:
        raise RuntimeError(f"No valid trajectories with keys {PLOT_KEYS} found under {data_dir}.")

    plot_records = _select_records_for_plot(
        records,
        max_trajectories=args.max_trajectories,
        min_duration_s=args.duration_s,
        seed=args.selection_seed,
    )
    figure_path = output_dir / args.output_name
    _plot_records(
        plot_records,
        figure_path,
        args.duration_s,
        legend_mode=args.legend_mode,
        legend_cols=args.legend_cols,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        show=args.show,
    )

    trajectory_summary_path = output_dir / "trajectory_summary.csv"
    directory_summary_path = output_dir / "directory_summary.csv"
    metrics_per_trajectory_path = output_dir / "target_position_metrics_per_trajectory.csv"
    metrics_summary_path = output_dir / "target_position_metrics_summary.csv"
    _write_trajectory_summary(records, trajectory_summary_path)
    directory_rows = _write_directory_summary(records, directory_summary_path)
    metric_records = _compute_target_position_metrics(
        plot_records,
        error_tolerance=args.error_tolerance,
        steady_window_s=args.steady_window_s,
        timeout_s=args.metric_timeout_s,
        duration_s=args.duration_s,
    )
    metrics_summary = _write_target_position_metrics(
        metric_records,
        metrics_per_trajectory_path,
        metrics_summary_path,
        error_tolerance=args.error_tolerance,
        steady_window_s=args.steady_window_s,
        duration_s=args.duration_s,
        timeout_s=args.metric_timeout_s,
    )

    print(f"[INFO] Loaded {len(pickle_files)} pickle file(s).")
    print(f"[INFO] Loaded {len(records)} trajectory/trajectories.")
    print(f"[INFO] Visualized {len(plot_records)} trajectory/trajectories.")
    for row in directory_rows:
        print(
            "[INFO] Directory: {directory} | num_trajectories={num_trajectories} | "
            "min_duration_s={min_duration_s:.6f}".format(**row)
        )
    print(f"[INFO] Figure saved to: {figure_path}")
    print(f"[INFO] Trajectory summary saved to: {trajectory_summary_path}")
    print(f"[INFO] Directory summary saved to: {directory_summary_path}")
    print(
        "[INFO] Target-position metrics: SR={success_rate_fraction}, "
        "SE={steady_state_error_mean:.6f}+/-{steady_state_error_std:.6f}, "
        "CONT={convergence_time_mean:.6f}+/-{convergence_time_std:.6f}, "
        "CLIT={climbing_time_mean:.6f}+/-{climbing_time_std:.6f}".format(**metrics_summary)
    )
    print(f"[INFO] Target-position per-trajectory metrics saved to: {metrics_per_trajectory_path}")
    print(f"[INFO] Target-position metrics summary saved to: {metrics_summary_path}")


def _find_pickle_files(data_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = data_dir.rglob(pattern) if recursive else data_dir.glob(pattern)
    return sorted(path.resolve() for path in iterator if path.is_file())


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle data to be a dict, got {type(data).__name__}: {path}")
    return data


def _load_trajectory_records(path: Path, sample_rate: float, duration_s: float) -> list[TrajectoryRecord]:
    try:
        data = _load_pickle(path)
    except Exception as exc:
        print(f"[WARN] Skipping {path}: failed to load pickle ({exc}).")
        return []

    missing_keys = [key for key in PLOT_KEYS if key not in data]
    if missing_keys:
        print(f"[WARN] Skipping {path}: missing required key(s): {', '.join(missing_keys)}.")
        return []

    key_trajectories = {key: _as_trajectory_list(data[key]) for key in PLOT_KEYS}
    step_trajectories = _as_trajectory_list(data["step_num"]) if "step_num" in data else None
    trajectory_count = min(len(values) for values in key_trajectories.values())
    if step_trajectories is not None:
        trajectory_count = min(trajectory_count, len(step_trajectories))
    if trajectory_count <= 0:
        print(f"[WARN] Skipping {path}: no trajectory data found.")
        return []

    records: list[TrajectoryRecord] = []
    max_plot_samples = max(1, int(duration_s * sample_rate))
    for trajectory_index in range(trajectory_count):
        raw_values = {key: _as_float_array(key_trajectories[key][trajectory_index]) for key in PLOT_KEYS}
        step_values = None
        if step_trajectories is not None:
            step_values = _as_float_array(step_trajectories[trajectory_index])

        lengths = [values.size for values in raw_values.values()]
        if step_values is not None:
            lengths.append(step_values.size)
        num_samples = min(lengths)
        if num_samples <= 0:
            continue

        aligned_values = {key: values[:num_samples] for key, values in raw_values.items()}
        if step_values is not None:
            step_values = step_values[:num_samples]
            time_s = (step_values - step_values[0]) / sample_rate
        else:
            time_s = np.arange(num_samples, dtype=np.float64) / sample_rate

        plotted_samples = min(num_samples, max_plot_samples)
        plot_values = {key: values[:plotted_samples] for key, values in aligned_values.items()}
        plot_time = time_s[:plotted_samples]
        records.append(
            TrajectoryRecord(
                directory=path.parent,
                file=path,
                trajectory_index=trajectory_index,
                time_s=plot_time,
                values=plot_values,
                num_samples=num_samples,
                duration_s=num_samples / sample_rate,
                plotted_samples=plotted_samples,
                plotted_duration_s=plotted_samples / sample_rate,
            )
        )
    return records


def _as_trajectory_list(value: Any) -> list[np.ndarray]:
    array = np.asarray(value, dtype=object if _is_ragged_sequence(value) else None)
    if isinstance(value, np.ndarray) and value.dtype != object:
        squeezed = np.squeeze(value)
        if squeezed.ndim == 0:
            return [squeezed.reshape(1)]
        if squeezed.ndim == 1:
            return [squeezed]
        if squeezed.ndim == 2:
            return [squeezed[index, :] for index in range(squeezed.shape[0])]
        return [np.ravel(squeezed[index]) for index in range(squeezed.shape[0])]

    if _is_scalar_sequence(value):
        return [np.asarray(value)]
    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 1:
        return [np.asarray(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]
    if array.ndim == 0:
        return [np.asarray([array.item()])]
    return [np.ravel(array)]


def _is_ragged_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    if _is_scalar(value[0]):
        return False
    try:
        lengths = [len(item) for item in value]
    except TypeError:
        return False
    return len(set(lengths)) > 1


def _is_scalar_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim <= 1 and value.dtype != object
    if not isinstance(value, (list, tuple)):
        return _is_scalar(value)
    if not value:
        return True
    return _is_scalar(value[0])


def _is_scalar(value: Any) -> bool:
    return np.isscalar(value) or isinstance(value, (str, bytes))


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _select_records_for_plot(
    records: list[TrajectoryRecord],
    max_trajectories: int,
    min_duration_s: float,
    seed: int,
) -> list[TrajectoryRecord]:
    """Select a bounded number of trajectories for clear visualization."""
    if len(records) <= max_trajectories:
        return records

    long_indices = [index for index, record in enumerate(records) if record.duration_s > min_duration_s]
    short_indices = [index for index, record in enumerate(records) if record.duration_s <= min_duration_s]
    rng = np.random.default_rng(seed)

    if len(long_indices) >= max_trajectories:
        selected_indices = _sample_indices(long_indices, max_trajectories, rng)
    else:
        remaining_count = max_trajectories - len(long_indices)
        selected_indices = list(long_indices)
        selected_indices.extend(_sample_indices(short_indices, remaining_count, rng))

    selected_indices = sorted(selected_indices)
    print(
        f"[INFO] Trajectory selection: total={len(records)}, "
        f"duration_gt_{min_duration_s:g}s={len(long_indices)}, visualized={len(selected_indices)}."
    )
    return [records[index] for index in selected_indices]


def _sample_indices(indices: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    sampled = rng.choice(np.asarray(indices, dtype=np.int64), size=count, replace=False)
    return [int(index) for index in sampled.tolist()]


def _plot_records(
    records: list[TrajectoryRecord],
    output_path: Path,
    duration_s: float,
    legend_mode: str,
    legend_cols: int,
    line_width: float,
    line_alpha: float,
    show: bool,
):
    fig_width = 19 if legend_mode == "figure" else 16
    fig, axes = plt.subplots(len(PLOT_KEYS), 1, figsize=(fig_width, 12), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    line_styles = _trajectory_line_styles(records)
    legend_handles = []
    legend_labels = []

    for ax, key in zip(axes_flat, PLOT_KEYS):
        for record in records:
            curve = record.values[key]
            if key == "e":
                curve = np.clip(curve, -POSITION_ERROR_LIMIT, POSITION_ERROR_LIMIT)
            style = line_styles[_trajectory_id(record)]
            (line,) = ax.plot(
                record.time_s,
                curve,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=line_width,
                alpha=line_alpha,
                label=style["label"],
            )
            if key == PLOT_KEYS[0]:
                legend_handles.append(line)
                legend_labels.append(style["label"])
        ax.set_ylabel(LABELS[key], fontsize=32)
        ax.tick_params(axis="y", which="major", labelsize=30)
        ax.tick_params(axis="x", labelbottom=False, bottom=True)
        ax.grid(True, which="major", linewidth=2.5, alpha=0.75)
        ax.set_xlim(0.0, duration_s)
    axes_flat[-1].tick_params(axis="x", labelbottom=True, bottom=True, which="major", labelsize=33)
    axes_flat[-1].set_xlabel("Time (s)", fontsize=42)
    if legend_mode == "axis":
        axes_flat[0].legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            ncols=max(1, legend_cols),
            title="Trajectory",
            title_fontsize=11,
        )
        fig.tight_layout()
    elif legend_mode == "figure":
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=12,
            frameon=True,
            ncols=max(1, legend_cols),
            title="Trajectory",
            title_fontsize=13,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _trajectory_line_styles(records: list[TrajectoryRecord]) -> dict[str, dict[str, Any]]:
    trajectory_ids = [_trajectory_id(record) for record in records]
    colormap = plt.get_cmap("tab20")
    linestyles = ("-", "--", "-.", ":")
    styles: dict[str, dict[str, Any]] = {}
    for index, trajectory_id in enumerate(trajectory_ids):
        styles[trajectory_id] = {
            "color": colormap(index % colormap.N),
            "linestyle": linestyles[(index // colormap.N) % len(linestyles)],
            "label": f"Trajectory {index + 1}",
        }
    return styles


def _trajectory_id(record: TrajectoryRecord) -> str:
    return f"{record.file.stem}:T{record.trajectory_index + 1}"


def _compute_target_position_metrics(
    records: list[TrajectoryRecord],
    error_tolerance: float,
    steady_window_s: float,
    timeout_s: float,
    duration_s: float,
) -> list[TargetPositionMetricRecord]:
    metric_records: list[TargetPositionMetricRecord] = []
    for record in records:
        time_s = np.asarray(record.time_s, dtype=np.float64)
        clipped_position_error = np.clip(
            np.asarray(record.values["e"], dtype=np.float64),
            -POSITION_ERROR_LIMIT,
            POSITION_ERROR_LIMIT,
        )
        error = np.abs(clipped_position_error)
        if time_s.size != error.size:
            size = min(time_s.size, error.size)
            time_s = time_s[:size]
            error = error[:size]
        valid = np.isfinite(time_s) & np.isfinite(error) & (time_s <= duration_s)
        time_s = time_s[valid]
        error = error[valid]

        if error.size == 0:
            steady_state_error = float("nan")
            convergence_time = timeout_s
            climbing_time = timeout_s
            success = False
        else:
            steady_state_error = _terminal_mean_error(time_s, error, steady_window_s)
            in_zone = error <= error_tolerance
            climbing_time = _first_entry_time(time_s, in_zone, timeout_s)
            convergence_time = _first_sustained_entry_time(time_s, in_zone, timeout_s)
            success = convergence_time < duration_s

        metric_records.append(
            TargetPositionMetricRecord(
                directory=record.directory,
                file=record.file,
                trajectory_index=record.trajectory_index,
                success=success,
                steady_state_error=steady_state_error,
                convergence_time=convergence_time,
                climbing_time=climbing_time,
            )
        )
    return metric_records


def _terminal_mean_error(time_s: np.ndarray, error: np.ndarray, steady_window_s: float) -> float:
    if error.size == 0:
        return float("nan")
    if time_s.size == 0 or time_s[-1] <= steady_window_s:
        return float(np.mean(error))
    window_start = time_s[-1] - steady_window_s
    window_error = error[time_s >= window_start]
    if window_error.size == 0:
        window_error = error
    return float(np.mean(window_error))


def _first_entry_time(time_s: np.ndarray, in_zone: np.ndarray, timeout_s: float) -> float:
    indices = np.flatnonzero(in_zone)
    if indices.size == 0:
        return float(timeout_s)
    return float(time_s[int(indices[0])])


def _first_sustained_entry_time(time_s: np.ndarray, in_zone: np.ndarray, timeout_s: float) -> float:
    indices = np.flatnonzero(in_zone)
    for index in indices:
        if np.all(in_zone[int(index) :]):
            return float(time_s[int(index)])
    return float(timeout_s)


def _write_target_position_metrics(
    metric_records: list[TargetPositionMetricRecord],
    per_trajectory_path: Path,
    summary_path: Path,
    error_tolerance: float,
    steady_window_s: float,
    duration_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    per_trajectory_fields = (
        "directory",
        "file",
        "trajectory_index",
        "success",
        "steady_state_error",
        "convergence_time",
        "climbing_time",
    )
    with per_trajectory_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_trajectory_fields)
        writer.writeheader()
        for record in metric_records:
            writer.writerow(
                {
                    "directory": str(record.directory),
                    "file": record.file.name,
                    "trajectory_index": record.trajectory_index,
                    "success": int(record.success),
                    "steady_state_error": _format_float(record.steady_state_error),
                    "convergence_time": _format_float(record.convergence_time),
                    "climbing_time": _format_float(record.climbing_time),
                }
            )

    summary = _summarize_target_position_metrics(
        metric_records,
        error_tolerance=error_tolerance,
        steady_window_s=steady_window_s,
        duration_s=duration_s,
        timeout_s=timeout_s,
    )
    summary_fields = (
        "num_trajectories",
        "success_count",
        "success_rate_fraction",
        "error_tolerance",
        "steady_window_s",
        "duration_s",
        "metric_timeout_s",
        "steady_state_error_mean",
        "steady_state_error_std",
        "convergence_time_mean",
        "convergence_time_std",
        "climbing_time_mean",
        "climbing_time_std",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({key: _format_float(value) if isinstance(value, float) else value for key, value in summary.items()})
    return summary


def _summarize_target_position_metrics(
    metric_records: list[TargetPositionMetricRecord],
    error_tolerance: float,
    steady_window_s: float,
    duration_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    num_trajectories = len(metric_records)
    success_count = sum(1 for record in metric_records if record.success)
    steady_state_error = np.asarray([record.steady_state_error for record in metric_records], dtype=np.float64)
    convergence_time = np.asarray([record.convergence_time for record in metric_records], dtype=np.float64)
    climbing_time = np.asarray([record.climbing_time for record in metric_records], dtype=np.float64)
    return {
        "num_trajectories": num_trajectories,
        "success_count": success_count,
        "success_rate_fraction": f"{success_count}/{num_trajectories}",
        "error_tolerance": float(error_tolerance),
        "steady_window_s": float(steady_window_s),
        "duration_s": float(duration_s),
        "metric_timeout_s": float(timeout_s),
        "steady_state_error_mean": _nanmean_or_nan(steady_state_error),
        "steady_state_error_std": _nanstd_or_nan(steady_state_error),
        "convergence_time_mean": _nanmean_or_nan(convergence_time),
        "convergence_time_std": _nanstd_or_nan(convergence_time),
        "climbing_time_mean": _nanmean_or_nan(climbing_time),
        "climbing_time_std": _nanstd_or_nan(climbing_time),
    }


def _nanmean_or_nan(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _nanstd_or_nan(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanstd(values, ddof=0))


def _format_float(value: float) -> str:
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return f"{float(value):.9f}"


def _write_trajectory_summary(records: list[TrajectoryRecord], output_path: Path):
    fieldnames = (
        "directory",
        "file",
        "trajectory_index",
        "num_samples",
        "duration_s",
        "plotted_samples",
        "plotted_duration_s",
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "directory": str(record.directory),
                    "file": record.file.name,
                    "trajectory_index": record.trajectory_index,
                    "num_samples": record.num_samples,
                    "duration_s": f"{record.duration_s:.9f}",
                    "plotted_samples": record.plotted_samples,
                    "plotted_duration_s": f"{record.plotted_duration_s:.9f}",
                }
            )


def _write_directory_summary(records: list[TrajectoryRecord], output_path: Path) -> list[dict[str, Any]]:
    grouped: dict[Path, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.directory].append(record)

    rows: list[dict[str, Any]] = []
    for directory, directory_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        durations = np.asarray([record.duration_s for record in directory_records], dtype=np.float64)
        files = {record.file for record in directory_records}
        rows.append(
            {
                "directory": str(directory),
                "num_files": len(files),
                "num_trajectories": len(directory_records),
                "min_duration_s": float(np.min(durations)),
                "max_duration_s": float(np.max(durations)),
                "mean_duration_s": float(np.mean(durations)),
            }
        )

    fieldnames = ("directory", "num_files", "num_trajectories", "min_duration_s", "max_duration_s", "mean_duration_s")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "min_duration_s": f"{row['min_duration_s']:.9f}",
                    "max_duration_s": f"{row['max_duration_s']:.9f}",
                    "mean_duration_s": f"{row['mean_duration_s']:.9f}",
                }
            )
    return rows


if __name__ == "__main__":
    main()
