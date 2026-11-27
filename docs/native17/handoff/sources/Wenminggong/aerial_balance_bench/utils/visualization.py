"""Rollout visualization helpers for Aerial-Balance-Bench."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import yaml

MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
import matplotlib.pyplot as plt


DEFAULT_STEP_DT = 1.0 / 60.0

STATE_KEYS = (
    "pb",
    "pg",
    "position_error",
    "vb",
    "ab",
    "theta",
    "omega",
    "alpha",
    "drz",
    "vrz",
    "arz",
)

ACTION_COMMAND_KEYS = (
    "actions",
    "step_last_action",
    "step_command_z",
    "step_executed_command_z",
    "step_vrz_cmd",
    "step_executed_vrz_cmd",
    "step_drz_cmd",
    "step_executed_drz_cmd",
    "step_target_position_z",
    "step_frz_cmd",
    "step_executed_frz_cmd",
    "step_delta_frz_cmd",
    "step_hover_force",
    "step_target_pitch",
    "step_target_height_acc",
    "step_beta",
)

GROUP_OUTPUT_FILES = {
    "states": "states.png",
    "actions": "actions_commands.png",
    "policy": "policy.png",
}

LABELS = {
    "pb": "Ball position pb (m)",
    "pg": "Goal position pg (m)",
    "position_error": "Position error pb - pg (m)",
    "vb": "Ball velocity vb (m/s)",
    "ab": "Ball acceleration ab (m/s^2)",
    "theta": "Theta (rad)",
    "omega": "Omega (rad/s)",
    "alpha": "Alpha (rad/s^2)",
    "drz": "Drone z displacement (m)",
    "vrz": "Drone z velocity (m/s)",
    "arz": "Drone z acceleration (m/s^2)",
    "actions": "Policy action",
    "step_last_action": "Last action",
    "step_command_z": "Command z",
    "step_executed_command_z": "Executed command z",
    "step_vrz_cmd": "Velocity command z (m/s)",
    "step_executed_vrz_cmd": "Executed velocity command z (m/s)",
    "step_drz_cmd": "Position command z (m)",
    "step_executed_drz_cmd": "Executed position command z (m)",
    "step_target_position_z": "Target position z (m)",
    "step_frz_cmd": "Force command z (N)",
    "step_executed_frz_cmd": "Executed force command z (N)",
    "step_delta_frz_cmd": "Delta force command z (N)",
    "step_hover_force": "Hover force (N)",
    "step_target_pitch": "Target pitch (rad)",
    "step_target_height_acc": "Target height acceleration (m/s^2)",
    "step_beta": "Beta",
    "rewards": "Reward",
    "terminated": "Terminated",
    "truncated": "Truncated",
    "policy_compute_time": "Policy compute time (s)",
}


@dataclass(frozen=True)
class PlotSeries:
    """One plottable rollout signal."""

    key: str
    label: str
    values: np.ndarray


def load_rollout_data(rollout_path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed rollout npz into a plain dictionary."""
    rollout_path = Path(rollout_path).expanduser().resolve()
    with np.load(rollout_path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def build_time_axis(
    rollout_data: dict[str, np.ndarray],
    rollout_path: str | Path | None = None,
    step_dt: float | None = None,
) -> np.ndarray:
    """Build a rollout time axis in seconds."""
    steps = _infer_step_count(rollout_data)
    dt = float(step_dt) if step_dt is not None else _infer_step_dt(rollout_path)
    return np.arange(steps, dtype=np.float64) * dt


def collect_plot_series(rollout_data: dict[str, np.ndarray], keys: Iterable[str]) -> list[PlotSeries]:
    """Collect available plottable series by key, skipping missing or all-NaN arrays."""
    series_map = _build_series_map(rollout_data)
    series: list[PlotSeries] = []
    for key in keys:
        values = series_map.get(key)
        if values is None or not _has_finite_data(values):
            continue
        series.append(PlotSeries(key=key, label=_label_for_key(key), values=values))
    return series


def plot_rollout_groups(
    rollout_data: dict[str, np.ndarray],
    output_dir: str | Path,
    time_axis: np.ndarray,
    groups: Iterable[str] = ("all",),
    keys: Iterable[str] | None = None,
    env_ids: str | Iterable[int] = "all",
    dpi: int = 150,
) -> list[Path]:
    """Plot selected rollout groups and return generated image paths."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if keys is not None:
        output_path = output_dir / "custom.png"
        series = collect_plot_series(rollout_data, keys)
        return [_plot_series(series, time_axis, output_path, "Custom rollout signals", env_ids, dpi)] if series else []

    group_names = _expand_groups(groups)
    output_paths: list[Path] = []
    for group in group_names:
        series = collect_plot_series(rollout_data, _keys_for_group(group, rollout_data))
        if not series:
            continue
        output_path = output_dir / GROUP_OUTPUT_FILES[group]
        output_paths.append(_plot_series(series, time_axis, output_path, _title_for_group(group), env_ids, dpi))
    return output_paths


def available_series_keys(rollout_data: dict[str, np.ndarray]) -> list[str]:
    """Return all plottable signal keys available in a rollout."""
    return sorted(key for key, values in _build_series_map(rollout_data).items() if _has_finite_data(values))


def _build_series_map(rollout_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}

    observations = rollout_data.get("observations")
    observation_fields = rollout_data.get("observation_fields")
    if observations is not None and observation_fields is not None and observations.ndim == 3:
        fields = [_decode_field_name(field) for field in observation_fields]
        for index, field in enumerate(fields):
            if index < observations.shape[-1]:
                series[field] = observations[..., index]

    for key, values in rollout_data.items():
        if key in {"observations", "observation_fields"} or values.dtype.kind in {"U", "S", "O"}:
            continue
        _add_array_series(series, key, values)

    pb = series.get("pb")
    pg = series.get("pg")
    if pb is None or pg is None:
        pb = series.get("step_pb")
        pg = series.get("step_pg")
    if pb is not None and pg is not None and pb.shape == pg.shape:
        series["position_error"] = pb - pg

    return series


def _add_array_series(series: dict[str, np.ndarray], key: str, values: np.ndarray):
    values = np.asarray(values)
    if values.ndim == 1 or values.ndim == 2:
        series[key] = values
    elif values.ndim == 3 and values.shape[-1] == 1:
        series[key] = values[..., 0]
    elif values.ndim == 3:
        for index in range(values.shape[-1]):
            series[f"{key}_{index}"] = values[..., index]


def _keys_for_group(group: str, rollout_data: dict[str, np.ndarray]) -> tuple[str, ...]:
    if group == "states":
        return STATE_KEYS
    if group == "actions":
        return ACTION_COMMAND_KEYS
    if group == "policy":
        return tuple(key for key in rollout_data if key.startswith("policy_") and key != "policy_compute_time")
    raise ValueError(f"Unsupported plot group: {group}")


def _expand_groups(groups: Iterable[str]) -> tuple[str, ...]:
    groups = tuple(groups)
    if not groups or "all" in groups:
        return ("states", "actions", "policy")
    allowed = set(GROUP_OUTPUT_FILES)
    unknown = [group for group in groups if group not in allowed]
    if unknown:
        raise ValueError(f"Unsupported plot group(s): {', '.join(unknown)}")
    return groups


def _plot_series(
    series: list[PlotSeries],
    time_axis: np.ndarray,
    output_path: Path,
    title: str,
    env_ids: str | Iterable[int],
    dpi: int,
) -> Path:
    fig_height = max(3.0 * len(series), 4.0)
    fig, axes = plt.subplots(len(series), 1, figsize=(16, fig_height), sharex=True, squeeze=False)
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, series):
        values = _align_steps(item.values, time_axis.size)
        x = time_axis[: values.shape[0]]
        _plot_values(ax, x, values, env_ids)
        ax.set_ylabel(item.label)
        ax.grid(True, linewidth=0.8, alpha=0.35)

    axes_flat[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _plot_values(ax, time_axis: np.ndarray, values: np.ndarray, env_ids: str | Iterable[int]):
    values = _as_numeric(values)
    if values.ndim == 1:
        ax.plot(time_axis, values, linewidth=1.8, color="tab:blue")
        return

    selected_envs = _resolve_env_ids(env_ids, values.shape[1])
    selected_values = values[:, selected_envs]
    alpha = 0.08 if selected_values.shape[1] > 50 else 0.25
    linewidth = 0.55 if selected_values.shape[1] > 50 else 0.9
    for env_index in range(selected_values.shape[1]):
        ax.plot(time_axis, selected_values[:, env_index], linewidth=linewidth, alpha=alpha, color="tab:blue")


def _resolve_env_ids(env_ids: str | Iterable[int], num_envs: int) -> np.ndarray:
    if isinstance(env_ids, str):
        if env_ids == "all":
            return np.arange(num_envs, dtype=np.int64)
        raw_ids = [part.strip() for part in env_ids.split(",") if part.strip()]
        selected = np.asarray([int(part) for part in raw_ids], dtype=np.int64)
    else:
        selected = np.asarray(list(env_ids), dtype=np.int64)

    if selected.size == 0:
        raise ValueError("At least one environment id must be selected.")
    if np.any(selected < 0) or np.any(selected >= num_envs):
        raise ValueError(f"Environment ids must be in [0, {num_envs - 1}].")
    return selected


def _align_steps(values: np.ndarray, step_count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] == step_count:
        return values
    return values[: min(values.shape[0], step_count)]


def _as_numeric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype == np.bool_:
        return values.astype(np.float32)
    return values


def _has_finite_data(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.dtype.kind not in {"b", "i", "u", "f"}:
        return False
    if values.size == 0:
        return False
    if values.dtype.kind == "f":
        return not np.all(np.isnan(values))
    return True


def _infer_step_count(rollout_data: dict[str, np.ndarray]) -> int:
    if "observations" in rollout_data:
        return int(rollout_data["observations"].shape[0])
    for values in rollout_data.values():
        if getattr(values, "ndim", 0) > 0 and values.dtype.kind not in {"U", "S", "O"}:
            return int(values.shape[0])
    raise ValueError("Unable to infer rollout step count.")


def _infer_step_dt(rollout_path: str | Path | None) -> float:
    if rollout_path is None:
        return DEFAULT_STEP_DT

    rollout_path = Path(rollout_path).expanduser().resolve()
    run_dir = rollout_path.parent
    resolved_run = _load_yaml(run_dir / "resolved_run.yaml")
    for key in ("policy_step_dt", "step_dt"):
        value = resolved_run.get(key)
        if value is not None:
            return float(value)

    for config_path in _candidate_config_paths(run_dir, resolved_run):
        config = _load_yaml(config_path)
        step_dt = _step_dt_from_config(config)
        if step_dt is not None:
            return step_dt

    return DEFAULT_STEP_DT


def _candidate_config_paths(run_dir: Path, resolved_run: dict) -> list[Path]:
    candidates = [run_dir / "input_env_config.yaml", run_dir / "input_config.yaml"]
    for key in ("env_config_path", "config_path"):
        value = resolved_run.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    return [path.resolve() for path in candidates if path.exists()]


def _step_dt_from_config(config: dict) -> float | None:
    if not config:
        return None
    sim_cfg = config.get("sim", {})
    dt = float(sim_cfg.get("dt", 1.0 / 180.0))
    decimation = int(sim_cfg.get("decimation", config.get("decimation", 3)))
    if "dt" in sim_cfg or "decimation" in sim_cfg or "decimation" in config:
        return dt * decimation
    return None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _decode_field_name(field) -> str:
    if isinstance(field, bytes):
        return field.decode("utf-8")
    return str(field)


def _label_for_key(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    if key.startswith("benchmark_"):
        return key.removeprefix("benchmark_").replace("_", " ")
    if key.startswith("policy_"):
        return key.removeprefix("policy_").replace("_", " ")
    if key.startswith("step_"):
        return key.removeprefix("step_").replace("_", " ")
    return key.replace("_", " ")


def _title_for_group(group: str) -> str:
    return {
        "states": "System states",
        "actions": "Actions and commands",
        "policy": "Policy internal signals",
    }[group]
