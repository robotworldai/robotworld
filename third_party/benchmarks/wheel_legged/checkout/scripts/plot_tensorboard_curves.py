#!/usr/bin/env python3

"""Export the main wheel-legged training metrics from a TensorBoard event file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PANELS = (
    (
        "Training performance",
        (
            ("Train/mean_reward", "Mean reward", None),
            ("Train/mean_episode_length", "Episode length", None),
        ),
    ),
    (
        "Moving-jump curriculum",
        (
            ("Episode/moving_jump_curriculum_level", "Curriculum level", 4.0),
            ("Episode/curriculum_score", "Velocity tracking score", 0.77),
        ),
    ),
    (
        "Curriculum pass metrics",
        (
            ("Episode/moving_jump_curriculum_success", "Jump success", 0.75),
            ("Episode/moving_jump_curriculum_soft", "Soft landing", 0.85),
            ("Episode/moving_jump_curriculum_heading", "Heading", 0.80),
        ),
    ),
    (
        "Jump capability",
        (
            ("Episode/jump_wheel_clearance", "Wheel clearance (m)", 0.10),
            ("Episode/jump_air_time", "Air time (s)", None),
            ("Episode/jump_apex_rise", "Base apex rise (m)", None),
        ),
    ),
    (
        "Velocity tracking",
        (
            ("Episode/cmd_vx_limit", "Command limit (m/s)", 1.0),
            ("Episode/vx_tracking_gain", "Tracking gain", None),
            ("Episode/vx_err_inst", "Instant error (m/s)", None),
        ),
    ),
    (
        "Safety",
        (
            ("Episode/torque_saturation", "Torque saturation", 0.05),
            ("Episode/jump_fail_recovery_rate", "Recovery failure", 0.02),
            ("Episode/tilt_angle", "Tilt angle (rad)", None),
        ),
    ),
)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), smoothed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=15)
    parser.add_argument("--title", default="Wheel-Legged Moving-Jump Training")
    args = parser.parse_args()

    accumulator = EventAccumulator(
        str(args.event_file.expanduser().resolve()), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle(args.title, fontsize=18, fontweight="bold")

    for axis, (panel_title, series) in zip(axes.flat, PANELS):
        thresholds: set[float] = set()
        for tag, label, threshold in series:
            if tag not in available:
                continue
            events = accumulator.Scalars(tag)
            steps = np.asarray([event.step for event in events])
            values = np.asarray([event.value for event in events], dtype=np.float64)
            axis.plot(
                steps,
                moving_average(values, args.smooth),
                linewidth=1.8,
                label=label,
            )
            if threshold is not None:
                thresholds.add(threshold)
        for threshold in sorted(thresholds):
            axis.axhline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
            )
        axis.set_title(panel_title, fontweight="bold")
        axis.set_xlabel("Iteration")
        axis.legend(fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training curves to: {args.output}")


if __name__ == "__main__":
    main()
