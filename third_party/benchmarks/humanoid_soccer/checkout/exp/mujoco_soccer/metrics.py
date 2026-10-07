from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .math_utils import as_jsonable
from .schemas import ExperimentConfig, TrialResult


def goal_crossed(
    prev_ball: np.ndarray,
    curr_ball: np.ndarray,
    goal_center: np.ndarray,
    goal_normal: np.ndarray,
    goal_width: float,
) -> bool:
    prev_rel = prev_ball[:2] - goal_center[:2]
    curr_rel = curr_ball[:2] - goal_center[:2]
    prev_s = float(np.dot(prev_rel, goal_normal))
    curr_s = float(np.dot(curr_rel, goal_normal))
    if prev_s > 0.0 or curr_s < 0.0:
        return False
    denom = prev_s - curr_s
    alpha = 1.0 if abs(denom) < 1.0e-9 else prev_s / denom
    alpha = float(np.clip(alpha, 0.0, 1.0))
    crossing = prev_ball[:2] + alpha * (curr_ball[:2] - prev_ball[:2])
    tangent = np.array([-goal_normal[1], goal_normal[0]], dtype=np.float64)
    lateral = abs(float(np.dot(crossing - goal_center[:2], tangent)))
    return lateral <= 0.5 * goal_width


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    def finite_mean(values: list[float]) -> float:
        finite = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
        return float(finite.mean()) if finite.size else math.nan

    return {
        "num_trials": len(results),
        "success_goal_rate": float(np.mean([r.success for r in results])) if results else math.nan,
        "is_succ_phc": finite_mean([r.is_succ_phc for r in results]),
        "is_succ_phc_rel": finite_mean([r.is_succ_phc_rel for r in results]),
        "E_gmpjpe": finite_mean([r.E_gmpjpe for r in results]),
        "E_mpjpe": finite_mean([r.E_mpjpe for r in results]),
        "E_dof_mpjpe": finite_mean([r.E_dof_mpjpe for r in results]),
        "E_vel": finite_mean([r.E_vel for r in results]),
        "E_acc": finite_mean([r.E_acc for r in results]),
        "E_contact_acc": finite_mean([r.E_contact_acc for r in results]),
        "seq_jerk": finite_mean([r.seq_jerk for r in results]),
        "seq_ref_jerk": finite_mean([r.seq_ref_jerk for r in results]),
        "global_mpjpe": finite_mean([r.global_mpjpe for r in results]),
        "mpjpe": finite_mean([r.mpjpe for r in results]),
        "vel_error": finite_mean([r.vel_error for r in results]),
        "ball_vel_dir_alignment_angle": finite_mean([r.ball_vel_dir_alignment_angle for r in results]),
        "max_ball_speed": finite_mean([r.max_ball_speed for r in results]),
        "first_contact_rate": float(np.mean([np.isfinite(r.first_contact_time) for r in results])) if results else math.nan,
        "mean_spawn_attempts": finite_mean([float(r.spawn_attempts) for r in results]),
    }


def write_outputs(cfg: ExperimentConfig, xml_path: Path, results: list[TrialResult]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = as_jsonable(cfg)
        payload["generated_mjcf"] = str(xml_path)
        json.dump(payload, f, indent=2)

    rows = [dataclasses.asdict(result) for result in results]
    if rows:
        with (cfg.output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with (cfg.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(as_jsonable(summarize(results)), f, indent=2)
