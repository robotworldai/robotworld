#!/usr/bin/env python3
"""Summarize evaluation summary.csv files with pooled benchmark statistics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


NORMAL_METRICS = (
    "benchmark_success_rate",
    "policy_compute_time_mean",
)

POOLED_METRICS = {
    "benchmark_steady_state_error": "benchmark_steady_state_error_std",
    "benchmark_convergence_time": "benchmark_convergence_time_std",
    "benchmark_climbing_time": "benchmark_climbing_time_std",
}

WEIGHT_COLUMNS = (
    "benchmark_completed_episodes",
    "completed_episodes",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute aggregate mean/std statistics from an evaluation summary.csv. "
            "Metrics with per-row std columns are combined using pooled variance."
        )
    )
    parser.add_argument("summary_csv", type=str, help="Path to the evaluation summary.csv file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Defaults to summary_statistics.csv next to the input file.",
    )
    return parser.parse_args()


def _to_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _population_mean_std(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(variance, 0.0))


def _select_weight_column(fieldnames: list[str]) -> str | None:
    for column in WEIGHT_COLUMNS:
        if column in fieldnames:
            return column
    return None


def _get_weights(rows: list[dict[str, str]], fieldnames: list[str]) -> tuple[list[float], str]:
    weight_column = _select_weight_column(fieldnames)
    if weight_column is None:
        print(
            "[WARN] Neither benchmark_completed_episodes nor completed_episodes was found. "
            "Using weight=1 for every row.",
            file=sys.stderr,
        )
        return [1.0] * len(rows), "unit_weight"

    weights = []
    invalid_count = 0
    for row in rows:
        weight = _to_float(row.get(weight_column))
        if not math.isfinite(weight) or weight <= 0.0:
            invalid_count += 1
            weight = math.nan
        weights.append(weight)
    if invalid_count:
        print(
            f"[WARN] Ignoring {invalid_count} row(s) with invalid {weight_column} values "
            "for pooled metrics.",
            file=sys.stderr,
        )
    return weights, weight_column


def _pooled_mean_std(
    rows: list[dict[str, str]],
    weights: list[float],
    mean_column: str,
    std_column: str,
) -> tuple[float, float, float]:
    samples: list[tuple[float, float, float]] = []
    for row, weight in zip(rows, weights, strict=True):
        mean = _to_float(row.get(mean_column))
        std = _to_float(row.get(std_column))
        if math.isfinite(weight) and math.isfinite(mean) and math.isfinite(std):
            samples.append((weight, mean, std))

    total_weight = sum(weight for weight, _, _ in samples)
    if total_weight <= 0.0:
        return math.nan, math.nan, 0.0

    pooled_mean = sum(weight * mean for weight, mean, _ in samples) / total_weight
    pooled_variance = (
        sum(weight * (std**2 + (mean - pooled_mean) ** 2) for weight, mean, std in samples) / total_weight
    )
    return pooled_mean, math.sqrt(max(pooled_variance, 0.0)), total_weight


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    return rows, list(reader.fieldnames)


def _validate_columns(fieldnames: list[str]):
    required = set(NORMAL_METRICS)
    for mean_column, std_column in POOLED_METRICS.items():
        required.add(mean_column)
        required.add(std_column)
    missing = sorted(required - set(fieldnames))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_text}")


def _write_summary(output_path: Path, summary: dict[str, float | int | str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _print_summary(summary: dict[str, float | int | str]):
    print("[INFO] Aggregate statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def summarize(summary_csv: Path, output_path: Path) -> dict[str, float | int | str]:
    rows, fieldnames = _read_rows(summary_csv)
    _validate_columns(fieldnames)
    weights, weight_column = _get_weights(rows, fieldnames)

    total_completed_episodes = sum(weight for weight in weights if math.isfinite(weight) and weight > 0.0)
    summary: dict[str, float | int | str] = {
        "source_csv": str(summary_csv),
        "num_rows": len(rows),
        "weight_column": weight_column,
        "total_completed_episodes": total_completed_episodes,
    }

    for metric in NORMAL_METRICS:
        mean, std = _population_mean_std([_to_float(row.get(metric)) for row in rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

    for mean_column, std_column in POOLED_METRICS.items():
        mean, std, metric_weight = _pooled_mean_std(rows, weights, mean_column, std_column)
        summary[f"{mean_column}_mean"] = mean
        summary[f"{mean_column}_std"] = std
        summary[f"{mean_column}_total_episodes"] = metric_weight

    _write_summary(output_path, summary)
    return summary


def main():
    args = _parse_args()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else summary_csv.parent / "summary_statistics.csv"
    )

    try:
        summary = summarize(summary_csv, output_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    _print_summary(summary)
    print(f"[INFO] Saved aggregate statistics to: {output_path}")


if __name__ == "__main__":
    main()
