"""Lightweight logging helpers for benchmark scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_yaml(data: dict[str, Any], path: str | Path):
    """Save a dictionary as YAML."""
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def append_csv_row(path: str | Path, row: dict[str, Any]):
    """Append one row to a CSV file, creating the header on first write."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fieldnames = list(row.keys())

    if not write_header:
        with open(csv_path, encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header

    with open(csv_path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
