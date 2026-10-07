from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = normalize_quat(q.astype(np.float64))
    return quat_mul(quat_mul(q, np.array([0.0, *v], dtype=np.float64)), quat_conj(q))[1:]


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = normalize_quat(q.astype(np.float64))
    return quat_mul(quat_mul(quat_conj(q), np.array([0.0, *v], dtype=np.float64)), q)[1:]


def gravity_orientation(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q.astype(np.float64))
    qw, qx, qy, qz = q
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def sample_disk(rng: np.random.Generator, radius: float) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    radial = radius * math.sqrt(rng.uniform(0.0, 1.0))
    return np.array([radial * math.cos(angle), radial * math.sin(angle)], dtype=np.float64)


def parse_csv_floats(value: str) -> list[float]:
    if value.strip() == "":
        return []
    return [float(x) for x in value.split(",") if x.strip()]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value):
        return {k: as_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: as_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    return value
