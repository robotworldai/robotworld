# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore a recorded environment configuration for episode playback.

Every run serializes its resolved env config to ``env_cfg.json`` (see
:func:`robolab.core.environments.runtime.create_env`). Replay can overlay that
recorded config onto a freshly built config so playback uses the exact values
(object poses, physics params, termination params, seed, ...) the episode was
recorded with, instead of whatever the current repo's task definitions produce.

The overlay is value-level, mirroring ``isaaclab.utils.dict.update_class_from_dict``
but lenient: fields that no longer exist or changed type are skipped and
reported instead of raising, since drift between the recording-time and
current config schema is exactly the scenario being handled. It restores
config *values*, not code — changed predicate implementations or changed USD
file contents on disk are out of scope.
"""

import json
import os
import re

from isaaclab.utils.string import string_to_callable

from robolab.constants import ASSET_DIR

# Namespaces that must come from the current invocation (host/runtime state),
# never from the recording. ``num_envs``/``env_spacing`` are CLI choices,
# ``sim/device`` and rendering are host-specific, and the replay run manages
# its own recorders.
_RUNTIME_KEY_NAMESPACES = {
    "/viewer",
    "/sim/device",
    "/sim/render",
    "/recorders",
    "/renderer",
    "/policy",
    "/num_envs",
    "/env_spacing",
    "/scene/num_envs",
    "/scene/env_spacing",
}


def load_recorded_env_cfg(hdf5_path: str) -> tuple[dict, str] | None:
    """Load the ``env_cfg.json`` sidecar recorded next to an HDF5 episode file.

    Args:
        hdf5_path: Path to the recorded ``data.hdf5``; the sidecar is expected
            in the same directory.

    Returns:
        ``(config_dict, sidecar_path)`` if the sidecar exists, else ``None``.
    """
    sidecar_path = os.path.join(os.path.dirname(os.path.abspath(hdf5_path)), "env_cfg.json")
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path) as f:
        return json.load(f), sidecar_path


def apply_recorded_env_cfg(env_cfg, recorded: dict) -> list[str]:
    """Overlay a recorded ``env_cfg.json`` dict onto a live env config instance.

    ``env_cfg`` must be a freshly built config for the same task (from
    ``parse_env_cfg``): it provides the class structure, types, and callable
    defaults that the JSON serialization is lossy about. Recorded values are
    written over it in place; runtime/host namespaces (device, num_envs,
    rendering, recorders) are left untouched.

    Args:
        env_cfg: The env config instance to update in place.
        recorded: The parsed ``env_cfg.json`` dict.

    Returns:
        List of skipped field namespaces with reasons (empty when the recorded
        config matched the current schema exactly).
    """
    skipped: list[str] = []
    _overlay(env_cfg, recorded, "", skipped)
    return skipped


def _string_to_slice(value: str) -> slice:
    """Parse ``str(slice(...))`` output (tolerates the spaces ``json.dump(default=str)`` keeps)."""
    match = re.match(r"slice\((.*),(.*),(.*)\)", value)
    if match is None:
        raise ValueError(f"Invalid slice string: {value}")
    parts = [part.strip() for part in match.groups()]
    return slice(*(None if part == "None" else int(part) for part in parts))


def _reroot_asset_path(value: str) -> str:
    """Rewrite a recorded absolute asset path onto the current checkout's asset dir."""
    marker = f"{os.sep}assets{os.sep}"
    if value.startswith(os.sep) and marker in value:
        return os.path.join(ASSET_DIR, value.split(marker, 1)[1])
    return value


def _assign(obj, key: str, value) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _overlay(obj, data: dict, ns: str, skipped: list[str]) -> None:
    """Recursively write recorded values onto ``obj``, collecting mismatches in ``skipped``."""
    for key, value in data.items():
        key_ns = f"{ns}/{key}"
        if key_ns in _RUNTIME_KEY_NAMESPACES:
            continue
        if isinstance(obj, dict):
            if key not in obj:
                # Plain dicts (event params, carb settings) accept recorded-only keys.
                obj[key] = value
                continue
            target = obj[key]
        else:
            if not hasattr(obj, key):
                skipped.append(f"{key_ns} (not in current config)")
                continue
            target = getattr(obj, key)

        # The recorded instruction is the resolved string; the current config may
        # still hold the pre-resolution variants dict. Faithful replay keeps the
        # recorded string (create_env's resolve_instruction passes strings through).
        if key_ns == "/instruction" and isinstance(value, str):
            _assign(obj, key, value)
            continue

        if isinstance(value, dict):
            if isinstance(target, dict) or hasattr(target, "__dict__"):
                _overlay(target, value, key_ns, skipped)
            else:
                skipped.append(f"{key_ns} (recorded a config group, current value is {type(target).__name__})")
            continue

        if isinstance(value, list):
            if any(isinstance(element, dict) for element in value):
                if isinstance(target, (list, tuple)) and len(target) == len(value):
                    for i, element in enumerate(value):
                        if isinstance(element, dict):
                            _overlay(target[i], element, f"{key_ns}[{i}]", skipped)
                else:
                    skipped.append(f"{key_ns} (nested list shape changed)")
                continue
            _assign(obj, key, tuple(value) if isinstance(target, tuple) else value)
            continue

        if isinstance(value, str):
            if value.startswith("slice("):
                try:
                    _assign(obj, key, _string_to_slice(value))
                except ValueError:
                    skipped.append(f"{key_ns} (unparsable slice string)")
                continue
            if target is not None and callable(target):
                try:
                    _assign(obj, key, string_to_callable(value))
                except Exception:
                    skipped.append(f"{key_ns} (callable '{value}' not importable)")
                continue
            _assign_scalar(obj, key, _reroot_asset_path(value), target, key_ns, skipped)
            continue

        _assign_scalar(obj, key, value, target, key_ns, skipped)


def _assign_scalar(obj, key: str, value, target, key_ns: str, skipped: list[str]) -> None:
    is_compatible = (
        value is None
        or target is None
        or isinstance(value, type(target))
        or (isinstance(value, int) and isinstance(target, float))
    )
    if is_compatible:
        _assign(obj, key, value)
    else:
        skipped.append(f"{key_ns} (type mismatch: current {type(target).__name__}, recorded {type(value).__name__})")
