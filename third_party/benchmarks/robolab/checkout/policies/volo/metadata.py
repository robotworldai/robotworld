# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin that adds VoLo orchestrator metadata to a backend client's requests.

Kept free of backend imports (openpi-client etc.) so it can be unit-tested
without any policy-server client library installed.
"""

import numpy as np

# An object counts as "lifted" once raised this far (meters) above the
# position it had when this client first saw it in the episode.
LIFT_THRESHOLD_M = 0.03
# Grasp detection only counts once the gripper is at least this closed.
GRASP_CLOSEDNESS_THRESHOLD = 0.25

# VoLo server wire keys, mapped from the observation terms that feed them.
# The pos/quat/K terms exist only for depth-enabled registrations (see
# robolab.core.observations.observation_utils). Camera pose is env-local;
# quaternion is world-frame OpenGL (w, x, y, z); K is in pixels.
ORCHESTRATOR_KEY_MAP: dict[str, str] = {
    "over_shoulder_left_camera_depth": "observation/depth_external",
    "egocentric_mirrored_camera_depth": "observation/depth_front",
    "egocentric_mirrored_camera": "observation/front_image_left_raw",
    "over_shoulder_left_camera_pos": "observation/camera_pos",
    "over_shoulder_left_camera_quat": "observation/camera_quat",
    "over_shoulder_left_camera_K": "observation/camera_K",
    "egocentric_mirrored_camera_pos": "observation/camera_pos_front",
    "egocentric_mirrored_camera_quat": "observation/camera_quat_front",
    "egocentric_mirrored_camera_K": "observation/camera_K_front",
}


class OrchestratorMetadataMixin:
    """Add inference-proxy metadata on top of a concrete ``InferenceClient``.

    Compose to the LEFT of the backend client so ``super()`` resolves to the
    backend implementation::

        class VoloCosmos3Client(OrchestratorMetadataMixin, Cosmos3Client):
            ...

    The resulting request is a strict superset of the backend's wire format:

    - depth, front RGB, camera calibration, and opt-in GT state, collected by
      :meth:`_orchestrator_keys` per :data:`ORCHESTRATOR_KEY_MAP` (the depth
      and calibration observation terms exist when environments were
      registered via :func:`policies.volo.registration.register_volo_envs`)
    - ``__episode_id`` so a persistent proxy connection can distinguish
      sequential episodes

    The core GT-state exporter emits a raw per-step snapshot (poses in the
    env-local frame, meters; quaternions world-frame (w, x, y, z)). This
    mixin derives the orchestrator's stateful fields from that stream —
    lift tracking against the episode-initial position and grasp detection
    gated on gripper closure — so the wire format matches what the VoLo
    server has always received. ``_extract_observation`` runs every env
    step (even when a cached action chunk skips the server round-trip), so
    the tracking never misses a step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Episode-initial object positions and running max lift, keyed
        # (env_id, object). Cleared by reset().
        self._gt_initial_positions: dict[tuple[int, str], np.ndarray] = {}
        self._gt_max_lift: dict[tuple[int, str], float] = {}

    # Class-level alias so subclasses can override the wire map.
    ORCHESTRATOR_KEY_MAP = ORCHESTRATOR_KEY_MAP

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        extracted["_orchestrator_keys"] = self._orchestrator_keys(raw_obs, env_id=env_id)
        return extracted

    def _orchestrator_keys(self, raw_obs, *, env_id: int = 0) -> dict:
        """Collect the VoLo proxy's metadata for one env's request.

        Uses the generic observation helpers from ``InferenceClient``
        (``_find_obs_term`` / ``_get_env_gt_state`` / ``_to_numpy``); the
        VoLo-specific part is only the wire-key mapping and the derived GT
        fields. Missing camera, depth, or ground-truth observations are
        intentionally ignored so the same client runs in ordinary RoboLab
        evaluations.
        """
        out: dict = {}
        for source, destination in self.ORCHESTRATOR_KEY_MAP.items():
            value = self._find_obs_term(raw_obs, source)
            if value is not None:
                out[destination] = self._to_numpy(value, env_id)
        gt_state = self._get_env_gt_state(raw_obs, env_id)
        if gt_state is not None:
            out["gt_state"] = self._derive_gt_fields(gt_state, env_id)
        return out

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["__episode_id"] = self._eval_episode_idx
        request.update(extracted_obs.get("_orchestrator_keys", {}))
        return request

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._gt_initial_positions.clear()
            self._gt_max_lift.clear()
        else:
            for key in [k for k in self._gt_initial_positions if k[0] == env_id]:
                self._gt_initial_positions.pop(key)
                self._gt_max_lift.pop(key, None)

    # ------------------------------------------------------------------
    # Derived GT fields (VoLo wire format on top of the raw core snapshot)
    # ------------------------------------------------------------------

    def _derive_gt_fields(self, state: dict, env_id: int) -> dict:
        """Add the orchestrator's derived fields to the raw core snapshot.

        Shallow-copies the dicts it amends so the obs-side payload (shared
        across clients in multi-env runs) is never mutated.
        """
        state = dict(state)
        state["objects"] = {
            name: self._track_object(name, obj, env_id)
            for name, obj in state.get("objects", {}).items()
        }
        if "robot" in state:
            state["robot"] = self._derive_grasp(dict(state["robot"]))
        return state

    def _track_object(self, name: str, obj: dict, env_id: int) -> dict:
        """Displacement / lift tracking vs. the episode-initial position."""
        obj = dict(obj)
        key = (env_id, name)
        if key not in self._gt_initial_positions:
            self._gt_initial_positions[key] = np.asarray(obj["pos"], dtype=np.float32).copy()
            self._gt_max_lift[key] = 0.0
        init_pos = self._gt_initial_positions[key]
        pos = np.asarray(obj["pos"], dtype=np.float32)
        z_lift = float(pos[2] - init_pos[2])
        self._gt_max_lift[key] = max(self._gt_max_lift[key], z_lift)
        obj["displacement"] = np.float32(np.linalg.norm(pos - init_pos))
        obj["z_lift"] = np.float32(z_lift)
        obj["max_z_lift"] = np.float32(self._gt_max_lift[key])
        obj["lifted"] = bool(self._gt_max_lift[key] > LIFT_THRESHOLD_M)
        return obj

    @staticmethod
    def _derive_grasp(robot: dict) -> dict:
        """Grasp detection, gated on gripper closure.

        Matches the historical wire behavior: with the gripper open, no
        contacts and no grasp are reported even if the contact sensor fires.
        """
        contacts = list(robot.get("objects_in_contact", []))
        if float(robot.get("gripper_closedness", 0.0)) > GRASP_CLOSEDNESS_THRESHOLD:
            robot["grasped_object"] = contacts[0] if contacts else None
            robot["objects_in_contact"] = contacts
        else:
            robot["grasped_object"] = None
            robot["objects_in_contact"] = []
        return robot
