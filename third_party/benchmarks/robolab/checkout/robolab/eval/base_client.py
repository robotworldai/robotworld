# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is the source of truth. A verbatim copy lives at
# droid_plus/eval/base_client.py — keep both in sync when editing.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceClient(ABC):
    """Root client for policy inference.

    Subclass override surface, in order of increasing commitment:

    1. Implement the four hooks (``_extract_observation``, ``_pack_request``,
       ``_query_server``, ``_unpack_response``). Chunking, env-id bookkeeping,
       visualization, and reset are handled by the base.
    2. Additionally override ``_postprocess_chunk`` or ``_build_visualization``
       for action-space / logging quirks (gripper binarization, 7->8 padding).
    3. Override ``infer`` entirely if your flow isn't query-then-step-chunk
       (e.g. server-side session state, pre-step caching).

    Two hooks are meant to be split per concern:

      ``_extract_observation``  <- repo-specific (real-robot flat numpy dict vs
                                   sim nested torch batched dict)
      ``_pack_request``         <- backend-specific (wire keys, image sizes)

    Keeping these separate lets the same backend client be paired with
    different observation sources without duplicating the wire format.
    """

    # Subclasses override to match their server's chunk length.
    # horizon=1 is correct for single-action servers.
    open_loop_horizon: int = 1

    def __init__(self) -> None:
        # Per-env chunking state. Subclasses may ignore and manage state however
        # they want.
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}
        # Set by begin_episode(); see below.
        self._eval_episode_idx: int = 0

    def begin_episode(self, episode_idx: int) -> None:
        """Notify the client that a new episode is starting.

        Called by ``run_episode`` before the first inference of each episode.
        The default stores the index on ``self._eval_episode_idx``. Clients
        whose server keeps state across a persistent connection can use it to
        mark episode boundaries in their requests; subclasses that need to
        notify a server-side session should override and call
        ``super().begin_episode(episode_idx)``.
        """
        self._eval_episode_idx = episode_idx


    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        """Return ``{"action": np.ndarray, "viz": np.ndarray | None}``.

        Default flow: extract -> pack -> query -> unpack -> postprocess ->
        cache chunk -> step one action. Override entirely if your client needs
        a different control loop.
        """
        extracted = self._extract_observation(obs, env_id=env_id)

        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            chunk = self._unpack_response(response)
            chunk = self._postprocess_chunk(chunk)
            self._set_chunk(env_id, chunk)

        action = self._next_action(env_id)
        viz = self._build_visualization(extracted)
        return {"action": action, "viz": viz}

    def infer_batch(
        self, obs: Any, instruction: str, *, env_ids: list[int]
    ) -> dict[int, dict]:
        """Infer actions for several envs in one call.

        Returns ``{env_id: {"action": ..., "viz": ...}}`` with one entry per
        requested env. Default implementation is a serial loop over
        :meth:`infer` — behavior-identical to the historical per-env eval
        loop, so every existing client works unchanged. Clients whose server
        supports batched payloads can override this to issue one request for
        all envs needing a replan.
        """
        return {
            env_id: self.infer(obs, instruction, env_id=env_id)
            for env_id in env_ids
        }

    def reset(self, *, env_id: int | None = None) -> None:
        """Clear per-episode state. ``env_id=None`` resets all envs.

        Subclasses with server-side session state should override to notify
        the server, then call ``super().reset(env_id=env_id)``.
        """
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def close(self) -> None:
        """Release transport resources. Default: no-op."""
        return None

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        """Public convenience wrapper for callers that want the viz image."""
        return self._build_visualization(self._extract_observation(obs, env_id=env_id))

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict:
        """Convert the caller's native obs into a flat dict of numpy arrays.

        Repo-specific seam. Return whatever keys ``_pack_request`` expects;
        the contract between these two methods is owned by the subclass pair.
        """

    @abstractmethod
    def _pack_request(self, extracted_obs: dict, instruction: str) -> Any:
        """Build the server's wire-format request. Backend-specific."""

    @abstractmethod
    def _query_server(self, request: Any) -> Any:
        """Send the request and return the raw response. Transport-specific."""

    @abstractmethod
    def _unpack_response(self, response: Any) -> np.ndarray:
        """Return a ``(horizon, action_dim)`` numpy array."""

    # ------------------------------------------------------------------
    # Observation helpers (generic; no backend or camera knowledge)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_obs_term(raw_obs: Any, term_name: str) -> Any | None:
        """Look up an observation term by name across all observation groups.

        Scans every mapping-valued entry of ``raw_obs`` (observation groups
        like ``image_obs`` / ``viewport_cam`` / ``proprio_obs``) and returns
        the first batched value stored under ``term_name``, or None if no
        group has it. Term names are unique across groups in RoboLab
        registrations.
        """
        if not isinstance(raw_obs, dict):
            return None
        for group in raw_obs.values():
            getter = getattr(group, "get", None)
            if getter is None:
                continue
            value = getter(term_name)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_env_gt_state(raw_obs: Any, env_id: int) -> dict | None:
        """Return this env's ground-truth state snapshot, or None.

        ``run_episode`` attaches ``obs["gt_state"] = {env_id: state}`` when
        ``--enable-gt-state`` is set (schema: ``robolab/eval/gt_state.py``).
        """
        gt_state = raw_obs.get("gt_state") if isinstance(raw_obs, dict) else None
        if isinstance(gt_state, dict) and env_id in gt_state:
            return gt_state[env_id]
        return None

    @staticmethod
    def _to_numpy(value: Any, env_id: int = 0) -> np.ndarray:
        """Convert a batched tensor/array to NumPy, selecting one env's row."""
        if hasattr(value, "detach"):
            return value[env_id].detach().cpu().numpy()
        array = np.asarray(value)
        return array[env_id] if array.ndim > 0 else array

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Action post-processing (binarization, padding, sign flips).

        Default: identity.
        """
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray | None:
        """Image for logging/recording. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Chunking helpers (usable or ignorable by subclasses)
    # ------------------------------------------------------------------

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action
