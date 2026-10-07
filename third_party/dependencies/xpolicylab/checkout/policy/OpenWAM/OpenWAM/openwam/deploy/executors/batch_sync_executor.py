"""Batched synchronous executor: per-env action buffers over one shared engine.

Serves N parallel environments stepping in lockstep (e.g. a RoboDojo client
running ``num_envs`` Isaac environments): each control step delivers one
observation per env, and envs whose buffer ran out are replanned together in a
single ``engine.generate_batch`` forward pass — the GPU sees a real batch, not
N sequential calls.

Buffers are keyed by caller-provided env ids (not list position), so the
mapping stays correct when the env set shrinks mid-run (e.g. PhysX instability
removes an env) or arrives in a different order.
"""

from collections import deque
from typing import Optional, Sequence

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine


class BatchSyncInferenceExecutor:
    """Receding-horizon execution of batched engine-generated action chunks.

    Args:
        engine: Inference engine exposing ``generate_batch(conditions_list)``.
        inference_horizon: Number of actions to execute before re-generating.
            ``None`` means consume the full chunk (greedy). Applied per env,
            identically to :class:`SyncInferenceExecutor`.
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        inference_horizon: Optional[int] = None,
    ):
        if not hasattr(engine, "generate_batch"):
            raise TypeError(f"{type(engine).__name__} does not implement generate_batch().")
        self.engine = engine
        if inference_horizon is not None and inference_horizon <= 0:
            raise ValueError("inference_horizon must be positive")
        self.inference_horizon = inference_horizon

        self._action_buffers: dict = {}

    def predict_action_batch(
        self,
        conditions_list: Sequence[dict],
        env_ids: Optional[Sequence] = None,
    ) -> list:
        """Return the next action for each env, replanning empty buffers in one batch.

        Args:
            conditions_list: one conditions dict per env, aligned with ``env_ids``.
            env_ids: hashable per-env keys; defaults to ``range(len(conditions_list))``.

        Returns:
            list of np.ndarray actions aligned with ``conditions_list``.
        """
        if env_ids is None:
            env_ids = list(range(len(conditions_list)))
        env_ids = list(env_ids)
        if len(env_ids) != len(conditions_list):
            raise ValueError(
                f"env_ids ({len(env_ids)}) and conditions_list ({len(conditions_list)}) must align."
            )
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"env_ids contains duplicates: {env_ids}")

        need = [i for i, eid in enumerate(env_ids) if not self._action_buffers.get(eid)]
        if need:
            self._generate_and_enqueue([conditions_list[i] for i in need], [env_ids[i] for i in need])

        return [self._action_buffers[eid].popleft() for eid in env_ids]

    def _generate_and_enqueue(self, conditions_list: Sequence[dict], env_ids: Sequence) -> None:
        """One batched inference; refill each env's buffer with its own chunk slice."""
        result = self.engine.generate_batch(list(conditions_list))
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[0] != len(env_ids):
            raise RuntimeError(
                f"generate_batch returned actions of shape {actions.shape}; "
                f"expected (B={len(env_ids)}, T, D)."
            )

        chunk_len = actions.shape[1]
        if chunk_len <= 0:
            raise RuntimeError("Batched inference result did not contain any actions")
        inference_horizon = self.inference_horizon if self.inference_horizon is not None else chunk_len
        if inference_horizon > chunk_len:
            raise ValueError(f"inference_horizon ({inference_horizon}) must be <= action horizon ({chunk_len})")

        for row, eid in enumerate(env_ids):
            buffer = deque()
            buffer.extend(actions[row, :inference_horizon])
            self._action_buffers[eid] = buffer

    def reset(self):
        """Clear every env's buffered actions between episode batches."""
        self._action_buffers.clear()

    def reset_env(self, env_id) -> None:
        """Clear one env's buffer (early termination / mid-run removal)."""
        self._action_buffers.pop(env_id, None)

    def shutdown(self):
        """No background resources; present for executor-interface symmetry."""
