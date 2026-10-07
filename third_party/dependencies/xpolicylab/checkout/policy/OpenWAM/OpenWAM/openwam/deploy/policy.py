"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

from typing import Optional, Sequence

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    BatchSyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine

        self._execution_config = normalize_execution_config(execution_config)
        self._async = self._execution_config.enabled
        if self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
            )
        self._batch_executor: Optional[BatchSyncInferenceExecutor] = None

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation.

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        action = self._executor.predict_action(self._build_conditions(obs))
        return self._project_binary_dims(action)

    def predict_action_batch(
        self,
        obs_list: Sequence[dict],
        env_ids: Optional[Sequence] = None,
    ) -> list:
        """Return the next action for each of N parallel envs (one batched forward).

        Envs whose per-env buffer is empty are replanned together through
        ``engine.generate_batch`` — one GPU forward for the whole set — and
        every env then pops one action from its own buffer. Buffers are keyed
        by ``env_ids`` (defaults to list position), so a shrinking env set
        stays correctly aligned.

        Batch execution is sync-only: the async executor's background prefetch
        holds single-stream state that cannot serve interleaved envs.
        """
        if self._async:
            raise RuntimeError(
                "predict_action_batch requires inference.inference_mode=sync; "
                "the async executor is single-stream."
            )
        if self._batch_executor is None:
            self._batch_executor = BatchSyncInferenceExecutor(
                engine=self.engine,
                inference_horizon=self._execution_config.inference_horizon,
            )
        conditions_list = [self._build_conditions(obs) for obs in obs_list]
        actions = self._batch_executor.predict_action_batch(conditions_list, env_ids=env_ids)
        return [self._project_binary_dims(a) for a in actions]

    def _project_binary_dims(self, action) -> np.ndarray:
        """Final legality projection for two-point command dims (see predict_action)."""
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            action = np.array(action)
            for d in dims:
                if d >= action.shape[-1]:
                    raise ValueError(
                        f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                        "the ckpt config and the served action width disagree."
                    )
                action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state between episodes."""
        self._executor.reset()
        if self._batch_executor is not None:
            self._batch_executor.reset()

    def reset_env(self, env_id) -> None:
        """Clear one env's batch buffer (early termination / mid-run removal)."""
        if self._batch_executor is not None:
            self._batch_executor.reset_env(env_id)

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()
        if self._batch_executor is not None:
            self._batch_executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
