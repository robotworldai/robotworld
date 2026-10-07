"""BatchSyncInferenceExecutor: env-keyed buffering and batched replan triggers."""

import numpy as np
import pytest

from openwam.deploy.executors import BatchSyncInferenceExecutor


class _FakeBatchEngine:
    """Returns per-request action chunks encoding (request marker, step index)."""

    def __init__(self, chunk_len=3, action_dim=2):
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.calls: list = []

    def generate_batch(self, conditions_list):
        self.calls.append([c["marker"] for c in conditions_list])
        actions = np.zeros((len(conditions_list), self.chunk_len, self.action_dim), dtype=np.float32)
        for row, cond in enumerate(conditions_list):
            for t in range(self.chunk_len):
                actions[row, t] = (cond["marker"], t)
        return {"actions": actions, "video": None}


def test_batch_executor_pops_chunks_and_replans_together():
    engine = _FakeBatchEngine(chunk_len=3)
    ex = BatchSyncInferenceExecutor(engine)
    env_ids = ["env0", "env1"]

    for step in range(3):
        conds = [{"marker": 10.0}, {"marker": 20.0}]
        out = ex.predict_action_batch(conds, env_ids=env_ids)
        assert [tuple(a) for a in out] == [(10.0, float(step)), (20.0, float(step))]
    assert engine.calls == [[10.0, 20.0]]

    # Buffers exhausted -> next step replans both envs in ONE batched call.
    out = ex.predict_action_batch([{"marker": 11.0}, {"marker": 21.0}], env_ids=env_ids)
    assert engine.calls == [[10.0, 20.0], [11.0, 21.0]]
    assert [tuple(a) for a in out] == [(11.0, 0.0), (21.0, 0.0)]


def test_batch_executor_keys_buffers_by_env_id_not_position():
    engine = _FakeBatchEngine(chunk_len=2)
    ex = BatchSyncInferenceExecutor(engine)

    ex.predict_action_batch([{"marker": 1.0}, {"marker": 2.0}, {"marker": 3.0}], env_ids=[0, 1, 2])
    # Env 1 removed mid-run; remaining envs arrive in a different order and
    # must keep consuming their own buffers without a replan.
    out = ex.predict_action_batch([{"marker": 99.0}, {"marker": 98.0}], env_ids=[2, 0])
    assert [tuple(a) for a in out] == [(3.0, 1.0), (1.0, 1.0)]
    assert len(engine.calls) == 1


def test_batch_executor_replans_only_empty_buffers():
    engine = _FakeBatchEngine(chunk_len=2)
    ex = BatchSyncInferenceExecutor(engine)

    ex.predict_action_batch([{"marker": 1.0}], env_ids=["a"])  # a: 1 action left
    out = ex.predict_action_batch([{"marker": 5.0}, {"marker": 6.0}], env_ids=["a", "b"])
    # a pops its buffered step-1 action; only b was replanned (fresh chunk).
    assert tuple(out[0]) == (1.0, 1.0)
    assert tuple(out[1]) == (6.0, 0.0)
    assert engine.calls == [[1.0], [6.0]]


def test_batch_executor_inference_horizon_truncates_chunks():
    engine = _FakeBatchEngine(chunk_len=3)
    ex = BatchSyncInferenceExecutor(engine, inference_horizon=1)

    ex.predict_action_batch([{"marker": 1.0}], env_ids=["a"])
    ex.predict_action_batch([{"marker": 2.0}], env_ids=["a"])
    # horizon=1 -> every step replans.
    assert engine.calls == [[1.0], [2.0]]


def test_batch_executor_reset_and_reset_env():
    engine = _FakeBatchEngine(chunk_len=3)
    ex = BatchSyncInferenceExecutor(engine)
    ex.predict_action_batch([{"marker": 1.0}, {"marker": 2.0}], env_ids=["a", "b"])

    ex.reset_env("a")
    ex.predict_action_batch([{"marker": 7.0}, {"marker": 8.0}], env_ids=["a", "b"])
    assert engine.calls[-1] == [7.0]  # only a replanned

    ex.reset()
    ex.predict_action_batch([{"marker": 9.0}, {"marker": 10.0}], env_ids=["a", "b"])
    assert engine.calls[-1] == [9.0, 10.0]


def test_batch_executor_rejects_duplicate_env_ids_and_misalignment():
    engine = _FakeBatchEngine()
    ex = BatchSyncInferenceExecutor(engine)
    with pytest.raises(ValueError, match="duplicates"):
        ex.predict_action_batch([{"marker": 1.0}, {"marker": 2.0}], env_ids=["a", "a"])
    with pytest.raises(ValueError, match="align"):
        ex.predict_action_batch([{"marker": 1.0}], env_ids=["a", "b"])
