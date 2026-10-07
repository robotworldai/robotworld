from types import SimpleNamespace
import numpy as np

from environment.benchmarks.robodojo.adapter import RoboDojoAdapter
from .test_eef_executor import observation, spec, plan


class FakeEnv:
    num_envs = 1
    step_lim = 100

    def __init__(self):
        self.take_action_cnt = [0]
        self.success = [True]
        self.state = dict(observation().state)
        self.actions = []

    def is_episode_end(self):
        return self.take_action_cnt[0] >= self.step_lim

    def get_obs(self):
        return {'state': self.state, 'instruction': 'test',
                'vision': {'head': {'color': np.zeros((8, 8, 3), dtype=np.uint8)}}}

    def take_action(self, data):
        self.actions.append(data)
        self.state.update(data)
        self.take_action_cnt[0] += 1


def test_native_action_and_rgb_roundtrip(tmp_path):
    env = FakeEnv()
    adapter = RoboDojoAdapter(env, output_dir=tmp_path, spec=spec())
    from environment.benchmarks.action_contracts import action_contract
    assert action_contract("robodojo") in adapter.instructions
    assert "There is no give_up tool" in adapter.instructions
    assert action_contract("robodojo") in adapter.tool_spec()["description"]
    adapter.executor._planner = plan
    adapter.observe_content()
    response, trace = adapter.move_eef({'targets': {'left_z': .91, 'left_gripper': 0}, 'note': 'test'})
    assert response['success']
    assert len(env.actions) == trace['planned_waypoints'] == trace['executed_waypoints']
    assert set(env.actions[0]) == {'left_arm_joint_state', 'right_arm_joint_state',
                                   'left_ee_joint_state', 'right_ee_joint_state'}
    assert any(x['type'] == 'inputImage' for x in response['contentItems'])
    assert adapter.official_success() is None


def test_rejection_executes_no_action(tmp_path):
    env = FakeEnv()
    adapter = RoboDojoAdapter(env, output_dir=tmp_path, spec=spec())
    response, trace = adapter.move_eef({'targets': {'left_z': 99}, 'note': 'test'})
    assert not response['success']
    assert not env.actions
    assert trace['executed_waypoints'] == 0


def test_history_samples_four_prior_rounds_without_padding(tmp_path):
    import json
    env = FakeEnv()
    adapter = RoboDojoAdapter(env, output_dir=tmp_path, spec=spec())
    first = adapter.observe_content()
    assert sum(p['type'] == 'inputImage' for p in first) == 1
    for i in range(1, 11):
        env.take_action_cnt[0] = i * 3
        parts = adapter.observe_content()
    history = json.loads((tmp_path / 'frames/10/history.json').read_text())
    assert [x['observation'] for x in history] == [2, 4, 6, 8, 10]
    assert [x['env_step'] for x in history] == [6, 12, 18, 24, 30]
    assert [x['role'] for x in history] == ['HISTORY'] * 4 + ['CURRENT']
    assert sum(p['type'] == 'inputImage' for p in parts) == 5
    assert len(adapter._image_frames) == 9


def test_deploy_uses_shared_request_window_and_rejects_invalid_audit(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    import pytest
    from environment.benchmarks.robodojo import deploy
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex/config.toml").write_text(
        'model="gpt-6-astra-azure"\nmodel_provider="test"\n[model_providers.test]\nbase_url="http://unused"\n')
    adapter = SimpleNamespace(visual_history=SimpleNamespace(snapshot=lambda: {"policy": "test-window"}), video=None)
    monkeypatch.setattr(deploy, "RoboDojoAdapter", lambda *args, **kwargs: adapter)
    seen = {}
    class Audit:
        valid = True
        overrides = ["test-window-enabled"]
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def controller_interrupt(self, reason):
            pass
    monkeypatch.setattr(deploy, "RequestAudit", Audit)
    def episode(*args, **kwargs):
        assert kwargs["config_overrides"][-1] == "test-window-enabled"
        return {"termination": "agent_completed"}
    monkeypatch.setattr(deploy, "run_episode", episode)
    result = deploy.eval_one_episode(None, manifest="unused", output_dir=tmp_path)
    assert result["request_boundary_valid"]
    assert seen["image_window"]() == {"policy": "test-window"}
    Audit.valid = False
    with pytest.raises(RuntimeError, match="not a valid score"):
        deploy.eval_one_episode(None, manifest="unused", output_dir=tmp_path)
    assert not json.loads((tmp_path / "episode.json").read_text())["request_boundary_valid"]
