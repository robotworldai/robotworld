import ast
import hashlib
import json
from pathlib import Path

import pytest

from environment.benchmarks.behavior_1k.tasks import TASKS, selection, verify_template
from environment.runtime import request_audit


def test_first_four_templates_and_wrong_selection(tmp_path):
    for number in TASKS:
        task = selection(number)
        assert task["feishu_id"] == number
        assert task["gpu_dynamics"] == (number == 3)
        assert not task["exact_feishu_instance_verified"]
        path = tmp_path / "2026-challenge-task-instances/scenes" / task["template"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test template")
        receipt = dict(kind="partial_zip_members", status="materialized",
                       tasks=[number], all_systems_included=True)
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_template(tmp_path, task, receipt)
        task["template_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert verify_template(tmp_path, task, receipt) == path
        receipt["tasks"] = []
        with pytest.raises(ValueError, match="receipt"):
            verify_template(tmp_path, task, receipt)


def test_worker_control_and_checker_unchanged():
    world = Path(__file__).resolve().parents[2]
    native = world.parent / "behavior1k/docker/eef_worker.py"
    if not native.exists():
        pytest.skip("Local original worker unavailable")
    derivative = world / "environment/benchmarks/behavior_1k/task_worker.py"
    def functions(path):
        return {node.name: ast.dump(node) for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef)}
    old, new = functions(native), functions(derivative)
    for name in ("checker", "observation", "execute", "save", "checkpoint"):
        assert old[name] == new[name], name


def make_audit(tmp_path, monkeypatch):
    config = dict(model_provider="test", model_providers={"test": dict(base_url="http://invalid")})
    return request_audit.RequestAudit(config, tmp_path / "audit.json", {"step"})


def test_only_expected_final_inflight_cancellation_valid(tmp_path, monkeypatch):
    audit = make_audit(tmp_path, monkeypatch)
    completed = dict(status=200, violations=[], terminal_event="response.completed", finished=True)
    pending = dict(status=200, violations=[])
    audit.records = [completed, pending]
    assert not audit.valid
    audit.controller_interrupt("timeout")
    assert pending["controller_interrupt"] == "timeout"
    assert not audit.valid
    pending.update(client_disconnected=True, finished=True)
    assert audit.valid
    pending["violations"] = ["current_rgb_missing"]
    assert not audit.valid
    pending["violations"] = []
    pending["terminal_event"] = "response.failed"
    assert not audit.valid
    del pending["terminal_event"]
    audit.records.append(dict(completed))
    assert not audit.valid


@pytest.mark.parametrize("reason", ["timeout", "infrastructure_error", "agent_completed"])
def test_finished_disconnection_cannot_be_requalified(tmp_path, monkeypatch, reason):
    audit = make_audit(tmp_path, monkeypatch)
    audit.records = [dict(status=200, violations=[], client_disconnected=True, finished=True)]
    audit.controller_interrupt(reason)
    assert "controller_interrupt" not in audit.records[0]
    assert not audit.valid


def test_runner_marks_cancellation_before_interrupt(monkeypatch, tmp_path):
    from environment.runtime import episode_runner as runner
    from environment.tests.test_episode_runner import Adapter, Session
    events = []
    class TimeoutSession(Session):
        def receive(self, *args):
            raise TimeoutError("budget")
        def request(self, method, params):
            events.append(method)
            super().request(method, params)
    monkeypatch.setattr(runner, "CodexSession", TimeoutSession)
    result = runner.run_episode(Adapter(), manifest="unused", output_dir=tmp_path,
                                on_interrupt=lambda reason: events.append(reason))
    assert result["termination"] == "timeout"
    assert events == ["timeout", "turn/interrupt"]
