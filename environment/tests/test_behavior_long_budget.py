import json
from pathlib import Path

import pytest

from environment.benchmarks.behavior_1k.adapter import BehaviorAdapter, load_protocol
from environment.benchmarks.behavior_1k.budgets import DEFAULT_BATCH_PROFILE, get_budget
from environment.runtime import episode_runner as runner
from environment.tests.test_episode_runner import Session, call


def protocol(name):
    root = Path(__file__).resolve().parents[3] / "behavior1k/docker"
    if not root.exists():
        pytest.skip("Local native protocol unavailable")
    return load_protocol(root, name)


def command(steps):
    return dict(observation_id="1", arm="left", delta_position=[0, 0, 0],
                delta_rotation=[0, 0, 0], gripper=1, base=[0, 0, 0], steps=steps)


@pytest.mark.parametrize("name,limit", [("steps5000", 5000), ("steps10000", 10000)])
def test_budget_crosses_legacy_limits_and_rejects_overshoot(name, limit):
    p = protocol(name)
    assert p.check_budget(command(15), "1", 41, 615)["steps"] == 15
    assert p.check_budget(command(10), "1", 666, limit - 10)["steps"] == 10
    with pytest.raises(ValueError, match="budget"):
        p.check_budget(command(15), "1", 666, limit - 10)
    with pytest.raises(ValueError, match="budget"):
        p.check_budget(command(1), "1", 666, limit)
    old = protocol("legacy")
    assert old.MAX_STEPS == 500 and old.MAX_COMMANDS == 40
    assert p.MAX_STEPS == limit and p.MAX_COMMANDS == 10000
    assert get_budget(name)["wall_seconds"] is None


@pytest.mark.parametrize("name,limit", [("steps5000", 5000), ("steps10000", 10000)])
def test_long_prompt_and_schema_agree(tmp_path, name, limit):
    p = protocol(name)
    adapter = BehaviorAdapter(tmp_path, None, p, budget_profile=name)
    assert "Budget: 600 wall seconds" not in adapter.instructions
    assert f"{limit} native control steps" in adapter.instructions
    assert f"{limit / 30:.2f} simulation seconds" in adapter.instructions
    assert "There is no give_up tool" in adapter.instructions
    assert p.STEP_SCHEMA["properties"]["steps"]["maximum"] == 15


def test_future_batch_budget_changes_only_control_step_limit():
    assert DEFAULT_BATCH_PROFILE == "steps5000"
    assert get_budget(DEFAULT_BATCH_PROFILE) == dict(get_budget("steps10000"), native_steps=5000)


class LongAdapter:
    instructions = "Keep acting to control step budget"
    task_instruction = "Test"
    read_only_tools = {"observe"}

    def __init__(self, limit=30):
        self.state = dict(native_steps=0)
        self.limit = limit

    def tool_specs(self):
        return [dict(type="function", name=n, inputSchema={}) for n in ("step", "observe")]

    def tool_handlers(self):
        return {"step": self.step, "observe": lambda args: (dict(success=True, contentItems=[]),
                                                               dict(executed_native_steps=0))}

    def step(self, args):
        self.state["native_steps"] += 15
        return dict(success=True, contentItems=[]), dict(executed_native_steps=15)

    def ended(self):
        return self.state["native_steps"] >= self.limit

    def official_success(self):
        return False

    def observe_content(self):
        return [dict(type="inputText", text="sensor state")]


def step(identifier, name="step"):
    item = call(identifier)
    item["params"]["tool"] = name
    item["params"]["arguments"] = {}
    return item


def completed():
    return dict(method="turn/completed", params=dict(turn=dict(status="completed")))


def run(monkeypatch, tmp_path, messages, **kwargs):
    Session.messages = messages
    monkeypatch.setattr(runner, "CodexSession", Session)
    adapter = LongAdapter()
    return runner.run_episode(adapter, manifest="unused", output_dir=tmp_path,
                              timeout_s=None, max_actions=10000, max_tools=20000,
                              allow_give_up=False, continue_on_completion=True, **kwargs)


def test_early_completion_continues_same_episode(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path, [step(1), completed(), step(2)])
    assert result["termination"] == "environment_end"
    assert result["action_calls"] == 2 and len(result["continuations"]) == 1
    assert result["continuations"][0]["after_native_steps"] == 15
    starts = [x for x in Session.last.sent if isinstance(x, tuple) and x[0] == "turn/start"]
    assert len(starts) == 2
    assert starts[0][1]["threadId"] == starts[1][1]["threadId"]
    tools = json.loads((tmp_path / "prompt.json").read_text())["dynamicTools"]
    assert {t["name"] for t in tools} == {"step", "observe"}


def test_unsolicited_give_up_does_not_end_long_run(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path, [step(1, "give_up"), step(2), step(3)])
    assert result["termination"] == "environment_end"
    assert result["action_calls"] == 2


def test_repeated_empty_completions_are_operational_failure(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path, [completed(), completed(), completed()])
    assert result["termination"] == "agent_no_action"
    assert result["action_calls"] == 0


def test_nonstepping_tool_loop_guard(monkeypatch, tmp_path):
    result = run(monkeypatch, tmp_path, [step(1, "observe"), step(2, "observe")], max_no_motion_calls=2)
    assert result["termination"] == "no_motion_guard" and result["action_calls"] == 0


def test_long_run_idle_timeout_is_not_task_timeout(monkeypatch, tmp_path):
    class Idle(Session):
        def receive(self, *args):
            raise TimeoutError()
    monkeypatch.setattr(runner, "CodexSession", Idle)
    result = runner.run_episode(LongAdapter(), manifest="unused", output_dir=tmp_path,
                                timeout_s=None, allow_give_up=False, continue_on_completion=True)
    assert result["termination"] == "model_idle_timeout"
