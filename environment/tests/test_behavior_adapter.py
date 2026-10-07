import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from environment.benchmarks.behavior_1k.adapter import BehaviorAdapter, load_protocol
from environment.runtime import episode_runner as runner
from environment.runtime.codex_session import CodexSession
from environment.tests.test_episode_runner import Session, call


@pytest.fixture
def protocol():
    root = Path(__file__).resolve().parents[3] / "behavior1k/docker"
    if not root.is_dir():
        pytest.skip("Native local BEHAVIOR protocol unavailable")
    return load_protocol(root)


def state():
    return dict(observation_id="2", protocol="behavior-r1pro-eef-v1",
                robot={}, base_world_pose=[], commands=0, native_steps=0,
                physics_events=0, success=False, ended=False, instruction=["Clean the desk"])


def command():
    return dict(observation_id="2", arm="left", delta_position=[.01, 0, 0],
                delta_rotation=[0, 0, 0], gripper=1, base=[0, 0, 0], steps=15)


@pytest.fixture
def adapter(tmp_path, protocol):
    (tmp_path / "ipc").mkdir()
    (tmp_path / "observations").mkdir()
    Image.new("RGB", (256, 296), "red").save(tmp_path / "observations/mosaic.png")
    adapter = BehaviorAdapter(tmp_path, SimpleNamespace(poll=lambda: None), protocol, request_timeout=.01)
    adapter.state = state()
    adapter.image = "observations/mosaic.png"
    return adapter


def test_native_step_and_public_boundary(adapter):
    observed = dict(state(), observation_id="3", native_steps=15,
                    image="observations/mosaic.png", privileged={"objects": "secret"}, q_score=1)
    (adapter.simulator / "ipc/response-00001.json").write_text(json.dumps(
        dict(ok=True, observation=observed, feedback={"reached": True})))
    reply, trace = adapter.call("step", command())
    sent = json.loads((adapter.simulator / "ipc/request-00001.json").read_text())
    assert sent["tool"] == "step" and sent["arguments"]["delta_rotation"] == [0., 0., 0.]
    assert trace["executed_native_steps"] == 15
    assert reply["success"] is True
    assert all(word not in json.dumps(reply) for word in ("privileged", "q_score", "secret"))
    assert reply["contentItems"][-1]["type"] == "inputImage"


@pytest.mark.parametrize("change", [
    {"observation_id": "1"}, {"steps": 16}, {"delta_position": [1, 0, 0]},
    {"base": [0.1, 0, 0]}, {"delta_rotation": [float("nan"), 0, 0]},
])
def test_invalid_request_never_reaches_worker(adapter, change):
    reply, trace = adapter.call("step", dict(command(), **change))
    assert reply["success"] is False and trace["executed_native_steps"] == 0
    assert not list((adapter.simulator / "ipc").glob("request-*"))


def test_unknown_ack_stops_without_replay(adapter):
    with pytest.raises(TimeoutError):
        adapter.call("step", command())
    assert adapter.uncertain
    assert adapter.official_success() is None
    with pytest.raises(RuntimeError, match="uncertain"):
        adapter.request("observe")
    assert len(list((adapter.simulator / "ipc").glob("request-*"))) == 1


def test_image_cannot_escape_observation_directory(adapter, tmp_path):
    Image.new("RGB", (256, 296)).save(tmp_path / "private.png")
    adapter.image = "private.png"
    with pytest.raises(RuntimeError, match="boundary"):
        adapter.content()


def test_behavior_observation_window_uses_rounds_not_physics_steps(adapter):
    for number in range(11):
        adapter.state.update(observation_id=str(number + 20), native_steps=number * 15)
        parts = adapter.content()
    window = adapter.visual_history.snapshot()
    assert [x["observation"] for x in window["rounds"]] == [2, 4, 6, 8, 10]
    assert [x["env_step"] for x in window["rounds"]] == [30, 60, 90, 120, 150]
    assert sum(p["type"] == "inputImage" for p in parts) == 5
    assert json.loads((adapter.simulator.parent / "image-history/00010.json").read_text()) == window
    assert "secret" not in json.dumps(parts) and "q_score" not in json.dumps(parts)


def test_main_routes_behavior_and_does_not_count_observe_as_motion(monkeypatch, tmp_path):
    class Multi:
        instructions = "Use native BEHAVIOR deltas"
        developer_instructions = "Use observe/step, not move_eef"
        task_instruction = "Clean the desk"
        read_only_tools = {"observe"}
        def tool_specs(self):
            return [dict(type="function", name=n, inputSchema={}) for n in ("observe", "step")]
        def tool_handlers(self):
            return {n: lambda args: (dict(success=True, contentItems=[]), {}) for n in ("observe", "step")}
        def observe_content(self):
            return [dict(type="inputText", text="RGB")]
        def ended(self):
            return False
        def official_success(self):
            return False
    calls = [call(i) for i in range(4)]
    for value, name in zip(calls, ("observe", "step", "observe", "step")):
        value["params"]["tool"] = name
    Session.messages = calls
    monkeypatch.setattr(runner, "CodexSession", Session)
    result = runner.run_episode(Multi(), manifest="unused", output_dir=tmp_path,
                                max_actions=1, max_tools=8)
    assert result["action_calls"] == 1
    assert [x["tool"] for x in result["events"]] == ["observe", "step", "observe"]
    assert result["termination"] == "action_budget"
    prompt = json.loads((tmp_path / "prompt.json").read_text())
    assert [t["name"] for t in prompt["dynamicTools"]] == ["observe", "step", "give_up"]
    Session.messages = [copy.deepcopy(calls[0]), copy.deepcopy(calls[0]), calls[2]]
    result = runner.run_episode(Multi(), manifest="unused", output_dir=tmp_path / "budget",
                                max_actions=1, max_tools=1)
    assert result["tool_calls"] == 1 and result["action_calls"] == 0
    assert result["termination"] == "tool_budget"


def test_source_build_accepts_local_cache_symlink(monkeypatch, tmp_path):
    import hashlib
    from environment.runtime import codex_session
    world = tmp_path / "world"
    runtime = world / "environment/runtime"
    runtime.mkdir(parents=True)
    source = world / "codex"
    source.mkdir()
    local = tmp_path / "local"
    (local / "build/codex/revision/debug").mkdir(parents=True)
    (world / "var").symlink_to(local, target_is_directory=True)
    binary = local / "build/codex/revision/debug/codex-app-server"
    binary.write_bytes(b"fixture")
    manifest = dict(source=str(source), commit="revision", binary=str(binary),
                    sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    path = tmp_path / "build.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(codex_session, "__file__", str(runtime / "codex_session.py"))
    monkeypatch.setattr(codex_session.subprocess, "check_output",
                        lambda cmd, **kwargs: "revision\n" if cmd[1] == "rev-parse" else b"")
    assert CodexSession.verify_build(path) == manifest
    manifest["binary"] = str(tmp_path / "outside")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local source build"):
        CodexSession.verify_build(path)


def test_request_audit_requires_exact_tools_and_current_rgb():
    import base64
    from environment.runtime.request_audit import summarize, violations
    payload = dict(tools=[dict(type="function", name="observe")], input=[
        dict(type="input_image", image_url="data:image/png;base64," + base64.b64encode(b"rgb").decode()),
        dict(type="input_text", text="An omitted parameter is allowed.")])
    record = summarize(payload)
    assert record["input_images"] == 1 and record["omission_notices"] == 0
    assert violations(record, {"observe"}, b"rgb") == []
    assert violations(record, {"observe"}, b"old") == ["current_rgb_missing"]
    payload["tools"].append(dict(type="function", name="spawn_agent"))
    assert "tool_allowlist_mismatch" in violations(summarize(payload), {"observe"})
    payload["tools"] = [dict(type="web_search")]
    assert "tool_allowlist_mismatch" in violations(summarize(payload), {"observe"})
    payload["input"].append(dict(type="input_text", text="Images were omitted."))
    assert summarize(payload)["omission_notices"] == 1


def test_audit_blocks_extra_tools_before_upstream(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request
    from environment.runtime.request_audit import RequestAudit
    config = dict(model_provider="fixture", model_providers={"fixture": dict(base_url="http://invalid/v1")})
    path = tmp_path / "audit.json"
    with RequestAudit(config, path, {"observe"}) as audit:
        def forbidden(*args, **kwargs):
            pytest.fail("Rejected request reached upstream")
        # Use a dedicated opener for the local client before replacing the upstream factory.
        client = urllib.request.build_opener()
        monkeypatch.setattr(urllib.request, "build_opener", forbidden)
        request = urllib.request.Request(f"http://127.0.0.1:{audit.server.server_port}/v1/responses",
                                         data=json.dumps(dict(tools=[dict(type="function", name="spawn_agent")])).encode(),
                                         headers={"Authorization": "Bearer secret-fixture"})
        with pytest.raises(urllib.error.HTTPError) as error:
            client.open(request)
        assert error.value.code == 422
    assert not audit.valid
    assert "secret-fixture" not in path.read_text()
    assert audit.records[0]["violations"] == ["tool_allowlist_mismatch"]


def test_behavior_overrides_disable_model_catalog_agents():
    from environment.benchmarks.behavior_1k.diagnostics import model_overrides
    overrides = model_overrides()
    assert "agents.enabled=false" in overrides
    assert "tools.experimental_request_user_input.enabled=false" in overrides
    assert "features.goals=false" in overrides


def test_isolated_thread_rejects_missing_boundary_before_model(monkeypatch, tmp_path):
    from environment.tests.test_episode_runner import Adapter
    class Probe(Adapter):
        require_isolated_thread = True
    Session.messages = []
    monkeypatch.setattr(runner, "CodexSession", Session)
    with pytest.raises(RuntimeError, match="isolation"):
        runner.run_episode(Probe(), manifest="unused", output_dir=tmp_path, model="fixture")
    assert not any(isinstance(x, tuple) and x[0] == "turn/start" for x in Session.last.sent)
    assert json.loads((tmp_path / "episode.json").read_text())["model_turn_started"] is False






def test_direct_audit_forwards_exact_inputs_and_requires_completion(monkeypatch, tmp_path):
    import io
    import urllib.request
    from environment.runtime import request_audit as module
    config = dict(model_provider="fixture", model_providers={"fixture": dict(base_url="http://legacy.invalid")})
    seen = []
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "text/event-stream"}
    class Opener:
        def open(self, request, **kwargs):
            seen.append(request)
            return Response(b'data: {"type":"response.completed"}\n\n')
    client = urllib.request.build_opener()
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    payload = dict(model="gpt-6-astra-azure", tools=[dict(type="function", name="observe")],
                   input=[dict(type="function_call_output", call_id=str(i), output=[
                       dict(type="input_image", image_url="data:image/png;base64,cmdi")]) for i in range(4)])
    with module.RequestAudit(config, tmp_path / "audit.json", {"observe"},
                             current_image=lambda: b"rgb") as audit:
        req = urllib.request.Request(f"http://127.0.0.1:{audit.server.server_port}/v1/responses",
                                     data=json.dumps(payload).encode(), headers={"Authorization": "Bearer fixture"})
        with client.open(req) as response:
            assert response.read() == b'data: {"type":"response.completed"}\n\n'
    assert audit.valid and len(seen) == 1
    assert seen[0].full_url == "http://legacy.invalid/responses"
    assert json.loads(seen[0].data)["input"] == payload["input"]
    assert seen[0].headers["Authorization"] == "Bearer fixture"
    assert "Bearer fixture" not in (tmp_path / "audit.json").read_text()
    audit.records[0]["terminal_event"] = "response.failed"
    assert not audit.valid
