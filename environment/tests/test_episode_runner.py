import json
from pathlib import Path

import pytest

from environment.runtime import episode_runner as runner
from environment.runtime.codex_session import CodexSession


class Adapter:
    instructions = 'test robot'
    task_instruction = 'test motion'

    def __init__(self):
        self.calls = []

    def tool_spec(self):
        return {'type': 'function', 'name': 'move_eef', 'description': '', 'inputSchema': {}}

    def observe_content(self):
        return [{'type': 'inputText', 'text': 'state'},
                {'type': 'inputImage', 'imageUrl': 'data:image/png;base64,AA=='}]

    def move_eef(self, arguments):
        self.calls.append(arguments)
        return {'success': True, 'contentItems': self.observe_content()}, {'executed_waypoints': 1}

    def ended(self):
        return False

    def official_success(self):
        return None


class Session:
    messages = []

    def __init__(self, *args, **kwargs):
        self.items = iter(self.messages)
        self.sent = []
        Session.last = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def rpc(self, method, params):
        self.sent.append((method, params))
        return {'thread': {'id': 'thread'}} if method == 'thread/start' else {'turn': {'id': 'turn'}}

    def receive(self, *args):
        return next(self.items)

    def send(self, message):
        self.sent.append(message)

    def request(self, method, params):
        self.sent.append((method, params))


def call(identifier):
    return {'id': identifier, 'method': 'item/tool/call', 'params': {
        'callId': str(identifier), 'threadId': 'thread', 'turnId': 'turn',
        'tool': 'move_eef', 'arguments': {'targets': {'left_z': .91}, 'note': 'test'}}}


def test_duplicate_is_not_reexecuted_and_budget_stops(monkeypatch, tmp_path):
    Session.messages = [call(1), call(1), call(2)]
    monkeypatch.setattr(runner, 'CodexSession', Session)
    adapter = Adapter()
    result = runner.run_episode(adapter, manifest='unused', output_dir=tmp_path, max_actions=1)
    assert len(adapter.calls) == 1
    assert result['termination'] == 'action_budget'
    assert result['official_success'] is None
    assert any(isinstance(x, dict) and x.get('result', {}).get('contentItems', [{}])[-1].get('type') == 'inputImage'
               for x in Session.last.sent)
    assert json.loads((tmp_path / 'episode.json').read_text())['action_calls'] == 1


def test_text_completion_does_not_claim_success(monkeypatch, tmp_path):
    Session.messages = [{'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}}]
    monkeypatch.setattr(runner, 'CodexSession', Session)
    result = runner.run_episode(Adapter(), manifest='unused', output_dir=tmp_path)
    assert result['termination'] == 'agent_completed'
    assert result['official_success'] is None


def test_rejects_global_binary_manifest(tmp_path):
    manifest = tmp_path / 'build.json'
    manifest.write_text(json.dumps({'source': '/tmp/another-codex', 'binary': '/usr/bin/codex'}))
    with pytest.raises(ValueError, match='World/codex'):
        CodexSession.verify_build(manifest)


def test_reused_call_id_with_different_target_fails(monkeypatch, tmp_path):
    second = call(1)
    second['params']['arguments']['targets']['left_z'] = .95
    Session.messages = [call(1), second]
    monkeypatch.setattr(runner, 'CodexSession', Session)
    adapter = Adapter()
    with pytest.raises(RuntimeError, match='reused'):
        runner.run_episode(adapter, manifest='unused', output_dir=tmp_path)
    assert len(adapter.calls) == 1
    assert json.loads((tmp_path / 'episode.json').read_text())['termination'] == 'infrastructure_error'


def test_tool_request_racing_rpc_response_is_preserved():
    import queue
    from collections import deque
    session = object.__new__(CodexSession)
    session.timeout = 1
    session.counter = 0
    session.messages = queue.Queue()
    session.pending = deque()
    session.send = lambda message: None
    request = call(99)
    session.messages.put(request)
    session.messages.put({'id': 1, 'result': {'turn': {'id': 'turn'}}})
    assert session.rpc('turn/start', {}) == {'turn': {'id': 'turn'}}
    assert session.receive() == request
