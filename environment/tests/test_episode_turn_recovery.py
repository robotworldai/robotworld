import json

import pytest

from environment.runtime import episode_runner as runner, episode_recovery
from environment.tests.test_episode_runner import Adapter, Session, call


class RecoverySession(Session):
    def rpc(self, method, params, *, timeout=None):
        self.last_timeout = timeout
        return super().rpc(method, params)


def failed(error):
    return {'method': 'turn/completed', 'params': {'threadId': 'thread',
            'turn': {'id': 'turn', 'status': 'failed', 'error': {'message': error}}}}


def run(monkeypatch, tmp_path, messages):
    RecoverySession.messages = messages
    monkeypatch.setattr(runner, 'CodexSession', RecoverySession)
    waits = []
    monkeypatch.setattr(episode_recovery.time, 'sleep', waits.append)
    adapter = Adapter()
    result = runner.run_episode(adapter, manifest='unused', output_dir=tmp_path,
                                max_actions=2, recover_failed_turns=True)
    return result, adapter, waits


def test_same_episode_recovery_does_not_replay_action(monkeypatch, tmp_path):
    result, adapter, waits = run(monkeypatch, tmp_path,
        [call(1), failed('stream disconnected'), call(1), call(2), call(3)])
    assert len(adapter.calls) == 2 and waits == [10]
    assert result['termination'] == 'action_budget' and result['recoveries'] == 1
    starts = [params for method, params in (x for x in Session.last.sent if isinstance(x, tuple))
              if method == 'turn/start']
    assert len(starts) == 2 and all(p['threadId'] == 'thread' for p in starts)
    assert 'Last completed tool receipt' in starts[1]['input'][0]['text']
    assert 0 < Session.last.last_timeout <= 300


@pytest.mark.parametrize('error', ['Image processing blocked due to content policy violation',
                                 'Incomplete response returned, reason: content_filter',
                                 'request_too_large', 'invalid api key'])
def test_terminal_error_never_recovers(monkeypatch, tmp_path, error):
    result, adapter, waits = run(monkeypatch, tmp_path, [call(1), failed(error)])
    assert waits == [] and len(adapter.calls) == 1
    assert result['termination'] == 'agent_failed'


def test_recovery_limit(monkeypatch, tmp_path):
    result, adapter, waits = run(monkeypatch, tmp_path, [failed('stream disconnected')] * 4)
    assert waits == [10, 20, 40] and adapter.calls == []
    assert result['termination'] == 'agent_failed' and result['recoveries'] == 3


def test_uncertain_action_error_exits_without_replay(monkeypatch, tmp_path):
    def crash(self, arguments):
        self.calls.append(arguments)
        raise ConnectionError('action completion uncertain')
    monkeypatch.setattr(Adapter, 'move_eef', crash)
    with pytest.raises(ConnectionError):
        run(monkeypatch, tmp_path, [call(1), failed('stream disconnected')])
    assert json.loads((tmp_path/'episode.json').read_text())['termination'] == 'infrastructure_error'
