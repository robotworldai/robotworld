"""RoboDojo does not advertise or accept the shared optional give_up tool."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment.benchmarks.robodojo import deploy
from environment.runtime import episode_runner
from environment.tests.test_episode_runner import Adapter, Session, call


@pytest.mark.parametrize('socket_backend', [False, True])
def test_deploy_omits_give_up_and_rejects_unsolicited_call(monkeypatch, tmp_path, socket_backend):
    if socket_backend:
        monkeypatch.setenv('WORLD_CODEX_SOCKET', '/unused-test-socket')
    else:
        monkeypatch.delenv('WORLD_CODEX_SOCKET', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    (tmp_path / '.codex').mkdir()
    (tmp_path / '.codex/config.toml').write_text('model="gpt-6-astra"\n')

    class Robot(Adapter):
        video = None
        @property
        def state(self):return {'native_steps':len(self.calls)}
        visual_history = SimpleNamespace(included=[], snapshot=lambda: {'policy': 'test-window'})

        def ended(self):
            return len(self.calls) == 1

        def official_success(self):
            return False if self.ended() else None

    adapter = Robot()
    monkeypatch.setattr(deploy, 'RoboDojoAdapter', lambda *args, **kw: adapter)
    audit_tools = []

    class Audit:
        valid = True
        overrides = []

        def __init__(self, config, output, tools, **kwargs):
            audit_tools.append(tools)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def controller_interrupt(self, reason):
            pass

    monkeypatch.setattr(deploy, 'RequestAudit', Audit)
    monkeypatch.setattr(episode_runner, 'CodexSession', Session)
    quit_call = call(1)
    quit_call['params'].update(tool='give_up', arguments={'reason': 'difficult', 'hindsight': 'none'})
    monkeypatch.setattr(Session, 'messages', [quit_call, call(2)])

    result = deploy.eval_one_episode(None, manifest='unused', output_dir=tmp_path,
                                     model='gpt-6-astra', max_actions=2)

    tools = json.loads((tmp_path / 'prompt.json').read_text())['dynamicTools']
    assert [tool['name'] for tool in tools] == ['move_eef']
    assert result['termination'] == 'environment_end'
    assert result['official_success'] is False
    assert result['action_calls'] == 1 and len(adapter.calls) == 1
    rejected = next(x for x in Session.last.sent if isinstance(x, dict) and x.get('id') == 1)
    assert rejected['result']['success'] is False
    assert 'give_up_arguments' not in result
    assert audit_tools == ([] if socket_backend else [{'move_eef'}])
