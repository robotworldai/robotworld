import json
import os
from pathlib import Path
import tomllib
from unittest.mock import patch

from environment.clients.harbor_entry import initialize, load_connection, KEYS


def test_bootstrap_survives_filtered_mcp_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_EPISODE_URL', 'http://simulator:8765')
    monkeypatch.setenv('WORLD_EPISODE_TOKEN', 'test-episode-token-not-model-auth')
    monkeypatch.setenv('WORLD_AGENT_INTERFACE', 'native-feedback-v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-persisted')
    connection, brief = tmp_path/'connection.json', tmp_path/'ROBOTWORLD.md'
    public = {'instruction':'Close the drawer', 'tools':[{'name':'move_eef'}],
              'action_contract':{'frame':'world', 'units':'metres'}}
    with patch('environment.clients.episode.EpisodeClient.request', return_value=public) as request:
        initialize(connection, brief)
        request.assert_called_once_with('describe')
    assert 'must-not-be-persisted' not in connection.read_text()
    assert 'test-episode-token' not in brief.read_text()
    assert 'action_contract' in brief.read_text()
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    load_connection(connection)
    assert os.environ['WORLD_EPISODE_URL'] == 'http://simulator:8765'
    assert os.environ['WORLD_EPISODE_TOKEN'] == 'test-episode-token-not-model-auth'


def test_all_tasks_use_the_same_harness_independent_entry():
    import yaml
    root = Path(__file__).resolve().parents[1]
    canonical = (root/'environment/clients/harbor_entry.py').read_bytes()
    tasks = list((root/'tasks').glob('*/task.toml'))
    assert len(tasks) == 84
    for spec in tasks:
        d = tomllib.loads(spec.read_text())
        server = d['environment']['mcp_servers'][0]
        assert server['args'] == ['/opt/robotworld-client/environment/clients/harbor_entry.py', 'mcp']
        main = yaml.safe_load((spec.parent/'environment/docker-compose.yaml').read_text())['services']['main']
        assert main['command'][-1] == 'initialize'
        assert (spec.parent/'environment/client/environment/clients/harbor_entry.py').read_bytes() == canonical
        assert '/workspace/ROBOTWORLD.md' in (spec.parent/'instruction.md').read_text()
