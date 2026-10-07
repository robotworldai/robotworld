import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from environment.runtime.agent_docker import AgentDocker
from environment.runtime.isolated_codex import IsolatedCodex, external_sandbox_message
from environment.runtime.codex_session import CodexSession


def test_docker_mount_boundary(tmp_path):
    relay = socket.socket(socket.AF_UNIX)
    relay.bind(str(tmp_path/'relay.sock'))
    workspace = tmp_path/'work'; workspace.mkdir()
    observations = tmp_path/'obs'; observations.mkdir()
    env = {'WORLD_AGENT_DOCKER_IMAGE': 'sha256:'+'a'*64,
           'WORLD_AGENT_MODEL_SOCKET': str(tmp_path/'relay.sock')}
    try:
        with patch.dict('os.environ', env), patch('os.chown') as chown:
            backend = AgentDocker({'binary': '/bin/true'}, workspace, observations)
        chown.assert_any_call(workspace, 65532, 65532)
        chown.assert_any_call(observations, 0, 65532)
        assert workspace.stat().st_mode & 0o777 == 0o700
        assert observations.stat().st_mode & 0o777 == 0o750
        cmd = backend.command
        assert cmd[cmd.index('--network')+1] == 'none'
        assert cmd[cmd.index('--user')+1] == '65532:65532'
        assert cmd[cmd.index('--memory')+1] == '20g'
        assert cmd[cmd.index('--memory-swap')+1] == '22g'
        assert cmd[cmd.index('--cpus')+1] == '2'
        assert '--privileged' not in cmd and '--pid' not in cmd
        mounts = [cmd[i+1] for i, x in enumerate(cmd) if x == '--mount']
        assert any('dst=/observations,readonly' in m for m in mounts)
        assert any('dst=/model-relay/relay.sock,readonly' in m for m in mounts)
        assert not any('docker.sock' in m or 'auth.json' in m or 'dst=/runs' in m for m in mounts)
    finally:
        relay.close()


def test_requires_pinned_image(tmp_path):
    with patch.dict('os.environ', {'WORLD_AGENT_DOCKER_IMAGE': 'python:latest'}):
        with pytest.raises(ValueError, match='pin'):
            AgentDocker({'binary': '/bin/true'}, tmp_path, tmp_path)


def test_external_policy_preserves_payload():
    original = {'id': 7, 'method': 'turn/start', 'params': {
        'model': 'chosen-model', 'input': [{'type': 'image', 'url': 'data:image/png;base64,test'}]}}
    converted = external_sandbox_message(original)
    assert 'sandboxPolicy' not in original['params']
    assert converted['params']['model'] == 'chosen-model'
    assert converted['params']['input'] == original['params']['input']
    assert converted['params']['sandboxPolicy'] == {'type': 'externalSandbox', 'networkAccess': 'restricted'}
    tool_reply = {'id': 9, 'result': {'contentItems': [{'type': 'inputImage', 'imageUrl': 'test'}]}}
    assert external_sandbox_message(tool_reply) is tool_reply


def test_initialization_failure_cleans_temporary_directory(tmp_path):
    with patch.object(CodexSession, 'verify_build', return_value={'binary': '/bin/true'}):
        instance = IsolatedCodex('unused', tmp_path, tmp_path)
    with patch.dict('os.environ', {'WORLD_AGENT_BACKEND': 'invalid'}):
        with pytest.raises(ValueError, match='Unknown'):
            instance.__enter__()
    assert not instance.control.exists()
