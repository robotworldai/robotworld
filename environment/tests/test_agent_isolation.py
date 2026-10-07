import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from environment.runtime.isolated_codex import sandbox_command


@pytest.mark.skipif(not shutil.which('bwrap'), reason='Linux bubblewrap integration test')
def test_real_filesystem_process_and_observation_boundary(tmp_path):
    workspace=tmp_path/'workspace'; workspace.mkdir()
    observations=tmp_path/'observations'; observations.mkdir()
    home=tmp_path/'home'; home.mkdir()
    secret=tmp_path/'simulator-hidden-state.json'; secret.write_text('privileged')
    (observations/'rgb.txt').write_text('approved-observation')
    # Even agent-created links must not make files outside the namespace visible.
    (workspace/'escape').symlink_to(secret)
    cmd=sandbox_command('/bin/true',workspace,observations,home)
    cmd=cmd[:cmd.index('/runtime/codex-app-server',cmd.index('--chdir'))]
    script="""
import json, os
from pathlib import Path
assert not Path(SECRET).exists()
assert not Path('/workspace/escape').exists()
assert not Path('/data').exists()
assert not Path('/runs').exists()
assert not Path('/behavior-src').exists()
assert not Path('/agent-bridge').exists()
assert not Path('/var/run/docker.sock').exists()
assert 'WORLD_PRIVATE_CANARY' not in os.environ
assert Path('/observations/rgb.txt').read_text() == 'approved-observation'
try:
    Path('/observations/forged').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('observations writable')
Path('/workspace/memory.json').write_text(json.dumps({'route': ['door', 'kitchen']}))
assert len([p for p in Path('/proc').iterdir() if p.name.isdigit()]) <= 3
print('boundary-ok')
""".replace('SECRET',repr(str(secret)))
    result=subprocess.run(cmd+['/usr/bin/python3','-c',script],capture_output=True,text=True,
                          env={**os.environ,'WORLD_PRIVATE_CANARY':'hidden'})
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='boundary-ok'
    assert json.loads((workspace/'memory.json').read_text())['route']==['door','kitchen']


def test_code_mode_helper_is_mounted_readonly(tmp_path):
    helper = tmp_path / "codex-code-mode-host"
    cmd = sandbox_command("/bin/true", tmp_path, tmp_path, tmp_path, helpers=[helper])
    i = cmd.index(str(helper))
    assert cmd[i - 1:i + 2] == ["--ro-bind", str(helper), "/runtime/codex-code-mode-host"]
    assert i < cmd.index("/runtime/codex-app-server", cmd.index("--chdir"))


def test_explicit_direct_tool_mode_preserves_robot_tools(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    from environment.runtime.isolated_codex import IsolatedCodex
    class Output(io.BytesIO):
        def close(self):
            self.saved = self.getvalue()
            super().close()
    original = {'id': 1, 'method': 'thread/start', 'params': {
        'config': {'features.code_mode': True, 'features.shell_tool': True},
        'dynamicTools': [{'name': 'move_eef'}]}}
    for disabled in (False, True):
        monkeypatch.setenv('WORLD_CODEX_DISABLE_CODE_MODE', '1' if disabled else '0')
        source = io.BytesIO((json.dumps(original) + '\n').encode())
        output = Output()
        runner = object.__new__(IsolatedCodex)
        runner.docker_backend = None
        runner.output = tmp_path
        runner.connection = SimpleNamespace(makefile=lambda _: source)
        runner.process = SimpleNamespace(stdin=output)
        runner._to_agent()
        sent = json.loads(output.saved)
        assert sent['params']['dynamicTools'] == original['params']['dynamicTools']
        assert sent['params']['config']['features.shell_tool'] is True
        assert sent['params']['config']['features.code_mode'] is (not disabled)
    assert json.loads((tmp_path/'runtime-tool-mode.json').read_text())['codex_code_mode'] is False


def test_direct_mode_overrides_catalog_without_changing_source(tmp_path, monkeypatch):
    from unittest.mock import patch
    from environment.runtime.isolated_codex import IsolatedCodex
    from environment.runtime.codex_session import CodexSession
    from scripts.configure_api import configure
    source = tmp_path/'catalog.json'
    source.write_text(json.dumps({'models': [{'slug': 'custom-model', 'tool_mode': 'code_mode_only'}]}))
    before = source.read_bytes()
    auth = tmp_path/'auth'
    configure(auth, 'https://example.invalid/v1', 'custom-model')
    monkeypatch.setenv('WORLD_MODEL_API_KEY', 'private-test-value')
    monkeypatch.setenv('WORLD_CODEX_DISABLE_CODE_MODE', '1')
    out = tmp_path/'run'; out.mkdir()
    with patch.object(CodexSession, 'verify_build', return_value={'binary': '/bin/true'}), \
         patch('environment.runtime.isolated_codex.subprocess.run'):
        with IsolatedCodex('unused', out, auth, source) as instance:
            catalog = json.loads(instance.catalog.read_text())
            assert catalog['models'][0] == {'slug': 'custom-model', 'tool_mode': 'direct'}
            assert source.read_bytes() == before
