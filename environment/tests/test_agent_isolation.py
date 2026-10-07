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
