"""Opt-in external sandbox; model authentication is owned by a host relay."""
import os
from pathlib import Path
import stat
import subprocess
import uuid
import re


class AgentDocker:
    def __init__(self, build, workspace, observations, catalog=None, model_config=None):
        self.docker = [os.environ.get('WORLD_AGENT_DOCKER_BIN', 'docker')]
        if os.environ.get('DOCKER_HOST'):
            self.docker += ['--host', os.environ['DOCKER_HOST']]
        self.name = 'world-agent-'+uuid.uuid4().hex[:12]
        image = os.environ.get('WORLD_AGENT_DOCKER_IMAGE', '')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('WORLD_AGENT_DOCKER_IMAGE must pin a local image ID')
        relay = Path(os.environ['WORLD_AGENT_MODEL_SOCKET']).resolve(strict=True)
        if relay.name != 'relay.sock' or not stat.S_ISSOCK(relay.stat().st_mode):
            raise ValueError('Expected a dedicated trusted relay.sock')
        os.chown(workspace, 65532, 65532)
        workspace.chmod(0o700)
        # The root simulator writes observations without DAC override capabilities.
        # The agent gets group read access, plus a read-only container mount.
        os.chown(observations, 0, 65532)
        observations.chmod(0o750)
        cmd = self.docker + ['run', '-i', '--name', self.name, '--network', 'none',
            '--read-only', '--user', '65532:65532', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--memory', '20g',
            '--memory-swap', '22g', '--cpus', '2',
            '--pids-limit', '256', '--ipc', 'private', '--workdir', '/workspace',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
            '--tmpfs', '/agent-home:rw,nosuid,nodev,size=32m,uid=65532,gid=65532,mode=700',
            '-e', 'HOME=/agent-home', '-e', 'CODEX_HOME=/agent-home',
            '-e', 'LD_LIBRARY_PATH=/runtime/lib', '-e', 'PYTHONDONTWRITEBYTECODE=1']
        if build.get('local_patch', {}).get('id') == 'robot-image-window-v1':
            cmd += ['-e', 'WORLD_CODEX_IMAGE_WINDOW=1']
        mounts = [(Path(build['binary']).resolve(), '/runtime/codex-app-server', True),
                  (Path(__file__).with_name('agent_container_entry.py'), '/runtime/entry.py', True),
                  (workspace, '/workspace', False), (observations, '/observations', True),
                  (relay, '/model-relay/relay.sock', True)]
        for lib in ('libssl.so.1.1', 'libcrypto.so.1.1', 'libbz2.so.1'):
            mounts.append((Path('/lib64', lib).resolve(strict=True), '/runtime/lib/'+lib, True))
        for helper in build.get('helpers', {}):
            if Path(helper).name != 'codex-code-mode-host':
                raise ValueError('Unsupported sandbox helper')
            mounts.append((Path(helper).resolve(strict=True), '/runtime/codex-code-mode-host', True))
        if catalog:
            mounts.append((Path(catalog).resolve(strict=True), '/runtime/models.json', True))
        if model_config:
            mounts.append((Path(model_config).resolve(strict=True), '/runtime/model.json', True))
        for source, target, readonly in mounts:
            cmd += ['--mount', f'type=bind,src={source},dst={target}'+(',readonly' if readonly else '')]
        self.command = cmd + ['--entrypoint', 'python3', image, '/runtime/entry.py']

    def close(self):
        result = subprocess.run(self.docker+['rm', '-f', self.name], capture_output=True, timeout=60)
        if result.returncode and b'No such container' not in result.stderr:
            raise RuntimeError('Agent Docker cleanup failed')
