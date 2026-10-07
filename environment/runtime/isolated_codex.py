"""Host-owned app-server relay with an allowlisted Linux filesystem/process view.

The simulator connects over a Unix socket. The agent never sees that socket,
the evaluator checkout, assets, simulator processes, or evaluator output.
"""
import os
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import tomllib

from .codex_session import CodexSession


def external_sandbox_message(message):
    """Select external enforcement without changing tools, observations or models."""
    if message.get('method') in ('turn/start', 'command/exec'):
        message = {**message, 'params': {**message.get('params', {}), 'sandboxPolicy': {
            'type': 'externalSandbox', 'networkAccess': 'restricted'}}}
    return message


def sandbox_command(binary, workspace, observations, runtime_home, catalog=None, *, helpers=()):
    if not shutil.which('bwrap'):
        raise RuntimeError('bubblewrap is required; refusing an unisolated agent')
    cmd = ['bwrap', '--die-with-parent', '--new-session', '--unshare-user',
           '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--cap-drop', 'ALL',
           '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
           '--setenv', 'HOME', '/agent-home', '--setenv', 'CODEX_HOME', '/agent-home',
           '--setenv', 'LANG', 'C.UTF-8', '--setenv', 'TERM', 'dumb']
    for path in ('/usr/bin', '/usr/lib', '/usr/share', '/lib', '/lib64',
                 '/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf'):
        if Path(path).exists():
            cmd += ['--ro-bind', path, path]
    cmd += ['--symlink', '/usr/bin', '/bin', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp', '--bind', str(workspace), '/workspace',
            '--ro-bind', str(observations), '/observations',
            '--bind', str(runtime_home), '/agent-home',
            '--ro-bind', str(binary), '/runtime/codex-app-server', '--chdir', '/workspace']
    if catalog:
        cmd += ['--ro-bind', str(catalog), '/runtime/models.json']
    for helper in helpers:
        if Path(helper).name != 'codex-code-mode-host':
            raise ValueError('Unsupported sandbox helper')
        cmd += ['--ro-bind', str(helper), '/runtime/codex-code-mode-host']
    cmd += ['/runtime/codex-app-server', '--listen', 'stdio://']
    if catalog:
        cmd += ['-c', 'model_catalog_json="/runtime/models.json"']
    return cmd


class IsolatedCodex:
    """One episode / one app-server. All configuration is chosen by the host."""
    def __init__(self, manifest, output, auth_home, catalog=None):
        self.build = CodexSession.verify_build(manifest)
        self.output = Path(output)
        self.auth_home = Path(auth_home)
        self.catalog = catalog
        self.process = None
        self.connection = None
        self.error = None
        self.docker_backend = None
        self.image_audit = None

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _enter(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='world-agent-')
        self.control = Path(self.temporary.name)
        if os.environ.get('WORLD_CODEX_DISABLE_CODE_MODE') == '1':
            source = self.catalog or (Path(__file__).resolve().parents[2] /
                'codex/codex-rs/models-manager/models.json')
            model_catalog = json.loads(Path(source).read_text())
            for entry in model_catalog.get('models', []):
                entry['tool_mode'] = 'direct'
            self.catalog = self.control / 'models-direct.json'
            self.catalog.write_text(json.dumps(model_catalog))
        self.home = self.control / 'home'
        self.home.mkdir(mode=0o700)
        # Carry only model/provider credentials, never plugins, MCPs or filesystem grants.
        from .model_config import prepare_model_home
        backend = os.environ.get('WORLD_AGENT_BACKEND', 'bubblewrap')
        if backend not in ('bubblewrap', 'docker'):
            raise ValueError('Unknown WORLD_AGENT_BACKEND')
        if backend == 'bubblewrap':
            prepare_model_home(self.auth_home, self.home)
        self.workspace = self.output / 'agent-workspace'
        self.observations = self.output / 'agent-observations'
        self.workspace.mkdir()
        self.observations.mkdir()
        self.socket_dir = self.control / 'bridge'
        self.socket_dir.mkdir()
        self.socket_path = self.socket_dir / 'app-server.sock'
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(str(self.socket_path))
        self.listener.listen(1)
        self.listener.settimeout(0.5)
        self.stop = threading.Event()
        if backend == 'docker':
            from .agent_docker import AgentDocker
            config_path = self.auth_home/'config.toml'
            config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
            public_config = {k:config[k] for k in ('model', 'model_reasoning_effort')
                             if isinstance(config.get(k), str)}
            model_path = self.control/'model.json'
            model_path.write_text(json.dumps(public_config))
            model_path.chmod(0o444)
            self.docker_backend = AgentDocker(self.build, self.workspace, self.observations, self.catalog, model_path)
            self.command = self.docker_backend.command
        else:
            config = tomllib.loads((self.home / 'config.toml').read_text())
            provider = config.get('model_provider', 'openai')
            if config.get('model_providers', {}).get(provider, {}).get('base_url'):
                from .request_audit import RequestAudit
                self.image_audit = RequestAudit(config, self.output/'request-images.json', None,
                    max_requests=1000000, allow_view_image=True,
                    image_window=lambda: json.loads((self.output/'image-window.json').read_text()))
                self.image_audit.__enter__()
            self.command = sandbox_command(self.build['binary'], self.workspace,
                                           self.observations, self.home, self.catalog,
                                           helpers=self.build.get("helpers", {}))
            if self.image_audit:
                for override in self.image_audit.overrides:
                    self.command += ['-c', override]
            executable = self.command.index('/runtime/codex-app-server', self.command.index('--chdir'))
            subprocess.run(self.command[:executable] + ['/bin/true'], check=True, timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def _serve(self):
        try:
            while not self.stop.is_set():
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.connection is None:
                return
            with (self.output / 'agent-runtime.stderr.log').open('wb') as log:
                self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=log)
                writer = threading.Thread(target=self._to_agent, daemon=True)
                writer.start()
                while data := os.read(self.process.stdout.fileno(), 65536):
                    self.connection.sendall(data)
                self.process.wait()
                self.connection.shutdown(socket.SHUT_RDWR)
                writer.join(timeout=5)
        except (OSError, ValueError) as error:
            if not self.stop.is_set():
                self.error = str(error)
        finally:
            if self.connection:
                self.connection.close()

    def _to_agent(self):
        try:
            with self.connection.makefile('rb') as stream:
                for line in stream:
                    message = json.loads(line)
                    if self.docker_backend:
                        message = external_sandbox_message(message)
                    if (message.get('method') == 'thread/start'
                            and os.environ.get('WORLD_CODEX_DISABLE_CODE_MODE') == '1'):
                        params = dict(message.get('params', {}))
                        params['config'] = {**params.get('config', {}), 'features.code_mode': False}
                        message = {**message, 'params': params}
                        (self.output / 'runtime-tool-mode.json').write_text(json.dumps({
                            'codex_code_mode': False,
                            'reason': 'explicit WORLD_CODEX_DISABLE_CODE_MODE=1',
                            'environment_code_control': 'unchanged',
                        }, indent=2))
                    self.process.stdin.write((json.dumps(message)+'\n').encode())
                    self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            self.process.stdin.close()

    def __exit__(self, *_):
        if hasattr(self, 'stop'):
            self.stop.set()
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        try:
            if self.docker_backend:
                self.docker_backend.close()
            if hasattr(self, 'thread'):
                self.thread.join(timeout=6)
        finally:
            if hasattr(self, 'listener'):
                self.listener.close()
            if self.image_audit:
                self.image_audit.__exit__(None, None, None)
            if hasattr(self, 'temporary'):
                self.temporary.cleanup()
