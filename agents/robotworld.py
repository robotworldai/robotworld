"""Harbor 0.24 installed agents. The control loop runs inside main, not on the host."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/opt/robotworld-agent'


def stage_client(destination):
    """Explicit allowlist: no simulator, asset, checker, output or credential files."""
    destination = Path(destination)
    for relative in ('environment/clients/episode.py', 'environment/bridges/mcp.py',
                     'harnesses/claude_code/run.py', 'agents/probe.py', 'agents/run_claude.py', 'agents/reference_replay.py',
                     'harnesses/codex/run.py', 'environment/runtime/codex_session.py',
                     'environment/runtime/events.py', 'environment/runtime/nonaction_budget.py',
                     'environment/runtime/observation_window.py',
                     'environment/runtime/observation_window_proxy.py',
                     'environment/runtime/messages_window_proxy.py'):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (destination/'client-manifest.json').write_text(json.dumps({
        str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob('*.py'))}, indent=2))


class RobotWorldProbeAgent(BaseInstalledAgent):
    """Non-solving infrastructure validation through the real episode API."""
    capabilities = AgentCapabilities(mcp_servers=True)
    mode = 'probe'

    @staticmethod
    def name(): return 'robotworld-probe'
    def version(self): return 'episode-v1-harbor-1'

    def populate_context_post_run(self, context):
        # Harbor calls this on the host after syncing logs, including remote workers.
        # Leave the model-agent context empty during run() so this standard hook runs.
        context.metadata = {'execution': 'inside-agent-container', 'harness': self.mode,
                            'native_trace': 'agent/robotworld', 'code_control': 'runtime_describe'}
        if self.mode not in ('codex', 'claude-code'):
            return
        try:
            from agents.trajectory import export_trajectory
            export_trajectory(self.logs_dir, self.mode, self.model_name)
            context.metadata['trajectory'] = 'agent/trajectory.json'
            context.metadata['trajectory_fidelity'] = 'limited_observable_events'
        except Exception as exc:
            # Trajectory conversion cannot overwrite the physical outcome or hide raw logs.
            details = {'status': 'export_failed', 'error_type': type(exc).__name__,
                       'error': str(exc), 'raw_trace_preserved': True}
            (self.logs_dir / 'trajectory-export.json').write_text(json.dumps(details, indent=2))
            context.metadata['trajectory_export'] = details

    async def install(self, environment):
        with tempfile.TemporaryDirectory() as temporary:
            payload = Path(temporary)/'client'
            stage_client(payload)
            await self.exec_as_root(environment, command=f'mkdir -p {REMOTE}')
            await environment.upload_dir(payload, REMOTE)
        await self.exec_as_agent(environment,
            command='python -c "import mcp; from environment.clients.episode import EpisodeClient"',
            env={'PYTHONPATH':REMOTE, 'WORLD_ALLOW_PRIVATE_MODEL_HTTP':os.environ.get('WORLD_ALLOW_PRIVATE_MODEL_HTTP','0'), 'WORLD_AGENT_INTERFACE':os.environ.get('WORLD_AGENT_INTERFACE','native-feedback-v1')})

    async def run(self, instruction, environment, context):
        logs = str(self.environment_logs_dir)
        await self._upload_config_text(environment, content=instruction,
            remote_path=logs+'/task-instruction.txt', filename='instruction.txt')
        await self.exec_as_agent(environment,
            command=f'python -m agents.probe --output {shlex.quote(logs+"/robotworld")}',
            env={'PYTHONPATH':REMOTE, 'WORLD_ALLOW_PRIVATE_MODEL_HTTP':os.environ.get('WORLD_ALLOW_PRIVATE_MODEL_HTTP','0'), 'WORLD_AGENT_INTERFACE':os.environ.get('WORLD_AGENT_INTERFACE','native-feedback-v1')}, timeout_sec=43200)
        context.metadata = {'execution':'inside-agent-container','harness':self.mode,
                            'classification':'infrastructure_probe_not_model_evaluation'}


class RobotWorldClaudeCodeAgent(RobotWorldProbeAgent):
    """Keep the already-tested persistent-episode wrapper, inside Harbor main."""
    mode = 'claude-code'
    DEFAULT_SETUP_TIMEOUT_SEC = 900.0

    @staticmethod
    def name(): return 'robotworld-claude-code'

    async def install(self, environment):
        await super().install(environment)
        local = os.environ.get('WORLD_HARBOR_CLAUDE_BINARY')
        if local:
            source = Path(local).expanduser().resolve(strict=True)
            await environment.upload_file(source, '/usr/local/bin/claude')
            await self.exec_as_root(environment, command='chmod 755 /usr/local/bin/claude')
            provenance={'provisioning':'reused-local-cli','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        else:
            # Version is explicit and fixed; use an existing pinned image to avoid downloads.
            await self.exec_as_agent(environment, command='command -v claude >/dev/null')
            provenance={'provisioning':'preinstalled-cli'}
        if os.environ.get('WORLD_HARBOR_CHAT_BASE_URL'):
            await self.exec_as_root(environment, command=(
                'python -m venv /opt/robotworld-provider && '
                '/opt/robotworld-provider/bin/pip install --disable-pip-version-check "litellm[proxy]==1.105.1"'),
                timeout_sec=600)
        r=await self.exec_as_agent(environment, command='claude --version')
        if not r.stdout.strip().startswith('2.1.292 '):
            raise RuntimeError('This profile requires Claude Code 2.1.292; pin a new profile to upgrade')
        provenance['version']=r.stdout.strip()
        await self._upload_config_text(environment, content=json.dumps(provenance,indent=2),
            remote_path=str(self.environment_logs_dir)+'/runtime.json', filename='runtime.json')

    async def run(self, instruction, environment, context):
        if not self.model_name: raise ValueError('Supply the exact provider model ID')
        env={'PYTHONPATH':REMOTE, 'WORLD_ALLOW_PRIVATE_MODEL_HTTP':os.environ.get('WORLD_ALLOW_PRIVATE_MODEL_HTTP','0'), 'WORLD_AGENT_INTERFACE':os.environ.get('WORLD_AGENT_INTERFACE','native-feedback-v1')}
        keys = (('WORLD_HARBOR_CHAT_BASE_URL','WORLD_MODEL_API_KEY') if
                os.environ.get('WORLD_HARBOR_CHAT_BASE_URL') else ('ANTHROPIC_BASE_URL','ANTHROPIC_API_KEY'))
        for key in keys:
            if not os.environ.get(key):raise ValueError(f'{key} required in orchestrator environment')
            env[key]=os.environ[key]
        # Provider bridge must be reachable from main. Never serialize these values.
        logs=str(self.environment_logs_dir)
        await self._upload_config_text(environment, content=instruction,
            remote_path=logs+'/task-instruction.txt', filename='instruction.txt')
        command=shlex.join(['python','-m','agents.run_claude','--model',
            self.model_name,'--output',logs+'/robotworld','--instruction-file',logs+'/task-instruction.txt'])
        await self.exec_as_agent(environment, command=command, env=env, timeout_sec=43200)
        # The wrapper archives provider errors; propagate them into Harbor error accounting.
        await self.exec_as_agent(environment, command=(
            "python -c "+shlex.quote(
                "import json; s=json.load(open("+repr(logs+'/robotworld/summary.json')+")); "
                "assert s['harness_stop'] not in ('harness_error',), 'RobotWorld harness error; inspect native trace'")))
        # Post-sync metadata and ATIF conversion use Harbor's populate_context_post_run hook.

class RobotWorldCodexAgent(RobotWorldProbeAgent):
    """Run the verified source-built Codex app-server inside main."""
    mode = 'codex'

    @staticmethod
    def name(): return 'robotworld-codex'

    async def install(self, environment):
        await super().install(environment)
        runtime = Path(os.environ['WORLD_HARBOR_CODEX_RUNTIME']).resolve(strict=True)
        if not (runtime/'provider.json').is_file():
            raise ValueError('Run scripts/prepare_harbor_codex.py first')
        await self.exec_as_root(environment,command='mkdir -p /opt/robotworld-runtimes /tmp/robotworld-codex-auth && chmod 700 /tmp/robotworld-codex-auth')
        await environment.upload_dir(runtime,'/opt/robotworld-runtimes')
        await self.exec_as_agent(environment,command='git config --global --add safe.directory /opt/robotworld-runtimes/codex')
        await self.exec_as_agent(environment,
            command="python -c "+shlex.quote("from environment.runtime.codex_session import CodexSession; CodexSession.verify_build('/opt/robotworld-runtimes/build.json')"),
            env={'PYTHONPATH':REMOTE,'WORLD_CODEX_SOURCE':'/opt/robotworld-runtimes/codex'})
        await environment.upload_file(runtime/'provenance.json',
                                      str(self.environment_logs_dir)+'/runtime.json')

    async def run(self, instruction, environment, context):
        if not self.model_name:raise ValueError('Supply the exact provider model ID')
        runtime=Path(os.environ['WORLD_HARBOR_CODEX_RUNTIME'])
        config=json.loads((runtime/'provider.json').read_text())
        key_env=config['observation_window']['api_key_env']
        env={'PYTHONPATH':REMOTE,'WORLD_CODEX_SOURCE':'/opt/robotworld-runtimes/codex',
             'WORLD_CODEX_PROVIDER_CONFIG':'/opt/robotworld-runtimes/provider.json',
             'CODEX_HOME':'/tmp/robotworld-codex-auth',key_env:os.environ[key_env], 'WORLD_ALLOW_PRIVATE_MODEL_HTTP':os.environ.get('WORLD_ALLOW_PRIVATE_MODEL_HTTP','0'), 'WORLD_AGENT_INTERFACE':os.environ.get('WORLD_AGENT_INTERFACE','native-feedback-v1')}
        logs=str(self.environment_logs_dir)
        await self._upload_config_text(environment,content=instruction,
            remote_path=logs+'/task-instruction.txt',filename='instruction.txt')
        await self.exec_as_agent(environment,command=shlex.join([
            'python','-m','harnesses.codex.run','--model',self.model_name,
            '--output',logs+'/robotworld','--wall-timeout','43200','--instruction-file',logs+'/task-instruction.txt']),
            env=env,timeout_sec=43200)
        await self.exec_as_agent(environment,command='python -c '+shlex.quote(
            "import json; s=json.load(open("+repr(logs+'/robotworld/summary.json')+")); "
            "assert s['harness_stop'] != 'harness_error', 'Codex runtime error; inspect native trace'"))
        # Post-sync metadata and ATIF conversion use Harbor's populate_context_post_run hook.


class RobotWorldReferenceReplayAgent(RobotWorldProbeAgent):
    """Private-reference migration diagnostic. Never include in model leaderboards."""
    mode = 'reference-replay'

    @staticmethod
    def name(): return 'robotworld-reference-replay'

    async def run(self, instruction, environment, context):
        reference = Path(os.environ['WORLD_REPLAY_REFERENCES']).resolve(strict=True)
        logs = str(self.environment_logs_dir)
        await environment.upload_file(reference, logs+'/reference-commands.json')
        await self.exec_as_agent(environment, command=shlex.join([
            'python','-m','agents.reference_replay','--references',logs+'/reference-commands.json',
            '--output',logs+'/robotworld']),
            env={'PYTHONPATH':REMOTE, 'WORLD_AGENT_INTERFACE':'native-feedback-v1'},timeout_sec=43200)
        context.metadata={'classification':'historical_reference_replay_not_model_evaluation'}
