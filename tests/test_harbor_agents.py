"""Check the payload boundary and standard Harbor job configuration offline."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('harbor')
from agents.robotworld import ROOT, stage_client
from harbor.models.job.config import JobConfig


def test_agent_payload_imports_without_simulator_or_host_files(tmp_path):
    stage_client(tmp_path)
    manifest = json.loads((tmp_path / 'client-manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert not any((tmp_path / path).exists() for path in
                   ['Assets', 'third_party', 'environment/service', 'tests', '.git', 'var'])
    # Execute from the staged tree, so a host checkout cannot satisfy missing imports.
    result = subprocess.run([
        sys.executable, '-c',
        'import harnesses.codex.run; import harnesses.claude_code.run; '
        'import agents.run_claude; import environment.bridges.mcp'
    ], cwd=tmp_path, env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(tmp_path)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_jobs_are_harbor_configs_with_existing_tasks():
    import yaml
    for path in (ROOT / 'configs/jobs').glob('*.yaml'):
        config = JobConfig.model_validate(yaml.safe_load(path.read_text()))
        assert config.n_attempts > 0
        for task in config.tasks:
            directory = ROOT / task.path
            assert (directory / 'instruction.md').is_file()
            assert (directory / 'task.toml').is_file()


def test_wrappers_accept_harbor_instruction_argument():
    # --help verifies the portable entrypoints accept Harbor's task instruction
    # without starting a model, requiring credentials, or importing a simulator.
    for module in ('harnesses.codex.run', 'harnesses.claude_code.run', 'agents.run_claude'):
        result = subprocess.run([sys.executable, '-m', module, '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '--instruction-file' in result.stdout


def test_post_sync_trajectory_hook_records_export_without_changing_score(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from harbor.models.agent.context import AgentContext
    from agents.robotworld import RobotWorldProbeAgent
    import agents.trajectory

    calls = []
    monkeypatch.setattr(agents.trajectory, 'export_trajectory',
                        lambda *args: calls.append(args))
    agent = SimpleNamespace(mode='codex', logs_dir=tmp_path, model_name='test-model')
    context = AgentContext()
    assert context.is_empty()  # Harbor's post-sync entry condition.
    RobotWorldProbeAgent.populate_context_post_run(agent, context)
    assert calls == [(tmp_path, 'codex', 'test-model')]
    assert context.metadata['trajectory'] == 'agent/trajectory.json'
    assert context.cost_usd is None

    def fail(*args):
        raise ValueError('partial trace')

    monkeypatch.setattr(agents.trajectory, 'export_trajectory', fail)
    context = AgentContext()
    RobotWorldProbeAgent.populate_context_post_run(agent, context)
    assert json.loads((tmp_path / 'trajectory-export.json').read_text())['status'] == 'export_failed'
    assert not (tmp_path / 'reward.txt').exists()


def test_transport_timeout_is_not_an_agent_budget_stop():
    from harnesses.codex.run import timeout_stop_reason
    assert timeout_stop_reason(None, now=10) == 'harness_error'
    assert timeout_stop_reason(20, now=10) == 'harness_error'
    assert timeout_stop_reason(20, now=20) == 'harness_timeout'
    assert timeout_stop_reason(20, now=21) == 'harness_timeout'
