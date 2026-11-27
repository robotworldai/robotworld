#!/usr/bin/env python3
"""Audit Harbor trial artifacts. Missing/invalid trials never become reward zero."""
import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


PROBE_AGENTS = {
    'agents.robotworld:RobotWorldReferenceReplayAgent', 'robotworld-reference-replay',
    'agents.smoke100:RobotWorldSmokeAgent',
    'agents.smoke100:RobotWorldNativeBudgetProbeAgent',
    'agents.smoke100:RobotWorldRandomProbeAgent', 'robotworld-random-probe',
    'nop', 'agents.robotworld:RobotWorldProbeAgent',
    'robotworld-probe', 'robotworld-smoke100', 'robotworld-native-budget-probe',
    'robotworld-robolab-probe', 'robotworld-soccer-probe', 'robotworld-native-probe',
    'robotworld-robodojo-probe', 'robotworld-task-probe',
    'agents.robolab_probe:RoboLabProbeAgent',
    'agents.soccer_probe_agent:SoccerProbeAgent',
    'agents.native_probe_agent:RobotWorldNativeProbeAgent',
    'agents.native_probe_agent:RobotWorldNativeBudgetProbeAgent',
    'agents.robodojo_probe_agent:RobotWorldDojoProbeAgent',
    'agents.task_probe_agent:RobotWorldTaskProbeAgent',
}


# A custom agent may identify itself as a diagnostic only in the finished
# result. Only these configured model agents have a known role before execution.
MODEL_AGENTS = {
    'codex', 'claude-code', 'antigravity-cli', 'robotworld-codex', 'robotworld-claude-code',
    'agents.robotworld:RobotWorldCodexAgent', 'agents.robotworld:RobotWorldClaudeCodeAgent',
}


def agent_identity(config):
    agent = config.get('agent') or {}
    return agent.get('import_path') or agent.get('name')


def is_diagnostic(config, result):
    """Host-owned Harbor config/context only; never agent-authored summary text."""
    identity = agent_identity(config)
    metadata = (result.get('agent_result') or {}).get('metadata') or {}
    return (identity in PROBE_AGENTS or
            metadata.get('classification') in {'infrastructure_probe_not_model_evaluation', 'historical_reference_replay_not_model_evaluation'})


def classify_trial(config, result, verified):
    diagnostic = is_diagnostic(config, result)
    error = (result.get('exception_info') or {}).get('exception_type')
    # Old diagnostic verifiers intentionally emitted no reward. No other exception
    # (including AgentTimeoutError) is silently converted to task failure.
    if error and not (diagnostic and error == 'RewardFileNotFoundError'):
        raise ValueError('Harbor execution exception: ' + error)
    if diagnostic:
        return {'status': 'diagnostic', 'reward': None,
                'score_basis': 'infrastructure_probe_excluded_from_model_evaluation'}
    if not agent_identity(config):
        raise ValueError('Missing trusted Harbor agent configuration')
    if verified['episode']['termination'] == 'agent_stopped':
        capped = verified['episode'].get('stop_reason') == 'control_step_limit'
        return {'status': 'valid_failure', 'reward': 0,
                'score_basis': ('goal_not_confirmed_before_control_step_limit' if capped else
                                'goal_not_confirmed_before_agent_return')}
    return {'status': verified['status'], 'reward': verified['reward'],
            'score_basis': verified.get('score_basis', 'trusted_simulator_completion')}


def check_trial(trial):
    trial = Path(trial)
    report = {'trial': str(trial), 'status': 'not_run', 'reward': None, 'errors': []}
    simulation = trial / 'artifacts/simulation'
    try:
        config_path = trial / 'config.json'
        result_path = trial / 'result.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        report['diagnostic'] = agent_identity(config) in PROBE_AGENTS
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        report['diagnostic'] = is_diagnostic(config, result)
        if result.get('id') is not None:
            if not isinstance(result['id'], str):
                raise ValueError('Invalid Harbor trial ID')
            report['trial_id'] = result['id']
        lock_path = trial / 'lock.json'
        if result and lock_path.is_file():
            lock = json.loads(lock_path.read_text())
            task, agent, model, _ = trial_identity(lock)
            if result.get('task_name') != task:
                raise ValueError('Harbor result task differs from trial lock')
            locked_path = lock['task'].get('path')
            result_path_id = (result.get('task_id') or {}).get('path')
            if (lock['task'].get('type') == 'local' and locked_path and result_path_id and
                    Path(result_path_id).name != Path(locked_path).name):
                raise ValueError('Harbor result task package differs from trial lock')
            for recorded in (config, result.get('config') or config):
                if (agent_identity(recorded) != agent or
                        (recorded['agent'].get('model_name') != model and 'model_name' in lock['agent'])):
                    raise ValueError('Harbor agent or model configuration differs from trial lock')
        if not (simulation / 'episode.json').exists():
            report['errors'].append('No collected simulation episode')
            if result_path.exists():
                report['status'] = 'invalid'
            return report
        if not result_path.exists() or not config_path.exists():
            raise ValueError('Missing Harbor trial result or configuration')
        task_path = Path(result.get('task_id', {}).get('path', ROOT / 'tasks' / result.get('task_name','')))
        if not task_path.is_dir():
            task_path = ROOT / 'tasks' / result.get('task_name','')
        expected_path = task_path / 'tests/task.json'
        if expected_path.exists():
            expected = json.loads(expected_path.read_text())
        elif result.get('task_name') == 'robocasa-close-drawer':
            expected = {'benchmark':'robocasa','task':'CloseDrawer','action_dimension':12}
        else:
            raise ValueError('Trusted task verifier specification unavailable')
        report.update(benchmark=expected['benchmark'], task=expected['task'])
        spec = importlib.util.spec_from_file_location('robotworld_close_drawer_verifier', ROOT / 'tasks/close-drawer/tests/grade.py')
        grader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grader)
        verified = grader.validate(simulation, expected)
        classification = classify_trial(config, result, verified)
        report.update(native_success=verified.get('native_success'),
                      environment_success=verified.get('environment_success', verified['episode']['success']),
                      outcome_file=verified.get('outcome_file', 'episode.json'),
                      score_basis=classification['score_basis'])
        if classification['status'] == 'diagnostic':
            report.update(**classification, termination=verified['episode']['termination'],
                          control_steps=verified['episode']['control_step'], video=verified['video'])
            return report
        reward = float((trial / 'verifier/reward.txt').read_text())
        if reward != verified['reward']:
            raise ValueError('Harbor reward differs from native evidence')
        if result.get('verifier_result', {}).get('rewards', {}).get('reward') != reward:
            raise ValueError('Harbor result reward differs from reward.txt')
        if not (trial / 'verifier/details.json').is_file():
            raise ValueError('Missing verifier details')
        if not (trial / 'agent').is_dir() or not any(p.is_file() and p.stat().st_size for p in (trial / 'agent').rglob('*')):
            raise ValueError('Missing agent artifacts')
        report.update(status=classification['status'], reward=reward,
                      termination=verified['episode']['termination'],
                      control_steps=verified['episode']['control_step'], video=verified['video'])
    except Exception as exc:
        report.update(status='invalid', reward=None)
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    return report


def trial_identity(lock):
    task = lock['task']['name']
    agent = agent_identity(lock)
    model = lock['agent'].get('model_name')
    digest = lock['task'].get('digest')
    if not all(isinstance(value, str) and value for value in (task, agent)):
        raise ValueError('Missing task or agent identity')
    if any(value is not None and not isinstance(value, str) for value in (model, digest)):
        raise ValueError('Invalid model name or task digest')
    return task, agent, model, digest


@dataclass
class JobPlanAudit:
    missing: list[dict] = field(default_factory=list)
    unattributed_missing: int = 0
    unexpected: int = 0
    errors: list[str] = field(default_factory=list)


def audit_job_plan(job, rows):
    audit = JobPlanAudit()
    lock_path = job / 'lock.json'
    result_path = job / 'result.json'
    try:
        plan = json.loads(lock_path.read_text())['trials']
        if not isinstance(plan, list):
            raise ValueError('Expected list of planned trials')
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        # Historical jobs may only record a count. Their individually graded
        # trials remain available, but the task/model coverage is unverified.
        audit.errors.append(f'Cannot verify scheduled trial identities: {exc}')
        if result_path.is_file():
            try:
                planned = json.loads(result_path.read_text())['n_total_trials']
                if type(planned) is not int or planned < 0:
                    raise ValueError('Invalid planned trial count')
                audit.unattributed_missing = max(0, planned - len(rows))
                audit.unexpected = max(0, len(rows) - planned)
            except (KeyError, TypeError, ValueError) as count_error:
                audit.errors.append(f'Cannot read planned trial count: {count_error}')
        return audit

    try:
        remaining = Counter(trial_identity(item) for item in plan)
        observed = [trial_identity(json.loads((Path(row['trial']) / 'lock.json').read_text()))
                    for row in rows]
    except (FileNotFoundError, KeyError, TypeError, AttributeError, ValueError) as exc:
        audit.errors.append(f'Cannot match job and trial locks: {exc}')
        audit.unattributed_missing = max(0, len(plan) - len(rows))
        audit.unexpected = max(0, len(rows) - len(plan))
        return audit

    seen_ids = set()
    for row, key in zip(rows, observed):
        trial_id = row.get('trial_id')
        if remaining[key] and (not trial_id or trial_id not in seen_ids):
            remaining[key] -= 1
            agent = key[1]
            # A failed or inconsistent probe result cannot change the role of
            # the scheduled model slot and silently remove it from coverage.
            if agent in MODEL_AGENTS and row.get('diagnostic'):
                row.setdefault('errors', []).append('Scheduled model attempt was classified as a diagnostic')
                row.update(status='invalid', reward=None, diagnostic=False)
            elif agent in PROBE_AGENTS:
                row['diagnostic'] = True
            if trial_id:
                seen_ids.add(trial_id)
            continue
        error = ('Duplicate Harbor trial ID' if trial_id and trial_id in seen_ids else
                 'No remaining scheduled attempt matches this task, agent, model and digest')
        row.setdefault('errors', []).append(error)
        row.update(status='unexpected', reward=None)
        audit.unexpected += 1

    for (task, agent, model, digest), count in remaining.items():
        if not count:
            continue
        if agent not in PROBE_AGENTS and agent not in MODEL_AGENTS:
            audit.unattributed_missing += count
            continue
        spec = ROOT / 'tasks' / Path(task).name / 'tests/task.json'
        expected = json.loads(spec.read_text()) if spec.is_file() else {}
        audit.missing.extend({'trial': None, 'status': 'not_run', 'reward': None,
                              'errors': ['No matching trial directory'], 'diagnostic': agent in PROBE_AGENTS,
                              'benchmark': expected.get('benchmark', 'unknown'), 'task': expected.get('task', task),
                              'agent': agent, 'model': model, 'task_digest': digest} for _ in range(count))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Harbor job or trial directory')
    parser.add_argument('--output', type=Path, help='Optional audit JSON file')
    args = parser.parse_args()
    plan = JobPlanAudit()
    config_file = args.path / 'config.json'
    if ((args.path / 'artifacts').exists() or (args.path / 'agent').exists() or
            (config_file.is_file() and 'task' in json.loads(config_file.read_text()))):
        rows = [check_trial(args.path)]
    else:
        trials = sorted(p for p in args.path.iterdir() if p.is_dir() and (p / 'config.json').exists())
        rows = [check_trial(p) for p in trials]
        plan = audit_job_plan(args.path, rows)
        rows.extend(plan.missing)
    coverage_unknown = bool(plan.unattributed_missing or plan.unexpected or plan.errors)
    evaluation_rows = [r for r in rows if r['status'] != 'unexpected' and not r.get('diagnostic')]
    valid = [r['reward'] for r in evaluation_rows if r['status'].startswith('valid_')]
    report = {'trials': rows, 'evaluation_trials': len(evaluation_rows), 'total_trials': len(rows) + plan.unattributed_missing,
              'missing_trial_directories': plan.unattributed_missing + len(plan.missing), 'valid_trials': len(valid),
              'invalid_trials': sum(r['status'] == 'invalid' for r in rows),
              'diagnostic_trials': sum(bool(r.get('diagnostic')) and r['status'] != 'unexpected' for r in rows),
              'unexpected_trials': plan.unexpected, 'plan_errors': plan.errors,
              'not_run_trials': plan.unattributed_missing + sum(r['status'] == 'not_run' for r in rows),
              'success_rate_over_valid_trials': statistics.mean(valid) if valid else None,
              'population_variance_over_valid_trials': statistics.pvariance(valid) if valid else None,
              'coverage': len(valid) / len(evaluation_rows) if evaluation_rows else 0,
              'note': 'Diagnostics are excluded from model performance. Invalid or missing evaluation attempts prevent a complete benchmark score. Completion reward is not necessarily an upstream native score.'}
    if plan.unattributed_missing:
        report['evaluation_trials'] = None
    if coverage_unknown:
        report['coverage'] = None
    groups = {}
    for row in evaluation_rows:
        key = row.get('benchmark','unknown') + '/' + row.get('task','unknown')
        groups.setdefault(key,[]).append(row)
    report['tasks'] = {}
    for key, group in groups.items():
        values = [r['reward'] for r in group if r['status'].startswith('valid_')]
        report['tasks'][key] = {'trials':len(group),'valid_trials':len(values),
            'success_rate':statistics.mean(values) if values else None,
            'population_variance':statistics.pvariance(values) if values else None,
            'coverage':None if coverage_unknown else len(values)/len(group)}
    report['task_macro_success_rate'] = statistics.mean([v['success_rate'] for v in report['tasks'].values()]) if report['tasks'] and all(v['coverage']==1 for v in report['tasks'].values()) else None
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    return 0 if not coverage_unknown and evaluation_rows and len(valid) == len(evaluation_rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
