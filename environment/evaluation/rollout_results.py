"""Extract one native episode result, never infer success from process exit or survival."""
from .task_names import named_record
import csv
import json
import math
import os
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def one(items, name):
    if len(items) != 1:
        raise ValueError(f'Expected one {name}, found {len(items)}')
    return items[0]


def scalar_metrics(value, prefix=''):
    """Flatten declared score/metric containers; singleton tensors are scalars, traces are not."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(scalar_metrics(item, f'{prefix}.{key}' if prefix else key))
        return out
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if type(value) in (int, float) and math.isfinite(value):
        return {prefix: value}
    return {}


def extract(bench, root):
    root = Path(root)
    if bench == 'robodojo':
        path = root/'episode.json'; data = load(path); success = data.get('official_success')
    elif bench == 'behavior_1k':
        path = root/'results.json'; data = one(list(load(path).values()), 'BEHAVIOR result'); success = data.get('success')
    elif bench == 'robocasa':
        path = one(list(root.glob('*/episode-*/episode.json')), 'RoboCasa episode')
        data = load(path); success = data.get('success')
    elif bench == 'robolab':
        stopped=root/'interaction-stop-result.json'
        path = stopped if stopped.exists() else root/'official/episode_results.jsonl'
        data = load(path) if stopped.exists() else one([json.loads(s) for s in path.read_text().splitlines() if s.strip()], 'RoboLab episode')
        success = data.get('success')
        if not load(root/'evaluation-finished.json').get('official_runner_completed'):
            raise ValueError('RoboLab runner did not finish')
    else:
        path = root/('evaluation-finished.json' if bench == 'humanoid_soccer' else 'result.json')
        data = load(path); success = data.get('success')
    if bench == 'ai_cps' and str(data.get('case')) == '24':
        from environment.benchmarks.ai_cps.scoring import peg_insertion_check
        native_success = data.get('native_success', success)
        trace_path = root/'native_trace.json'
        check = (peg_insertion_check(load(trace_path), data.get('native_episode_complete') is True)
                 if trace_path.is_file() else {'valid': False, 'success': None, 'reason': 'missing_native_trace'})
        data = {**data, 'native_success': native_success, 'success': check['success'],
                'insertion_evaluation': check, 'scoring_profile': 'peg-xy-z-window-v2',
                'success_definition': check.get('definition')}
        success = data['success']
    if data.get('diagnostic_only') or data.get('not_model_performance'):
        raise ValueError('Diagnostic/reference validation must not enter model success-rate statistics')
    if data.get('termination') in {'wall_timeout', 'timeout', 'action_budget', 'infrastructure_error'}:
        raise ValueError('Policy ended before a scored environment boundary: '+str(data['termination']))
    if success is not None and type(success) is not bool:
        raise ValueError(f'Unexpected success type in {path}')
    # Explicit terminal dictionaries only; no arbitrary observation/state flattening.
    metrics = {}
    for field in ('metrics', 'native_metrics', 'q_score', 'normalized_agent_distance', 'agent_distance',
                  'native_weighted_reward_component_sums', 'weighted_reward_component_sums', 'robotworld'):
        metrics.update(scalar_metrics(data.get(field, {}), field))
    for field in ('score', 'reward', 'return', 'reward_sum', 'native_reward_sum', 'robustness', 'dangerous_rate', 'completion_time'):
        metrics.update(scalar_metrics(data.get(field), field))
    for container in ('evaluation', 'last_evaluation'):
        inner = data.get(container, {})
        for field in ('reward', 'return', 'score', 'metrics', 'stats', 'native_stats_after_reward'):
            metrics.update(scalar_metrics(inner.get(field), f'{container}.{field}'))
    if bench == 'behavior_1k':
        metrics.update(scalar_metrics(data.get('time', {}).get('normalized_time'), 'normalized_time'))
    if bench == 'humanoid_soccer' and (root/'summary.json').exists():
        native_scores = load(root/'summary.json')
        native_scores.pop('num_trials', None)
        metrics.update(scalar_metrics(native_scores, 'native_scores'))
    world=data.get('world_evaluation')
    if data.get('scoring_profile')=='world-state-v1':
        if not isinstance(world,dict) or not world.get('valid'):
            success=None
        else:
            success=world.get('world_success')
            if type(success) is not bool:raise ValueError('Valid World evaluation must have binary success')
    from environment.runtime.nonaction_budget import VERSION, CALIBRATED, REASONS, limits
    budget=data.get('interaction_budget')
    if budget is None:
        candidates=list(root.glob('**/interaction-budget.json'))
        if len(candidates)==1:budget=load(candidates[0])
    if budget and budget.get('version') in (VERSION, *CALIBRATED) and data.get('stop_reason',data.get('termination')) in REASONS:
        consecutive, total = limits(budget['version'])
        if (budget.get('mode', 'enforce') != 'enforce' or
                budget.get('stop_reason') != data.get('stop_reason',data.get('termination')) or
                (budget['version'] in CALIBRATED and
                 (budget.get('total_limit') != total or budget.get('consecutive_limit') != consecutive)) or
                (budget.get('total',0)<total and budget.get('consecutive',0)<consecutive)):
            raise ValueError('Invalid interaction-budget termination')
        success=success is True or (data.get('scoring_profile')!='world-state-v1' and data.get('success') is True)
    # Keep the exact result and its native success explanation available to the reviewer.
    record = {'success': success, 'scores': metrics, 'result_file': str(path.relative_to(root)),
            'scoring_profile':data.get('scoring_profile','native'),
            'native_success':data.get('native_success',data.get('success')),
            'insertion_evaluation':data.get('insertion_evaluation'),
            'world_evaluation':world,
            'success_definition': data.get('success_definition'),
            'stop_reason': data.get('stop_reason', data.get('termination')),
            'control_steps': data.get('control_steps', data.get('steps', data.get('episode_step')))}
    if budget is not None:record['interaction_budget']=budget
    return record


def moments(values):
    return {'n': len(values), 'mean': statistics.fmean(values) if values else None,
            'variance': statistics.pvariance(values) if values else None,
            'sample_variance': statistics.variance(values) if len(values) > 1 else None}


def summarize(records):
    valid = [r for r in records if r['status'] == 'completed']
    binary = [int(r['result']['success']) for r in valid if r['result']['success'] is not None]
    keys = sorted({k for r in valid for k in r['result']['scores']})
    return {'requested': len(records), 'completed': len(valid),
            'infrastructure_errors': sum(r['status'] == 'infrastructure_error' for r in records),
            'not_run': sum(r['status'] == 'not_run' for r in records),
            'interrupted': sum(r['status'] == 'interrupted' for r in records),
            'unscored_success': sum(r['result']['success'] is None for r in valid),
            'artifact_warnings': sum(bool(r.get('artifact_warnings')) for r in records),
            'success_rate': moments(binary),
            'scores': {k: moments([r['result']['scores'][k] for r in valid if k in r['result']['scores']]) for k in keys}}


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def index_artifacts(run):
    """Expose native files without copies; preserve complete original launch hierarchy."""
    root = run/'artifacts'
    if not root.exists(): return ['No environment artifacts']
    groups = {'videos': sorted(root.rglob('*.mp4')),
              'events': sorted(p for p in root.rglob('*.jsonl') if 'events' in p.parts),
              'programs': sorted(p for p in root.rglob('*') if p.is_file() and
                                 ('programs' in p.parts or 'controllers' in p.parts) and p.suffix in {'.py', '.json'})}
    warnings = []
    if not groups['videos']: warnings.append('No video was produced; inspect launcher.log')
    if not any(p.name == 'codex.jsonl' and 'no-images' not in p.parts for p in groups['events']):
        warnings.append('No full codex.jsonl trajectory found')
    for name, paths in groups.items():
        for source in paths:
            target = run/name/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.symlink_to(os.path.relpath(source, target.parent))
    workspace = root/'agent-workspace'
    if workspace.exists() and not (run/'agent-workspace').exists():
        (run/'agent-workspace').symlink_to('artifacts/agent-workspace', target_is_directory=True)
    save_json(run/'artifact-index.json', {k: [str(p.relative_to(run)) for p in v] for k, v in groups.items()})
    return warnings


def write_summary(folder, batch, records, config):
    records = [named_record(r) for r in records]
    by_task = {}
    for record in records: by_task.setdefault(record['task'], []).append(record)
    report = {'batch': batch, 'configuration': config,
              'statistics': 'variance: population (ddof=0); sample_variance: ddof=1, null for n<2. SR is Bernoulli mean over valid outcomes under the selected scoring profile; diagnostic runs are excluded.',
              'tasks': {t: summarize(rows) for t, rows in by_task.items()}, 'runs': records}
    for task, rows in by_task.items():
        save_json(folder/task/'summaries'/f'{batch}.json', {**report['tasks'][task], 'configuration': config, 'runs': rows,
            'task':task,'task_title':rows[0].get('task_title',task),
            'benchmark':rows[0].get('benchmark'), 'legacy_task_id':rows[0].get('legacy_task_id')})
    # Task-equal headline plus pooled Bernoulli dispersion; never pool incompatible raw scores.
    task_rates=[v['success_rate']['mean'] for v in report['tasks'].values() if v['success_rate']['n']]
    complete=all(v['success_rate']['n']==v['requested'] for v in report['tasks'].values())
    pooled=summarize(records)
    pooled.pop('scores',None)  # Raw scores with identical names need not have the same task scale.
    report['overall']={'coverage_complete':complete,'tasks_requested':len(by_task),
        'tasks_with_binary_results':len(task_rates),
        'task_equal_success_rate':moments(task_rates),
        'full_benchmark_success_rate':statistics.fmean(task_rates) if complete and task_rates else None,
        'pooled_rollouts':pooled,
        'note':'task_equal variance is dispersion across task SRs; pooled variance is dispersion of rollout 0/1. Raw scores remain per task; partial coverage is not a full benchmark score.'}
    target = folder/'summaries'/batch
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_name(target.name+'-runs.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['task','rollout','status','success','steps','code_control','scoring_profile','scores_json','path','error'])
        for r in records:
            writer.writerow([r['task'],r.get('rollout'),r['status'],r.get('result',{}).get('success'),
                r.get('step_limit'),r.get('code_control_effective'),r.get('scoring_profile'),
                json.dumps(r.get('result',{}).get('scores',{})),r.get('path'),r.get('error')])
    save_json(target.with_suffix('.json'), report)
    with target.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'requested', 'completed', 'infrastructure_errors', 'unscored_success', 'metric', 'n', 'mean', 'variance', 'sample_variance'])
        for task, stats in report['tasks'].items():
            for metric, value in {'success_rate': stats['success_rate'], **stats['scores']}.items():
                writer.writerow([task, stats['requested'], stats['completed'], stats['infrastructure_errors'], stats['unscored_success'], metric, value['n'], value['mean'], value['variance'], value['sample_variance']])
    return report
