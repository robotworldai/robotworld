"""Pair model-generated actions with a no-model Harbor replay of the SAME run.

Private validation artifacts belong under var/validation, never task solutions.
No historical leaderboard result is used as a target.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def extract(trial, spec):
    trial = Path(trial)
    result = read(trial / 'result.json')
    if result.get('exception_info'):
        raise ValueError('Source model trial is invalid; do not replay it as a task failure')
    if (result.get('agent_info') or {}).get('name') in {'robotworld-reference-replay', 'robotworld-probe', 'robotworld-smoke100', 'robotworld-native-budget-probe'}:
        raise ValueError('Source must be a model trial, not a probe or replay')
    runs = trial / 'artifacts/simulation'
    episode = read(runs / 'episode.json')
    if not episode.get('terminal') or type(episode.get('success')) is not bool:
        raise ValueError('Source environment has no terminal outcome')
    config_path = runs / 'service-config.json'
    if not config_path.exists(): config_path = runs / 'evaluation-config.json'
    config = read(config_path)
    if any(config.get(k) != spec[k] for k in ('benchmark', 'task')):
        raise ValueError('Source task identity mismatch')
    commands, batches, pending = [], [], None
    for line in (runs / 'events.jsonl').read_text().splitlines():
        event = json.loads(line); data = event['data']
        if event['event'] == 'execution_requested':
            if pending is not None: raise ValueError('Overlapping source executions')
            pending = data
        elif event['event'] == 'execution_completed':
            if pending is None or pending['request_id'] != data['request_id']:
                raise ValueError('Unmatched source receipt')
            if data.get('execution_uncertain') or data.get('error'):
                raise ValueError('Uncertain source action; cannot build a faithful replay')
            batches.append(pending['commands'])
            commands.extend(pending['commands']); pending = None
    if pending is not None or (not commands and episode['control_step'] != 0):
        raise ValueError('Incomplete or empty source action trajectory')
    return {'benchmark': spec['benchmark'], 'task_aliases': [str(spec['task'])],
            'historical_id': 'paired-source:' + trial.name,
            'expected': {'success': episode['success'], 'control_steps': episode['control_step']},
            'commands': commands, 'command_batches': batches}, episode


def completed_trial(job):
    candidates = list(Path(job).glob('*/result.json'))
    if len(candidates) != 1: raise ValueError('Expected one trial per paired job')
    return candidates[0].parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--harbor', default='harbor')
    p.add_argument('--continue-on-error', action='store_true', help='Record a blocked task and continue other tasks')
    p.add_argument('--task', action='append', help='Task directory name; default: all tasks in model config')
    p.add_argument('--source-trial', type=Path, help='Reuse one existing model trial; requires one --task')
    args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.model_config.read_text())
    task_paths = [Path(t['path']) for t in config['tasks']]
    task_paths = [t if t.is_absolute() else root/t for t in task_paths]
    if args.task: task_paths = [t for t in task_paths if t.name in args.task]
    if not task_paths or (args.source_trial and len(task_paths) != 1): raise ValueError('Invalid task selection')
    jobs = out/'jobs'; jobs.mkdir(exist_ok=True)
    results = []
    def run(task, phase, agent, env):
        job_name = task.name+'--'+phase
        job = jobs/job_name
        if (job/'result.json').exists() and read(job/'result.json').get('finished_at'):
            return completed_trial(job)
        cfg = out/(job_name+'.yaml')
        cfg.write_text(yaml.safe_dump({'jobs_dir': str(jobs), 'n_attempts':1,
            'n_concurrent_trials':1, 'agents':[agent], 'tasks':[{'path':str(task)}]}))
        with (out/(job_name+'.log')).open('a') as log:
            subprocess.run([args.harbor,'run','--config',str(cfg),'--job-name',job_name],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return completed_trial(job)
    env = dict(os.environ)
    for task in task_paths:
        row = {'task':task.name, 'classification':'same_run_model_output_replay'}
        try:
            spec = read(task/'tests/task.json')
            fingerprint = {str(f.relative_to(task)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in task.rglob('*') if f.is_file()}
            source = args.source_trial.resolve() if args.source_trial else run(task,'model',config['agents'][0],env)
            row['source_trial'] = str(source)
            reference, original = extract(source,spec)
            ref_path = out/(task.name+'--actions.json'); ref_path.write_text(json.dumps([reference],indent=2))
            current = {str(f.relative_to(task)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in task.rglob('*') if f.is_file()}
            if current != fingerprint: raise ValueError('Task changed between generation and replay')
            replay_env = {**env,'WORLD_REPLAY_REFERENCES':str(ref_path)}
            source_cap = original.get('control_step_limit')
            if source_cap is not None: replay_env['WORLD_CONTROL_STEP_LIMIT']=str(source_cap)
            else: replay_env.pop('WORLD_CONTROL_STEP_LIMIT',None)
            row['source_control_step_limit']=source_cap
            row['source_action_batches']=len(reference['command_batches'])
            replay = run(task,'replay',{'import_path':'agents.robotworld:RobotWorldReferenceReplayAgent'},
                         replay_env)
            result = read(replay/'result.json')
            if result.get('exception_info'): raise ValueError('Replay infrastructure error: '+str(result['exception_info']['exception_type']))
            final = read(replay/'artifacts/simulation/episode.json')
            summary = read(replay/'agent/robotworld/summary.json')
            if summary['errors']: raise ValueError('Replay action errors: '+str(summary['errors']))
            reward = (result.get('verifier_result') or {}).get('rewards',{}).get('reward')
            row.update(replay_trial=str(replay),source_outcome=original,replay_outcome=final,
                       replay_reward=reward,same_success=original['success']==final['success'],
                       same_steps=original['control_step']==final['control_step'],
                       verifier_matches_environment=reward==int(final['success']),task_sha256=fingerprint)
            row['status']='match' if row['same_success'] and row['same_steps'] and row['verifier_matches_environment'] else 'divergence'
            if not reference['command_batches']: row['status']='no_action_source'
        except Exception as exc:
            row.update(status='blocked_or_invalid',error=str(exc))
        results.append(row)
        (out/'pairs.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k in ('task','status','error','same_success','same_steps')}),flush=True)
        # Do not repeat provider/configuration failures over an entire paid suite.
        if row['status']=='blocked_or_invalid' and not args.continue_on_error: break

if __name__=='__main__': main()
