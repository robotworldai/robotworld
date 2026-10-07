"""Repeat isolated source-Codex episodes with native scoring and durable per-run artifacts."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from .task_names import canonical
from .resume import batch_lock, resume_plans
from .rollout_catalog import WORLD, CORE, catalog
from .rollout_results import extract, index_artifacts, save_json, write_summary


def command_for(bench, task, args, output, auth, steps, seed):
    from .runner import run_command
    common = ['--output', str(output), '--codex-home', str(auth)]
    if task['suite_row'] is None or bench == 'bench2dex':
        command = [sys.executable, '-m', 'environment.runtime.native_project_launch', '--project', bench,
                   '--task', task['task'], '--runtime-profile', task['runtime_profile'], '--mode', 'codex',
                   '--model', args.model, '--steps', str(steps), '--seed', str(seed),
                   '--wall-timeout', str(args.timeout), *common]
    elif bench == 'wheeledlab':
        command=[sys.executable,str(WORLD/'third_party/benchmarks/wheeledlab/docker/run.py'),*common,
                 '--case',task['task_key'],'--model',args.model,'--seed',str(seed),
                 '--steps',str(steps),'--wall-timeout',str(args.timeout)]
    elif bench == 'robolab':
        command = [sys.executable, str(WORLD/'third_party/benchmarks/robolab/docker/run.py'), *common,
                   '--task', task['task'], '--control-mode', 'absolute_ik', '--world-timeout', str(args.timeout),
                   '--wall-timeout', str(args.timeout + 1800), '--max-tool-calls', '1000000', '--world-seed', str(seed)]
        if args.runtime == 'isaac601': command.append('--isaac601')
        if steps != task['native_steps']: command += ['--world-step-cap', str(steps)]
    elif bench == 'humanoid_soccer':
        row = task['suite_row']
        command = [sys.executable, str(WORLD/'third_party/benchmarks/humanoid_soccer/docker/run.py'), *common,
                   '--mode', row['mode'], '--model', args.model, '--seed', str(seed), '--sim-time', str(steps / 50),
                   '--wall-timeout', str(args.timeout), '--scenery', row['scenery'], '--balance-assist', row['balance_assist']]
        # Prior model solutions are not injected into fresh evaluation episodes.
    else:
        a = SimpleNamespace(runtime=args.runtime, steps=None if steps == task['native_steps'] else steps,
                            seed=seed, seed_explicit=True, episode_index=args.episode_index, split=args.split,
                            model=args.model, timeout=args.timeout)
        # These launchers already consume an explicit native step cap.
        if bench in {'ai_cps', 'wheeledlab'}: a.steps = steps
        command = run_command(bench, task['suite_row'], a, output, auth)
        if '--max-actions' in command:
            command[command.index('--max-actions')+1] = '1000000'
        if bench == 'robocasa': command += ['--max-actions', '1000000']
    from .world_success.profiles import get_profile
    profile=get_profile(bench,task['task'])
    requested=getattr(args,'scoring_profile','native')
    scoring='world-state-v1' if requested=='auto' and profile else 'native' if requested=='auto' else requested
    if profile and scoring!='native':command += ['--scoring-profile',scoring]
    control = getattr(args,'task_code_controls',{}).get(task['task'], args.code_control)
    if control == 'off' and task['code_control_supported']:
        command.append('--disable-coding-control')
    return list(map(str, command))


def control_overrides(values, tasks):
    known={r['task_key']:r['task'] for r in tasks}
    known.update({r['task']:r['task'] for r in tasks})
    result={}
    for entry in values:
        key,sep,value=entry.partition('=')
        if not sep or key not in known or value not in ('on','off','default'):
            raise ValueError('--task-code-control expects a known TASK=on|off|default')
        key=known[key]
        if value=='default':result.pop(key,None)
        else:result[key]=value
    return result


def step_overrides(values, tasks):
    known = {r['task'] for r in tasks}
    result = {}
    for entry in values:
        if '=' not in entry: raise ValueError('--task-steps expects TASK=native or TASK=positive_integer')
        key, value = entry.split('=', 1)
        key = next((r['task'] for r in tasks if r.get('task_key') == key), key)
        if key not in known: raise ValueError('Unknown step-override task: '+key)
        if value == 'profile':result[key]=None
        elif value == 'native': result[key] = next(r['native_steps'] for r in tasks if r['task']==key)
        else:
            number = int(value)
            if number <= 0: raise ValueError('Step limits must be positive')
            result[key] = number
    return result


def execute(record, command, runner=subprocess.run):
    """One leaf directory = one rollout; no in-place retries or reuse of agent memory."""
    run = Path(record['path']); run.mkdir(parents=True, exist_ok=False)
    record.update(status='running', command=command)
    save_json(run/'run.json', record)
    try:
        with (run/'orchestrator.log').open('w') as log:
            completed = runner(command, cwd=WORLD, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = completed.returncode
        native_exit = run/'artifacts/exit.json'
        if native_exit.exists():
            native = json.loads(native_exit.read_text())
            if native.get('infrastructure_ok') is False or native.get('returncode', 0) != 0:
                raise RuntimeError('Native launcher exit reports an infrastructure error; see artifacts/exit.json')
        if completed.returncode:
            raise RuntimeError(f'Launcher exit {completed.returncode}; see orchestrator.log and artifacts/launcher.log')
        if any((run/'artifacts'/name).exists() for name in ('error.txt', 'failure.json', 'episode-aborted.json')):
            raise RuntimeError('Native launcher retained an error/abort record')
        record['result'] = extract(record['benchmark'], run/'artifacts')
        record['status'] = 'completed'
    except KeyboardInterrupt:
        record.update(status='interrupted', error='Interrupted; partial artifacts retained, excluded from scores')
        raise
    except Exception as error:
        record.update(status='infrastructure_error', error=str(error))
    finally:
        record['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        record['artifact_warnings'] = index_artifacts(run)
        save_json(run/'run.json', record)
    return record


def _main(argv=None, lock_stack=None):
    tasks_by_bench = catalog()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', required=True, choices=list(tasks_by_bench))
    p.add_argument('action', nargs='?', default='list', choices=['list', 'run', 'summarize'])
    p.add_argument('--tasks', help='Comma-separated task names/IDs printed by list')
    p.add_argument('--task-steps', action='append', default=[], metavar='TASK=native|N')
    p.add_argument('--rollouts', type=int)
    p.add_argument('--code-control', choices=['on', 'off'], default='off')
    p.add_argument('--task-code-control', action='append', default=[], metavar='TASK=on|off|default')
    p.add_argument('--scoring-profile',choices=['auto','native','world-state-v1'],default='auto',help='auto: approved World state rules for 16 redesigned tasks; native criteria for other tasks')
    p.add_argument('--model'); p.add_argument('--codex-home', type=Path)
    p.add_argument('--output-root', type=Path, default=WORLD/'outputs')
    p.add_argument('--batch', help='Unique label per experiment/model configuration; automatic timestamp by default')
    p.add_argument('--seed', type=int); p.add_argument('--seed-stride', type=int, default=0)
    p.add_argument('--runtime', choices=['official', 'isaac601'], default='isaac601', help='Core runtime; native projects use their pinned task profile, printed by list')
    p.add_argument('--timeout', type=int, default=43200, help='Wall-clock/model time limit in seconds, separate from simulated steps')
    p.add_argument('--episode-index', type=int, default=0); p.add_argument('--split', choices=['pretrain', 'target'], default='pretrain')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Skip completed rollouts; restart interrupted attempts from reset in new directories')
    args = p.parse_args(argv)
    if not args.model and args.codex_home is not None:
        import tomllib
        config_path = args.codex_home / 'config.toml'
        if config_path.is_file():
            args.model = tomllib.loads(config_path.read_text()).get('model')
    if args.batch and not re.fullmatch(r'[A-Za-z0-9_-]+', args.batch): p.error('Invalid batch label')
    if args.resume and (args.action != 'run' or not args.batch): p.error('--resume requires run and --batch')
    prior = None
    if args.resume:
        manifest = args.output_root.resolve()/args.bench/'summaries'/f'{args.batch}.json'
        if not manifest.is_file(): p.error('No batch summary to resume: '+str(manifest))
        if not args.dry_run: lock_stack.enter_context(batch_lock(manifest.with_suffix('.lock')))
        prior = json.loads(manifest.read_text())
        if args.tasks is None: args.tasks = ','.join(dict.fromkeys(r['task'] for r in prior['runs']))
        if args.rollouts is None: args.rollouts = prior['configuration']['rollouts']
        for field in ('episode_index','split'):
            if getattr(args, field) != prior['configuration'].get(field): p.error('Resume '+field+' differs; use a new batch')
    if not args.model and prior is not None:
        args.model = prior['configuration'].get('model')
    if not args.model:
        if args.action == 'run' and not args.dry_run:
            p.error('Configure a model with scripts/configure_api.py or pass --model')
        args.model = 'YOUR_MODEL_ID'
    if args.rollouts is None: args.rollouts = 3
    if args.rollouts < 1 or args.timeout < 1 or args.episode_index < 0: p.error('Invalid rollout count, timeout or episode index')
    all_tasks = tasks_by_bench[args.bench]
    try:
        overrides = step_overrides(args.task_steps, all_tasks)
        args.task_code_controls = control_overrides(args.task_code_control, all_tasks)
    except ValueError as error: p.error(str(error))
    tasks = all_tasks
    if args.tasks:
        keys = {canonical(args.bench, key) for key in args.tasks.split(',')}
        missing = keys - {t['task'] for t in tasks}
        if missing: p.error('Unknown tasks: '+', '.join(sorted(missing)))
        tasks = [t for t in tasks if t['task'] in keys]
    batch = args.batch or dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', batch): p.error('--batch accepts letters, digits, underscore and hyphen only')
    folder = args.output_root.resolve()/args.bench
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'codex_home'}
    if args.action == 'summarize':
        if not args.batch: p.error('summarize requires --batch to avoid mixing models/budgets')
        manifest = folder/'summaries'/f'{batch}.json'
        lock_stack.enter_context(batch_lock(manifest.with_suffix('.lock')))
        prior = json.loads(manifest.read_text())
        records = []
        for old in prior['runs']:
            path = Path(old['path'])/'run.json'
            records.append(json.loads(path.read_text()) if path.exists() else old)
        write_summary(folder, batch, records, prior['configuration']); print(manifest); return 0
    plans = []
    for task in tasks:
        from .world_success.profiles import get_profile
        profile=get_profile(args.bench,task['task'])
        scoring='world-state-v1' if profile and args.scoring_profile!='native' else 'native'
        horizon=profile['steps'] if scoring=='world-state-v1' else task['native_steps']
        steps = overrides.get(task['task']) or horizon
        error = None
        if steps is None:
            error = 'Native horizon unavailable: supply explicit --task-steps '+task['task']+'=N (custom budget)'
        elif horizon is not None and steps > horizon:
            p.error(f"{task['task']}: {steps} exceeds {scoring} horizon {horizon}; native failure termination is preserved")
        if task['scene_seed_fixed'] and (args.seed is not None or args.seed_stride):
            p.error(f"{args.bench} uses a fixed prepared scene/instance; --seed/--seed-stride would be misleading. Each rollout starts a fresh agent.")
        control = args.task_code_controls.get(task['task'], args.code_control)
        seed = args.seed if args.seed is not None else task['seed']
        if args.action == 'list':
            print(f"{task['task']} ({task['task_title']}) | native={task['native_steps']} effective={steps} scoring={scoring} | code_control={control if task['code_control_supported'] else 'unavailable (ordinary actions only)'} | profile={task['runtime_profile']} | {task['protocol_kind']}")
            print('  '+task['budget_source']+(f' | {error}' if error else ''))
        for rollout in range(args.rollouts):
            path = folder/task['task']/f'run-{batch}-{rollout+1:04d}'
            record = {'benchmark': args.bench, 'task': task['task'], 'task_title':task.get('task_title',task['task']),
                      'legacy_task_id':task.get('legacy_task_id'), 'batch': batch, 'rollout': rollout+1,
                      'path': str(path), 'status': 'not_run', 'model': args.model,
                      'step_limit': steps, 'scoring_profile':scoring, 'native_steps': task['native_steps'], 'budget_source': task['budget_source'],
                      'budget_override': steps != task['native_steps'], 'protocol_kind': task['protocol_kind'],
                      'runtime': args.runtime if args.bench in CORE else task['runtime_profile'],
                      'code_control_requested': control == 'on',
                      'code_control_effective': control == 'on' and task['code_control_supported'],
                      'seed': None if args.bench in {'robodojo', 'behavior_1k'} else seed+rollout*args.seed_stride,
                      'scene_seed_fixed': task['scene_seed_fixed'], 'scene_config': task['suite_row'],
                      'started_at': None}
            if error: record['error'] = error
            plans.append((task, record, seed+rollout*args.seed_stride))
    if args.action == 'list': return 0
    if prior is not None:
        try: plans = resume_plans(plans, prior)
        except ValueError as error: p.error(str(error))
    if args.dry_run:
        for task, record, seed in plans:
            if record['status'] == 'completed':
                print('SKIP completed: '+record['path']); continue
            if record.get('error'): print(record['task']+': NOT RUN: '+record['error']); continue
            cmd = command_for(args.bench, task, args, Path(record['path'])/'artifacts', Path('/private/codex-home'), record['step_limit'], seed)
            print(shlex.join(cmd))
        return 1 if any(r.get('error') for _, r, _ in plans) else 0
    records = [r for _, r, _ in plans]
    if all(r['status'] == 'completed' for r in records):
        write_summary(folder, batch, records, prior['configuration'])
        print('All rollouts already completed; no model or environment launched.'); return 0
    if args.codex_home is None: p.error('run requires --codex-home pointing to local Codex login/provider configuration')
    if not args.resume:
        lock_stack.enter_context(batch_lock(folder/'summaries'/f'{batch}.lock'))
    if not args.resume and ((folder/'summaries'/f'{batch}.json').exists() or any(Path(r['path']).exists() for _, r, _ in plans)):
        p.error('Batch already exists. Use --resume with the same settings or choose a new --batch.')
    records = [r for _, r, _ in plans]
    with tempfile.TemporaryDirectory(prefix='world-rollouts-auth-') as temp:
        from environment.runtime.model_config import prepare_model_home
        auth = Path(temp)
        config['model_info'] = prepare_model_home(args.codex_home.resolve(), auth, args.model)
        write_summary(folder, batch, records, config)
        for task, record, seed in plans:
            if record['status'] == 'completed':
                print('SKIP completed: '+record['path'], flush=True); continue
            if record.get('error'):
                Path(record['path']).mkdir(parents=True)
                save_json(Path(record['path'])/'run.json', record)
                continue
            cmd = command_for(args.bench, task, args, Path(record['path'])/'artifacts', auth, record['step_limit'], seed)
            print(f"{args.bench}/{task['task']} rollout {record['rollout']}/{args.rollouts}, steps={record['step_limit']}, code_control={record['code_control_effective']}", flush=True)
            record['started_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            try: execute(record, cmd)
            finally: write_summary(folder, batch, records, config)
            print(f"  {record['status']}: {record.get('result', {}).get('success')} -> {record['path']}", flush=True)
    write_summary(folder, batch, records, config)
    print('Summary:', folder/'summaries'/f'{batch}.json')
    return 1 if any(r['status'] != 'completed' for r in records) else 0

def main(argv=None):
    from contextlib import ExitStack
    with ExitStack() as stack:
        return _main(argv, stack)


if __name__ == '__main__':
    raise SystemExit(main())
