"""One source-Codex rollout per benchmark, with artifact and control-path checks."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

from .rollout_catalog import WORLD, catalog
from .rollout_results import save_json
from .task_names import canonical, named_record, title
from .resume import batch_lock, campaign_record

# Prefer previously loaded scenes. Known-blocked projects remain visible and are attempted.
CASES = {
    'robocasa': 'CoffeeSetupMug', 'robodojo': 'deposit_coin',
    'behavior_1k': 'clean_up_your_desk', 'robolab': 'RubiksCubeLeftOfBowlTask',
    'humanoid_soccer': 'play-soccer', 'ai_cps': '22', 'wheeledlab': 'mushr-drift',
    'bench2dex': '41', 'steadytray': 'T01', 'ttrl': 'T02', 'reflexbench': 'T03',
    'volleybots': 'T05-single', 'wheel_legged': 'T07',
    'wheeled_quadruped': 'T09', 'go2_push': 'T10', 'robot_lab': 'T11',
    'digit': 'T13', 'omniisaacgymenvs': 'T14', 'flamingo': 'T15', 'omnidrones': 'T16',
}

CASES = {bench: canonical(bench, task) for bench, task in CASES.items()}


def events(path):
    if not path.exists(): return
    with path.open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def audit(run):
    run = Path(run); record = named_record(json.loads((run/'run.json').read_text()))
    raw = run/'artifacts'
    result = {'benchmark': record['benchmark'], 'task': record['task'], 'run': str(run),
              'rollout_status': record['status'], 'task_success': record.get('result', {}).get('success'),
              'smoke_passed': False, 'checks': {}, 'videos': [], 'errors': []}
    checks = result['checks']
    checks['scored_launcher_exit'] = record['status'] == 'completed'
    checks['artifact_index'] = (run/'artifact-index.json').is_file()
    checks['prompt_saved'] = any(raw.rglob('prompt.json'))
    # RoboDojo disables shell entirely; its source-built Codex runs inside the Docker image.
    checks['workspace_retained_or_not_applicable'] = (raw/'agent-workspace').is_dir() or record['benchmark'] == 'robodojo'
    codex = [p for p in raw.rglob('codex.jsonl') if 'no-images' not in p.parts]
    checks['full_codex_events'] = bool(codex) and all(p.stat().st_size > 0 for p in codex)
    checks['no_images_codex_events'] = bool(codex) and all((p.parent/'no-images'/p.name).is_file() for p in codex)
    tool_logs = [p for p in raw.rglob('tools.jsonl') if 'no-images' not in p.parts]
    requested = []
    try:
        for path in tool_logs:
            requested.extend(e['payload'] for e in events(path) if e.get('kind') == 'tool_requested')
        # Check actual registered tools, rather than trusting the requested CLI flag.
        registered = []
        for path in raw.rglob('prompt.json'):
            registered.extend(t['name'] for t in json.loads(path.read_text()).get('dynamicTools', []))
        checks['code_control_schema_matches'] = ('coding_control' in registered) == record['code_control_effective'] if registered else False
    except (ValueError, KeyError) as error:
        result['errors'].append('Event/prompt parse: '+str(error))
    result['tools_requested'] = [r.get('tool') for r in requested]
    checks['model_requested_robot_tool'] = any(r.get('tool') not in {'observe', 'give_up', None} for r in requested)
    steps = record.get('result', {}).get('control_steps')
    if steps is None:
        steps = 0
        for path in raw.rglob('environment.jsonl'):
            if 'no-images' in path.parts: continue
            try:
                steps += sum(e.get('kind') in {'action_completed','environment_step'} for e in events(path))
            except ValueError as error: result['errors'].append('Environment events: '+str(error))
    # RoboDojo reports executed planner waypoints in its authoritative episode trace.
    if record['benchmark'] == 'robodojo' and (raw/'episode.json').exists():
        ep = json.loads((raw/'episode.json').read_text())
        steps = sum(e.get('execution', {}).get('executed_waypoints', 0) for e in ep.get('events', []))
    result['recorded_steps'] = steps
    checks['physics_advanced'] = isinstance(steps, (int, float)) and steps > 0
    for path in sorted(raw.rglob('*.mp4')):
        video = {'path': str(path), 'bytes': path.stat().st_size, 'valid': False}
        try:
            proc = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                                   '-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(path)],
                                  capture_output=True,text=True,timeout=60)
            streams = json.loads(proc.stdout).get('streams', []) if proc.returncode == 0 else []
            if streams:
                video.update(streams[0]); video['valid'] = int(streams[0].get('nb_read_frames', 0)) > 0
            else: video['error'] = proc.stderr[-500:]
        except (ValueError, subprocess.TimeoutExpired) as error: video['error'] = str(error)
        result['videos'].append(video)
    checks['readable_videos'] = bool(result['videos']) and all(v['valid'] for v in result['videos'])
    checks['video_index_links'] = bool(result['videos']) and all(
        (run/'videos'/Path(v['path']).relative_to(raw)).is_file() for v in result['videos'])
    checks['event_index_links'] = bool(codex) and all((run/'events'/p.relative_to(raw)).is_file() for p in codex)
    if record.get('error'): result['errors'].append(record['error'])
    result['smoke_passed'] = all(checks.values()) and not result['errors']
    save_json(run/'smoke-audit.json', result)
    return result


def report(folder, campaign):
    # User-retired case; historical artifacts remain outside the active result table.
    campaign['cases'] = [r for r in campaign['cases'] if r['benchmark'] != 'aerial_balance']
    for row in campaign['cases']:
        row.update(named_record(row))
        if row.get('status') == 'running':
            campaign_record(row, folder.parents[1], campaign['batch'])
        if Path(row['run']).exists(): row['run'] = str(Path(row['run']).resolve())
        if row.get('audit'):
            row['audit'] = named_record(row['audit'])
            row['audit']['run'] = row['run']
    save_json(folder/'campaign.json', campaign)
    with (folder/'README.md').open('w') as stream:
        stream.write('# One model smoke episode per benchmark\n\n')
        stream.write('This campaign uses native step limits to check the source-built agent, robot tools, simulator, and result archive. A smoke check can pass even when the task fails. These are execution checks, not official benchmark success rates.\n\n')
        stream.write(f"Campaign status: {campaign.get('status', 'unknown')}.\n\n")
        stream.write('| Benchmark | Task | Step limit | Status | Smoke check | Task success | Run directory |\n|---|---|---:|---|---|---|---|\n')
        for row in campaign['cases']:
            check = row.get('audit', {})
            stream.write(f"| {row['benchmark']} | {title(row['benchmark'], row['task'])} (`{row['task']}`) | {row['steps']} | {row['status']} | {check.get('smoke_passed', 'Pending review')} | {check.get('task_success', '—')} | [{Path(row['run']).name}]({row['run']}) |\n")
        stream.write('\n## Videos\n\n')
        for row in campaign['cases']:
            for video in row.get('audit', {}).get('videos', []):
                stream.write(f"- {row['benchmark']} / [{Path(video['path']).name}]({video['path']}): {video.get('nb_read_frames','?')} frames, {video.get('duration','?')} seconds\n")
        stream.write('\n## Failed checks\n\n')
        for row in campaign['cases']:
            check = row.get('audit', {})
            if check and not check.get('smoke_passed'):
                stream.write(f"- {row['benchmark']}：{', '.join(k for k,v in check.get('checks',{}).items() if not v)}；{' / '.join(check.get('errors', []))}\n")


def _main(argv=None, lock_stack=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='list', choices=['list','run','audit'])
    p.add_argument('--benchmarks', help='Comma-separated subset; default all 20')
    # Smoke reduces task/rollout count only; the benchmark supplies each horizon.
    p.add_argument('--model', default='gpt-6-astra'); p.add_argument('--codex-home', type=Path)
    p.add_argument('--code-control', choices=['on','off'], default='on')
    p.add_argument('--timeout', type=int, default=43200)
    p.add_argument('--batch'); p.add_argument('--output-root', type=Path, default=WORLD/'outputs')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Skip completed cases and restart incomplete rollouts from initial state')
    a = p.parse_args(argv)
    if a.resume and (a.action != 'run' or not a.batch): p.error('--resume requires run and --batch')
    if a.timeout < 1: p.error('Timeout must be positive')
    batch = a.batch or 'smoke-'+dt.datetime.now().strftime('%Y%m%dT%H%M%S')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', batch): p.error('Invalid batch label')
    folder = a.output_root.resolve()/'_smoke'/batch
    prior = None
    if a.resume:
        if not (folder/'campaign.json').is_file(): p.error('No campaign.json to resume')
        if not a.dry_run: lock_stack.enter_context(batch_lock(folder/'.lock'))
        prior = json.loads((folder/'campaign.json').read_text())
        if a.model != prior['model']: p.error('Resume model differs; use a new batch')
        if a.benchmarks is None: a.benchmarks = ','.join(r['benchmark'] for r in prior['cases'])
        if set(a.benchmarks.split(',')) != {r['benchmark'] for r in prior['cases']}: p.error('Resume must use the same benchmark set')
    if a.action == 'audit':
        if not a.batch: p.error('audit requires --batch')
        lock_stack.enter_context(batch_lock(folder/'.lock'))
        campaign = json.loads((folder/'campaign.json').read_text())
        for row in campaign['cases']:
            record = campaign_record(row, a.output_root.resolve(), batch)
            if (Path(row['run'])/'run.json').exists():
                row['status'] = record['status']; row['audit'] = audit(row['run'])
        report(folder,campaign); print(folder/'README.md'); return
    registry = catalog()
    benches = a.benchmarks.split(',') if a.benchmarks else list(CASES)
    if len(set(benches)) != len(benches) or set(benches)-CASES.keys(): p.error('Invalid/duplicate benchmarks')
    campaign = {'batch':batch, 'model':a.model, 'purpose':'end-to-end source Codex smoke, not task success-rate evaluation', 'cases':[]}
    if prior is not None: campaign = {**prior, 'cases':[]}
    for bench in benches:
        task = next(t for t in registry[bench] if t['task'] == CASES[bench])
        steps = task['native_steps']
        if steps is None: p.error('Selected smoke task lacks an official horizon: '+task['task'])
        run = a.output_root.resolve()/bench/task['task']/f'run-{batch}-0001'
        command = [sys.executable,'-m','environment.evaluation.rollouts','--bench',bench,'run',
                   '--tasks',task['task'],'--task-steps',f"{task['task']}=native",'--rollouts','1',
                   '--code-control',a.code_control,'--model',a.model,'--batch',batch,
                   '--output-root',str(a.output_root.resolve()),'--timeout',str(a.timeout)]
        if a.codex_home: command += ['--codex-home',str(a.codex_home.resolve())]
        row = {'benchmark':bench,'task':task['task'],'steps':steps,'run':str(run),'status':'pending','command':command}
        if prior is not None:
            old = next(r for r in prior['cases'] if r['benchmark'] == bench)
            if canonical(bench, old['task']) != task['task'] or old['steps'] != steps: p.error('Smoke task/budget changed; use a new batch')
            previous_command = old['command']
            if previous_command[previous_command.index('--code-control')+1] != a.code_control:
                p.error('Resume code-control setting differs; use a new batch')
            row = {**named_record(old), 'command':command}
            record = campaign_record(row, a.output_root.resolve(), batch)
            row['status'] = record['status']
            if (a.output_root.resolve()/bench/'summaries'/f'{batch}.json').exists(): command.append('--resume')
        campaign['cases'].append(row)
    if a.action == 'list':
        for row in campaign['cases']: print(f"{row['benchmark']}: {row['task']} ({row['steps']} steps)")
        return
    if a.dry_run:
        import shlex
        for row in campaign['cases']:
            if row['status'] == 'completed': print('SKIP completed: '+row['run'])
            else: print(shlex.join(row['command']+['--dry-run']))
        return
    if not a.codex_home and any(r['status'] != 'completed' for r in campaign['cases']): p.error('run requires --codex-home')
    if not a.resume:
        folder.mkdir(parents=True,exist_ok=False)
        lock_stack.enter_context(batch_lock(folder/'.lock'))
    campaign['status'] = 'running'
    report(folder,campaign)
    for row in campaign['cases']:
        if row['status'] == 'completed':
            row['audit'] = audit(row['run'])
            print('SKIP completed: '+row['run'],flush=True); report(folder,campaign); continue
        row['status']='running'; row['started_at']=dt.datetime.now(dt.timezone.utc).isoformat(); report(folder,campaign)
        print(f"START {row['benchmark']} / {row['task']}",flush=True)
        try:
            with (folder/(row['benchmark']+'.log')).open('a' if a.resume else 'w') as log:
                result = subprocess.run(row['command'],cwd=WORLD,stdout=log,stderr=subprocess.STDOUT)
            row['status']='launcher_completed' if result.returncode==0 else 'launcher_error'
            row['returncode']=result.returncode
            record = campaign_record(row, a.output_root.resolve(), batch)
            if (Path(row['run'])/'run.json').exists():
                row['status'] = record['status']; row['audit']=audit(row['run'])
            else: row['audit']={'smoke_passed':False,'errors':['No run.json; see campaign launcher log']}
        except KeyboardInterrupt:
            campaign_record(row, a.output_root.resolve(), batch)
            row['status']='interrupted'; campaign['status']='paused'; report(folder,campaign); raise
        except Exception as error:
            row['status']='audit_or_launch_error';row['audit']={'smoke_passed':False,'errors':[str(error)]}
        row['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();report(folder,campaign)
        print(f"END {row['benchmark']} smoke_passed={row['audit']['smoke_passed']} {row['run']}",flush=True)
    campaign['status']='finished'; report(folder,campaign)
    print('Report:',folder/'README.md',flush=True)
    raise SystemExit(0 if all(row['audit']['smoke_passed'] for row in campaign['cases']) else 1)

def main(argv=None):
    from contextlib import ExitStack
    with ExitStack() as stack:
        return _main(argv, stack)


if __name__ == '__main__': main()
