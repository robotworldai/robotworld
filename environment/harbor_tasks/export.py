"""Export executable adapters, explicitly inventory unported benchmark tasks.

A source benchmark being registered is not evidence that its Harbor adapter works.
The inventory keeps implementation readiness separate from runtime validation.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OTHER_SERVICES = {'ai_cps','wheeledlab','robolab','behavior_1k','robodojo','humanoid_soccer'}
NATIVE_PROJECTS = {'bench2dex','digit','flamingo','go2_push','omnidrones','omniisaacgymenvs','reflexbench','robot_lab','steadytray','ttrl','volleybots','wheel_legged','wheeled_quadruped'}


SEMANTIC_TASK_NAMES = {
    'ai_cps': {'22':'catch-moving-ball','23':'ball-tray-balance','24':'peg-in-hole'},
    'bench2dex': {'41':'bimanual-piano-melody','42':'soup-serving',
                 '43':'screwdriver-box-and-hammer','44':'fridge-wine-interhand-pour',
                 '45':'faucet-cup-water-fill','46':'microwave-bowl-loading',
                 '47':'gaming-desk-setup','48':'jigsaw-puzzle-assembly',
                 '49':'wine-glass-plate-balance'},
}

RUNTIME_VALIDATION = {
    ('robodojo','deposit_coin'): ('harbor_macro_diagnostic_verified', 'robodojo-deposit-macro-probe02: 1/300 native steps; three videos each 2 frames; independent verifier diagnostic/no score'),
    ('robocasa','CloseDrawer'): ('prior_close_drawer_pilot', 'close-drawer source-Codex and Claude Code pilots'),
    ('digit','digit_walk_hand_tracking'): ('harbor_partial_diagnostic_verified', 'digit-native-probe01: 10/700 steps; 11 frames; no score'),
    ('reflexbench','cup_ball_catching'): ('harbor_native_terminal_verified', 'reflex-native-budget-probe02: native failure at 30/100; 31 frames; verifier reward 0'),
    ('robolab','RubiksCubeLeftOfBowlTask'): ('harbor_partial_diagnostic_verified', 'robolab-probe01: 10/450 steps; 11 frames; no score'),
    ('humanoid_soccer','play-soccer'): ('harbor_native_terminal_verified', 'soccer-probe01: 1000/1000 steps; 1001 frames; verifier reward 0'),
}


def inventory():
    from environment.evaluation.rollout_catalog import catalog
    rows = []
    for benchmark, tasks in catalog().items():
        for task in tasks:
            ready = (benchmark == 'robocasa' or benchmark in NATIVE_PROJECTS or benchmark in OTHER_SERVICES) and type(task['native_steps']) is int and task['native_steps'] > 0
            from environment.evaluation.world_success.profiles import get_profile
            profile = get_profile(benchmark, task['task'])
            scoring = 'world-state-v1' if profile else 'native'
            if benchmark == 'ai_cps' and str(task['task']) == '24':
                scoring = 'peg-xy-z-window-v2'
            evaluation_steps = profile['steps'] if profile else task['native_steps']
            if benchmark == 'behavior_1k' and evaluation_steps is not None:
                evaluation_steps = min(evaluation_steps,2000)
            budget_override_reason = None
            if benchmark == 'robocasa' and task['task'] == 'CountertopCleanup':
                evaluation_steps = 600
                ready = True
                budget_override_reason = 'Existing World handoff budget from scripts/run_robocasa.sh: CountertopCleanup=600; no verified official horizon.'
            if benchmark == 'humanoid_soccer' and task['suite_row'].get('fixed_protocol'):
                evaluation_steps = task['suite_row']['steps']
                budget_override_reason = 'Selected World fixed protocol: 20 seconds at 50 Hz; upstream CLI default is 6 seconds.'
            if benchmark == 'behavior_1k':
                budget_override_reason = 'Selected evaluation cap: min(original task budget, 2000).'
            rows.append({
                'benchmark': benchmark, 'task': task['task'], 'task_key': task['task_key'],
                'name': slug(benchmark, task['task']), 'title': task['task_title'],
                'budget_override_reason':budget_override_reason, 'native_steps': task['native_steps'], 'horizon':evaluation_steps, 'scoring_profile':scoring, 'budget_source': task['budget_source'],
                'runtime_profile': task['runtime_profile'], 'seed': task['suite_row']['seed'] if benchmark == 'robodojo' else task['seed'],
                'scene_selection': {k:v for k,v in (task['suite_row'] or {}).items() if k in ('seed','layout','instance_index','instance_id')},
                'scene_seed_fixed': task['scene_seed_fixed'],
                'harbor_status': 'adapter_ready' if ready else 'legacy_pending',
                'runtime_validation': RUNTIME_VALIDATION.get((benchmark,task['task']), ('not_verified',None))[0],
                'validation_evidence': RUNTIME_VALIDATION.get((benchmark,task['task']), ('not_verified',None))[1],
                'reason': ('Episode adapter implemented; runtime validation tracked separately.' if ready else
                           'Native horizon/task identity unresolved.' if benchmark == 'robocasa' else
                           'Legacy runner exists; independent Episode service has not been exported.'),
                'assets_validation':'not_checked_at_export',
                'code_control': False if ready else None,
                'legacy_code_control_supported': task['code_control_supported'],
            })
    return rows


def slug(benchmark, task):
    task = SEMANTIC_TASK_NAMES.get(benchmark, {}).get(str(task), task)
    return re.sub(r'[^a-z0-9]+', '-', re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '-', task).lower()).strip('-')


def select(rows, benches=None, tasks=None):
    benches = set(benches or [])
    tasks = set(tasks or [])
    if benches - {r['benchmark'] for r in rows}:
        raise ValueError('Unknown benchmark: ' + ', '.join(sorted(benches - {r['benchmark'] for r in rows})))
    subset = [r for r in rows if not benches or r['benchmark'] in benches]
    if tasks:
        unknown = tasks - {key for r in subset for key in (r['task'], r['name'])}
        if unknown:
            raise ValueError('Unknown task selection: ' + ', '.join(sorted(unknown)))
        subset = [r for r in subset if r['task'] in tasks or r['name'] in tasks]
        unsupported = [r['name'] for r in subset if r['harbor_status'] != 'adapter_ready']
        if unsupported:
            raise ValueError('No executable Harbor adapter for: ' + ', '.join(unsupported))
    ready = [r for r in subset if r['harbor_status'] == 'adapter_ready']
    if not ready:
        raise ValueError('No executable Harbor adapters in this selection; use --list to inspect legacy_pending tasks.')
    return ready


def native_service(row, seed, template):
    """Construct the same pinned runtime configuration as the legacy native launcher."""
    from environment.runtime.native_project_launch import load_project
    _, config = load_project(row['benchmark'], row['runtime_profile'])
    root = '/opt/robotworld'
    source = root + '/third_party/benchmarks/' + row['benchmark']
    paths = [x if x.startswith('/') else root + '/' + x for x in config.get('runtime_python_paths', [])]
    if config.get('engine', 'isaaclab') != 'custom':
        paths += ['/opt/isaaclab22/source/' + x for x in ('isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl')]
    paths += [root] + [source + '/checkout/' + x for x in config.get('python_paths', [])]
    entry = config.get('entrypoint', ['python'])
    if isinstance(entry, str):
        entry = [entry]
    env = {'NVIDIA_DRIVER_CAPABILITIES':'all','ACCEPT_EULA':'Y','OMNI_KIT_ACCEPT_EULA':'YES',
           'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(paths),
           **config.get('environment',{}),
           'WORLD_CONTROL_STEP_LIMIT':'${WORLD_CONTROL_STEP_LIMIT:-}',
           'WORLD_EPISODE_TOKEN':'${WORLD_EPISODE_TOKEN:?Set a random per-job token}'}
    if row['benchmark'] == 'volleybots' and row['runtime_profile'] == 'isaac6-scripted-1v1':
        env.update(WORLD_VOLLEY_PLAYER='1', WORLD_VOLLEY_CODING_CONTROL='0')
    return {'image':config['image'], 'gpus':'all', 'shm_size':'4gb',
            'entrypoint':entry, 'working_dir':root, 'environment':env,
            'volumes':['${WORLD_ROOT}:/opt/robotworld:ro', '${WORLD_ASSETS:?Set Assets path}:/opt/robotworld/Assets:ro'],
            'command':['-m','environment.integrations.native_project_eval', '--project',row['benchmark'],
                       '--task',row['task_key'],'--mode','service','--model','unused', '--output','/runs',
                       '--runtime-profile',row['runtime_profile'],'--scoring-profile',row['scoring_profile'],
                       '--steps',str(row['horizon']),'--seed',str(seed),'--disable-coding-control'],
            'healthcheck':{'test':['CMD-SHELL', 'test -f /runs/episode.json'],
                           'interval':'5s','timeout':'3s','retries':120,'start_period':'30s'}}


def export_task(row, directory, template, seed=None):
    import yaml  # Harbor itself depends on PyYAML; no simulator import required.
    shutil.copytree(template, directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # A portable task bundles one common implementation; task templates must not
    # introduce their own scoring semantics or preserve stale grader copies.
    for source, target in (('grading.py', 'shared_grade.py'), ('grade_entrypoint.py', 'grade.py')):
        shutil.copyfile(ROOT/'environment/harbor_tasks'/source, directory/'tests'/target)
    compose_path = directory / 'environment/docker-compose.yaml'
    compose = yaml.safe_load(compose_path.read_text())
    if seed is not None and row['scene_seed_fixed'] and seed != row['seed']:
        raise ValueError('This task has a fixed scene/instance protocol; --seed cannot override '+row['name'])
    selected_seed = row['seed'] if seed is None else seed
    if row['benchmark'] in NATIVE_PROJECTS:
        compose['services']['simulator'] = native_service(row, selected_seed, compose['services']['simulator'])
    elif row['benchmark'] in OTHER_SERVICES:
        from .services import service
        compose['services']['simulator'] = service(row, selected_seed)
    else:
        command = compose['services']['simulator']['command']
        command[command.index('--task') + 1] = row['task']
        command[command.index('--seed') + 1] = str(selected_seed)
        if '--seed-protocol' not in command:
            command += ['--seed-protocol', 'legacy-task-index']
        if row['task'] == 'CountertopCleanup':
            command += ['--horizon',str(row['horizon'])]
    compose['services']['main'].setdefault('environment',{})['PYTHONPATH'] = '/opt/robotworld-client'
    compose['services']['main']['environment']['WORLD_AGENT_INTERFACE'] = '${WORLD_AGENT_INTERFACE:-native-feedback-v1}'
    compose['services']['simulator'].setdefault('environment',{})['WORLD_CODE_CONTROL'] = '${WORLD_CODE_CONTROL:-0}'
    compose['services']['main']['command'] = ['python', '/opt/robotworld-client/environment/clients/harbor_entry.py', 'initialize']
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    for relative in ('environment/clients/episode.py','environment/bridges/mcp.py','environment/clients/harbor_entry.py'):
        target = directory/'environment/client'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    dockerfile=directory/'environment/Dockerfile'
    dockertext=dockerfile.read_text()
    if 'COPY client/' not in dockertext:
        dockerfile.write_text(dockertext+'\nCOPY client/ /opt/robotworld-client/\nENV PYTHONPATH=/opt/robotworld-client\n')
    toml_path = directory / 'task.toml'
    text = toml_path.read_text().replace('args = ["-m", "environment.bridges.mcp"]', 'args = ["/opt/robotworld-client/environment/clients/harbor_entry.py", "mcp"]')
    # Replace only metadata.name, not similarly named server configuration keys.
    start = text.index('[metadata]')
    end = text.find('\n[', start + 1)
    if end == -1:
        end = len(text)
    section = re.sub(r'(?m)^name\s*=\s*"[^"]+"', 'name = ' + json.dumps(row['name']), text[start:end], count=1)
    section = re.sub(r'(?m)^tags\s*=.*$', 'tags = '+json.dumps([row['benchmark'],row['scoring_profile'],'mcp']), section)
    toml_path.write_text(text[:start] + section + text[end:])
    (directory / 'instruction.md').write_text(
        f"Complete the {row['benchmark']} {row['task']} task. Read `describe` for the exact scene instruction, "
        "action definitions, units, reference frames, and native controller assistance. Use `observe` "
        "for the allowed images and proprioception. Use exposed native robot tools and their returned observations. If `execute` is exposed instead, send ordered commands through it and query `observe` afterwards; use `status` "
        "and `receipt` to inspect execution.\n\n"
        f"The configured {row['scoring_profile']} budget is {row['horizon']} native control steps. A command's `steps` repeats its native "
        "control input only where describe explicitly defines repetition. Some benchmark commands are planned macros, "
        "whose physical cost is measured from execution feedback. An action chunk can contain multiple commands. "
        "Execution stops on native success, native termination, or the step budget.\n\n"
        "Continue until the episode is terminal. Textual claims cannot complete a task. No reset, "
        "privileged state, simulator filesystem, direct simulator handle is exposed. Feedback code is available only when describe lists coding_control. "
        "Only the configured simulator-side success checker determines reward.\n"
    )
    from .instructions import control_brief
    instruction_path = directory / 'instruction.md'
    instruction_path.write_text('Before acting, read `/workspace/ROBOTWORLD.md` for the full public task and robot specification, or call the `describe` tool. This file is generated from the live environment and is identical across harnesses.\n\n' + instruction_path.read_text() + '\n' + control_brief(row['benchmark']))
    record = {**row, 'seed': selected_seed, 'template': str(template.relative_to(ROOT)) if template.is_relative_to(ROOT) else template.name,
              'schema_version': 'robotworld.harbor-task.v1'}
    (directory / 'tests/task.json').write_text(json.dumps({k:row[k] for k in ('benchmark','task','horizon','scoring_profile')},indent=2)+'\n')
    (directory / 'robotworld-task.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    (directory / 'README.md').write_text(
        f"# {row['name']}\n\nGenerated standard Harbor task; see `robotworld-task.json` for provenance.\n\n"
        f"Evaluation horizon: {row['horizon']} control steps; configured seed: {selected_seed}. "
        f"Fixed scene/instance selectors: `{json.dumps(row['scene_selection'])}`. "
        "Repeating this package repeats the fixed scene, not different environment seeds.\n\n"
        "Start through `harbor run --path <this-directory> --agent <agent>`. Configure the "
        "RobotWorld assets, images, Episode token and agent dependencies as documented in the repository. "
        "Generated packages share the checked-in simulator service and trusted verifier.\n\n"
        f"Readiness: {row['harbor_status']}; runtime validation: {row['runtime_validation']}. "
        "Export/schema validation does not prove simulation initialization or task success.\n"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='Print the full migration inventory without writing task directories')
    p.add_argument('--bench', action='append', help='Select benchmark; repeat or comma-separate values')
    p.add_argument('--tasks', help='Comma-separated task names or exported names')
    p.add_argument('--output', type=Path, default=ROOT/'tasks')
    p.add_argument('--seed', type=int, help='Fixed scene seed; default is each task registry seed')
    p.add_argument('--template', type=Path, default=ROOT/'tasks/close-drawer')
    args = p.parse_args(argv)
    rows = inventory()
    if args.list:
        print(json.dumps({'benchmarks':len({r['benchmark'] for r in rows}), 'tasks':rows}, indent=2, ensure_ascii=False))
        return
    try:
        chosen = select(rows, [b for value in args.bench or [] for b in value.split(',')], args.tasks.split(',') if args.tasks else None)
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f'Refusing to overwrite existing output: {output}; choose a new export directory.')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.harbor-export-', dir=output.parent) as temporary:
            staged = Path(temporary)/'tasks'
            staged.mkdir()
            for row in chosen:
                export_task(row, staged/row['name'], args.template.resolve(), args.seed)
            report = {'schema_version':'robotworld.harbor-catalog.v1', 'task_count':len(rows),
                      'benchmark_count':len({r['benchmark'] for r in rows}), 'exported':[r['name'] for r in chosen],
                      'tasks':rows}
            (staged/'catalog.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            staged.rename(output)
        print(json.dumps({'output':str(output), 'exported':len(chosen), 'legacy_pending':sum(r['harbor_status']!='adapter_ready' for r in rows)}, indent=2))
    except ValueError as error:
        p.error(str(error))

if __name__ == '__main__':
    main()
