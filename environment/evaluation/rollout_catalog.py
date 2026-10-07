"""Selected task registry and native horizons, read without importing simulators."""
from .task_names import canonical, title, routing_key
import ast
import json
import re
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]
CORE = {'robodojo', 'behavior_1k', 'robocasa', 'robolab'}

def read_json(path):
    return json.loads(path.read_text())

def profile(bench, task):
    if bench == 'volleybots' and task == 'T05': return 'isaac6-scripted-1v1'
    if bench == 'bench2dex': return 'anchored'
    if task == 'T11': return 'a1-feet'
    if task in {'T02', 'T14', 'T16', 'T17', 'T05-single'}: return 'isaac6'
    return 'default'

def catalog():
    suites = read_json(WORLD/'environment/evaluation/suites.json')
    selected = read_json(WORLD/'environment/validation/selected-tasks.json')['tasks']
    # Read only syntax/constants: importing RoboCasa would require MuJoCo and assets.
    casa_source = WORLD/'third_party/benchmarks/robocasa/checkout/robocasa/utils/dataset_registry.py'
    casa = {}
    if casa_source.exists():
        for node in ast.walk(ast.parse(casa_source.read_text())):
            if isinstance(node, ast.keyword) and isinstance(node.value, ast.Call):
                for field in node.value.keywords:
                    if field.arg == 'horizon' and isinstance(field.value, ast.Constant):
                        casa[node.arg] = field.value.value
    stats_path = WORLD/'Assets/behavior_1k/data/2026-challenge-task-instances/metadata/task.jsonl'
    pinned=read_json(WORLD/'environment/evaluation/native-budgets.json')['tasks']
    human = {r['task_name']: r for r in (json.loads(s) for s in stats_path.read_text().splitlines() if s.strip())} if stats_path.exists() else {}
    result = {}
    for entry in selected:
        bench, key = entry['benchmark'], routing_key(entry['benchmark'], entry['case'])
        row = None
        for candidate in suites.get(bench, {}).get('cases', []):
            if candidate['id'] == key or (bench in CORE and candidate['task'] == key):
                row = dict(candidate); break
        horizon, source, kind = None, '', 'native'
        if bench == 'robodojo':
            src = WORLD/'third_party/benchmarks/RoboDojo/task/RoboDojo/tasks'/f'{key}.py'
            match = re.search(r'self\.step_lim\s*=\s*(\d+)', src.read_text()) if src.exists() else None
            horizon = int(match[1]) if match else None
            source = str(src.relative_to(WORLD))+': self.step_lim'
        elif bench == 'behavior_1k':
            horizon = int(human[key]['length'] * 1.5) if key in human else pinned.get(bench,{}).get(key,{}).get('steps')
            source = 'OmniGibson/eval/evaluator.py: int(task human mean length * EVAL_TIMEOUT_MULTIPLIER=1.5)'
        elif bench == 'robocasa':
            horizon = casa.get(key)
            source = str(casa_source.relative_to(WORLD))+': get_task_horizon'
        elif bench in {'robolab', 'ai_cps', 'wheeledlab'}:
            horizon = row['steps']
            source = 'environment/evaluation/suites.json; pinned native task horizon'
            if key.startswith('rw-') or (bench == 'ai_cps' and key == '34'):
                kind = 'RobotWorld-authored'; source = 'versioned RobotWorld task protocol (not an upstream task)'
        elif bench == 'humanoid_soccer':
            # Keep the selected scene/control mode, but restore the upstream 6s / 50Hz budget.
            horizon = 300
            source = 'HumanoidSoccer/exp/mujoco_soccer/cli.py: sim_time=6s, control_dt=0.02s'
            kind = 'upstream duration; selected World scene/controller profile'
        else:
            project = read_json(WORLD/'third_party/benchmarks'/bench/'project.json')
            horizon = project['tasks'][key]['steps']
            source = f'third_party/benchmarks/{bench}/project.json: tasks.{key}.steps'
        result.setdefault(bench, []).append({
            'task': canonical(bench, key), 'task_key': key, 'task_title': title(bench,key),
            'legacy_task_id': key if canonical(bench,key) != key else None, 'native_task': entry.get('native_task', key), 'native_steps': horizon,
            'budget_source': source, 'protocol_kind': kind,
            'code_control_supported': bench not in (CORE - {'behavior_1k'}),
            'runtime_profile': profile(bench, key), 'suite_row': row,
            'seed': 100000000 if bench == 'bench2dex' else 2 if bench == 'humanoid_soccer' else 7,
            'scene_seed_fixed': bench in {'robodojo', 'behavior_1k', 'bench2dex'} or (bench == 'volleybots' and key == 'T05'),
        })
    return result
