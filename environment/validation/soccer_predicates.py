"""Exercise the pinned upstream goal function without importing the simulator."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

WORLD = Path(__file__).resolve().parents[2]


def audit():
    path = WORLD / 'third_party/benchmarks/humanoid_soccer/checkout/exp/mujoco_soccer/metrics.py'
    source = path.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'goal_crossed')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    goal_crossed = namespace['goal_crossed']
    cases = [
        ('forward_center', [-1., 0., 0.1], [1., 0., 0.1], True),
        ('reverse_center', [1., 0., 0.1], [-1., 0., 0.1], False),
        ('still_before_plane', [-2., 0., 0.1], [-1., 0., 0.1], False),
        ('outside_width', [-1., 1.001, 0.1], [1., 1.001, 0.1], False),
        ('exact_width_boundary', [-1., 1., 0.1], [1., 1., 0.1], True),
        ('very_high_ball', [-1., 0., 10.], [1., 0., 10.], True),
        ('stationary_on_plane', [0., 0., 0.1], [0., 0., 0.1], True),
        ('cross_at_center_end_outside', [-1., -2., 0.1], [1., 2., 0.1], True),
    ]
    rows = []
    for name, prev, curr, expected in cases:
        actual = bool(goal_crossed(np.array(prev), np.array(curr), np.zeros(3), np.array([1., 0.]), 2.))
        rows.append({'case': name, 'expected_native_behavior': expected, 'actual': actual, 'passed': actual == expected})
    return {'source': str(path), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'passed': all(row['passed'] for row in rows), 'tests': rows,
            'scope': 'Exact upstream function on synthetic positions; not physical goal-crossing certification.',
            'limitations': ['Only XY goal plane and width are tested by upstream.',
                            'No ball height/crossbar test.',
                            'Stationary points exactly on the goal plane count as crossed.']}


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
