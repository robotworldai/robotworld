import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from environment.benchmarks.robodojo import eef_executor as module
from environment.benchmarks.robodojo.eef_executor import EefExecutor
from environment.benchmarks.robodojo.spec import RoboDojoActionSpec
from environment.benchmarks.robodojo.types import Observation
from environment.robots.arx_x5 import pose


def spec():
    labels = tuple(x for arm in ('left', 'right')
                   for x in [*(f'{arm}_joint{i}' for i in range(1, 7)), f'{arm}_gripper'])
    return RoboDojoActionSpec(labels, np.array([-3.] * 6 + [0.] + [-3.] * 6 + [0.]),
                             np.array([3.] * 6 + [1.] + [3.] * 6 + [1.]), 25, '',
                             tuple([.05] * 6 + [.25] + [.05] * 6 + [.25]))


def observation():
    state = {}
    for arm, x in [('left', -.2), ('right', .2)]:
        state[f'{arm}_arm_joint_state'] = np.zeros(6, dtype=np.float32)
        state[f'{arm}_ee_joint_state'] = np.array([1.], dtype=np.float32)
        state[f'{arm}_ee_pose'] = pose.values_to_pose(dict(x=x, y=-.1, z=.9,
                                               pitch_deg=0., roll_deg=0., yaw_deg=0.))
    return Observation({}, state, 'smoke')


def plan(*, arm, target_pose):
    # Known non-straight path catches accidental endpoint-only resampling.
    scale = 1. if arm == 'left' else 2.
    return {'status': 'Success', 'position': scale * np.array([
        [0.] * 6, [.04, -.04, .08, 0, .02, 0], [.10, -.02, .06, 0, .03, 0]])}


@pytest.fixture
def reference():
    source = Path(__file__).resolve().parents[3] / 'RoboProbe/policy/RoboDojo_Agent_L3_Inspect_EEF/policy.py'
    if not source.exists():
        pytest.skip('Read-only RoboProbe reference checkout not present')
    tree = ast.parse(source.read_text())
    original = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'EefAgentPolicy')
    names = {'_handle_motion', '_parse_targets', '_plan_arm', '_arm_steps', '_gripper_steps',
             '_resample_path', '_held_action_data', '_accepted_text', '_system_message'}
    cls = ast.ClassDef(name='Reference', bases=[ast.Name(id='EefExecutor', ctx=ast.Load())],
                       keywords=[], body=[x for x in original.body if getattr(x, 'name', None) in names],
                       decorator_list=[])
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['Reference'](spec(), plan)


def test_system_prompt_matches_roboprobe(reference):
    actual = EefExecutor(spec(), plan)
    actual._max_llm_calls = reference._max_llm_calls = 64
    assert actual._system_message() == reference._system_message()


def test_tool_matches_roboprobe():
    source = Path(__file__).resolve().parents[3] / 'RoboProbe/policy/RoboDojo_Agent_L3_Inspect_EEF/policy.py'
    if not source.exists():
        pytest.skip('Reference checkout absent')
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'EefAgentPolicy')
    method = next(n for n in cls.body if getattr(n, 'name', None) == '_build_tools')
    # Only replace inherited terminal-tool collection; compare motion verbatim.
    method.body[0].value = ast.List(elts=[], ctx=ast.Load())
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), 'exec'), namespace)
    executor = EefExecutor(spec(), plan)
    assert executor._build_tools(executor.action_spec) == namespace['_build_tools'](executor, executor.action_spec)


@pytest.mark.parametrize('targets', [
    {'left_z': .91}, {'left_gripper': 0}, {'left_z': .91, 'left_gripper': 0},
    {'left_z': .91, 'right_z': .92, 'right_gripper': .3},
    {'left_z': 99}, {'bad_axis': 0}, {'left_x': float('nan')},
])
def test_reference_actions(reference, targets):
    actual = EefExecutor(spec(), plan).execute_plan({'targets': targets, 'note': 'test'}, observation())
    expected = reference.execute_plan({'targets': targets, 'note': 'test'}, observation())
    assert actual.tool_result == expected.tool_result
    assert actual.repairable == expected.repairable
    if expected.chunk is None:
        assert actual.chunk is None
        return
    assert actual.chunk.meta == expected.chunk.meta
    assert len(actual.chunk.actions) == len(expected.chunk.actions)
    for a, b in zip(actual.chunk.actions, expected.chunk.actions, strict=True):
        for key in a.data:
            np.testing.assert_array_equal(a.data[key], b.data[key])


def test_gripper_only_after_arrival_and_persists():
    executor = EefExecutor(spec(), plan)
    obs = observation()
    result = executor.execute_plan({'targets': {'left_z': .91, 'left_gripper': 0}}, obs)
    arm_steps = result.chunk.meta['trace']['arm_waypoints']
    assert arm_steps > 0
    assert all(a.data['left_ee_joint_state'][0] == 1 for a in result.chunk.actions[:arm_steps])
    assert all(a.data['left_ee_joint_state'][0] == 0 for a in result.chunk.actions[arm_steps:])
    next_result = executor.execute_plan({'targets': {'left_z': .92}}, obs)
    assert all(a.data['left_ee_joint_state'][0] == 0 for a in next_result.chunk.actions)


def test_unreachable_other_arm_does_not_execute():
    def failing(**kwargs):
        return {'status': 'Fail'} if kwargs['arm'] == 'right' else plan(**kwargs)
    result = EefExecutor(spec(), failing).execute_plan(
        {'targets': {'left_z': .91, 'right_z': .91}}, observation())
    assert result.chunk is None
    assert 'Neither arm moved' in result.tool_result


def test_pose_roundtrip():
    values = dict(x=.12, y=-.2, z=.9, pitch_deg=12., roll_deg=-15., yaw_deg=40.)
    restored = pose.pose_to_values(pose.values_to_pose(values))
    np.testing.assert_allclose(list(restored.values()), list(values.values()), atol=1e-10)


def test_stationary_pose_plus_gripper_has_nonempty_arm_phase():
    def stationary(**kwargs):
        return {'status': 'Success', 'position': np.zeros((1, 6))}
    result = EefExecutor(spec(), stationary).execute_plan(
        {'targets': {'left_z': .9, 'left_gripper': 0}}, observation())
    assert result.chunk.meta['trace']['arm_waypoints'] == 1
    assert result.chunk.actions[0].data['left_ee_joint_state'][0] == 1
    assert result.chunk.actions[-1].data['left_ee_joint_state'][0] == 0
