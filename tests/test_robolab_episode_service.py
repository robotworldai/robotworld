"""Protocol semantics that do not need Isaac: preserve native frozen results."""
from types import SimpleNamespace
import numpy as np
import pytest
from environment.service.robolab import RoboLabAdapter, native_outcome


def test_initial_termination_artifact_is_not_success():
    env = SimpleNamespace(all_terminated=False, get_env_results=lambda: [{'env_id': 0, 'success': None}])
    assert native_outcome(env) == {'success':False,'terminated':False,'truncated':False}


@pytest.mark.parametrize('success', [True, False])
def test_native_frozen_result_is_authoritative(success):
    env = SimpleNamespace(all_terminated=True, get_env_results=lambda: [{'env_id': 0, 'success': success}])
    result = native_outcome(env)
    assert result['success'] is success
    assert result['terminated'] is success
    assert result['truncated'] is not success


def test_missing_terminal_result_is_infrastructure_error():
    env = SimpleNamespace(all_terminated=True, get_env_results=lambda: [{'env_id': 0, 'success': None}])
    with pytest.raises(RuntimeError):
        native_outcome(env)


def adapter():
    obj = RoboLabAdapter.__new__(RoboLabAdapter)
    obj.mode, obj.gripper = 'joint_position', 0
    obj.state = lambda: {'arm_joint_pos':[0.0]*7}
    return obj


def test_whole_chunk_rejected_without_changing_gripper():
    obj = adapter()
    commands = [
        {'tool':'set_gripper','arguments':{'note':'close','steps':1,'targets':{'gripper_close':1}}},
        {'tool':'move_joints','arguments':{'note':'bad','steps':31,'targets':{'joint_positions':[0.0]*7}}}]
    with pytest.raises(ValueError):obj.prepare(commands)
    assert obj.gripper == 0


def test_prepared_command_defers_measured_hold_until_execution():
    obj = adapter()
    commands = [{'tool':'set_gripper','arguments':{'note':'open','steps':2,'targets':{'gripper_close':0}}}]
    prepared = obj.prepare(commands)
    assert prepared[0][1] == 2
    assert prepared[0][0]['resolved'] is None
    commands[0]['arguments']['targets']['gripper_close'] = 1
    assert prepared[0][0]['command']['arguments']['targets']['gripper_close'] == 0
