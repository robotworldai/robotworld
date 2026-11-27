from types import SimpleNamespace
import queue
import threading
import numpy as np
import pytest
from environment.service.humanoid_soccer import HumanoidSoccerAdapter


def adapter():
    obj=HumanoidSoccerAdapter.__new__(HumanoidSoccerAdapter)
    obj.mode='direct'
    obj.snapshot={'names':['joint'],'limits':[[-1,1]],'state':{'joint_positions':{'joint':0.2}}}
    obj.commands=queue.Queue()
    return obj


def test_validation_cannot_mutate_snapshot_or_execute():
    obj=adapter()
    good={'tool':'move_joints','arguments':{'note':'hold','steps':2,'joint_positions':{'joint':0.3}}}
    bad={'tool':'move_joints','arguments':{'note':'bad','steps':2,'joint_positions':{'joint':2}}}
    with pytest.raises(ValueError):obj.prepare([good,bad])
    assert obj.commands.empty()
    assert obj.snapshot['state']['joint_positions']['joint']==0.2


def test_one_step_submits_one_command_and_waits_for_native_result():
    obj=adapter()
    obj._receive=lambda:{'success':False,'terminated':False,'truncated':False}
    command={'tool':'move_joints','arguments':{'note':'hold','steps':2,'joint_positions':{'joint':0.3}}}
    action,count=obj.prepare([command])[0]
    assert count==2
    assert obj.step(action)=={'success':False,'terminated':False,'truncated':False}
    assert obj.commands.qsize()==1
    assert obj.commands.get()['targets'].tolist()==[0.3]


def test_native_final_success_propagates_without_invented_checker():
    obj=adapter()
    obj._receive=lambda:{'success':True,'terminated':True,'truncated':False}
    action,_=obj.prepare([{'tool':'move_joints','arguments':{'note':'hold','steps':1,'joint_positions':{'joint':0.3}}}])[0]
    assert obj.step(action)['success'] is True
