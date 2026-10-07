import numpy as np
import pytest
from environment.benchmarks.robocasa.control import action_for,tool_specs,state_of,STATE_KEYS

def request(**targets):return {'note':'test','steps':3,'targets':targets}

def test_native_layout_and_gripper_persistence():
    a,n,g=action_for('move_robot',request(eef_delta=[.1,.2,.3,-.4,0,.6],base_motion=[.2,-.2,.1],torso=.4,gripper_close=1,control_mode=1),0)
    np.testing.assert_allclose(a,[.1,.2,.3,-.4,0,.6,1,.2,-.2,.1,.4,1]);assert n==3 and g==1
    b,_,_=action_for('move_eef',request(eef_delta=[0]*6),g)
    assert b[6]==1 and b[11]==0 and np.count_nonzero(b)==1

def test_base_following_and_torso():
    a,_,_=action_for('move_base',request(base_motion=[0,.1,0]),0);assert a[8]==pytest.approx(.1) and a[11]==1
    a,_,_=action_for('move_torso',request(torso=.1),0);assert a[10]==pytest.approx(.1) and a[11]==1

@pytest.mark.parametrize('name,args',[
 ('move_eef',request(eef_delta=[0]*5)),('move_base',request(base_motion=[2,0,0])),
 ('move_eef',request(eef_delta=[float('nan')]*6)),('set_gripper',request(gripper_close=.5)),
 ('move_torso',request(torso=True)),('move_base',request(base_motion=[0]*3,torso=0)),
 ('give_up',request(gripper_close=0)),('move_robot',request()),
 ('set_gripper',{**request(gripper_close=0),'steps':31})])
def test_rejects_invalid_without_clipping(name,args):
    with pytest.raises((ValueError,TypeError)):action_for(name,args,0)

def test_observation_allowlist():
    obs={k:np.zeros(3) for k in STATE_KEYS};obs.update(hidden_object_pose=[1,2,3],success=True)
    assert set(state_of(obs))==set(STATE_KEYS)
    assert {s['name'] for s in tool_specs()}=={'move_robot','move_eef','move_base','move_torso','set_gripper'}
