import numpy as np
import pytest
from environment.benchmarks.robolab.control import specs,action_for
STATE={'arm_joint_pos':[0]*7,'ee_pos':[.3,.1,.5],'ee_quat':[1,0,0,0]}

def test_native_profiles_and_simultaneous_gripper():
 for mode,tool,targets,expected in [
  ('joint_position','move_robot',{'joint_positions':[.1]*7,'gripper_close':1},[.1]*7+[1]),
  ('absolute_ik','move_robot',{'position':[.4,.2,.6],'quaternion_wxyz':[2,0,0,0],'gripper_close':1},[.4,.2,.6,1,0,0,0,1]),
  ('relative_ik','move_robot',{'delta_pose':[.1]*6,'gripper_close':0},[.1]*6+[0])]:
  a,n,g=action_for(mode,tool,{'note':'test','steps':4,'targets':targets},STATE,0)
  np.testing.assert_allclose(a,expected);assert n==4 and g==targets['gripper_close']
  assert {x['name'] for x in specs(mode)}=={'move_robot','set_gripper','move_joints' if mode=='joint_position' else 'move_eef'}

def test_hold_and_gripper_persistence():
 a,_,_=action_for('absolute_ik','set_gripper',{'note':'close','steps':2,'targets':{'gripper_close':1}},STATE,0)
 np.testing.assert_allclose(a,[.3,.1,.5,1,0,0,0,1])
 a,_,g=action_for('relative_ik','move_eef',{'note':'delta','steps':1,'targets':{'delta_pose':[0]*6}},STATE,1)
 assert a[-1]==g==1

@pytest.mark.parametrize('targets',[{'base_vx':1},{'quaternion_wxyz':[0]*4},{'position':[0,float('nan'),0]},{'gripper_close':True},{'position':[1,2]},{'position':['1',2,3]},{}])
def test_reject_invalid(targets):
 with pytest.raises(ValueError):action_for('absolute_ik','move_robot',{'note':'bad','steps':1,'targets':targets},STATE,0)
