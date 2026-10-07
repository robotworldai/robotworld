import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from environment.benchmarks.wheeledlab.control import specs,validate_action
from environment.benchmarks.wheeledlab.simulator import Simulator
from environment.benchmarks.wheeledlab.policy import Agent
from environment.evaluation.runner import selected,run_command

@pytest.mark.parametrize('bad',[[True,0],[0,float('nan')],[0,1.001],[0],{'speed':1},[float('inf'),0]])
def test_reject_unsafe_actions(bad):
    with pytest.raises(ValueError):validate_action(bad)

def test_mask_enforced_before_program_execution(tmp_path):
    sim=SimpleNamespace(steps=0)
    agent=Agent(sim,tmp_path,None,'test',10,10,coding=False)
    assert [t['name'] for t in specs(False)]==['observe','drive']
    with pytest.raises(ValueError,match='unavailable'):
        agent.execute('coding_control',{'note':'','code':'','max_steps':1})

class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value

def test_native_observation_only_and_visual_decode(tmp_path):
    env=SimpleNamespace(step_dt=.2,observation_manager=SimpleNamespace(active_terms={'policy':['camera','base_lin_vel']},group_obs_term_dim={'policy':[(3200,),(3,)]}),max_episode_length=50)
    sim=Simulator(env,'visual',tmp_path)
    sim._observe({'policy':[Tensor([-1]*1600+[1]*1600+[.1,.2,.3])]})
    assert sim.policy_image.shape==(40,80)
    assert sim.policy_image[0,0]==0 and sim.policy_image[-1,-1]==255
    assert sim.observation()['native_policy_terms']['base_lin_vel']==[.1,.2,.3]
    sim.last_evaluation={'reward':100,'native_termination_terms':{'out_range':False}}
    assert 'reward' not in str(sim.observation())
    assert sim.result()['success'] is None

def test_native_goal_does_not_hide_simultaneous_failure(tmp_path):
    sim=Simulator(SimpleNamespace(step_dt=.1,max_episode_length=200),'elevation',tmp_path);sim.done=True
    sim.last_evaluation={'native_termination_terms':{'at_goal':True,'rollover':True}}
    assert sim.result()['success'] is False
    sim.last_evaluation['native_termination_terms']['rollover']=False
    assert sim.result()['success'] is True

def test_suite_native_budgets_and_override_guard():
    rows=selected('wheeledlab',None)
    assert [r['steps'] for r in rows]==[250,250,200,50]
    a=argparse.Namespace(steps=None,model='model-x',seed=7,timeout=120)
    cmd=run_command('wheeledlab',rows[-1],a,Path('/out'),Path('/auth'))
    assert cmd[cmd.index('--steps')+1]=='50'
    a.steps=51
    with pytest.raises(ValueError,match='horizon'):run_command('wheeledlab',rows[-1],a,Path('/out'),Path('/auth'))
