"""Boundary tests independent of Isaac: native bounds, actor isolation and episode lifecycle."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from environment.benchmarks.native_project.control import validate_action, specs
from environment.benchmarks.native_project.policy import Agent, EnvironmentExecutionError
from environment.benchmarks.native_project.simulator import Simulator


def test_runtime_profile_preserves_source_and_task_contract(tmp_path,monkeypatch):
    from environment.runtime import native_project_launch as launch
    source=tmp_path/'third_party/benchmarks/example'
    (source/'runtime-profiles').mkdir(parents=True)
    base={'key':'example','commit':'fixed','tasks':{'T01':{'steps':123,'action':'native'}},
          'image':'original','environment':{'KEEP':'1'}}
    (source/'project.json').write_text(json.dumps(base))
    (source/'runtime-profiles/isaac6.json').write_text(json.dumps({
        'image':'experimental','runtime':'Isaac6 experimental; not equivalent',
        'environment':{'EXPERIMENTAL':'1'}}))
    monkeypatch.setattr(launch,'WORLD',tmp_path)
    _,profile=launch.load_project('example','isaac6')
    _,original=launch.load_project('example')
    assert profile['tasks']==original['tasks']==base['tasks']
    assert profile['commit']==original['commit']=='fixed'
    assert original['image']=='original' and profile['image']=='experimental'
    assert profile['environment']=={'KEEP':'1','EXPERIMENTAL':'1'}
    assert json.loads((source/'project.json').read_text())==base


@pytest.mark.parametrize('field',['tasks','commit','repository','key'])
def test_runtime_profile_rejects_task_or_source_replacement(tmp_path,monkeypatch,field):
    from environment.runtime import native_project_launch as launch
    source=tmp_path/'third_party/benchmarks/example'
    (source/'runtime-profiles').mkdir(parents=True)
    (source/'project.json').write_text(json.dumps({'key':'example'}))
    (source/'runtime-profiles/isaac6.json').write_text(json.dumps({
        'image':'experiment','runtime':'experimental',field:'replacement'}))
    monkeypatch.setattr(launch,'WORLD',tmp_path)
    with pytest.raises(ValueError,match='never source identity or tasks'):
        launch.load_project('example','isaac6')


def test_native_bounds_do_not_invent_normalization():
    metadata={'dim':2,'lower':[None,-.008333],'upper':[None,.008333]}
    assert validate_action([5,.001],metadata)==[5,.001]
    for value in ([0,.1],[True,0],[float('nan'),0],[0],[0,float('inf')]):
        with pytest.raises(ValueError):
            validate_action(value,metadata)
    assert [x['name'] for x in specs(metadata,False)]==['observe','apply_action']


class FakeSim:
    def __init__(self):
        self.steps=0
        self.done=False
        self.dt=.01
        self.last_evaluation={'secret_evaluation':123}
        self.policy_images={}
        self.action_metadata={'dim':2,'lower':[None,None],'upper':[None,None]}
        self.actions=[]
    def observation(self):
        return {'control_step':self.steps,'native_policy_terms':{'angle':[.1]}}
    def step(self,action):
        self.actions.append(action)
        self.steps+=1
        self.done=self.steps==3
        return self.observation()


@pytest.mark.parametrize('cameras', [0, 2])
def test_native_packet_publishes_exact_image_window(tmp_path, cameras):
    from environment.runtime.image_history import select_request_images
    from environment.runtime.request_audit import summarize
    sim=FakeSim()
    agent=Agent(sim,tmp_path,None,'unused',1000,60,'test',coding=False)
    payload={'input':[]}
    for i in range(12):
        sim.steps=i*3
        sim.policy_images={str(c):np.full((4,4,3),i,dtype=np.uint8) for c in range(cameras)}
        parts=agent.content()
        content=[{'type':'input_image','image_url':p['imageUrl']} if p['type']=='inputImage'
                 else {'type':'input_text','text':p['text']} for p in parts]
        payload['input'].append({'type':'function_call_output','call_id':str(i),'output':content})
    window=json.loads((tmp_path/'image-window.json').read_text())
    assert [r['env_step'] for r in window['rounds']]==[9,15,21,27,33]
    if cameras:
        outgoing,_=select_request_images(payload,window)
        assert summarize(payload)['input_images']>50
        assert summarize(outgoing)['input_images']==10
        assert outgoing['input'][-1]==payload['input'][-1]
    else:
        assert window['image_sha256']==[]
        assert summarize(payload)['input_images']==0
    assert 'coding_control' not in [t['name'] for t in agent.tool_schema]


def test_batched_native_actions_stop_on_first_terminal(tmp_path):
    sim=FakeSim()
    agent=Agent(sim,tmp_path,None,'unused',100,60,'test')
    result=agent.execute('apply_action',{'note':'test','action':[2,3],'steps':50})
    assert result=={'executed_steps':3,'reason':'native_termination'}
    assert sim.actions==[[2.,3.]]*3
    assert (tmp_path/'events/no-images/environment.jsonl').exists()


def test_environment_failure_after_advancing_aborts_and_records_attempt(tmp_path):
    class BrokenSim(FakeSim):
        def step(self, action):
            self.steps += 1
            raise ValueError('non-finite observation after physics')
    sim = BrokenSim()
    agent = Agent(sim,tmp_path,None,'unused',100,60,'test')
    with pytest.raises(EnvironmentExecutionError):
        agent.execute('apply_action',{'note':'test','action':[0,0],'steps':50})
    assert sim.steps == 1
    log = (tmp_path/'events/no-images/environment.jsonl').read_text()
    assert 'environment_step_error' in log
    assert 'non-finite observation after physics' in log


def test_feedback_program_receives_current_observation_each_step(tmp_path):
    sim=FakeSim()
    agent=Agent(sim,tmp_path,None,'unused',100,60,'test')
    result=agent.execute('coding_control',{'note':'feedback','max_steps':20,
        'code':'def control(obs, memory):\n    return {"action": [obs["control_step"], 0]}\n'})
    assert result['executed_steps']==3
    assert sim.actions==[[0.,0.],[1.,0.],[2.,0.]]


def test_content_history_and_review_camera_exclusion(tmp_path):
    sim=FakeSim()
    sim.camera=SimpleNamespace(image=np.zeros((3,3,3),dtype=np.uint8))
    sim.policy_images={'front':np.zeros((2,2,3),dtype=np.uint8)}
    agent=Agent(sim,tmp_path,None,'unused',100,60,'test')
    for i in range(10):
        sim.steps=i*5
        parts=agent.content()
    obs=[p['text'] for p in parts if p['type']=='inputText' and 'observation=' in p['text']]
    assert [int(x.split('observation=')[1].split('\n')[0]) for x in obs]==[1,3,5,7,9]
    assert len([p for p in parts if p['type']=='inputImage'])==5
    assert 'secret_evaluation' not in json.dumps(parts)
    assert all('Native sensor: front'==p['text'] for p in parts if p['type']=='inputText' and 'Native sensor:' in p['text'])


def test_explicit_actor_groups_never_export_critic():
    class Tensor:
        def __getitem__(self,i):return self
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return [1,2]
    sim=Simulator.__new__(Simulator)
    sim.project=SimpleNamespace(policy_groups=['stack_policy','none_stack_policy'])
    sim._observe({'stack_policy':Tensor(),'none_stack_policy':Tensor(),'critic':'private'})
    assert set(sim.policy_terms)=={'stack_policy','none_stack_policy'}
