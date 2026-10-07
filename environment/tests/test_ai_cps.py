import math
from types import SimpleNamespace
import pytest
from environment.benchmarks.ai_cps.control import ContactRecovery,validate_action,specs
from environment.evaluation.runner import SUITES,selected,run_command

def test_ai_cps_packet_image_window(tmp_path):
    import json
    import numpy as np
    from environment.benchmarks.ai_cps.policy import Agent
    from environment.runtime.image_history import select_request_images
    from environment.runtime.request_audit import summarize
    sim=SimpleNamespace(steps=0,camera=SimpleNamespace(image=np.zeros((4,4,3),dtype=np.uint8)))
    sim.observation=lambda:{'control_step':sim.steps}
    agent=Agent(sim,tmp_path,None,'unused',300,60,coding_control_enabled=False)
    payload={'input':[]}
    for i in range(18):
        sim.steps=i*2
        parts=agent.content()
        content=[{'type':'input_image','image_url':p['imageUrl']} if p['type']=='inputImage'
                 else {'type':'input_text','text':p['text']} for p in parts]
        payload['input'].append({'type':'function_call_output','call_id':str(i),'output':content})
    window=json.loads((tmp_path/'image-window.json').read_text())
    outgoing,_=select_request_images(payload,window)
    assert summarize(payload)['input_images']>50
    assert summarize(outgoing)['input_images']==5
    assert [r['env_step'] for r in window['rounds']]==[18,22,26,30,34]

def test_native_action_rejects_bad_values():
    assert validate_action([0.]*7)==[0.]*9
    for bad in ([True]*7,[math.nan]*7,[math.inf]*7,[1.01]*7,[0.]*9,{},None):
        with pytest.raises(ValueError):validate_action(bad)

def test_contact_requires_real_sustained_force_cancel_and_release():
    r=ContactRecovery()
    for _ in range(20):r.contact(0,1)
    assert r.result()['status']=='not_covered' and r.result()['success'] is None
    with pytest.raises(ValueError):r.cancel(1)
    r.contact(12,1);r.contact(0,1);assert r.trigger_step is None
    r.contact(11,2);r.contact(12,2);assert r.pending
    for _ in range(20):r.tick(0)
    assert r.result()['status']=='failed'
    r.cancel(2)
    for _ in range(9):r.tick(0)
    assert not r.recovered
    r.tick(6);r.tick(0);assert not r.recovered
    for _ in range(9):r.tick(0)
    assert r.result()['status']=='recovered'
    r.tick(8);assert r.result()['status']=='failed'
    assert 'success' not in r.observation() and 'status' not in r.observation()
    for bad in (math.nan,-1):
        with pytest.raises(ValueError):r.contact(bad,100)

def test_specs_and_suite(tmp_path):
    assert [r['id'] for r in selected('ai_cps',None)]==['22','23','24','34']
    assert {r['name'] for r in specs('24')}=={'observe','move_joints','coding_control'}
    assert 'cancel_action' in {r['name'] for r in specs('34')}
    args=SimpleNamespace(model='model-b',seed=9,steps=None,timeout=42)
    cmd=run_command('ai_cps',SUITES['ai_cps']['cases'][0],args,tmp_path,tmp_path/'auth')
    assert cmd[cmd.index('--model')+1]=='model-b'
    assert cmd[cmd.index('--steps')+1]=='300'

def test_scoring_incomplete_does_not_claim_success():
    from environment.benchmarks.ai_cps.scoring import score
    r=score('FrankaBallCatching',[[0.]*27]*3,False,'missing')
    assert r['success'] is None and r['status']=='incomplete_diagnostic'

def test_joint_only_mask_and_dispatch(tmp_path):
    from environment.benchmarks.ai_cps.policy import Agent,instructions
    assert {r['name'] for r in specs('22',False)}=={'observe','move_joints'}
    assert {r['name'] for r in specs('34',False)}=={'observe','move_joints','cancel_action'}
    a=Agent(SimpleNamespace(steps=0),tmp_path,None,'test',300,60,coding_control_enabled=False)
    with pytest.raises(ValueError,match='disabled'):
        a.execute('coding_control',{'note':'bypass','max_steps':1,'code':'def control(obs,memory): return {"arm_action":[0]*7}'})
    assert a.sim.steps==0 and not (tmp_path/'programs').exists()
    for case in ('22','23','24','34'):
        text=instructions(case,'task',False)
        assert 'coding_control' not in text and 'def control' not in text
    assert '180-degree' in instructions('22','task',False)
    assert 'Use coding_control for fast feedback' in instructions('22','task',True)

def test_code_segments_validate_before_step_and_budget(tmp_path):
    from environment.benchmarks.ai_cps.policy import Agent
    from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
    from environment.runtime.events import EventLog
    import time
    class FakeSim:
        steps=0;done=False;recovery=None
        def observation(self):return {'control_step':self.steps,'dt':.0166}
        def step(self,action):self.steps+=1;return self.observation()
    a=object.__new__(Agent);a.sim=FakeSim();a.steps=2;a.out=tmp_path;a.deadline=time.monotonic()+30;a.program=0
    a.events=EventLog(tmp_path/'events/environment.jsonl')
    r=a.execute('coding_control',{'note':'feedback','max_steps':100,'code':'def control(obs,memory):\n    return {"arm_action":[0,0,0,0,0,0,0]}\n'})
    assert r['executed_steps']==2 and a.sim.steps==2
    a.steps=10
    r=a.execute('coding_control',{'note':'stop','max_steps':10,'code':'def control(obs,memory):\n    return {"done":True}\n'})
    assert r['executed_steps']==0 and a.sim.steps==2
    r=a.execute('coding_control',{'note':'bad','max_steps':10,'code':'def control(obs,memory):\n    return {"arm_action":[2,0,0,0,0,0,0]}\n'})
    assert r['executed_steps']==0 and 'error' in r

def test_scoring_upstream_first_sample_alias_and_completion(monkeypatch):
    import sys
    from environment.benchmarks.ai_cps import scoring
    # Lightweight torch substitute verifies what samples the scorer sends, not GPU math.
    class Tensor:
        def __init__(self,trace):self.trace=trace
        def __getitem__(self,key):return [row[key[1]] for row in self.trace]
    class Norm:
        def __init__(self,rows):self.values=[math.hypot(*r) for r in rows]
        def tolist(self):return self.values
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float32='float32',tensor=lambda t,dtype:Tensor(t),linalg=SimpleNamespace(vector_norm=lambda r,dim:Norm(r))))
    import json
    seen={}
    def run(cmd,**kwargs):
        seen.update(json.loads(kwargs['input']));return SimpleNamespace(stdout=json.dumps({'success':True}))
    monkeypatch.setattr(scoring.subprocess,'run',run)
    trace=[[0.]*27 for _ in range(8)];trace[0][21]=1
    r=scoring.score('FrankaBallCatching',trace,True,'source')
    assert seen['trace'][0]==[1,0.] and seen['trace'][1]==[2,0.]
    assert r['completion_time']==0 and trace[0][21]==1
