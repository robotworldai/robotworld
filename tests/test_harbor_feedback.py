from environment.service.episode import Episode
from environment.service.feedback import FeedbackAdapter

class Adapter:
    def __init__(self): self.n=0
    def describe(self): return {'tools':[{'name':'move'}]}
    def observe(self): return {'state':{'n':self.n},'images':[]}
    def prepare(self, commands):
        result=[]
        for c in commands:
            if c['tool']!='move': raise ValueError('unsupported')
            result.append((c['arguments']['x'],c['arguments'].get('steps',1)))
        return result
    def step(self,x):
        self.n+=x
        return {'success':self.n>=3,'terminated':False,'truncated':False}
    def close(self): pass

def run(code, cap=None):
    a=Adapter(); events=[]
    e=Episode(FeedbackAdapter(a,lambda k,v:events.append((k,v))),10,'test',control_step_limit=cap)
    r=e.execute({'request_id':'p','episode_id':'test','observation_id':'test:0','commands':[
        {'tool':'coding_control','arguments':{'note':'test','max_steps':10,'code':code}}]})
    return a,e,r,events

def test_feedback_steps_and_native_success():
    a,e,r,events=run('def control(obs,memory):\n return {"tool":"move","arguments":{"x":1}}')
    assert r['success'] and r['executed_steps']==3 and len(events)==3
    assert events[1][1]['observation']['state']['n']==1

def test_feedback_cap():
    _,_,r,_=run('def control(obs,memory):\n return {"tool":"move","arguments":{"x":1}}',2)
    assert r['executed_steps']==2 and r['stop_reason']=='control_step_limit'

def test_done_is_not_success():
    _,_,r,_=run('def control(obs,memory):\n return {"done":True}')
    assert r['executed_steps']==0 and not r['terminal'] and not r['success']

def test_invalid_program_does_not_end_episode():
    _,_,r,_=run('import os\ndef control(obs,memory):\n return {"done":True}')
    assert r['executed_steps']==0 and not r['terminal']
    assert 'program_error' in r['feedback'][0]['details']

def test_disabled_adapter_rejects_code():
    import pytest
    a=Adapter()
    with pytest.raises(ValueError): a.prepare([{'tool':'coding_control','arguments':{}}])
