from concurrent.futures import ThreadPoolExecutor
import pytest
from environment.service.episode import Episode

class Fake:
    def __init__(self, success_at=999): self.n=0; self.success_at=success_at
    def describe(self): return {'tools':[]}
    def observe(self): return {'n':self.n}
    def prepare(self, commands):
        for c in commands:
            if type(c) is not int or c<1: raise ValueError('bad command')
        return [(0,c) for c in commands]
    def step(self, action):
        self.n+=1
        return dict(success=self.n==self.success_at,terminated=False,truncated=False)

def request(commands=[3], rid='a', observation='e:0'):
    return dict(request_id=rid,episode_id='e',observation_id=observation,commands=commands)

def test_duplicate_parallel():
    e=Episode(Fake(),10,'e')
    with ThreadPoolExecutor(2) as pool: results=list(pool.map(e.execute,[request(),request()]))
    assert results[0]==results[1]; assert e.steps==3

def test_reuse_rejected():
    e=Episode(Fake(),10,'e');e.execute(request())
    with pytest.raises(ValueError):e.execute(request([4]))

def test_validate_whole_chunk():
    e=Episode(Fake(),10,'e')
    with pytest.raises(ValueError):e.execute(request([3,-1]))
    assert e.adapter.n==0

@pytest.mark.parametrize('success_at,horizon,reason,count',[(2,10,'success',2),(999,4,'budget_exhausted',4)])
def test_partial_chunk(success_at,horizon,reason,count):
    e=Episode(Fake(success_at),horizon,'e');r=e.execute(request([3,3]))
    assert r['executed_steps']==count;assert r['termination']==reason
    with pytest.raises(ValueError):e.execute(request([1],rid='b',observation=f'e:{count}'))

def test_query_and_stale():
    e=Episode(Fake(),10,'e');assert e.observe()==e.observe();assert e.steps==0
    e.execute(request())
    with pytest.raises(ValueError):e.execute(request(rid='b'))

def test_uncertain_action_not_retried():
    class Broken(Fake):
        def step(self, action): self.n+=1;raise RuntimeError('native error')
    e=Episode(Broken(),10,'e');a=e.execute(request());b=e.execute(request())
    assert a==b;assert a['execution_uncertain'];assert e.adapter.n==1

def test_macro_native_steps_and_zero_step_planning_rejection():
    class Macro(Fake):
        def step(self,action):
            self.n+=1
            return {'success':False,'terminated':False,'truncated':False,
                    'executed_steps':0 if self.n==1 else 4,
                    'details':{'planned':self.n!=1}}
    episode=Episode(Macro(),8,'e')
    result=episode.execute(request([3]))
    assert result['executed_steps']==8
    assert result['executed_steps_per_command']==[8]
    assert result['termination']=='budget_exhausted'
    assert len(result['feedback'])==3
    assert episode.execute(request([3]))==result

def test_bad_adapter_overrun_is_infrastructure_error():
    class Macro(Fake):
        def step(self,action):
            return {'success':False,'terminated':False,'truncated':False,'executed_steps':11}
    result=Episode(Macro(),10,'e').execute(request([1]))
    assert result['termination']=='infrastructure_error'
    assert result['execution_uncertain']

@pytest.mark.parametrize('cap,horizon,success_at,reason,count', [
    (4,10,999,'agent_stopped',4), (20,5,999,'budget_exhausted',5),
    (4,10,4,'success',4), (4,4,999,'budget_exhausted',4)])
def test_optional_cap_preserves_native_budget_and_terminal_priority(cap,horizon,success_at,reason,count):
    e=Episode(Fake(success_at),horizon,'e',control_step_limit=cap)
    r=e.execute(request([3,8,2]))
    assert r['control_step']==count and r['horizon']==horizon
    assert r['termination']==reason
    assert r['stop_reason']==('control_step_limit' if reason=='agent_stopped' else None)
    assert sum(r['executed_steps_per_command'])==count
    assert e.execute(request([3,8,2]))==r
    assert e.describe()['execution_budget']['effective_control_step_limit']==min(horizon,cap)


def test_atomic_macro_cap_overshoot_reported_and_no_next_macro():
    class Macro(Fake):
        def step(self,action):
            self.n+=1
            return dict(success=False,terminated=False,truncated=False,executed_steps=3)
    e=Episode(Macro(),20,'e',control_step_limit=4)
    r=e.execute(request([10]))
    assert r['control_step']==6 and r['termination']=='agent_stopped'
    assert r['control_step_limit_overshoot']==2 and e.adapter.n==2


@pytest.mark.parametrize('cap',[0,-1,True,1.5,'400'])
def test_invalid_control_step_cap(cap):
    with pytest.raises(ValueError):Episode(Fake(),10,'e',control_step_limit=cap)


def test_feedback_history_samples_tool_boundaries_not_physics():
    class Camera(Fake):
        def describe(self):return {'benchmark':'robolab','tools':[]}
        def observe(self):return {'state':{'n':self.n},'images':[
            {'control_step':i,'camera':'front','base64':str(i)} for i in range(max(0,self.n-8),self.n+1)]}
    e=Episode(Camera(),1000,'e')
    e.observe()
    for i in range(1,7):
        e.execute(request([10],rid=str(i),observation=f'e:{(i-1)*10}'))
        obs=e.observe()
        assert obs==e.observe()
    assert [x['control_step'] for x in obs['observation']['images']]==[0,20,40,60]
    assert e.describe()['observation_history']['interval_tool_feedback_rounds']==2


def test_robocasa_retains_native_physical_frame_history():
    class Camera(Fake):
        def describe(self):return {'benchmark':'robocasa','tools':[]}
        def observe(self):return {'images':[{'control_step':i} for i in (2,4,6,8,10)]}
    e=Episode(Camera(),100,'e')
    assert e.observe()['observation']['images']==e.adapter.observe()['images']


def test_native_projects_keep_historical_public_numeric_state():
    class Camera(Fake):
        def describe(self):return {'benchmark':'wheeledlab','tools':[]}
        def observe(self):return {'state':{'x':self.n},'images':[{'control_step':self.n}]}
    e=Episode(Camera(),100,'e');e.observe()
    for i in (1,2):
        e.execute(request([10],rid=str(i),observation=f'e:{(i-1)*10}'));obs=e.observe()
    assert [x['state']['x'] for x in obs['observation']['history_states']]==[0,20]
    assert obs['observation']['state']['episode_steps_remaining']==80
