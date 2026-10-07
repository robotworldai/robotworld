import json
from types import SimpleNamespace

import pytest

from environment.runtime import episode_recovery as recovery
from environment.benchmarks.native_project import policy


@pytest.mark.parametrize('error,kind', [
    ({'message':'stream disconnected: error sending request'}, 'transient_api'),
    ({'codexErrorInfo':{'responseStreamDisconnected':{'httpStatusCode':429}}}, 'transient_api'),
    ({'message':'stream disconnected', 'httpStatusCode':400}, 'invalid_request'),
    ({'message':'stream disconnected: content policy violation'}, 'content_policy_violation'),
    ({'message':'stream disconnected: Incomplete response returned, reason: content_filter'}, 'content_policy_violation'),
    ({'message':'request_too_large'}, 'request_too_large'),
    ({'message':'invalid api key'}, 'authentication_error'),
    ({'message':'process disappeared'}, 'unknown'),
])
def test_failure_classification(error, kind):
    assert recovery.failure_kind(error) == kind


def test_deadline_and_recovery_budget(monkeypatch):
    clock = [0]
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(recovery.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0]+delay))
    events = SimpleNamespace(write=lambda *a: None)
    r = recovery.TurnRecovery(100, events)
    args = dict(thread='t', turn='u', control_step=42)
    for _ in range(3):
        assert r.wait({'message':'rate limit exceeded'}, **args)
    assert clock[0] == 70
    assert not r.wait({'message':'rate limit exceeded'}, **args)
    with pytest.raises(TimeoutError):
        recovery.TurnRecovery(75, events).wait({'message':'timeout'}, **args)
    assert clock[0] == 70


@pytest.mark.parametrize('error_message', [
    'rate limit exceeded',
    'Image processing blocked due to content policy violation',
])
def test_failed_turn_keeps_sim_thread_receipt_and_exact_packet(tmp_path, monkeypatch, error_message):
    clock = [0]
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(recovery.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0]+delay))
    monkeypatch.setenv('WORLD_CODEX_SOCKET', 'fixture')
    class Sim:
        steps = 0
        done = False
        dt = .01
        policy_images = {}
        action_metadata = {'dim':1,'lower':[None],'upper':[None]}
        last_evaluation = {}
        actions = []
        def observation(self): return {'control_step':self.steps}
        def step(self, action):
            self.actions.append(action)
            self.steps += 1
            self.done = self.steps == 2
            return self.observation()
    sim = Sim()
    class Session:
        def __init__(self,*args): self.turns=[];self.sent=[];self.n=0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def rpc(self,method,params,**kwargs):
            if method=='thread/start': return {'thread':{'id':'same-thread'}}
            self.turns.append(params)
            return {'turn':{'id':str(len(self.turns))}}
        def send(self,item): self.sent.append(item)
        def receive(self,remaining):
            self.n += 1
            if self.n==2:
                return {'method':'turn/completed','params':{'threadId':'same-thread','turn':{
                    'id':'1','status':'failed','error':{'message':error_message}}}}
            return {'id':self.n,'method':'item/tool/call','params':{
                'threadId':'same-thread','turnId':str(len(self.turns)),
                'callId':str(self.n),'tool':'apply_action',
                'arguments':{'note':'fixture','action':[self.n],'steps':1}}}
    session=Session()
    monkeypatch.setattr(policy,'CodexSession',lambda *a: session)
    agent=policy.Agent(sim,tmp_path,None,'fixture',10,100,'fixture',coding=False)
    agent.run()
    assert sim.actions==[[1.],[3.]]
    assert clock[0]==10 and agent.deadline==100
    assert agent.seq==3  # Recovery did not introduce an extra observation round.
    assert len(session.turns)==2
    assert all(t['threadId']=='same-thread' for t in session.turns)
    note=session.turns[1]['input'][0]['text']
    assert '"call_id": "1"' in note and 'control step 1' in note
    packet=session.sent[0]['result']['contentItems'][1:]
    assert session.turns[1]['input'][1:]==[
        {'type':'text','text':p['text'],'text_elements':[]} for p in packet]
    assert 'turn_recovery_started' in (tmp_path/'events/no-images/environment.jsonl').read_text()


def test_image_error_has_three_attempt_cap_and_shared_deadline(monkeypatch):
    clock=[0]; rows=[]
    monkeypatch.setattr(recovery.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(recovery.time,'sleep',lambda d:clock.__setitem__(0,clock[0]+d))
    events=SimpleNamespace(write=lambda name,payload:rows.append((name,payload)))
    args=dict(thread='same',turn='failed',control_step=604)
    error={'message':'Image processing blocked due to content policy violation'}
    r=recovery.TurnRecovery(100,events)
    assert r.wait(error,**args)
    assert r.wait(error,**args)
    assert r.wait(error,**args)
    assert not r.wait(error,**args)
    assert clock[0]==70 and r.attempts==3
    assert [p['delay_seconds'] for name,p in rows if name=='turn_recovery_wait']==[10,20,40]
    assert rows[0][1]['reason']=='content_policy_violation'
    assert not r.wait({'message':'timeout'},**args)
    with pytest.raises(TimeoutError):
        recovery.TurnRecovery(clock[0]+5,events).wait(error,**args)


def test_image_recovery_shares_cap_with_prior_disconnect(monkeypatch):
    clock=[0]
    monkeypatch.setattr(recovery.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(recovery.time,'sleep',lambda d:clock.__setitem__(0,clock[0]+d))
    r=recovery.TurnRecovery(100,SimpleNamespace(write=lambda *args:None))
    args=dict(thread='t',turn='u',control_step=42)
    error={'message':'Image processing blocked due to content policy violation'}
    assert r.wait({'message':'timeout'},**args)
    assert r.wait(error,**args)
    assert r.wait(error,**args)
    assert not r.wait(error,**args)
    assert r.attempts==3 and r.image_attempts==2


@pytest.mark.parametrize('message', ['content policy violation', 'content_filter',
    'invalid api key', 'request_too_large'])
def test_other_nontransient_errors_are_not_retried(message):
    r=recovery.TurnRecovery(1e12,SimpleNamespace(write=lambda *args:None))
    assert not r.wait({'message':message},thread='t',turn='u',control_step=1)
