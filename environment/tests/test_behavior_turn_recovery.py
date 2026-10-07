import json
import sys
import types

import pytest

from environment.benchmarks.behavior_1k import policy as module
from environment.runtime import episode_recovery
from environment.tests.test_behavior_bridge import FakeSession, profile, obs


class RecoverySession(FakeSession):
    error={'message':'stream disconnected before completion'}
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.turns=0;self.index=0
    def rpc(self,method,params,*,timeout=None):
        self.sent.append((method,params))
        if method=='thread/start':return {'thread':{'id':'t'}}
        self.turn_timeout=timeout
        self.turns+=1;return {'turn':{'id':f'u{self.turns}'}}
    def receive(self,*args):
        self.index+=1
        if self.index==2:
            return {'method':'turn/completed','params':{'threadId':'t',
                'turn':{'id':'u1','status':'failed','error':self.error}}}
        return {'id':self.index,'method':'item/tool/call','params':{'threadId':'t',
            'turnId':f'u{self.turns}','callId':f'c{self.index}','tool':'move_base',
            'arguments':{'note':'attempt','targets':{'base_vx':.1},'steps':1}}}


def setup(monkeypatch,tmp_path,session=RecoverySession):
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake')
    monkeypatch.setattr(module,'CodexSession',session)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    waits=[];monkeypatch.setattr(episode_recovery.time,'sleep',waits.append)
    p=module.WorldPolicy(tmp_path,'unused','clean_up_your_desk')
    p.bind(profile());p.reset()
    return p,waits


def test_failed_turn_resumes_without_reset_or_extra_observation(monkeypatch,tmp_path):
    p,waits=setup(monkeypatch,tmp_path)
    p.forward(obs());deadline=p.deadline
    p.forward(obs())
    assert p.step==1 and p.calls==2 and p.observation==2
    assert p.thread=='t' and p.turn=='u2' and p.deadline==deadline
    assert waits==[10] and p.recovery.attempts==1
    assert p.last_receipt['call_id']=='c1' and p.last_receipt['control_step']==1
    starts=[v[1] for v in p.session.sent if isinstance(v,tuple) and v[0]=='turn/start']
    assert len(starts)==2 and 'Last completed tool receipt' in starts[1]['input'][0]['text']
    records=[json.loads(l) for l in (p.run_dir/'events/environment.jsonl').read_text().splitlines()]
    assert sum(r['kind']=='action_completed' for r in records)==1


def test_policy_refusal_does_not_recover(monkeypatch,tmp_path):
    class Refusal(RecoverySession):error={'message':'Image processing blocked due to content policy violation'}
    p,waits=setup(monkeypatch,tmp_path,Refusal)
    p.forward(obs())
    with pytest.raises(RuntimeError,match='agent_failed'):p.forward(obs())
    assert waits==[] and p.calls==1 and p.step==1


def test_turn_start_uses_remaining_original_deadline(monkeypatch,tmp_path):
    p,_=setup(monkeypatch,tmp_path)
    p.forward(obs())
    monkeypatch.setattr(module.time,'monotonic',lambda:p.deadline-2)
    p._begin_turn(p.latest_content)
    assert p.session.turn_timeout==2
    starts=p.session.turns
    monkeypatch.setattr(module.time,'monotonic',lambda:p.deadline)
    with pytest.raises(TimeoutError,match='before turn start'):
        p._begin_turn(p.latest_content)
    assert p.session.turns==starts
