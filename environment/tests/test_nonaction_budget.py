import io
import json
from collections import deque
from types import SimpleNamespace
import pytest
from environment.runtime.nonaction_budget import (
    NonActionBudget, NonActionBudgetExceeded, attach, check, apply_result)
from environment.runtime.codex_session import CodexSession


@pytest.fixture(autouse=True)
def enable_protocol(monkeypatch):
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL','nonaction-20-150-v1')


def event(method, **params):
    return {'method':method,'params':params}


def aux(identifier, kind='commandExecution'):
    return event('item/completed',turnId='t',item={'type':kind,'id':identifier})


def test_twenty_inclusive_latched_and_deduplicated(tmp_path):
    step=[0];b=NonActionBudget(tmp_path,lambda:step[0])
    for i in range(20):
        b.observe(aux(str(i)));b.observe(aux(str(i)))
        assert b.total==i+1
        assert bool(b.reason)==(i==19)
    step[0]=1;b.progress()
    assert b.consecutive==0 and b.total==20 and b.reason=='nonaction_consecutive_limit'


def test_total_budget_survives_progress(tmp_path):
    step=[0];b=NonActionBudget(tmp_path,lambda:step[0])
    for i in range(150):
        b.observe(aux(str(i),'imageView'))
        step[0]+=1;b.progress()
    assert b.total==150 and b.reason=='nonaction_total_limit'


def test_empty_turn_failed_turn_and_no_double_charge(tmp_path):
    b=NonActionBudget(tmp_path,lambda:0)
    b.start_turn('a');b.observe(event('turn/completed',turn={'id':'a','status':'failed'}))
    assert b.total==0
    b.start_turn('b');b.observe(event('turn/completed',turn={'id':'b','status':'completed'}))
    assert b.total==1
    b.start_turn('c');b.observe(aux('shell'))
    b.observe(event('turn/completed',turn={'id':'c','status':'completed'}))
    assert b.total==2


def test_zero_step_call_counts_but_executed_call_resets(tmp_path):
    step=[0];b=NonActionBudget(tmp_path,lambda:step[0]);b.start_turn('t')
    for i in (1,2):
        b.observe({'id':i,'method':'item/tool/call','params':{'callId':str(i)}})
        if i==2:step[0]=4
        b.reply({'id':i,'result':{'success':i==2}})
    assert b.total==1 and b.consecutive==0
    b.observe(event('turn/completed',turn={'id':'t','status':'completed'}))
    assert b.total==1


def test_inflight_robot_receipt_drained_before_interrupt(tmp_path):
    b=NonActionBudget(tmp_path,lambda:0,consecutive_limit=1)
    b.start_turn('t');b.observe({'id':1,'method':'item/tool/call','params':{'callId':'c'}})
    sent=[];s=SimpleNamespace(nonaction_budget=b,_budget_thread='thread',request=lambda *x:sent.append(x))
    b.observe(aux('a'));check(s)
    assert not sent
    b.reply({'id':1,'result':{'success':False}})
    with pytest.raises(NonActionBudgetExceeded):check(s)
    assert sent==[('turn/interrupt',{'threadId':'thread','turnId':'t'})]


def test_session_stops_within_turn_not_next_turn(tmp_path):
    s=CodexSession.__new__(CodexSession);s.pending=deque([aux(str(i)) for i in range(21)])
    s.events=SimpleNamespace(write=lambda *x:None);s.input_stream=io.StringIO();s.counter=0
    s._budget_thread='thread';b=attach(s,tmp_path,lambda:0);b.start_turn('t')
    for _ in range(19):s.receive()
    with pytest.raises(NonActionBudgetExceeded):s.receive()
    assert len(s.pending)==1
    assert 'turn/interrupt' in s.input_stream.getvalue()


def test_success_and_world_validity_not_fabricated():
    snapshot={'stop_reason':'nonaction_total_limit'}
    assert apply_result({'success':True},snapshot)['success'] is True
    assert apply_result({'success':None},snapshot)['success'] is False
    assert apply_result({'success':True,'scoring_profile':'world-state-v1',
                         'world_evaluation':{'valid':False}},snapshot)['success'] is False


def test_step_regression_is_infrastructure_error(tmp_path):
    step=[2];b=NonActionBudget(tmp_path,lambda:step[0]);step[0]=1
    with pytest.raises(RuntimeError):b.progress()


def test_robodojo_final_answers_continue_to_shared_budget(monkeypatch,tmp_path):
    from environment.runtime import episode_runner as runner
    from environment.tests.test_episode_runner import Adapter, Session
    class Robot(Adapter):
        state={'native_steps':0}
    class BudgetSession(Session):
        def rpc(self,method,params,**kwargs):
            if method=='turn/start':
                self.n=getattr(self,'n',0)+1
                self.nonaction_budget.start_turn(str(self.n))
                return {'turn':{'id':str(self.n)}}
            return super().rpc(method,params)
        def receive(self,*args):
            message=event('turn/completed',turn={'id':str(self.n),'status':'completed'})
            self.nonaction_budget.observe(message);check(self)
            return message
    monkeypatch.setattr(runner,'CodexSession',BudgetSession)
    result=runner.run_episode(Robot(),manifest='unused',output_dir=tmp_path,
        continue_on_completion=True,nonaction_protocol=True,allow_give_up=False)
    assert result['termination']=='nonaction_consecutive_limit'
    assert result['interaction_budget']['total']==20
    assert len(result['continuations'])==19
    assert result['action_calls']==0


def test_budget_result_extraction_keeps_native_incomplete_separate(tmp_path):
    from environment.evaluation.rollout_results import extract
    from environment.runtime.nonaction_budget import VERSION
    snapshot=dict(version=VERSION,stop_reason='nonaction_total_limit',total=150,consecutive=2)
    data={'success':None,'case':'24','scoring_profile':'peg-xy-z-window-v2',
          'native_episode_complete':False,'interaction_budget':snapshot,
          'stop_reason':snapshot['stop_reason'],'control_steps':30}
    (tmp_path/'result.json').write_text(json.dumps(data))
    result=extract('ai_cps',tmp_path)
    assert result['success'] is False
    assert not result['insertion_evaluation']['valid']
    assert result['stop_reason']=='nonaction_total_limit'


def test_observe_exceeds_both_limits_without_stopping(monkeypatch, tmp_path):
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL', 'observe')
    s=SimpleNamespace(request=lambda *args:pytest.fail('Observation must not interrupt'))
    b=attach(s,tmp_path,lambda:0)
    for i in range(160):
        b.start_turn(str(i))
        b.observe(event('turn/completed',turn={'id':str(i),'status':'completed'}))
        check(s)
    snapshot=json.loads((tmp_path/'interaction-budget.json').read_text())
    assert snapshot['total']==160 and snapshot['consecutive_peak']==160
    assert snapshot['stop_reason'] is None
    assert snapshot['first_threshold']['total']==20
    assert snapshot['first_threshold']['turn_id']=='19'
    assert snapshot['missing_turn_end_count']==0
    assert snapshot['counts_complete'] is False
    result=apply_result({'success':None,'stop_reason':'native_reason'},snapshot)
    assert result['success'] is None and result['stop_reason']=='native_reason'


def test_aux_started_failed_completed_deduplicated_and_identical_polls_count(tmp_path):
    b=NonActionBudget(tmp_path,lambda:0,observe_only=True)
    b.start_turn('t')
    b.observe(event('item/started',turnId='t',item={'type':'commandExecution','id':'x'}))
    b.observe(event('item/completed',turnId='t',item={
        'type':'commandExecution','id':'x','status':'failed'}))
    poll=event('item/commandExecution/terminalInteraction',turnId='t',stdin='')
    b.observe(poll);b.observe(poll)
    b.observe(event('turn/completed',turn={'id':'t','status':'completed'}))
    assert b.total==3 and b.counts=={'aux':1,'stdin':2}


def test_observe_does_not_inject_prompt_and_off_does_not_attach(monkeypatch,tmp_path):
    from environment.runtime.nonaction_budget import enabled
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL','observe')
    assert not enabled()
    s=CodexSession.__new__(CodexSession)
    s.counter=0;s.events=SimpleNamespace(write=lambda *args:None)
    s.input_stream=io.StringIO()
    attach(s,tmp_path,lambda:0)
    params={'developerInstructions':'original'}
    s.request('thread/start',params)
    assert json.loads(s.input_stream.getvalue())['params']==params
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL','off')
    assert attach(SimpleNamespace(),tmp_path,lambda:0) is None


def test_observe_robodojo_retains_final_answer_termination(monkeypatch,tmp_path):
    from environment.runtime import episode_runner as runner
    from environment.tests.test_episode_runner import Adapter, Session
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL','observe')
    class Robot(Adapter):
        state={'native_steps':0}
    class ObservingSession(Session):
        def receive(self,*args):
            self.nonaction_budget.start_turn('t')
            message=event('turn/completed',turn={'id':'t','status':'completed'})
            self.nonaction_budget.observe(message)
            check(self)
            return message
    monkeypatch.setattr(runner,'CodexSession',ObservingSession)
    result=runner.run_episode(Robot(),manifest='unused',output_dir=tmp_path,
                             allow_give_up=False)
    assert result['termination']=='agent_completed'
    assert result['interaction_budget']['total']==1
    assert result['interaction_budget']['mode']=='observe'
    assert not result.get('continuations')


def test_observe_snapshot_reports_missing_turn_and_peak_survives_progress(tmp_path):
    steps=[0]
    b=NonActionBudget(tmp_path,lambda:steps[0],observe_only=True)
    b.start_turn('missing')
    b.observe(aux('a'));b.observe(aux('b'))
    steps[0]=1;b.progress()
    assert b.snapshot()['consecutive_peak']==2
    assert b.snapshot()['consecutive']==0
    assert b.snapshot()['missing_turn_end_count']==1
