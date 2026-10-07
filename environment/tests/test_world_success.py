import copy
import json
from pathlib import Path
import pytest
from environment.evaluation.world_success.monitor import Monitor
from environment.evaluation.world_success.profiles import PROFILES
from environment.evaluation.world_success.validation import passing_state, run_trace


@pytest.mark.parametrize('key',PROFILES)
def test_all_approved_checks_have_positive_witness_and_fail_on_native_failure(key):
    p=PROFILES[key];s=passing_state(p)
    assert run_trace(p,s)['world_success'] is True
    assert run_trace(p,s,lambda i,s:{'native_failure':i==p['steps']})['world_success'] is False


def test_hold_requires_real_elapsed_duration_not_sample_count():
    p=PROFILES['go2_push/quadruped_push_recovery'];s=passing_state(p)
    m=Monitor(p)
    for i in range(101):
        m.update(i,{**s,'goal_distance':1. if i==0 else 0.})
    assert m.records['S01']['passed'] is False # 100 true samples span only 1.98 s
    m.update(101,s)
    assert m.records['S01']['passed'] is True
    assert m.report(scene_verified=True)['world_success'] is None # full episode still required


def test_missing_duplicate_and_out_of_order_ticks_rejected():
    p=PROFILES['go2_push/quadruped_push_recovery'];m=Monitor(p);s=passing_state(p)
    with pytest.raises(ValueError):m.update(1,s)
    m.update(0,s)
    with pytest.raises(ValueError):m.update(0,s)
    with pytest.raises(ValueError):m.update(2,s)


def test_single_always_violation_not_erased_by_later_recovery():
    p=PROFILES['digit/digit_walk_hand_tracking'];s=passing_state(p)
    def violation(i,s):
        if i==200:s['left_wrist_position_error']=.081
    assert run_trace(p,s,violation)['world_success'] is False


def test_always_window_includes_bracketing_samples_at_62_hz():
    p=PROFILES['omnidrones/drone_inverted_pendulum_tracking'];s=passing_state(p)
    def violation(i,s):
        if i==31:s['tip_error']=.21
    assert run_trace(p,s,violation)['world_success'] is False


def test_event_end_requires_all_five_completed_serves():
    p=PROFILES['ttrl/humanoid_table_tennis_return'];m=Monitor(p)
    m.update(0,dict(scored_serves=4,valid_returns=4))
    assert m.report(scene_verified=True,event_complete=True)['world_success'] is False
    m.update(1,dict(scored_serves=5,valid_returns=3))
    assert m.report(scene_verified=True,event_complete=True)['world_success'] is True


def test_boolean_not_accepted_for_numeric_state():
    p=PROFILES['go2_push/quadruped_push_recovery'];s=passing_state(p);s['speed']=False
    assert run_trace(p,s)['world_success'] is None


def test_executable_profile_matches_approved_design_inventory():
    data=json.loads((Path(__file__).resolve().parents[2]/'docs/scoring/success-proposals.json').read_text())
    assert len(PROFILES)==len(data['proposals'])==16
    for proposal in data['proposals']:
        p=PROFILES[proposal['benchmark']+'/'+proposal['task']]
        assert p['steps']==proposal['environment_settings']['step_limit']
        assert [(x['id'],x['operator'],x['seconds']) for x in p['checks']]==[(x['id'],x['operator'],x['seconds']) for x in proposal['state_checks']]


def test_directed_gate_order_and_reverse_motion():
    from environment.evaluation.world_success.routes import Gates
    g=Gates([((1,0),(1,0),.5),((2,0),(1,0),.5)])
    for p in [(2.1,0),(.5,0),(.9,1.0),(1.1,1.0)]:g.update(p)
    assert g.next==0
    for p in [(0,0),(1.1,0),(2.1,0)]:g.update(p)
    assert g.complete
    for p in [(0,0),(3,0),(0,0),(3,0)]:g.update(p)
    assert g.next==2


def test_drift_requires_two_distinct_continuous_corners_and_moving_forward():
    import math
    from environment.evaluation.world_success.routes import Stadium
    track=Stadium(.02)
    for i in range(16):track.update(i*.02,(.4,1.2),math.pi,1.,math.tan(.35))
    assert len(track.completed)==1
    for i in range(16):track.update(1+i*.02,(.4,1.2),math.pi,1.,math.tan(.35))
    assert len(track.completed)==1
    for i in range(16):track.update(2+i*.02,(-.4,-1.2),0,-1.,0.)
    assert len(track.completed)==1
    for i in range(16):track.update(3+i*.02,(-.4,-1.2),0,1.,math.tan(.35))
    assert len(track.completed)==2
    assert not track.gates.complete


def test_jump_counts_each_command_once_and_requires_rise_then_landing():
    from environment.evaluation.world_success.scenes import JumpCounter
    c=JumpCounter()
    for t,z,air,ground in [(3,.5,False,True),(3.1,.57,True,False),(3.5,.5,False,True)]:c.update(t,z,air,ground)
    assert c.completed==0
    for row in [(3.6,.60,True,False),(3.9,.5,False,True),(4,.7,True,False),(4.3,.5,False,True)]:c.update(*row)
    assert c.completed==1
    for row in [(10,.5,False,True),(10.2,.7,True,False),(10.4,.5,False,True)]:c.update(*row)
    assert c.completed==2


def test_reference_results_cannot_enter_model_statistics(tmp_path):
    from environment.evaluation.rollout_results import extract
    (tmp_path/'result.json').write_text(json.dumps(dict(success=True,diagnostic_only=True)))
    with pytest.raises(ValueError,match='Diagnostic'):extract('omnidrones',tmp_path)


def test_world_missing_judge_cannot_reuse_native_success(tmp_path):
    from environment.evaluation.rollout_results import extract
    (tmp_path/'result.json').write_text(json.dumps(dict(success=True,scoring_profile='world-state-v1')))
    assert extract('omnidrones',tmp_path)['success'] is None


def test_world_rollout_uses_approved_budget_and_explicit_profile(capsys):
    from environment.evaluation.rollouts import main
    main(['--bench','omnidrones','run','--dry-run','--tasks','drone_inverted_pendulum_tracking','--rollouts','1'])
    out=capsys.readouterr().out
    assert '620' in out and 'world-state-v1' in out


def test_all_model_facing_world_instructions_are_english():
    from environment.evaluation.world_success.runtime import instruction
    for profile in PROFILES.values():
        text=instruction(profile)
        assert not any('\u4e00'<=c<='\u9fff' for c in text)
        assert profile['version'] in text


def test_runtime_keeps_private_judging_out_of_actor_and_declares_added_goal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from environment.evaluation.world_success import runtime
    profile=PROFILES['go2_push/quadruped_push_recovery']
    class Scene:
        def __init__(self, sim, profile, seed, enabled):
            self.enabled=enabled;self.goal=[2.5,0];self.verified=True
            self.failure=None;self.events=[]
        def initialize_contact(self, step): pass
        def after_reset(self): pass
        def before_step(self): pass
        def public_observation(self): return {'world_goal_relative_body_m':[2.5,0]}
    sim=SimpleNamespace(dt=.02,output=tmp_path,steps=0,done=False,
                        terminated=False,last_evaluation={},observation_metadata={'policy':{'shape':[3]}})
    sim.reset=lambda: None
    def step(action): sim.steps+=1
    sim.step=step
    sim.observation=lambda: {'actor':[1,2,3]}
    sim.result=lambda: {'success':None,'reward':1.25}
    monkeypatch.setattr(runtime,'Scene',Scene)
    monkeypatch.setattr(runtime,'measure',lambda *args:passing_state(profile))
    runtime.attach(sim,'go2_push','quadruped_push_recovery',7,'world-state-v1')
    obs=sim.reset()
    assert set(obs)=={'actor','world_goal_relative_body_m'}
    assert sim.observation_metadata['policy']=={'shape':[3]}
    assert sim.observation_metadata['world_added_fields']['world_goal_relative_body_m']['units']=='m'
    for _ in range(profile['steps']):sim.step([0])
    result=sim.result()
    assert result['success'] is True and result['native_success'] is None
    assert result['reward']==1.25
    assert result['stop_reason']=='world_evaluation_end'
    assert result['world_added_observation_fields']['world_goal_relative_body_m']['shape']==[2]
    assert 'world_evaluation' not in sim.observation()
    assert len((tmp_path/'events/world-success.jsonl').read_text().splitlines())==profile['steps']+1
