import numpy as np
import pytest
from environment.benchmarks.humanoid_soccer.control import specs, validate
from environment.benchmarks.humanoid_soccer.control import compose_action

NAMES=['hip','knee','ankle']
LIMITS=np.array([[-1,1],[0,2],[-.5,.5]])


def test_native_order_roundtrip_accept_is_exact_and_corrections_reach_correct_joint():
    # Native order ankle, hip, knee; named target order hip, knee, ankle.
    proposal=np.array([.3,.4,.2],dtype=np.float32)
    default=np.array([0,1,0]);scale=np.array([.5,.2,.1])
    to_mj=np.array([1,2,0]);to_native=np.array([2,0,1])
    actual=compose_action('hybrid','accept',proposal,None,default,scale,LIMITS,to_mj,to_native)
    assert np.array_equal(actual,proposal)
    corrected=compose_action('hybrid','modify',proposal,np.array([.1,0,0]),default,scale,LIMITS,to_mj,to_native)
    assert np.allclose(default+scale*corrected[to_mj],[.3,1.04,.03])
    direct=compose_action('direct','direct',None,np.array([-.4,1.2,.1]),default,scale,LIMITS,to_mj,to_native)
    assert np.allclose(default+scale*direct[to_mj],[-.4,1.2,.1])


def test_supervisor_requires_explicit_accept_or_bounded_modification():
    n,offset=validate('review_action',{'note':'keep original','steps':20,'decision':'accept','joint_offsets':{}},NAMES,LIMITS,[0,1,0],'hybrid')
    assert n==20 and np.array_equal(offset,[0,0,0])
    _,offset=validate('review_action',{'note':'correct','steps':1,'decision':'modify','joint_offsets':{'knee':-.1}},NAMES,LIMITS,[0,1,0],'hybrid')
    assert np.allclose(offset,[0,-.1,0])
    for decision,offsets in [('accept',{'hip':.1}),('modify',{}),('modify',{'hip':float('nan')}),('modify',{'ankle':.3})]:
        with pytest.raises(ValueError):validate('review_action',{'note':'bad','steps':1,'decision':decision,'joint_offsets':offsets},NAMES,LIMITS,[0,1,0],'hybrid')


def test_direct_targets_hold_omitted_measured_joints_and_reject_bad_commands():
    _,target=validate('move_joints',{'note':'move','steps':5,'joint_positions':{'hip':.2}},NAMES,LIMITS,[0,1,.3],'direct')
    assert np.allclose(target,[.2,1,.3])
    for targets in ({'hip':2},{'hip':True},{'ankle':float('inf')},{'teleport':1},{}):
        with pytest.raises(ValueError):validate('move_joints',{'note':'bad','steps':1,'joint_positions':targets},NAMES,LIMITS,[0,1,0],'direct')
    assert [t['name'] for t in specs(NAMES,LIMITS,'direct')]==['move_joints','coding_control']
    with pytest.raises(ValueError):validate('review_action',{'note':'hidden policy','steps':1,'decision':'accept','joint_offsets':{}},NAMES,LIMITS,[0,1,0],'direct')


def test_soccer_suite_has_separate_modes_and_preserves_native_budget():
    import argparse
    from pathlib import Path
    from environment.evaluation.runner import selected, run_command
    rows=selected('humanoid_soccer','baseline,hybrid,direct')
    assert [r['mode'] for r in rows]==['baseline','hybrid','direct']
    assert all(r['steps']==300 for r in rows)
    args=argparse.Namespace(steps=None,model='gpt-6-astra',seed=7,timeout=7200)
    cmd=run_command('humanoid_soccer',rows[2],args,Path('/out'),Path('/auth'))
    assert cmd[cmd.index('--mode')+1]=='direct'
    args.steps=500
    with pytest.raises(ValueError):run_command('humanoid_soccer',rows[2],args,Path('/out'),Path('/auth'))


def test_fixed_soccer_case_is_default_and_rejects_protocol_changes():
    import argparse
    from pathlib import Path
    from environment.evaluation.runner import selected, run_command, WORLD
    rows=selected('humanoid_soccer',None)
    assert len(rows)==1 and rows[0]['id']=='play-soccer'
    args=argparse.Namespace(steps=None,model='another-model',seed=7,seed_explicit=False,timeout=7200)
    cmd=run_command('humanoid_soccer',rows[0],args,Path('/out'),Path('/auth'))
    for flag,value in [('--mode','direct'),('--seed','2'),('--sim-time','20.0'),('--scenery','training-pitch'),('--balance-assist','ankle-com'),('--model','another-model')]:
        assert cmd[cmd.index(flag)+1]==value
    notes=Path(cmd[cmd.index('--controller-notes')+1])
    assert notes.is_file() and notes.is_relative_to(WORLD/'third_party')
    args.seed_explicit=True
    with pytest.raises(ValueError,match='seed=2'):run_command('humanoid_soccer',rows[0],args,Path('/out'),Path('/auth'))
    args.seed=2;args.steps=500
    with pytest.raises(ValueError,match='fixed step budget'):run_command('humanoid_soccer',rows[0],args,Path('/out'),Path('/auth'))
    args.steps=1000
    assert '--sim-time' in run_command('humanoid_soccer',rows[0],args,Path('/out'),Path('/auth'))
