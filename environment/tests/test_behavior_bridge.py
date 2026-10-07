import json
import sys
import types
import numpy as np
import pytest
from environment.benchmarks.behavior_1k.control import action_for, axis_angle, split_proprio
from environment.benchmarks.behavior_1k import policy as module


def profile():
    return {'action_dim':21,'controller_indices':{'base':[0,1,2], 'arm_left':list(range(3,9)),
            'gripper_left':[9], 'arm_right':list(range(10,16)), 'gripper_right':[16], 'trunk':[17,18,19,20], 'camera':[]},
            'proprio_fields':{'eef_left_pos':3,'eef_left_quat':4,'eef_right_pos':3,'eef_right_quat':4,'trunk_qpos':4},
            'proprio_key':'robot::proprio','camera_keys':{'head':'robot::head','left_wrist':'robot::left','right_wrist':'robot::right'}}


def obs():
    o={'robot::proprio':np.array([[.3,.2,.8,0,0,0,1,.3,-.2,.8,0,0,0,1,0,0,0,0]])}
    for prefix in profile()['camera_keys'].values():
        o[prefix+'::rgb']=np.zeros((1,4,4,3),dtype=np.uint8)
        o[prefix+'::depth_linear']=np.ones((1,4,4))
    o['privileged_object_positions']='MUST_NOT_LEAK'
    return o


def test_absolute_ik_hold_gripper_and_bounded_base():
    p=profile();state=split_proprio(obs()['robot::proprio'],p['proprio_fields'])
    action,grip=action_for({'left_z':.9,'left_gripper':0,'base_vx':.1},state,p,{'left':1,'right':.4})
    np.testing.assert_allclose(action[[3,4,5]],[.3,.2,.9]);np.testing.assert_allclose(action[10:13],[.3,-.2,.8])
    assert action[9]==-1 and grip=={'left':0,'right':.4}
    assert np.isclose(action[16],-.2)
    held,_=action_for({'left_z':.9},state,p,grip)
    assert held[0]==0 and held[9]==-1
    for targets in ({'left_x':float('nan')},{'base_vx':1},{'left_gripper':2},{'world_pose':0}):
        with pytest.raises(ValueError):action_for(targets,state,p,grip)


def test_quaternion_sign_equivalence_and_shape():
    np.testing.assert_allclose(axis_angle([0,0,1,0]),axis_angle([0,0,2,0]))
    np.testing.assert_allclose(axis_angle([0,0,0,-1]),[0,0,0])
    with pytest.raises(ValueError):axis_angle([0,0,0,0])


class FakeSession:
    def __init__(self,*args,**kwargs):self.sent=[];self.closed=False;self.received=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def rpc(self,method,params,*,timeout=None):
        self.sent.append((method,params))
        return {'thread':{'id':'t'}} if method=='thread/start' else {'turn':{'id':'u'}}
    def receive(self,*args):
        if self.received:raise AssertionError('Should not ask model again within action')
        self.received=True
        return {'id':7,'method':'item/tool/call','params':{'threadId':'t','turnId':'u','callId':'c',
                'tool':'move_arms','arguments':{'note':'probe','targets':{'left_z':.9},'steps':3}}}
    def send(self,message):self.sent.append(message)
    def request(self,method,params):self.sent.append((method,params))


def test_official_step_boundary_terminal_feedback_and_no_privileged_input(monkeypatch,tmp_path):
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake-isolated-transport')
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(tmp_path/'export'))
    monkeypatch.setattr(module,'CodexSession',FakeSession)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    policy=module.WorldPolicy(tmp_path,'unused','turning_on_radio')
    policy.bind(profile());policy.reset()
    actions=[policy.forward(obs()) for _ in range(3)]
    for a in actions: assert a.shape==(1,21)
    session=policy.session
    assert not any(isinstance(v,dict) and v.get('id')==7 for v in session.sent)
    policy.finish(obs(),{'success':True,'q_score':{'final':1.}})
    assert session.closed
    result=json.loads((tmp_path/'rollout-000/episode.json').read_text())
    assert result['env_steps']==3 and result['events'][0]['executed_steps']==3
    assert result['official_success'] is True
    assert 'MUST_NOT_LEAK' not in repr(session.sent)
    params=next(v[1] for v in session.sent if isinstance(v,tuple) and v[0]=='thread/start')
    from environment.benchmarks.action_contracts import action_contract
    assert action_contract('behavior_1k') in params['baseInstructions']
    assert all(action_contract('behavior_1k') in t['description'] for t in params['dynamicTools'])
    assert params['config']['features.shell_tool'] is True
    assert params['config']['features.code_mode'] is True
    assert params['cwd']=='/workspace'
    assert 'give_up' not in {tool['name'] for tool in params['dynamicTools']}
    assert 'MUST_NOT_LEAK' not in (tmp_path/'export/rollout-000/0/observation.txt').read_text()
    records=[json.loads(x) for x in (tmp_path/'rollout-000/events/environment.jsonl').read_text().splitlines()]
    assert sum(x['kind']=='action_completed' for x in records)==3
    assert sum(x['kind']=='action_requested' for x in records)==3
    assert (tmp_path/'rollout-000/events/no-images/tools.jsonl').exists()


def test_proprio_rejects_batch_or_shape_change():
    with pytest.raises(ValueError):split_proprio([1,2],{'arm':3})


def test_benchmark_tools_separate_base_arms_grippers_and_torso():
    from environment.benchmarks.behavior_1k.control import tool_specs,validate_tool
    p=profile();p['trunk_limits']=[[-1]*4,[1]*4]
    assert {x['name'] for x in tool_specs(p)}=={'move_base','move_arms','set_grippers','move_torso','move_robot'}
    for name,targets in [('move_base',{'base_vx':.1}),('move_arms',{'right_z':.8}),
                         ('set_grippers',{'left_gripper':0}),('move_torso',{'trunk_qpos':[0]*4})]:
        validate_tool(name,{'note':'test','targets':targets,'steps':2},p)
    with pytest.raises(ValueError):validate_tool('move_base',{'note':'test','targets':{'left_z':.8},'steps':2},p)
    state=split_proprio(obs()['robot::proprio'],p['proprio_fields'])
    with pytest.raises(ValueError):action_for({'trunk_qpos':[2]*4},state,p,{'left':1,'right':1})


def test_native_joint_robot_is_accepted_only_for_observation_probe():
    from environment.integrations.behavior_eval import robot_profile
    robot = types.SimpleNamespace(
        name='robot', base_control_idx=np.arange(3), trunk_control_idx=np.arange(4),
        trunk_joint_names=['trunk_0','trunk_1','trunk_2','trunk_3'],
        proprioception_dim=3, action_dim=23,
        control_limits={'position': (np.full(4, -1), np.ones(4))},
        controller_action_idx={'base': np.arange(3), 'trunk': np.arange(3, 7),
            'arm_left': np.arange(7, 14), 'gripper_left': np.array([14]),
            'arm_right': np.arange(15, 22), 'gripper_right': np.array([22]), 'camera': np.array([], dtype=int)})
    config = {'proprio_obs': ['base_qvel'],
              'controller_config': {'arm_left': {'name': 'JointController'}},
              'eval': {'camera_sensor_names': {}}}
    declared=robot_profile(robot, config, observation_only=True)
    assert declared['action_dim'] == 23
    assert declared['trunk_joint_names'] == robot.trunk_joint_names
    with pytest.raises(ValueError, match='arm_left action dimension'):
        robot_profile(robot, config)


def test_give_up_and_final_answer_do_not_end_active_episode(monkeypatch,tmp_path):
    class ContinuingSession(FakeSession):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.turns=0;self.index=0
        def rpc(self,method,params,*,timeout=None):
            self.sent.append((method,params))
            if method=='thread/start':return {'thread':{'id':'t'}}
            self.turns+=1
            return {'turn':{'id':f'u{self.turns}'}}
        def receive(self,*args):
            self.index+=1
            if self.index==1:
                return {'id':8,'method':'item/tool/call','params':{'threadId':'t','turnId':'u1',
                    'callId':'quit','tool':'give_up','arguments':{'reason':'grasp failed'}}}
            if self.index==2:
                return {'method':'turn/completed','params':{'threadId':'t','turn':{'id':'u1','status':'completed'}}}
            assert self.index==3
            return {'id':9,'method':'item/tool/call','params':{'threadId':'t','turnId':'u2',
                'callId':'retry','tool':'move_base','arguments':{'note':'change approach','targets':{'base_vx':.1},'steps':1}}}
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake-isolated-transport')
    monkeypatch.setattr(module,'CodexSession',ContinuingSession)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    p=module.WorldPolicy(tmp_path,'unused','clean_up_your_desk',step_budget=5000)
    p.bind(profile());p.reset()
    action=p.forward(obs());session=p.session
    assert p.stopped is None and p.step==0 and p.calls==1
    assert action[0,0]==pytest.approx(.1)  # No idle step substituted for a retry.
    assert session.turns==2
    assert len([x for x in session.sent if isinstance(x,tuple) and x[0]=='thread/start'])==1
    assert next(x for x in session.sent if isinstance(x,dict) and x.get('id')==8)['result']['success'] is False
    assert [x[1]['threadId'] for x in session.sent if isinstance(x,tuple) and x[0]=='turn/start']==['t','t']
    p.finish(obs(),{'success':False})
    assert p.result['env_steps']==1 and p.result['events'][0]['executed_steps']==1


@pytest.mark.parametrize('status',['failed','interrupted'])
def test_runtime_failure_or_interrupt_is_not_restarted(monkeypatch,tmp_path,status):
    class FailedSession(FakeSession):
        def receive(self,*args):
            return {'method':'turn/completed','params':{'threadId':'t','turn':{
                'id':'u','status':status,'error':{'message':'runtime unavailable'}}}}
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake-isolated-transport')
    monkeypatch.setattr(module,'CodexSession',FailedSession)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    p=module.WorldPolicy(tmp_path,'unused','clean_up_your_desk');p.bind(profile());p.reset()
    with pytest.raises(RuntimeError, match='Agent stopped before native episode end'):
        p.forward(obs())
    assert p.last_action is None  # Runtime failure must not generate silent hold steps.
    assert p.stopped=='agent_'+status
    assert p.result['agent_turn_error']['message']=='runtime unavailable'
    p.finish(obs())


def test_combined_tool_all_components_and_profile_gating():
    from environment.benchmarks.behavior_1k.control import tool_specs, validate_tool
    p=profile();p['trunk_limits']=[[-1]*4,[1]*4]
    state=split_proprio(obs()['robot::proprio'],p['proprio_fields'])
    targets={'left_x':.4,'left_z':.9,'left_quat_xyzw':[0,0,1,1],
             'right_y':-.3,'right_quat_xyzw':[0,0,0,1],
             'left_gripper':0,'right_gripper':.25,'base_vx':.1,'base_vy':-.1,
             'base_wz':.2,'trunk_qpos':[.1,.2,.3,.4]}
    args={'note':'combined','targets':targets,'steps':3}
    validate_tool('move_robot',args,p)
    action,grip=action_for(targets,state,p,{'left':1,'right':1})
    np.testing.assert_allclose(action[:3],[.1,-.1,.2])
    np.testing.assert_allclose(action[3:6],[.4,.2,.9])
    np.testing.assert_allclose(action[6:9],[0,0,np.pi/2])
    np.testing.assert_allclose(action[10:13],[.3,-.3,.8])
    np.testing.assert_allclose(action[17:21],[.1,.2,.3,.4])
    assert action[9]==-1 and action[16]==-.5
    held,_=action_for({'right_z':.7},state,p,grip)
    np.testing.assert_allclose(held[:3],0)
    np.testing.assert_allclose(held[17:21],state['trunk_qpos'])
    assert held[9]==-1 and held[16]==-.5
    p['controller_indices']['trunk']=[]
    specs={s['name']:s for s in tool_specs(p)}
    assert 'move_torso' not in specs
    assert 'trunk_qpos' not in specs['move_robot']['inputSchema']['properties']['targets']['properties']
    with pytest.raises(ValueError):validate_tool('move_robot',args,p)


@pytest.mark.parametrize('targets',[
    {}, {'base_vx':True}, {'left_x':'0.4'}, {'base_vy':.31},
    {'left_z':float('nan')}, {'right_x':2.1}, {'left_quat_xyzw':[0,0,0,0]},
    {'right_quat_xyzw':[0,0,1]}, {'trunk_qpos':[0]*3},
    {'trunk_qpos':[0,0,0,2]}, {'camera':0},
])
def test_combined_tool_rejects_invalid_targets(targets):
    from environment.benchmarks.behavior_1k.control import validate_tool
    p=profile();p['trunk_limits']=[[-1]*4,[1]*4]
    with pytest.raises(ValueError):
        validate_tool('move_robot',{'note':'invalid','targets':targets,'steps':2},p)


def test_combined_action_uses_one_segment_and_one_feedback(monkeypatch,tmp_path):
    class CombinedSession(FakeSession):
        def receive(self,*args):
            m=super().receive(*args)
            m['params']['tool']='move_robot'
            m['params']['arguments']['targets']={'left_z':.9,'right_z':.7,'base_vx':.1,
                'trunk_qpos':[.1]*4,'left_gripper':0}
            return m
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake-isolated-transport')
    monkeypatch.setattr(module,'CodexSession',CombinedSession)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    p=profile();p['trunk_limits']=[[-1]*4,[1]*4]
    policy=module.WorldPolicy(tmp_path,'unused','combined_probe');policy.bind(p);policy.reset()
    for _ in range(3):
        a=policy.forward(obs())[0]
        np.testing.assert_allclose(a[[0,5,12,17,9]],[.1,.9,.7,.1,-1])
    policy.finish(obs(),{'success':False})
    assert policy.result['env_steps']==3 and policy.calls==1
    assert len(policy.result['events'])==1 and policy.result['events'][0]['executed_steps']==3


def make_program_policy(monkeypatch, tmp_path, code, max_steps=3, enabled=True):
    class ProgramSession(FakeSession):
        def receive(self,*args):
            if not self.received:
                self.received=True
                return {'id':7,'method':'item/tool/call','params':{'threadId':'t','turnId':'u','callId':'p',
                    'tool':'coding_control','arguments':{'note':'feedback probe','code':code,'max_steps':max_steps}}}
            return {'id':8,'method':'item/tool/call','params':{'threadId':'t','turnId':'u','callId':'fallback',
                'tool':'move_base','arguments':{'note':'after program','targets':{'base_vx':.12},'steps':2}}}
    monkeypatch.setenv('WORLD_CODEX_SOCKET','fake-isolated-transport')
    monkeypatch.setattr(module,'CodexSession',ProgramSession)
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=lambda x:x))
    p=module.WorldPolicy(tmp_path,'unused','clean_up_your_desk',coding_control_enabled=enabled,step_budget=32126)
    pr=profile();pr['trunk_limits']=[[-1]*4,[1]*4]
    p.bind(pr);p.reset()
    return p


def test_program_fresh_feedback_budget_and_terminal_archive(monkeypatch,tmp_path):
    code="""def control(obs, memory):
    memory['ticks'] = memory.get('ticks', 0) + 1
    memory['last_z'] = obs['proprio']['eef_left_pos'][2]
    return {'targets': {'left_z': obs['proprio']['eef_left_pos'][2] + 0.01,
                        'base_vx': 0.1, 'trunk_qpos': [0.1, 0.1, 0.1, 0.1], 'left_gripper': 0}}
"""
    p=make_program_policy(monkeypatch,tmp_path,code,max_steps=2)
    assert p.forward(obs())[0,5]==pytest.approx(.81)
    worker=p.pending['worker']; session=p.session
    updated=obs();updated['robot::proprio'][0,2]=.85
    assert p.forward(updated)[0,5]==pytest.approx(.86)
    assert worker.memory['last_z']==.85  # Measured feedback, not a cached starting frame.
    assert p.forward(updated)[0,0]==pytest.approx(.12)  # Budget exhausted: ordinary next tool.
    assert worker.worker.process.poll() is not None
    assert p.result['events'][0]['executed_steps']==2
    assert p.result['events'][0]['reason']=='segment_complete'
    p.finish(updated,{'success':False})
    assert p.result['env_steps']==3
    root=tmp_path/'rollout-000'
    assert (root/'programs/program-0001.py').read_text()==code
    events=[json.loads(x) for x in (root/'events/environment.jsonl').read_text().splitlines()]
    ticks=[e['payload'] for e in events if e['kind']=='program_tick']
    assert len(ticks)==2 and ticks[1]['observation']['program_step']==1
    assert ticks[1]['observation']['nominal_episode_steps_remaining']==32125
    assert 'MUST_NOT_LEAK' not in json.dumps(ticks)
    assert len(ticks[0]['validated_native_action'])==21
    assert ticks[0]['observation']['depth']['head']['values_m'][0][0]==1
    assert (root/'events/no-images/environment.jsonl').exists()
    params=next(x[1] for x in session.sent if isinstance(x,tuple) and x[0]=='thread/start')
    assert 'coding_control' in {x['name'] for x in params['dynamicTools']}


@pytest.mark.parametrize('code,reason,ok',[
    ("def control(obs, memory):\n    return {'done': True}",'program_done',True),
    ("def control(obs, memory):\n    return {'targets': {'base_vx': 50}}",'program_error',False),
    ("def control(obs, memory):\n    return {'targets': {'hidden_object_pose': 0}}",'program_error',False),
])
def test_program_done_or_invalid_action_never_advances_a_hidden_step(monkeypatch,tmp_path,code,reason,ok):
    p=make_program_policy(monkeypatch,tmp_path,code)
    assert p.forward(obs())[0,0]==pytest.approx(.12)
    assert p.step==0 and p.result['events'][0]['executed_steps']==0
    assert p.result['events'][0]['reason']==reason
    reply=next(x for x in p.session.sent if isinstance(x,dict) and x.get('id')==7)
    assert reply['result']['success'] is ok
    p.finish(obs(),{'success':False})
    assert p.step==1


def test_program_mask_rejects_calls_and_native_end_closes_worker(monkeypatch,tmp_path):
    code="def control(obs, memory):\n    return {'targets': {'base_vx': 0.1}}"
    p=make_program_policy(monkeypatch,tmp_path/'off',code,enabled=False)
    assert p.forward(obs())[0,0]==pytest.approx(.12)
    params=json.loads((p.run_dir/'prompt.json').read_text())
    assert 'coding_control' not in {x['name'] for x in params['dynamicTools']}
    assert not (p.run_dir/'programs').exists()
    p.finish(obs(),{'success':False})
    p=make_program_policy(monkeypatch,tmp_path/'on',code,max_steps=600)
    p.forward(obs());worker=p.pending['worker']
    p.finish(obs(),{'success':True})
    assert worker.worker.process.poll() is not None
    assert p.result['events'][0]['executed_steps']==1
    assert p.result['events'][0]['reason']=='environment_end'
    assert p.result['official_success'] is True


def test_program_latched_targets_base_defaults_and_worker_boundary():
    from environment.benchmarks.behavior_1k.code_control import FeedbackController, depth_grid, validate_program
    state=split_proprio(obs()['robot::proprio'],profile()['proprio_fields'])
    code="""def control(obs, memory):
    if obs['program_step'] == 0:
        return {'targets': {'left_z': 0.95, 'left_gripper': 0, 'base_vx': 0.1}}
    return {'targets': {'right_z': 0.7}}
"""
    pr=profile();pr['trunk_limits']=[[-1]*4,[1]*4]
    program=FeedbackController({'note':'test','max_steps':2,'code':code},state,pr)
    try:
        grip={'left':1,'right':1}
        first=program.observation(state,{},grip,0,0,2)
        _,grip,_=program.tick(first,state,grip)
        second=program.observation(state,{},grip,1,1,2)
        a,grip,_=program.tick(second,state,grip)
        assert a[5]==pytest.approx(.95) and a[12]==pytest.approx(.7)
        assert a[0]==0 and grip['left']==0
    finally:program.close()
    for bad in (0,601,True):
        with pytest.raises(ValueError): validate_program({'note':'x','max_steps':bad,'code':code})
    for code in ("import os\ndef control(obs, memory): return {}", "def control(obs, memory): return open('/etc/passwd').read()"):
        try:
            bad=FeedbackController({'note':'x','max_steps':1,'code':code},state,profile())
        except ValueError:continue
        try:
            with pytest.raises(ValueError): bad.tick(first,state,grip)
        finally:bad.close()
    grid=depth_grid({'head_depth_linear':np.array([[0,np.nan],[np.inf,2.]])})
    assert grid['head']['values_m']==[[None,None],[None,2.]]
