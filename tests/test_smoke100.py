import json
import random
import pytest
from agents.smoke100 import command_for


def install_probe_client(monkeypatch, packet, modalities=None, *, stop_at=135, termination='budget_exhausted', horizon=135):
    class Client:
        step = 0
        receipts = {}
        def request(self, method, body=None):
            state = {'episode_id':'test', 'observation_id':str(self.step),
                     'control_step':self.step, 'horizon':horizon, 'terminal':self.step == stop_at,
                     'termination':termination if self.step == stop_at else None}
            if method == 'describe':
                return {**({'observation_modalities': modalities} if modalities else {}),
                        'benchmark':'digit', 'native_horizon':horizon,
                        'tools':[{'name':'apply_action'}],
                        'action_metadata':{'lower':[-1], 'upper':[1]}}
            if method == 'observe':return {**state, 'observation':packet}
            if method == 'status':return state
            if method.startswith('receipts/'):return self.receipts[method.split('/')[1]]
            key = body['request_id']
            if key not in self.receipts:
                steps = min(body['commands'][0]['arguments']['steps'], stop_at - self.step)
                self.step += steps
                self.receipts[key] = {'control_step':self.step,'executed_steps':steps,'execution_uncertain':False}
            return self.receipts[key]
    monkeypatch.setattr('environment.clients.episode.EpisodeClient', Client)


@pytest.mark.parametrize('modalities,packet,error', [
    (None, {'images': [{}]}, None),
    (['state'], {'state': {'speed': 0}, 'images': []}, None),
    (['state'], {'state': {'episode_steps_remaining': 135}}, 'Initial state observation missing'),
    (['state'], {'state': {'dt': 0.01, 'control_step': 0, 'native_policy_terms': {}}}, 'Initial state observation missing'),
    (['state'], {'state': {'native_actor_history': [[]]}}, 'Initial state observation missing'),
    (['state'], {'state': {'native_policy_observation': [0, 0]}}, None),
    (['state', 'images'], {'state': {'speed': 0}, 'images': []}, 'Initial camera images missing'),
    (None, {'state': {'speed': 0}, 'images': []}, 'Initial camera images missing'),
])
def test_native_budget_probe(tmp_path, monkeypatch, modalities, packet, error):
    from agents.smoke100 import run
    install_probe_client(monkeypatch, packet, modalities)
    if error:
        with pytest.raises(AssertionError, match=error):
            run(tmp_path, 'native')
        return
    run(tmp_path, 'native')
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['passed'] and summary['actual_control_steps'] == 135
    assert summary['classification'] == 'infrastructure_probe_not_model_evaluation'


def test_optimized_probe_checks_observations(tmp_path, monkeypatch):
    from pathlib import Path
    from agents import smoke100

    # Compile the probe as the agent container would under PYTHONOPTIMIZE=1.
    path = Path(smoke100.__file__)
    module = {'__name__': 'agents.smoke100', '__file__': str(path)}
    exec(compile(path.read_bytes(), str(path), 'exec', optimize=1), module)
    install_probe_client(monkeypatch, {'state': {'episode_steps_remaining': 1}}, ['state'])
    with pytest.raises(AssertionError, match='Initial state observation missing'):
        module['run'](tmp_path, 1)
    assert json.loads((tmp_path/'summary.json').read_text())['passed'] is False


@pytest.mark.parametrize('horizon,stop_at,termination,passed', [
    (135, 3, 'agent_stopped', False),
    (135, 0, 'agent_stopped', False),
    (135, 3, 'native_failure', True),
    (135, 3, 'budget_exhausted', False),
    (50, 50, 'budget_exhausted', True),
])
def test_smoke_early_stop(tmp_path, monkeypatch, horizon, stop_at, termination, passed):
    from agents.smoke100 import run
    install_probe_client(monkeypatch, {'images': [{}]}, horizon=horizon, stop_at=stop_at, termination=termination)
    if passed:
        run(tmp_path, 100)
    else:
        with pytest.raises(AssertionError):
            run(tmp_path, 100)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['passed'] is passed
    assert summary['actual_control_steps'] == stop_at
    assert summary['early_native_termination'] is passed


def description(bench, tool, **kwargs):
    return dict(benchmark=bench, tools=[{'name':tool}], **kwargs)


def test_native_bounds():
    d=description('digit','apply_action',action_metadata={'lower':[-1,1,None],'upper':[1,2,None]})
    assert command_for(d,{},7)['arguments']['action']==[0,1,0]


def test_random_actions_are_reproducible_and_within_native_bounds():
    from agents.smoke100 import random_command

    spec = description('digit', 'apply_action', action_metadata={'lower': [-1, 1, None], 'upper': [1, 2, None]})

    def sample(seed):
        rng = random.Random(seed)
        return [random_command(spec, {}, 10, None, index, rng)['arguments']['action'] for index in range(10)]

    sequence = sample(23)
    assert sequence == sample(23)
    assert sequence != sample(24)
    assert all(-0.05 <= a <= 0.05 and 1 <= b <= 1.05 and -0.05 <= c <= 0.05 for a, b, c in sequence)


def test_benchmark_actions():
    for bench,tool in [('robocasa','set_gripper'),('robolab','set_gripper'),('behavior_1k','set_grippers'),('ai_cps','move_joints'),('wheeledlab','drive')]:
        result=command_for(description(bench,tool),{},3)
        assert result['tool']==tool and result['arguments']['steps']==3


def test_soccer_holds_public_joint():
    observation={'observation':{'state':{'joint_positions':{'ankle':0.25}}}}
    assert command_for(description('humanoid_soccer','move_joints'),observation,10)['arguments']['joint_positions']=={'ankle':0.25}


def test_dojo_uses_latest_public_pose():
    keys=('left_x','left_y','left_z','left_pitch_deg','left_roll_deg','left_yaw_deg','left_gripper')
    observation={'observation':{'feedback_text':[f'{key}=0.1' for key in keys]+['left_z=0.4']}}
    args=command_for(description('robodojo','move_eef'),observation,10)['arguments']
    assert args['targets']['left_z']==0.405 and 'steps' not in args


def test_dojo_probe_stays_near_initial_pose_instead_of_accumulating():
    keys=('left_x','left_y','left_z','left_pitch_deg','left_roll_deg','left_yaw_deg','left_gripper')
    origin={'observation':{'feedback_text':[f'{key}=0.1' for key in keys]}}
    current={'observation':{'feedback_text':[f'{key}=0.7' for key in keys]}}
    for index in range(20):
        args=command_for(description('robodojo','move_eef'),current,10,origin=origin,call_index=index)['arguments']
        assert abs(args['targets']['left_z']-0.1)<0.005001
        assert args['targets']['left_x']==0.1
