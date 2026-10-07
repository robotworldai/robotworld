"""Action descriptions must preserve real ordering, units and clipping semantics."""
from types import SimpleNamespace
import pytest

from environment.benchmarks.native_project.action_prompt import build_action_guide
from environment.benchmarks.native_project.policy import Agent


def metadata():
    return {'dim':2,'lower':[None,None],'upper':[None,None],'terms':[
        {'dim':2,'name':'joints','config':{'class_type':'mdp:JointPositionAction',
          'clip':{'second_joint':[-.1,.3]}},
         'resolved_joint_names':['second_joint','first_joint'],
         'scale':[[.5,-.25]],'offset':[[.29,-.2]]}]}


def test_actual_order_negative_scale_and_processed_clip():
    text = build_action_guide('T01', metadata(), .02)
    assert '| 0 | second_joint | rad | [-inf, +inf] | 0.29 | 0.5 | 0.29 | 0.3 | 0.3 | [-0.1,0.3] rad |' in text
    assert '| 1 | first_joint | rad | [-inf, +inf] | -0.2 | -0.25 | -0.2 | -0.225 | -0.45 | none configured |' in text
    assert 'NOT recommended motions' in text
    assert 'steps=10 lasts 0.2' in text


def test_missing_resolved_names_do_not_guess_regex_joint_order():
    data = metadata()
    data['terms'][0]['resolved_joint_names'] = None
    with pytest.raises(ValueError,match='Resolved joint order'):
        build_action_guide('T01',data,.02)


def test_ttrl_input_clipping_not_processed_clipping():
    data={'dim':1,'lower':[-100],'upper':[100],'names':['hip'],
          'scale':.25,'default_joint_position':[.2]}
    text=build_action_guide('T02',data,.02)
    assert 'BEFORE scaling' in text
    assert '| 0 | hip | rad | [-100, 100] | 0.2 | 0.25 | 0.2 | 0.225 | 0.45 | none configured |' in text


@pytest.mark.parametrize('task',['T05','T16','T17'])
def test_rotor_thrust_and_observed_throttle_are_different_quantities(task):
    data={'dim':4,'names':[f'rotor_{i}' for i in range(4)],'lower':[-1]*4,'upper':[1]*4}
    text=build_action_guide(task,data,1/62)
    assert 'u=0 requests 50%' in text and 'Zero is NOT hover' in text
    assert '((v+1)/2)^2' in text
    assert 'sqrt(clip((u+1)/2,0,1))' in text


def test_vmc_zero_leg_length_and_limit_asymmetry():
    data={'dim':6,'lower':[-1]*6,'upper':[1]*6,
          'control_names':['left_angle','left_length','left_wheel','right_angle','right_length','right_wheel']}
    text=build_action_guide('T07',data,.01)
    assert '| 1 | left_length | length=clip(0.237+0.06*u,0.18,0.30) m | 0.18 / 0.237 / 0.297 m |' in text
    assert 'NOT six raw joint angles' in text


def test_absolute_eef_and_delayed_increment_contracts():
    eef=build_action_guide('T03',{'dim':8},.04)
    assert '[0,0,0,0] is invalid' in eef
    assert 'gripper | >0 opens' in eef
    inc=build_action_guide('T04',{'dim':1,'lower':[-1/120],'upper':[1/120]},1/60)
    assert '10 applied steps adds +0.01 m/s' in inc
    assert '15-control-step delay' in inc


def test_guide_is_in_actual_prompt_builder_without_evaluator_data(tmp_path):
    sim=SimpleNamespace(action_metadata=metadata(),dt=.025,policy_images={},
                        observation_metadata={'allowed':'policy'},last_evaluation={'secret':'judge'})
    agent=Agent(sim,tmp_path,None,'unused',100,60,'native task',coding=False,task_id='T01')
    prompt=agent.build_prompt()
    assert agent.action_guide in prompt['baseInstructions']
    assert 'coding_control is disabled' in prompt['baseInstructions']
    assert 'secret' not in prompt['baseInstructions']
    assert [x['name'] for x in prompt['dynamicTools']]==['observe','apply_action']


def test_existing_driving_and_soccer_controls_keep_mode_specific_zero():
    from environment.benchmarks.wheeledlab.control import action_guide as car
    from environment.benchmarks.humanoid_soccer.control import action_guide as soccer
    assert 'cannot reverse' in car(.02,False)
    assert 'requests -3/-0.3 m/s (reverse)' in car(.02,True)
    guide=soccer(['hip'],[[-.3,.7]],'hybrid',.02)
    assert 'zero leaves that joint unmodified' in guide
    assert '| 0 | hip | [-0.3,0.7] |' in guide
