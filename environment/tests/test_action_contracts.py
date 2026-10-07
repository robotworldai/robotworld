"""Cross-check published examples against converters and pinned upstream methods."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from environment.benchmarks.action_contracts import action_contract
from environment.benchmarks.robocasa.control import action_for, tool_specs
from environment.robots.arx_x5 import pose

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT/'third_party/dependencies/robosuite/checkout/robosuite'


def upstream_methods(path, classname, names, namespace):
    """Execute the actual pure calculation methods without starting a simulator."""
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    cls.bases=[ast.Name(id='Parent',ctx=ast.Load())]
    cls.decorator_list=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    tree.body=[cls]
    namespace.update(np=np,Parent=type('Parent',(),{'run_controller':lambda self:None}))
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace[classname]()


def test_robocasa_base_examples_against_upstream_transform_and_actuator_ranges():
    import xml.etree.ElementTree as ET
    xml=ET.parse(SUITE/'models/assets/bases/omron_mobile_base.xml')
    actuators={x.attrib['name']:list(map(float,x.attrib['ctrlrange'].split())) for x in xml.findall('.//actuator/velocity')}
    bounds=np.array([actuators['actuator_mobile_'+name] for name in ['forward','side','yaw']])
    obj=upstream_methods(SUITE/'controllers/parts/mobile_base/joint_vel.py',
                        'MobileBaseJointVelocityController',{'set_goal','run_controller'},
                        {'T':SimpleNamespace(mat2euler=lambda yaw:[0,0,yaw])})
    obj.update=lambda:None;obj.impedance_mode='fixed';obj.qpos_index=[0,1,2]
    obj.joint_dim=3;obj.scale_action=lambda x:x;obj.init_ori=0;obj.interpolator=None
    obj.actuator_min=bounds[:,0];obj.actuator_max=bounds[:,1]
    for yaw,expected in [(0,[.1,0,.15]),(math.pi/2,[0,-.1,.15])]:
        obj.get_base_pose=lambda:(None,yaw)
        obj.set_goal(np.array([.1,0,.1]))
        np.testing.assert_allclose(obj.run_controller(),expected,atol=1e-9)
    assert '[0,-0.1]' in action_contract('robocasa')


def test_robocasa_torso_is_slide_and_uses_measured_position_not_previous_goal():
    import xml.etree.ElementTree as ET
    xml=ET.parse(SUITE/'models/assets/bases/omron_mobile_base.xml')
    torso=xml.find(".//joint[@name='joint_torso_height']")
    assert torso.attrib['type']=='slide' and torso.attrib['axis']=='0 0 1'
    cfg=json.loads((SUITE/'controllers/config/robots/default_pandaomron.json').read_text())
    assert cfg['body_parts']['torso']['type']=='JOINT_POSITION'
    tree=ast.parse((SUITE/'controllers/parts/generic/joint_pos.py').read_text())
    init=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    assert ast.literal_eval(defaults['output_max'])==.05
    assert ast.literal_eval(defaults['output_min'])==-.05
    obj=upstream_methods(SUITE/'controllers/parts/generic/joint_pos.py','JointPositionController',{'set_goal'},
        {'set_goal_position':lambda delta,current,**kw:current+delta})
    obj.update=lambda:None;obj.input_type='delta';obj.impedance_mode='fixed';obj.qpos_index=[0];obj.joint_dim=1
    obj.position_limits=None;obj.interpolator=None;obj.scale_action=lambda x:.05*x
    obj.joint_pos=np.array([.2]);obj.goal_qpos=np.array([.3])
    obj.set_goal(np.array([.1]));np.testing.assert_allclose(obj.goal_qpos,[.205])
    obj.set_goal(np.array([.1]));np.testing.assert_allclose(obj.goal_qpos,[.205])


def test_dojo_special_pitch_and_grasp_point_roundtrip():
    np.testing.assert_allclose(pose.tool_axis(pose.angles_to_quat(0,0,0)),[0,0,-1],atol=1e-6)
    np.testing.assert_allclose(pose.tool_axis(pose.angles_to_quat(90,0,0)),[0,1,0],atol=1e-6)
    values={'x':-.3,'y':.4,'z':.92,'pitch_deg':0,'roll_deg':0,'yaw_deg':10}
    rebuilt=pose.pose_to_values(pose.values_to_pose(values))
    np.testing.assert_allclose([rebuilt[k] for k in values],list(values.values()),atol=1e-6)


def test_tool_contracts_are_shared_and_native_action_is_unchanged():
    from environment.benchmarks.robolab.control import specs as lab
    from environment.benchmarks.ai_cps.control import specs as cps
    from environment.benchmarks.behavior_1k.control import tool_specs as behavior
    for bench,specs in [('robocasa',tool_specs()),('behavior_1k',behavior()),('ai_cps',cps('22'))]:
        for tool in specs:
            if tool['name']!='observe':assert action_contract(bench) in tool['description']
    for mode in ['joint_position','absolute_ik','relative_ik']:
        assert all(action_contract('robolab',mode) in t['description'] for t in lab(mode))
    a,steps,grip=action_for('move_robot',{'note':'example','steps':10,'targets':{'eef_delta':[0,0,.1,0,0,0],'torso':.1}},1)
    np.testing.assert_allclose(a,[0,0,.1,0,0,0,1,0,0,0,.1,0]);assert steps==10 and grip==1


def test_actual_casa_prompt_and_tools_share_contract(monkeypatch,tmp_path):
    from environment.benchmarks.robocasa import policy as module
    monkeypatch.setattr(module, "CodexSession", lambda *args: None)
    Policy=module.Policy
    calls=[]
    p=Policy(None,tmp_path)
    p.session=SimpleNamespace(rpc=lambda method,params:calls.append(params) or {'thread':{'id':'test'}})
    p.content=lambda:[];p.begin_turn=lambda content:None;p.start()
    saved=json.loads((tmp_path/'prompt.json').read_text())
    assert action_contract('robocasa') in saved['baseInstructions']
    assert all(action_contract('robocasa') in t['description'] for t in saved['dynamicTools'])


def test_native_tool_contract_includes_actual_joint_order(tmp_path):
    from environment.benchmarks.native_project.policy import Agent
    data={'dim':2,'lower':[-1,-1],'upper':[1,1],'terms':[{'dim':2,'config':{'class_type':'JointPositionAction'},'resolved_joint_names':['right','left'],'scale':[-.2,.1],'offset':[.3,.4]}]}
    agent=Agent(SimpleNamespace(action_metadata=data,dt=.02),tmp_path,None,'unused',30,30,'test',task_id='T01')
    prompt=agent.build_prompt()
    assert agent.action_guide in prompt['baseInstructions']
    for tool in prompt['dynamicTools']:
        if tool['name']!='observe':assert agent.action_guide in tool['description']


def test_model_facing_contracts_and_driving_titles_are_english():
    import re
    from environment.benchmarks.action_contracts import CONTRACTS
    from environment.benchmarks.wheeledlab.catalog import CASES, MODEL_TITLES
    from environment.benchmarks.wheeledlab.control import action_guide
    assert set(CASES)==set(MODEL_TITLES)
    texts=[action_contract(b) for b in CONTRACTS]
    texts += [action_contract('robolab',mode) for mode in ['joint_position','absolute_ik','relative_ik']]
    texts += list(MODEL_TITLES.values())+[action_guide(.02,True)]
    assert not re.search(r'[\u4e00-\u9fff]', '\n'.join(texts))
