import math
from types import SimpleNamespace
import numpy as np
from third_party.benchmarks.wheeledlab.robotworld.precision_specs import scenario,NAMES
from third_party.benchmarks.wheeledlab.robotworld.precision_scoring import Evaluator
from third_party.benchmarks.wheeledlab.robotworld.precision_prompt import task
from environment.benchmarks.wheeledlab.control import specs
from environment.benchmarks.wheeledlab.policy import Agent


def state(step,x,y,yaw=0,z=0,body=0):
    return dict(step=step,time=step*.02,position=[x,y,z],yaw=yaw,roll=0.,pitch=0.,
                linear_velocity=[body*math.cos(yaw),body*math.sin(yaw),0],body_velocity=[body,0,0])


def test_painted_sides_and_aisle_boundary_are_strict():
    s=scenario('rw-reverse-bay')
    for x,y,yaw in [(.065,-.65,math.pi/2),(0,.93,0)]:
        e=Evaluator(s);e.update(state(1,x,y,yaw));assert e.failure.startswith('forbidden_line_touch:')
    e=Evaluator(s);e.update(state(1,0,-.12,math.pi/2))
    assert not e.failure  # Dashed entry is the only traversable bay boundary.


def test_reverse_bay_success_requires_real_reverse_entry_and_dwell():
    s=scenario('rw-reverse-bay');e=Evaluator(s)
    e.previous=state(0,0,.1,math.pi/2)
    for i in range(1,76):e.update(state(i,0,.1-.01*i,math.pi/2,body=-.5))
    assert e.reverse_entry and e.reverse_inside>.45 and not e.terminal
    for i in range(76,126):e.update(state(i,0,-.65,math.pi/2))
    assert e.success,e.report()
    e=Evaluator(s)
    for i in range(1,100):e.update(state(i,0,-.65,math.pi/2))
    assert not e.success  # Merely being at the final pose is not the task.


def test_painted_line_rule_uses_ground_projection_at_realistic_root_heights():
    s=scenario('rw-reverse-bay')
    for z in [-.0065,.04,.12]:
        e=Evaluator(s);e.update(state(1,.065,-.65,math.pi/2,z=z))
        assert e.failure.startswith('forbidden_line_touch:')
        e=Evaluator(s);e.update(state(1,0,-.12,math.pi/2,z=z))
        assert not e.failure  # White dashed entrance remains permitted.


def test_bridge_tread_cannot_touch_painted_rail_edges():
    s=scenario('rw-twin-beam');p=state(1,1,0,z=.3)
    p['wheel_positions']={f'{end}_{side}_wheel_link':[1+dx,dy,.3488] for end,dx in [('front',.1385),('back',-.158)] for side,dy in [('left',.115),('right',-.115)]}
    e=Evaluator(s);e.update(p);assert not e.failure
    p['wheel_positions']['front_left_wheel_link'][1]+=.02
    e=Evaluator(s);e.update(p);assert e.failure=='tire_touched_rail_edge:front_left_wheel_link'


def test_precision_task_does_not_supply_coordinates_and_reverse_is_explicit():
    for n in NAMES:
        s=scenario(n);assert s['reverse_enabled'] and s['steps']==2000
        for k in ['start','goal','bay','bridge','obstacles','forbidden_lines']:s[k]='PRIVATE'
        assert 'PRIVATE' not in task(s)
    tools=specs(onboard=True,reverse=True)
    assert 'clamped' not in tools[1]['description'] and 'reverse wheel motion' in tools[1]['description']


def test_only_vehicle_camera_images_are_given_to_agent(tmp_path):
    front=np.full((360,640,3),50,dtype=np.uint8);rear=np.full((360,640,3),150,dtype=np.uint8)
    sim=SimpleNamespace(case='rw-parallel-park',steps=0,spec={'reverse_enabled':True},policy_sensor=True,
        policy_image=front,rear_sensor=SimpleNamespace(image=rear),observation=lambda:{'sensors':{},'control_step':0})
    agent=Agent(sim,tmp_path,None,'test',2000,10);parts=agent.content()
    assert len([x for x in parts if x['type']=='inputImage'])==2
    assert any(x.get('text')=='FRONT onboard' for x in parts)
    assert any(x.get('text')=='REAR onboard (not mirrored)' for x in parts)
    assert not (tmp_path/'observations/0/camera.mp4').exists()
