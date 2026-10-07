import math
import pytest
from third_party.benchmarks.wheeledlab.robotworld.specs import scenario,NAMES
from third_party.benchmarks.wheeledlab.robotworld.scoring import Evaluator,gate_state,project,overlaps

def state(step,p,yaw=0,v=(0,0,0),body=None):
    return dict(step=step,time=step*.02,position=list(p),yaw=yaw,roll=0.,pitch=0.,linear_velocity=list(v),body_velocity=list(body or v))

def traverse(spec,drifting=False):
    e=Evaluator(spec);step=0
    for a,b in zip(spec['path'],spec['path'][1:]):
        length=math.dist(a,b);n=max(1,math.ceil(length/.02));yaw=math.atan2(b[1]-a[1],b[0]-a[0])
        for j in range(n):
            step+=1;p=[a[k]+(b[k]-a[k])*(j+1)/n for k in range(3)]
            e.update(state(step,p,yaw,(math.cos(yaw),math.sin(yaw),0),(1,.4 if drifting else 0,0)))
            if e.terminal:return e
    for _ in range(60):step+=1;e.update(state(step,spec['goal'],spec['goal_yaw']))
    return e

def test_baseline_success_requires_route_and_parking():
    spec=scenario('rw-courtyard');e=traverse(spec)
    assert e.success,e.report()
    e=Evaluator(spec)
    for step in range(1,2001):e.update(state(step,spec['start']))
    assert e.failure=='time_limit' and not e.success
    e=Evaluator(spec)
    for step in range(1,100):e.update(state(step,spec['goal']))
    assert not e.success and e.next==0

def test_hairpins_traversable_and_height_checked():
    spec=scenario('rw-hairpins');e=traverse(spec)
    assert e.success,e.report()
    e=Evaluator(spec);e.update(state(1,[spec['path'][20][0],spec['path'][20][1],-.4]))
    assert not e.success and e.failure

def test_teleport_and_wrong_order_do_not_pass():
    spec=scenario('rw-courtyard');e=Evaluator(spec);e.update(state(1,spec['start']));e.update(state(2,spec['goal']))
    assert e.failure=='discontinuous_pose'

def test_drift_tracking_without_side_slip_does_not_pass():
    spec=scenario('rw-drift-switch');e=traverse(spec,False)
    assert not e.success
    e=traverse(spec,True)
    assert e.success,e.report()
    assert e.completed_laps==1

def test_gate_clearance_and_stop_are_independent():
    spec=scenario('rw-gate-dock');assert not gate_state(spec,0)['clear'] and gate_state(spec,4)['clear']
    e=Evaluator(spec);e.update(state(199,[-.01,0,0],v=(.2,0,0)));e.update(state(200,[.01,0,0],v=(.2,0,0)))
    assert e.failure=='missed_mandatory_stop'
    e=Evaluator(spec)
    for i in range(1,26):e.update(state(i,[-1.8,0,0]))
    assert e.stop_done

def test_obstacle_safety_envelope_rotation():
    o=dict(center=[0,0,.5],size=[1,1,1])
    assert overlaps([.7,0,0],0,[.3,.18],o,.015)
    assert not overlaps([2,0,0],0,[.3,.18],o,.015)

def test_protocol_fixed_before_model_evaluation():
    specs=[scenario(n) for n in NAMES]
    assert [s['difficulty'] for s in specs].count('hard_target')==3
    assert all(s['steps']==2000 and s['dt']==.02 for s in specs)

def test_gate_valid_stop_window_and_dock_can_succeed():
    spec=scenario('rw-gate-dock');e=Evaluator(spec);step=0
    def sample(p,yaw=0,v=(0,0,0)):
        nonlocal step
        step+=1;e.update(state(step,p,yaw,v));assert not e.failure,e.report()
    def straight(a,b,speed):
        n=math.ceil(math.dist(a,b)/(speed*.02));yaw=math.atan2(b[1]-a[1],b[0]-a[0])
        for j in range(n):sample([a[k]+(b[k]-a[k])*(j+1)/n for k in range(3)],yaw,(speed*math.cos(yaw),speed*math.sin(yaw),0))
    straight(spec['start'],[-1.8,0,0],1.)
    for _ in range(26):sample([-1.8,0,0])
    straight([-1.8,0,0],[-.6,0,0],.5)
    while not 3.3<=(step*.02)%6<=3.4:sample([-.6,0,0])
    straight([-.6,0,0],[.8,0,0],.5)
    straight([.8,0,0],[2,0,0],1.)
    start=spec['path'].index([2.,0.,0.])
    for a,b in zip(spec['path'][start:],spec['path'][start+1:]):straight(a,b,1.)
    for _ in range(51):sample(spec['goal'])
    assert e.success,e.report()
