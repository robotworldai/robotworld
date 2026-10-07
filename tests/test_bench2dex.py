"""Native budget, observation boundary and physics-rate metric regression checks."""
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import subprocess
import numpy as np
import pytest

from environment.benchmarks.bench2dex.project import ROOT, Simulator
from environment.benchmarks.native_project.control import validate_action


def test_budgets_match_official_script():
    tasks=json.loads((ROOT/'tasks.json').read_text())
    for task in tasks.values():
        text=subprocess.check_output(['bash','-c',
            'source script/eval_budget.sh; eval_budget_compute "$1"; printf "%s %s" "$EVAL_EPISODE_STEPS" "$EVAL_MAX_STEPS"',
            'budget',task['id']],cwd=ROOT/'checkout',text=True)
        control,physics=map(int,text.split())
        assert task['steps']==control
        assert control*3==physics


def test_observation_excludes_evaluator_state():
    sim=Simulator.__new__(Simulator)
    sim.steps=3;sim.physics_steps=9;sim.physics_dt=1/60;sim.robot=object();sim.active=object()
    sim.up=SimpleNamespace(read_joint_state=lambda _: {'qpos':np.array([.1,.2]),'private_object':123},
                           select_active=lambda q,a:q)
    sim.last_evaluation={'hidden_stage':8}
    obs=sim.observation()
    assert set(obs)=={'control_step','physics_steps','elapsed_sim_seconds','qpos'}
    assert obs['qpos']==[.1,.2]


@pytest.mark.parametrize('success_at,expected_ticks',[(None,3),(2,2)])
def test_metrics_every_physics_tick_and_native_early_stop(success_at,expected_ticks):
    sim=Simulator.__new__(Simulator)
    sim.done=False;sim.steps=0;sim.physics_steps=0;sim.physics_dt=1/60;sim.np=np
    sim.robot=object();sim.active=object();sim.contacts=None;sim.objects={};sim.object_ids=[];sim.limits={}
    sim.action_metadata={'dim':1,'lower':[-1.],'upper':[1.]};sim.task_spec={'steps':10}
    class Target:
        dtype='float';device='cpu'
        def __setitem__(self,index,value):self.value=value
    sim.hold={id(sim.robot):Target()}
    sim.torch=SimpleNamespace(as_tensor=lambda x,**kw:x)
    sim.up=SimpleNamespace(expand_to_full=lambda raw,a:raw,_joint_command_context=lambda *args:{},
        _get_object_states=lambda *args:{},read_joint_state=lambda *args:{})
    calls=[]
    class Tracker:
        success=False
        def update(self,*args,**kw):
            calls.append(kw['sim_step'])
            self.success=len(calls)==success_at
    sim.tracker=Tracker();sim._physics=lambda:None;sim._capture=lambda:None;sim.observation=lambda:{'steps':sim.steps}
    sim.native_events=SimpleNamespace(write=lambda *args:None)
    sim.step([.25])
    assert calls==list(range(expected_ticks))
    assert sim.physics_steps==expected_ticks and sim.steps==1
    assert sim.done==(success_at is not None)
    assert sim.hold[id(sim.robot)].value.tolist()==[.25]


def test_joint_action_rejects_wrong_dimension_and_nan():
    metadata={'dim':2,'lower':[-1.,-1.],'upper':[1.,1.]}
    for value in ([0.], [float('nan'),0.], [2.,0.]):
        with pytest.raises(ValueError):validate_action(value,metadata)


@pytest.mark.parametrize('steps,success,reason',[(4,False,'external_stop'),(10,False,'max_steps'),(4,True,'stable_success')])
def test_short_probe_does_not_claim_native_budget_exhaustion(steps,success,reason):
    sim=Simulator.__new__(Simulator)
    sim.steps=steps;sim.physics_steps=steps*3;sim.physics_dt=1/60
    sim.task_spec={'steps':10};sim.robot_key='test';sim.seed=1;sim.profile='smoke';sim.anchor=None
    @dataclass
    class Report:
        terminated_reason:str
    def finalize(**kwargs):
        assert kwargs['max_steps']==30
        return Report(kwargs['terminated_reason'])
    sim.tracker=SimpleNamespace(success=success,finalize=finalize)
    result=sim.result()
    assert result['native_metrics']['terminated_reason']==reason
    assert result['success']==success
