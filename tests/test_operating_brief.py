import json
from types import SimpleNamespace
import pytest
from environment.benchmarks.operating_brief import operating_brief, SCENES, NATIVE_SCENES
from environment.benchmarks.native_project.policy import Agent
from environment.benchmarks.native_project.review_scene import camera_offsets


@pytest.mark.parametrize('key', list(SCENES))
def test_benchmark_brief_keeps_information_and_termination_boundaries(key):
    brief=operating_brief(key)
    assert 'Scene and surroundings:' in brief
    assert 'Robot and interaction geometry:' in brief
    assert 'native failure and timeout remain terminal' in brief
    assert 'unknown unless explicitly supplied' in brief


@pytest.mark.parametrize('task', list(NATIVE_SCENES))
def test_native_brief_in_actual_prompt_without_private_state(tmp_path,task):
    sim=SimpleNamespace(action_metadata={'dim':4,'lower':[-1.]*4,'upper':[1.]*4,
                       'names':['rotor_0','rotor_1','rotor_2','rotor_3']},dt=.02)
    # Use direct brief construction for unrelated custom action encoders; this
    # test checks prompt assembly rather than their separately tested mechanics.
    agent=Agent(sim,tmp_path,None,'test',4,60,'Native task contract.',task_id=None)
    agent.instructions+='\n'+operating_brief('native_project',task)
    prompt=agent.build_prompt()
    assert NATIVE_SCENES[task][0] in prompt['baseInstructions']
    assert [t['name'] for t in prompt['dynamicTools']]==['observe','apply_action','coding_control']
    assert 'RECOVERY DIAGNOSTIC OVERRIDE' not in json.dumps(prompt)


def test_driving_and_balancing_are_not_conflated():
    assert 'not an ordinary four-wheel supported car' in operating_brief('native_project','T09')
    assert 'no carried box' in operating_brief('native_project','T13')
    assert 'independent of T05 1v1' in operating_brief('native_project','T05-single')
    assert sum(v*v for v in camera_offsets('T11')[0]) < 2.5**2+2.5**2+2**2


def test_robocasa_current_plus_four_history(tmp_path,monkeypatch):
    import numpy as np
    from environment.benchmarks.robocasa.policy import Policy
    from environment.benchmarks.robocasa.control import CAMERAS, STATE_KEYS
    monkeypatch.setattr('environment.benchmarks.robocasa.policy.CodexSession',lambda *args: None)
    policy=Policy(None,tmp_path)
    obs={k:np.zeros((2,2,3),dtype=np.uint8) for k in CAMERAS}
    obs.update({k:np.zeros(3) for k in STATE_KEYS})
    obs['annotation.human.task_description']='test'
    for i in range(10):
        policy.ingest(obs,i*3)
        policy.content()
    selected=json.loads((tmp_path/'frames/9/history.json').read_text())
    assert selected==[3,9,15,21,27]
