import pytest
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
from environment.benchmarks.humanoid_soccer.control import coding_spec


def test_program_uses_new_observations_and_persists_memory():
    p=ControllerProgram('''import math
def blend(a, b, t):
    return a + (b-a)*t
def control(obs, memory):
    memory["ticks"] = memory.get("ticks", 0) + 1
    q = {name: blend(angle, 0, 0.1) for name, angle in obs["joint_positions"].items()}
    q["hip"] += math.sin(obs["time_s"])*0.01
    if obs["stop"]:
        return {"done": True}
    return {"joint_positions": q}
''')
    try:
        first=p.request({'obs':{'joint_positions':{'hip':.2},'time_s':0,'stop':False}})
        second=p.request({'obs':{'joint_positions':{'hip':.4},'time_s':0,'stop':False}})
        assert first['action']['joint_positions']['hip']==pytest.approx(.18)
        assert second['action']['joint_positions']['hip']==pytest.approx(.36)
        assert second['memory']=={'ticks':2}
        assert p.request({'obs':{'joint_positions':{'hip':.4},'time_s':0,'stop':True}})['action']=={'done':True}
    finally:p.close()
    assert p.process.poll() is not None


@pytest.mark.parametrize('body',[
    'return open("/etc/passwd").read()',
    'return obs.__class__',
    'return __import__("os").environ',
    'exec("print(1)")',
    'while True:\n        pass',
    'return [0] * 1000000000',
    'return 2 ** 999999',
    'return {"x": float("nan")}',
])
def test_worker_rejects_escape_and_resource_abuse(body):
    p=ControllerProgram('def control(obs, memory):\n    '+body)
    try:
        with pytest.raises(ValueError):p.request({'obs':{}})
    finally:p.close()


def test_unknown_import_rejected_and_worker_cleaned_up():
    with pytest.raises(ValueError):ControllerProgram('import os\ndef control(obs,memory):\n    return {}')


def test_schema_limits_are_shared_but_returns_depend_on_mode():
    assert 'joint_positions' in coding_spec('direct')['description']
    assert 'joint_offsets' in coding_spec('hybrid')['description']
    assert coding_spec('direct')['inputSchema']['properties']['max_steps']['maximum']==500
