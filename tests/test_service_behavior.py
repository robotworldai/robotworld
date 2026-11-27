"""Chunk preparation preserves per-command absolute-target anchoring."""
import io
from types import SimpleNamespace
import numpy as np
import pytest
import environment.service.behavior_1k as module


def test_command_targets_anchor_when_execution_starts(monkeypatch):
    adapter=module.BehaviorAdapter.__new__(module.BehaviorAdapter)
    adapter.policy=SimpleNamespace(profile={},state={'position':1},grippers={'left':1,'right':1})
    monkeypatch.setattr(module,'validate_tool',lambda name,args,profile:args)
    anchors=[]
    def action_for(targets,state,profile,grippers):
        anchors.append(state['position'])
        return np.array([state['position'],targets['x']]),dict(grippers)
    monkeypatch.setattr(module,'action_for',action_for)
    prepared=adapter.prepare([{'tool':'move_arms','arguments':{'note':'x','targets':{'x':2},'steps':2}},
                              {'tool':'move_arms','arguments':{'note':'x','targets':{'x':3},'steps':1}}])
    assert all(command['action'] is None for command,repeat in prepared)
    # Static validation is non-mutating, and retains a separate deferred target per command.
    assert anchors==[1,1]
    adapter.policy.state={'position':9}
    command=prepared[1][0]
    command['action'],_=module.action_for(command['targets'],adapter.policy.state,{},adapter.policy.grippers)
    assert command['action'].tolist()==[9,3]


def test_prepare_rejects_unknown_command_envelope():
    adapter=module.BehaviorAdapter.__new__(module.BehaviorAdapter)
    adapter.policy=SimpleNamespace(profile={})
    with pytest.raises(ValueError):adapter.prepare([{'tool':'move_arms','extra':1}])


def test_observe_exports_only_allowed_raw_depth(tmp_path):
    import base64
    adapter=module.BehaviorAdapter.__new__(module.BehaviorAdapter)
    adapter.output=tmp_path;adapter.steps=3;adapter.instruction='public';adapter.history=[]
    adapter.policy=SimpleNamespace(images={'wrist_depth_linear':None,'wrist_rgb':None},
        state={'joint':np.array([0.2])})
    folder=tmp_path/'observations/3';folder.mkdir(parents=True)
    depth=np.array([[0.5,np.inf]],dtype=np.float32)
    np.save(folder/'wrist_depth_linear.npy',depth)
    (folder/'hidden-checker.json').write_text('{}')
    observation=adapter.observe()
    assert len(observation['sensor_files'])==1
    sensor=observation['sensor_files'][0]
    assert sensor['sensor']=='wrist_depth_linear' and sensor['units']=='metres'
    decoded=np.load(io.BytesIO(base64.b64decode(sensor['base64'])),allow_pickle=False)
    np.testing.assert_array_equal(decoded,depth)
