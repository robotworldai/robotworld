import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from environment.benchmarks.humanoid_soccer.policy import AgentPolicy
from environment.runtime.image_history import ObservationHistory, select_request_images
from environment.runtime.request_audit import summarize


def test_soccer_packet_and_request_crop_match(tmp_path, monkeypatch):
    constants = ModuleType('mujoco_soccer.constants')
    constants.ISAACLAB_TO_MUJOCO_REINDEX = []
    monkeypatch.setitem(sys.modules, 'mujoco_soccer.constants', constants)
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS', str(tmp_path/'export'))
    policy = AgentPolicy.__new__(AgentPolicy)
    policy.out = tmp_path
    policy.mode = 'direct'
    policy.cfg = SimpleNamespace(sim_time=6, control_dt=0.02)
    policy.history = ObservationHistory()
    policy.state = lambda: {'control_step': policy.step_count}
    policy.capture = SimpleNamespace(scenery='plain', images=lambda: {
        name: np.full((4, 4, 3), policy.seq, dtype=np.uint8)
        for name in ('front', 'side')})
    payload = {'input': [], 'tools': [
        {'type': 'namespace', 'name': 'functions', 'tools': []},
        {'type': 'custom', 'name': 'apply_patch'},
        {'type': 'function', 'name': 'move_joints'}]}
    for number in range(12):
        policy.seq = number
        policy.step_count = number*10
        parts = policy.content()
        content = [{'type': 'input_image', 'image_url': p['imageUrl']}
                   if p['type'] == 'inputImage' else {'type': 'input_text', 'text': p['text']}
                   for p in parts]
        payload['input'].append({'type': 'function_call_output', 'call_id': str(number), 'output': content})
        window = json.loads((tmp_path/'image-window.json').read_text())
        outgoing, report = select_request_images(payload, window)
        assert outgoing['tools'] == payload['tools']
        assert outgoing['input'][-1] == payload['input'][-1]
        assert summarize(outgoing)['image_sha256'] == window['image_sha256']
    assert summarize(payload)['input_images'] > 50
    assert summarize(outgoing)['input_images'] == 10
    assert report['removed_image_occurrences'] > 0
    assert [r['observation'] for r in window['rounds']] == [3, 5, 7, 9, 11]
    assert (tmp_path/'export/0/front.png').is_file()
    assert (tmp_path/'observations/11/side.png').is_file()
