import json
from types import SimpleNamespace

import numpy as np
import pytest

from environment.benchmarks.wheeledlab.policy import Agent
from environment.runtime.image_history import select_request_images


@pytest.mark.parametrize('visual', [False, True])
def test_wheeledlab_publishes_matching_window(tmp_path, monkeypatch, visual):
    export=tmp_path/'agent-observations'
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(export))
    sim=SimpleNamespace(steps=0,case='mushr-drift',policy_image=None)
    sim.observation=lambda: {'control_step':sim.steps}
    agent=Agent(sim,tmp_path,None,'fixture',250,60,coding=False)
    for seq in range(11):
        sim.steps=seq*3
        if visual:sim.policy_image=np.full((4,4,3),seq,dtype=np.uint8)
        parts=agent.content()
    window=json.loads((tmp_path/'image-window.json').read_text())
    assert [r['observation'] for r in window['rounds']]==[2,4,6,8,10]
    assert [r['env_step'] for r in window['rounds']]==[6,12,18,24,30]
    assert len(window['image_sha256'])==(5 if visual else 0)
    assert not any(t['name']=='coding_control' for t in agent.tool_schema)
    if visual:
        payload={'input':[{'role':'user','content':[
            {'type':'input_image','image_url':p['imageUrl']} if p['type']=='inputImage'
            else {'type':'input_text','text':p['text']} for p in parts]}]}
        selected,_=select_request_images(payload,window)
        assert selected==payload
