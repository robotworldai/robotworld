import json
from collections import deque

import numpy as np

from environment.benchmarks.robocasa.control import CAMERAS, STATE_KEYS
from environment.benchmarks.robocasa.policy import Policy
from environment.runtime.image_history import select_request_images


def test_publishes_existing_control_step_window_without_changing_sampling(tmp_path, monkeypatch):
    export=tmp_path/'agent-observations';export.mkdir()
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(export))
    policy=Policy.__new__(Policy)
    policy.out=tmp_path/'episode';policy.obs_index=0;policy.history=deque(maxlen=9)
    for step in range(12):
        obs={k:np.zeros(3) for k in STATE_KEYS}
        obs.update({k:np.full((8,8,3),step,dtype=np.uint8) for k in CAMERAS})
        obs['annotation.human.task_description']='Close drawer'
        policy.ingest(obs,step)
    parts=policy.content()
    window=json.loads((tmp_path/'image-window.json').read_text())
    assert [r['env_step'] for r in window['rounds']]==[3,5,7,9,11]
    assert window['history_unit']=='control_steps'
    wire=[dict(type='input_image',image_url=p['imageUrl']) if p['type']=='inputImage'
          else dict(type='input_text',text=p['text']) for p in parts]
    payload=dict(input=[dict(role='user',content=wire)])
    outgoing,report=select_request_images(payload,window)
    assert outgoing==payload and report['outgoing_images']==15
    assert (export/'0'/'robot0_agentview_left.png').exists()
