import json

from environment.benchmarks.behavior_1k.policy import WorldPolicy
from environment.runtime.image_history import select_request_images
from environment.tests.test_behavior_bridge import profile, obs


def test_publishes_current_and_sampled_history(tmp_path, monkeypatch):
    export=tmp_path/'agent-observations';export.mkdir()
    monkeypatch.setenv('WORLD_AGENT_OBSERVATIONS',str(export))
    p=WorldPolicy(tmp_path/'policy',tmp_path/'manifest','test')
    p.bind(profile());p.reset()
    for i in range(10):
        p.step=i;p._ingest(obs());parts=p._content()
    window=json.loads((tmp_path/'image-window.json').read_text())
    assert [r['env_step'] for r in window['rounds']]==[1,3,5,7,9]
    assert len(window['image_sha256'])==30
    wire=[{'type':'input_image','image_url':p['imageUrl']} if p['type']=='inputImage'
          else {'type':'input_text','text':p['text']} for p in parts]
    request={'input':[{'role':'user','content':wire}]}
    outgoing,_=select_request_images(request,window)
    assert outgoing==request
