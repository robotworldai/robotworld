"""All products share one render; every control tick remains in every video."""
import sys,types,json
import numpy as np
import pytest
from environment.benchmarks.wheeledlab.capture import capture_cameras

def fixture(tmp_path,monkeypatch,advance=False,unready=False):
    settings={'/app/player/playSimulations':True};poses=[];renders=[]
    class Camera:
        def __init__(self,name,height):
            self.stem=name;self.frames=0
            self.rgb=types.SimpleNamespace(get_data=lambda:np.zeros((height,640,4),dtype=np.uint8) if not unready or len(renders)>1 else np.empty(0))
        def update_pose(self):poses.append(self.stem)
        def capture(self,frame=None):
            assert frame is not None and frame.shape[-1]==3
            self.frames+=1
    sim=types.SimpleNamespace(policy_sensor=Camera('front',360),rear_sensor=Camera('rear',360),camera=Camera('review',480),output=tmp_path,steps=1,env=types.SimpleNamespace(sim=types.SimpleNamespace(current_time=.02)))
    def render(**kw):
        assert poses[-3:]==['front','rear','review']
        assert kw=={'delta_time':0.,'pause_timeline':False}
        assert settings['/app/player/playSimulations'] is False
        renders.append(kw)
        if advance:sim.env.sim.current_time+=.02
    fake_settings=types.SimpleNamespace(get=settings.get,set=settings.__setitem__)
    monkeypatch.setitem(sys.modules,'carb',types.SimpleNamespace(settings=types.SimpleNamespace(get_settings=lambda:fake_settings)))
    core=types.ModuleType('omni.replicator.core');core.orchestrator=types.SimpleNamespace(step=render)
    rep=types.ModuleType('omni.replicator');rep.core=core
    omni=types.ModuleType('omni');omni.replicator=rep
    for name,module in [('omni',omni),('omni.replicator',rep),('omni.replicator.core',core)]:monkeypatch.setitem(sys.modules,name,module)
    return sim,settings,renders

def test_one_render_per_step_preserves_all_camera_frames(tmp_path,monkeypatch):
    sim,settings,renders=fixture(tmp_path,monkeypatch)
    for step in range(1,4):sim.steps=step;capture_cameras(sim)
    assert len(renders)==3 and settings['/app/player/playSimulations'] is True
    assert [c.frames for c in [sim.policy_sensor,sim.rear_sensor,sim.camera]]==[3,3,3]
    rows=[json.loads(x) for x in (tmp_path/'camera-capture.jsonl').read_text().splitlines()]
    assert [r['control_step'] for r in rows]==[1,2,3]
    assert all(r['physics_time_before']==r['physics_time_after'] for r in rows)

def test_waits_for_all_products_before_writing(tmp_path,monkeypatch):
    sim,settings,renders=fixture(tmp_path,monkeypatch,unready=True);capture_cameras(sim)
    assert len(renders)==2 and sim.camera.frames==1

def test_rejects_render_that_advances_physics_and_restores_settings(tmp_path,monkeypatch):
    sim,settings,renders=fixture(tmp_path,monkeypatch,advance=True)
    with pytest.raises(RuntimeError,match='advanced physics'):capture_cameras(sim)
    assert settings['/app/player/playSimulations'] is True and sim.camera.frames==0
