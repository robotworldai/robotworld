"""One synchronized render for all review/onboard products, once per control tick.
Physics and camera intrinsics/poses are unchanged. Every camera keeps its own
continuous video and records the same current control state.
"""
import json
from pathlib import Path
import numpy as np

def capture_cameras(sim):
    import carb
    import omni.replicator.core as rep
    cameras=[c for c in (sim.policy_sensor,sim.rear_sensor,sim.camera) if c is not None]
    if not cameras:return
    for camera in cameras:camera.update_pose()
    before=sim.env.sim.current_time
    settings=carb.settings.get_settings();key='/app/player/playSimulations';previous=settings.get(key)
    settings.set(key,False)
    try:
        for attempt in range(3):
            rep.orchestrator.step(delta_time=0.,pause_timeline=False)
            raw=[np.asarray(c.rgb.get_data()) for c in cameras]
            expected=[(480,640,4) if c is sim.camera else (360,640,4) for c in cameras]
            if all(a.shape==shape for a,shape in zip(raw,expected)):break
        else:raise RuntimeError('Synchronized camera buffers unavailable: '+str([a.shape for a in raw]))
    finally:settings.set(key,previous)
    after=sim.env.sim.current_time
    if after!=before:raise RuntimeError('Synchronized camera capture advanced physics')
    for camera,frame in zip(cameras,raw):camera.capture(frame=np.ascontiguousarray(frame[:,:,:3]))
    record={'control_step':sim.steps,'physics_time_before':float(before),'physics_time_after':float(after),
            'render_passes':attempt+1,'frames':{getattr(c,'stem','review'):c.frames for c in cameras}}
    with (Path(sim.output)/'camera-capture.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
