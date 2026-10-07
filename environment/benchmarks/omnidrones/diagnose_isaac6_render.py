"""Spectator-only render convergence audit with zero episode physics ticks."""
import json
from pathlib import Path
import sys
from .project_isaac6 import create_sim


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    sim=create_sim('T16',7,out)
    import carb
    import imageio.v2 as imageio
    import omni.replicator.core as rep
    settings=carb.settings.get_settings()
    keys=['/rtx/post/aa/op','/rtx/post/motionblur/enabled','/rtx-transient/dlssg/enabled']
    report={'diagnostic_only':True,'policy_images':False,'phases':{}}
    try:
        sim.reset(7)
        initial_time=sim.env.sim.current_time
        initial_obs=sim.observation()
        for phase in ['unchanged_32_renders','motion_history_disabled','replicator_zero_dt']:
            if phase=='motion_history_disabled':
                settings.set_int('/rtx/post/aa/op',0)
                settings.set_bool('/rtx/post/motionblur/enabled',False)
                settings.set_bool('/rtx-transient/dlssg/enabled',False)
            report['phases'][phase]={'settings':{key:settings.get(key) for key in keys}}
            for n in range(1,33):
                if phase=='replicator_zero_dt':
                    previous=settings.get('/app/player/playSimulations')
                    settings.set_bool('/app/player/playSimulations',False)
                    try:
                        rep.orchestrator.step(delta_time=0.,pause_timeline=False)
                    finally:
                        settings.set('/app/player/playSimulations',previous)
                else:
                    sim.env.sim.render()
                if sim.env.sim.current_time != initial_time:
                    raise RuntimeError('Render-only diagnostic advanced physics')
                if n in (1,4,8,16,32):
                    imageio.imwrite(out/f'{phase}-{n:02}.png',sim.env.render('rgb_array'))
            report['phases'][phase]['physical_time_delta']=sim.env.sim.current_time-initial_time
            report['phases'][phase]['episode_control_steps']=sim.steps
        report['cached_observation_unchanged']=sim.observation()==initial_obs
        (out/'render-diagnostic.json').write_text(json.dumps(report,indent=2))
    finally:
        sim.close()


if __name__=='__main__':main()
