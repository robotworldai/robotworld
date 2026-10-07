"""Load unchanged task configurations; externally manage one evaluation episode."""
import argparse,importlib,importlib.util,json,random,sys,traceback,types
from pathlib import Path

def main():
    from environment.benchmarks.wheeledlab.catalog import CASES
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['probe','zero','codex'],default='codex');p.add_argument('--steps',type=int,required=True)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--model',default='gpt-6-astra');p.add_argument('--manifest',type=Path)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--scoring-profile',choices=['native','audit-state','world-state-v1'],default='native');a=p.parse_args()
    import os
    os.environ['WORLD_SCORING_PROFILE']=a.scoring_profile
    from environment.evaluation.world_success.profiles import get_profile
    profile=get_profile('wheeledlab',a.case)
    horizon=profile['steps'] if profile and a.scoring_profile=='world-state-v1' else CASES[a.case]['steps']
    if not 1<=a.steps<=horizon:p.error('Budget must be within the configured horizon')
    import faulthandler
    faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
    from isaaclab.app import AppLauncher
    import isaacsim
    launcher=AppLauncher(headless=True,enable_cameras=True,experience=str(Path(isaacsim.__file__).parent/'apps/isaacsim.exp.full.kit'))
    app=launcher.app;env=None;camera=None;onboard=None;rear=None
    try:
        W=Path(__file__).resolve().parents[2];source=W/'third_party/benchmarks/wheeledlab'
        path=source/'compat/isaac601.py';spec=importlib.util.spec_from_file_location('world_wheeledlab_compat',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install(a.output)
        import torch,numpy as np
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        # Avoid importing all task families (visual import generates a large map).
        package=types.ModuleType('wheeledlab_tasks');package.__path__=[str(source/'checkout/source/wheeledlab_tasks/wheeledlab_tasks')];sys.modules['wheeledlab_tasks']=package
        choices={'mushr-drift':('drifting.mushr_drift_env_cfg','MushrDriftRLEnvCfg'),'f1tenth-drift':('drifting.f1tenth_drift_env_cfg','F1TenthDriftRLEnvCfg'),'elevation':('elevation.mushr_elevation_env_cfg','MushrElevationRLEnvCfg'),'visual':('visual.mushr_visual_env_cfg','MushrVisualRLEnvCfg')}
        custom=CASES[a.case]['family']=='custom'
        if custom:
            from third_party.benchmarks.wheeledlab.robotworld.specs import scenario
            from third_party.benchmarks.wheeledlab.robotworld.environment import build
            if CASES[a.case].get('precision'):
                from third_party.benchmarks.wheeledlab.robotworld.precision_specs import scenario
                from third_party.benchmarks.wheeledlab.robotworld.precision_environment import build
            import hashlib,shutil
            snapshot=a.output/'protocol-source';snapshot.mkdir()
            hashes={}
            for f in (source/'robotworld').glob('*.py'):
                shutil.copy2(f,snapshot/f.name);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
            (a.output/'protocol-hashes.json').write_text(json.dumps(hashes,indent=2))
            interface=a.output/'interface-source';interface.mkdir();interface_hashes={}
            for f in (W/'environment/benchmarks/wheeledlab').glob('*.py'):
                shutil.copy2(f,interface/f.name);interface_hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
            (a.output/'interface-hashes.json').write_text(json.dumps(interface_hashes,indent=2))
            from third_party.benchmarks.wheeledlab.robotworld.onboard import PROFILE,DESCRIPTION
            spec=scenario(a.case);spec['observation_profile']=PROFILE;cfg=build(spec,a.output)
            (a.output/'observation-profile.json').write_text(json.dumps({'profile':PROFILE,'description':DESCRIPTION,
                'camera':'vehicle-mounted front RGB640x360; callbacks48x27RGB8','sensor_noise':'none; ideal encoder/IMU readings',
                'review_camera_exposed':False,'global_map_exposed':False,'scoring_state_exposed':False,
                'rear_camera':{'enabled':bool(spec.get('rear_camera')),'pitch_down_deg':35,'mirrored':False},
                'scene_overlay':'visible red stopping box in gate task; no collision/scoring changes'},indent=2))
            (a.output/'scenario.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
        else:
            mod,cls=choices[a.case];cfg=getattr(importlib.import_module('wheeledlab_tasks.'+mod),cls)()
        from environment.evaluation.world_success.scenes import configure
        configure('wheeledlab',a.case,cfg,a.output)
        cfg.num_envs=1;cfg.scene.num_envs=1;cfg.seed=a.seed
        # Kit110 assets_loading remains true indefinitely (probe03 stack).
        # Avoid that reset busy-loop; use bounded render warmup without physics.
        cfg.wait_for_textures=False
        (a.output/'configuration.json').write_text(json.dumps({'case':a.case,'registered_id':CASES[a.case]['id'],'seed':a.seed,'runtime':'isaac6.0.1/isaaclab2.2-experimental','upstream':'Isaac4.5/IsaacLab2.0.2','config':cfg.to_dict()},default=str,indent=2))
        from isaaclab.envs import ManagerBasedRLEnv
        class SingleEpisodeEnv(ManagerBasedRLEnv):
            """Keep the terminal state instead of starting a second episode."""
            capture_terminal=False
            def _reset_idx(self,env_ids):
                if self.capture_terminal:return
                return super()._reset_idx(env_ids)
        env=SingleEpisodeEnv(cfg=cfg,render_mode=None)
        from environment.benchmarks.wheeledlab.simulator import Simulator
        if custom:
            from environment.benchmarks.wheeledlab.custom import CustomSimulator
            sim=CustomSimulator(env,a.case,a.output,spec)
        else:sim=Simulator(env,a.case,a.output)
        from environment.evaluation.world_success.runtime import attach
        attach(sim,'wheeledlab',a.case,a.seed,a.scoring_profile)
        sim.reset(a.seed)
        if custom:
            from third_party.benchmarks.wheeledlab.robotworld.onboard import OnboardCamera
            onboard=sim.policy_sensor=OnboardCamera(sim,a.output/'video')
            if spec.get('rear_camera'):rear=sim.rear_sensor=OnboardCamera(sim,a.output/'video',facing='rear')
        if a.mode!='probe':
            faulthandler.dump_traceback_later(60,repeat=True)
            from environment.benchmarks.wheeledlab.camera import Camera
            camera=sim.camera=Camera(sim,a.output/'video');camera.capture()
        faulthandler.cancel_dump_traceback_later()
        if a.mode=='codex':
            from environment.benchmarks.wheeledlab.policy import Agent
            from environment.runtime.nonaction_budget import run_bounded, apply_result
            interaction_stop=run_bounded(Agent(sim,a.output,a.manifest,a.model,a.steps,a.timeout,not a.disable_coding_control))
        else:
            from environment.runtime.events import EventLog
            log=EventLog(a.output/'events/environment.jsonl');log.write('initial_observation',sim.observation())
            for _ in range(a.steps):
                if sim.done:break
                action=[0.,0.]
                before=sim.observation();state=sim.step(action);log.write('environment_step',{'requested_action':action,'before':before,'after':state,'evaluation':sim.last_evaluation})
        result=sim.result();result.update(policy=a.mode,requested_steps=a.steps,stop_reason=result.get('stop_reason',('robotworld_terminal' if custom and sim.judge.terminal else 'native_termination') if sim.done else 'requested_step_budget'),simulated_seconds=sim.steps*sim.dt,coding_control_enabled=a.mode=='codex' and not a.disable_coding_control)
        if a.mode=='codex':apply_result(result,interaction_stop)
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (a.output/'final-observation.json').write_text(json.dumps(sim.observation(),indent=2,allow_nan=False))
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc();raise
    finally:
        if camera:camera.close()
        if onboard:onboard.close()
        if rear:rear.close()
        if env:env.close()
        faulthandler.cancel_dump_traceback_later()
        app.close(skip_cleanup=True)

if __name__=='__main__':main()
