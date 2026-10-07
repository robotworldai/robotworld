"""Dispatch independent original benchmark engines; never replace their task logic."""
import argparse
import faulthandler
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import traceback
from environment.runtime.events import EventLog
from environment.runtime.native_project_launch import WORLD, load_project


def main():
    p = argparse.ArgumentParser()
    for key in ('project','task','mode','model'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--runtime-profile',default='default')
    p.add_argument('--scoring-profile',choices=['native','audit-state','world-state-v1'],default='native')
    a = p.parse_args()
    import os
    os.environ['WORLD_SCORING_PROFILE']=a.scoring_profile
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    source, config = load_project(a.project,a.runtime_profile)
    task = config['tasks'][a.task]
    from environment.evaluation.world_success.profiles import get_profile
    world_profile=get_profile(a.project,a.task)
    horizon=world_profile['steps'] if world_profile and a.scoring_profile=='world-state-v1' else task['steps']
    if not 1<=a.steps<=horizon:
        raise ValueError('Native horizon exceeded')
    out = a.output
    out.mkdir(parents=True,exist_ok=True)
    hashes = {}
    snapshot_paths = [WORLD/'environment/benchmarks'/a.project, WORLD/'environment/benchmarks/native_project',
                      WORLD/'environment/benchmarks/operating_brief.py',
                      source/'compat',
                      WORLD/'environment/runtime', WORLD/'environment/evaluation/world_success', Path(__file__),
                      WORLD/'environment/benchmarks/wheeledlab/camera.py',
                      WORLD/'environment/benchmarks/humanoid_soccer/coding.py',
                      WORLD/'environment/benchmarks/humanoid_soccer/control_program.py',
                      WORLD/'environment/benchmarks/humanoid_soccer/program_worker.py']
    if a.project == 'reflexbench':
        snapshot_paths.append(WORLD/'environment/benchmarks/robolab')
        snapshot_paths.append(WORLD/'third_party/benchmarks/robolab/compat/isaac601.py')
    for folder in snapshot_paths:
        if not folder.exists():
            continue
        for f in ([folder] if folder.is_file() else folder.rglob('*.py')):
            rel = f.relative_to(WORLD)
            target = out/'interface-source'/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,target)
            hashes[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'interface-hashes.json').write_text(json.dumps(hashes,indent=2))
    app = sim = None
    try:
        project = importlib.import_module(config.get('adapter','environment.benchmarks.'+a.project+'.project'))
        if config.get('engine','isaaclab')=='custom':
            sim = project.create_sim(a.task,a.seed,out,headless=True)
        else:
            from isaaclab.app import AppLauncher
            import isaacsim
            experience = config.get('experience',str(Path(isaacsim.__file__).parent/'apps/isaacsim.exp.full.kit'))
            if hasattr(project, 'launch_app'):
                app = project.launch_app(experience)
            else:
                app = AppLauncher(headless=True,enable_cameras=True,experience=experience).app
            import numpy as np
            import torch
            random.seed(a.seed)
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            project.setup(source,out)
            cfg = project.build(a.task,a.seed,out)
            from environment.evaluation.world_success.scenes import configure
            configure(a.project,a.task,cfg)
            (out/'configuration.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
            if hasattr(project,'make_env'):
                env = project.make_env(cfg)
            else:
                from isaaclab.envs import ManagerBasedRLEnv
                class SingleEpisodeEnv(ManagerBasedRLEnv):
                    capture_terminal = False
                    def _reset_idx(self, env_ids):
                        if not self.capture_terminal:
                            return super()._reset_idx(env_ids)
                env = SingleEpisodeEnv(cfg=cfg,render_mode=None)
            from environment.benchmarks.native_project.simulator import Simulator
            sim = Simulator(env,task,out,project)
        from environment.evaluation.world_success.runtime import attach, instruction as world_instruction
        attached_profile=attach(sim,a.project,a.task,a.seed,a.scoring_profile)
        print('[World] Environment constructed; resetting native episode',flush=True)
        faulthandler.enable()
        faulthandler.dump_traceback_later(90, repeat=True)
        sim.reset(a.seed)
        print('[World] Native reset complete',flush=True)
        actual = {'python':sys.version,'requested_runtime':config.get('runtime'),
                  'runtime_profile':a.runtime_profile,
                  'physics_paused_during_model_reasoning':True,
                  'actual_control_dt':sim.dt,'packages':{},'module_paths':{}}
        for name in ['isaacsim','isaaclab','torch','numpy','torchrl','tensordict']:
            try:
                actual['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual['packages'][name]=None
        for name in ['isaacsim','isaaclab','omni.isaac.lab','omni_drones','omniisaacgymenvs']:
            module=sys.modules.get(name)
            if module is not None:
                actual['module_paths'][name]=str(getattr(module,'__file__',None))
        (out/'runtime-actual.json').write_text(json.dumps(actual,indent=2))
        # Read-only audit: configuration text alone missed legacy friction in A1.
        # Preserve randomization and record actual values; do not silently repair
        # other benchmarks or infer equivalence from this small parameter set.
        if config.get('engine','isaaclab')!='custom':
            parameter_audit={'read_only':True,'actuators':{},'physx':{}}
            try:
                robot=env.scene['robot'];view=robot.root_physx_view
                parameter_audit['joint_names']=robot.joint_names
                for name,act in robot.actuators.items():
                    parameter_audit['actuators'][name]={'joint_names':act.joint_names,
                        'configured_friction':getattr(act.cfg,'friction',None)}
                    for field in ['stiffness','damping','friction','effort_limit','velocity_limit']:
                        value=getattr(act,field,None)
                        parameter_audit['actuators'][name][field]=value.detach().cpu().tolist() if hasattr(value,'detach') else value
                for method in ['get_dof_friction_coefficients','get_dof_friction_properties','get_dof_stiffnesses','get_dof_dampings']:
                    try:parameter_audit['physx'][method]=getattr(view,method)().detach().cpu().tolist()
                    except Exception as exc:parameter_audit['physx'][method]={'unavailable':str(exc)}
            except Exception as exc:parameter_audit['unavailable']=str(exc)
            (out/'physics-parameter-audit.json').write_text(json.dumps(parameter_audit,default=str,indent=2))
        (out/'action-metadata.json').write_text(json.dumps(sim.action_metadata,default=str,indent=2))
        (out/'observation-metadata.json').write_text(json.dumps(getattr(sim,'observation_metadata',{}),default=str,indent=2))
        if config.get('engine','isaaclab')!='custom':
            from environment.benchmarks.native_project.camera import Camera
            sim.case = a.task
            sim.camera = Camera(sim,out/'video')
            sim.camera.capture()
        instructions = getattr(sim,'prompt',task.get('action',''))
        if hasattr(project,'instruction'):
            instructions = project.instruction(a.task,getattr(sim,'env',None))
        elif hasattr(project,'instructions'):
            instructions = project.instructions(a.task)
        if attached_profile and a.scoring_profile=='world-state-v1':
            instructions += '\n\n'+world_instruction(attached_profile)
        print('[World] Observation and review interfaces ready; starting '+a.mode,flush=True)
        faulthandler.cancel_dump_traceback_later()
        if a.mode=='codex':
            from environment.benchmarks.native_project.policy import Agent
            from environment.runtime.nonaction_budget import run_bounded, apply_result
            interaction_stop=run_bounded(Agent(sim,out,a.manifest,a.model,a.steps,a.timeout,instructions,not a.disable_coding_control,task_id=a.task))
        else:
            events = EventLog(out/'events/environment.jsonl')
            events.write('initial_observation',sim.observation())
            action = [0.]*sim.action_metadata['dim']
            if a.mode == 'probe' and hasattr(project,'probe_action'):
                action = project.probe_action(a.task,sim)
            for _ in range(a.steps):
                if sim.done:
                    break
                before = sim.observation()
                after = sim.step(action)
                events.write('environment_step',{'requested_action':action,'before':before,
                                                'after':after,'evaluation':sim.last_evaluation})
        result = sim.result()
        if a.mode=='codex':apply_result(result,interaction_stop)
        result.setdefault('control_steps',sim.steps)
        result.setdefault('control_dt',sim.dt)
        result.setdefault('simulated_seconds',sim.steps*sim.dt)
        result.update(project=a.project,task=a.task,registered_id=task['id'],policy=a.mode,
                      benchmark_commit=config['commit'],runtime=config.get('runtime',task.get('runtime')),
                      runtime_profile=a.runtime_profile,
                      requested_steps=a.steps,stop_reason=result.get('stop_reason','native_termination' if sim.done else 'requested_step_budget'),
                      upstream_source_modified=False,coding_control_enabled=a.mode=='codex' and not a.disable_coding_control)
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        (out/'final-observation.json').write_text(json.dumps(sim.observation(),indent=2,allow_nan=False))
    except BaseException:
        (out/'error.txt').write_text(traceback.format_exc())
        if sim is not None:
            try:
                partial = sim.result()
                partial.update(infrastructure_complete=False,policy=a.mode,project=a.project,task=a.task)
                (out/'partial-result.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2,allow_nan=False))
            except Exception:
                pass
        raise
    finally:
        try:
            if sim:
                sim.close()
        finally:
            if app:
                # Video and environment are already closed. Kit 110 shutdown can
                # hang in unrelated UI task groups; use its process teardown path.
                if importlib.metadata.version('isaacsim').startswith('6.'):
                    app.close(skip_cleanup=True)
                else:
                    app.close()


if __name__=='__main__':
    main()
