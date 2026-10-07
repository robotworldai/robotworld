"""A1 physical diagnostics; no model calls and no benchmark success claims.

Run inside a matching Isaac image. Each invocation is a fresh process so seeds,
startup material sampling and initial randomization do not depend on prior cases.
"""
import argparse
import json
import os
from pathlib import Path
import random
import traceback
from types import SimpleNamespace


def plain(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--asset', choices=['compatible', 'original', 'old-import', 'old-usd'], default='compatible')
    p.add_argument('--shared-usd', type=Path)
    p.add_argument('--control', choices=['hold', 'pulse', 'replay'], required=True)
    p.add_argument('--native', action='store_true')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--legacy-runtime', action='store_true')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--zero-joint-friction', action='store_true', help='Diagnostic ablation only; changes native physics')
    p.add_argument('--friction-compat', action='store_true', help='Validate production A1 zero-friction compatibility')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'request.json').write_text(json.dumps(vars(a), default=str, indent=2))
    app = env = None
    try:
        from isaaclab.app import AppLauncher
        import isaacsim
        if a.legacy_runtime:
            # Avoid unrelated CAD/IRAY UI extensions in the 4.5 full app. The
            # CPU cross-version attempt requires no renderer or review camera.
            app = AppLauncher(headless=True, enable_cameras=False, device=a.device).app
        else:
            app = AppLauncher(headless=True, enable_cameras=True, device=a.device,
                              experience=str(Path(isaacsim.__file__).parent/'apps/isaacsim.exp.full.kit')).app
        import numpy as np
        import torch
        from environment.benchmarks.robot_lab import project
        from environment.runtime.native_project_launch import WORLD
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        source = WORLD/'third_party/benchmarks/robot_lab'
        if a.legacy_runtime:
            # The NGC 4.5 archive has no pip distribution metadata for isaacsim.
            # No Kit6 shim applies; only establish the unchanged task namespaces.
            project.SOURCE=source/'checkout'; project.OUTPUT=a.output; project._packages()
        else:
            project.setup(source, a.output)
        # Exact native replay must preserve the original construction sequence:
        # conversion itself can consume RNG before startup randomization.
        if a.native and a.asset == 'compatible':
            os.environ['WORLD_A1_REIMPORT'] = '1'
        else:
            os.environ.pop('WORLD_A1_REIMPORT', None)
        # Preserve historical baselines unless the production repair is requested.
        os.environ['WORLD_A1_FRICTION_COMPAT']='1' if a.friction_compat else '0'
        cfg = project.build('T11', a.seed, a.output)
        cfg.sim.device = a.device
        if a.asset == 'compatible' and not a.native:
            cfg.scene.robot.spawn.usd_path = str(a.reference/'a1-reimport/a1-compatible.usdc')
        elif a.asset == 'old-usd':
            if a.shared_usd is None or not a.shared_usd.is_file():
                raise ValueError('--shared-usd must point to the completed 4.5 import')
            cfg.scene.robot.spawn.usd_path = str(a.shared_usd)
        elif a.asset == 'old-import':
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
            # Same foot-preserving derived URDF used by Isaac6, not a different
            # model. The older importer also ignores dont_collapse under merge.
            urdf = a.reference/'a1-reimport/a1-foot-preserving.urdf'
            conv = UrdfConverter(UrdfConverterCfg(asset_path=str(urdf), usd_dir=str(a.output/'old-import'),
                usd_file_name='a1.usd', fix_base=False, merge_fixed_joints=False, force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.,damping=1.))))
            cfg.scene.robot.spawn.usd_path = conv.usd_path
        if not a.native:
            # Isolate physics, not task scoring; both assets get identical overrides.
            cfg.events = SimpleNamespace()
            cfg.rewards = SimpleNamespace()
            cfg.terminations = SimpleNamespace()
            cfg.curriculum = SimpleNamespace()
            cfg.observations.policy.enable_corruption = False
        if not a.native:
            cfg.commands.base_velocity.debug_vis = False
        # No diagnostic sensor consumes the extra height scan. Its CUDA-only ray
        # query is not useful for comparing CPU PhysX. Neither actor nor critic
        # uses this scanner in the original flat configuration.
        if not a.native:
            cfg.scene.height_scanner_base = None
        from isaaclab.sim import GroundPlaneCfg
        init = GroundPlaneCfg.__init__
        def local_ground(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self.usd_path = str(WORLD/'third_party/benchmarks/wheeledlab/assets/Isaac/Environments/Grid/default_environment.usd')
        GroundPlaneCfg.__init__ = local_ground
        (a.output/'configuration.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
        from isaaclab.envs import ManagerBasedRLEnv
        class SingleEpisode(ManagerBasedRLEnv):
            capture_terminal = False
            def _reset_idx(self, ids):
                if not self.capture_terminal:
                    return super()._reset_idx(ids)
        env = SingleEpisode(cfg=cfg)
        obs, _ = env.reset(seed=a.seed)
        env.capture_terminal = True
        robot = env.scene['robot']; sensor = env.scene['contact_forces']
        if not a.native:
            state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
            env.sim.forward(); env.scene.update(0.)
        view = robot.root_physx_view
        friction_ablation=None
        if a.zero_joint_friction:
            if a.native: raise ValueError('Friction ablation is not a native benchmark run')
            ids=torch.tensor([0],dtype=torch.int32,device='cpu')
            if a.legacy_runtime:
                before=view.get_dof_friction_coefficients().clone()
                view.set_dof_friction_coefficients(torch.zeros_like(before),ids)
                after=view.get_dof_friction_coefficients().clone()
            else:
                before=view.get_dof_friction_properties().clone()
                view.set_dof_friction_properties(torch.zeros_like(before),ids)
                after=view.get_dof_friction_properties().clone()
                # Old USD can retain the legacy load-dependent coefficient even
                # when the new three-component friction API reads all zeros.
                legacy_before=view.get_dof_friction_coefficients().clone()
                view.set_dof_friction_coefficients(torch.zeros_like(legacy_before),ids)
                legacy_after=view.get_dof_friction_coefficients().clone()
                if torch.any(legacy_after!=0): raise RuntimeError('Legacy friction remains active')
            if torch.any(after!=0): raise RuntimeError('Friction ablation did not take effect')
            friction_ablation={'before':plain(before),'after':plain(after)}
            if not a.legacy_runtime:
                friction_ablation.update(legacy_before=plain(legacy_before),legacy_after=plain(legacy_after))
        runtime = {'body_names': robot.body_names, 'joint_names': robot.joint_names,
                   'actuators': {}, 'physx': {}, 'device': str(env.device), 'control_dt': env.step_dt,
                   'friction_ablation':friction_ablation}
        for name, act in robot.actuators.items():
            runtime['actuators'][name] = {k: plain(getattr(act,k,None)) for k in
                ['joint_names','stiffness','damping','effort_limit','velocity_limit']}
        for name in ['get_dof_stiffnesses','get_dof_dampings','get_dof_limits','get_dof_max_velocities',
                     'get_dof_max_forces','get_masses','get_inertias','get_coms','get_material_properties',
                     'get_dof_friction_coefficients','get_dof_friction_properties','get_dof_armatures']:
            try: runtime['physx'][name] = plain(getattr(view,name)())
            except Exception as exc: runtime['physx'][name] = {'unavailable': str(exc)}
        import omni.usd
        from pxr import UsdPhysics
        stage=omni.usd.get_context().get_stage()
        physics_attributes={}
        for prim in stage.Traverse():
            attrs={attr.GetName(): {'value':str(attr.Get()),'authored':attr.HasAuthoredValueOpinion()}
                   for attr in prim.GetAttributes()
                   if attr.GetName().startswith(('physics:','physx'))}
            if attrs: physics_attributes[str(prim.GetPath())]={'type':prim.GetTypeName(),'schemas':list(prim.GetAppliedSchemas()),'attributes':attrs}
        (a.output/'physics-stage-attributes.json').write_text(json.dumps(physics_attributes,indent=2))
        term = env.action_manager.get_term('joint_pos')
        runtime['action_joint_names'] = term._joint_names
        (a.output/'effective-parameters.json').write_text(json.dumps(runtime,indent=2))
        def telemetry(step, action=None):
            forces = sensor.data.net_forces_w_history[0]
            maximum = torch.linalg.vector_norm(forces,dim=-1).max(dim=0).values
            contact = {name: float(value) for name,value in zip(sensor.body_names,maximum)}
            return {'step': step, 'action': action, 'root_state': plain(robot.data.root_state_w[0]),
                    'joint_pos': plain(robot.data.joint_pos[0]), 'joint_vel': plain(robot.data.joint_vel[0]),
                    'joint_target': plain(robot.data.joint_pos_target[0]),
                    'applied_torque': plain(robot.data.applied_torque[0]),
                    'gravity': plain(robot.data.projected_gravity_b[0]),
                    'body_pos_w': plain(robot.data.body_pos_w[0]),
                    'contact_history_peak_N': contact,
                    'nonfoot_contact_over_1N': {n:f for n,f in contact.items() if not n.endswith('_foot') and f>1.},
                    'termination_terms': {n: bool(env.termination_manager.get_term(n)[0]) for n in env.termination_manager.active_terms}}
        source_steps=[]
        if a.control=='replay':
            for line in (a.reference/'events/environment.jsonl').open():
                event=json.loads(line)
                if event['kind']=='environment_step': source_steps.append(event['payload'])
            assert len(source_steps)==101
        rows=[telemetry(0)]
        initial_actor_match = None
        if source_steps and a.native:
            # Check initial noisy actor vector against the actual model episode.
            from environment.benchmarks.native_project.simulator import Simulator
            sim=Simulator(env,{},a.output,project);sim._observe(obs)
            reference=source_steps[0]['before']['native_policy_terms']
            initial_actor_match=max(abs(float(x)-float(y)) for n in reference
                for x,y in zip(np.array(reference[n]).ravel(),np.array(sim.policy_terms[n]).ravel()))
        budget=min(a.steps,len(source_steps)) if source_steps else a.steps
        with (a.output/'physics.jsonl').open('w') as stream:
            stream.write(json.dumps(rows[0])+'\n')
            for i in range(budget):
                action=[0.]*12
                if a.control=='replay': action=source_steps[i]['requested_action']
                elif a.control=='pulse':
                    for idx,start in [(0,40),(1,90),(2,140)]:
                        if start<=i<start+20: action[idx]=.2
                with torch.inference_mode(): env.step(torch.tensor([action],device=env.device))
                row=telemetry(i+1,action);rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                if a.native and any(row['termination_terms'].values()): break
        first=next((r for r in rows[1:] if r['nonfoot_contact_over_1N']),None)
        summary={'infrastructure_ok':True,'benchmark_score':False,'native_task_rules':a.native,
                 'steps':len(rows)-1,'initial_actor_max_abs_error_vs_model_run':initial_actor_match,
                 'first_nonfoot_contact': None if first is None else {'step':first['step'],'bodies':first['nonfoot_contact_over_1N']},
                 'final_gravity':rows[-1]['gravity'],'final_termination_terms':rows[-1]['termination_terms'],
                 'overrides': ['review camera omitted'] + (['joint friction zeroed: diagnostic ablation'] if a.zero_joint_friction else []) + ([] if a.native else
                   ['debug rendering off','unused height_scanner_base omitted','events/rewards/terminations/curriculum disabled','actor noise disabled','default root/joint pose; zero velocities']),
                 'warning': 'Merged calf contains foot collision; nonfoot force label cannot be scored as illegal contact.' if not any(n.endswith('_foot') for n in robot.body_names) else None}
        (a.output/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc());raise
    finally:
        if env: env.close()
        if app:
            if not a.legacy_runtime: app.close(skip_cleanup=True)
            else: app.close()


if __name__=='__main__': main()
