"""External synchronous adapter around original Bench2Dex scene and metrics helpers."""
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]/'third_party/benchmarks/bench2dex'
SOURCE=ROOT/'checkout'
ROBOT='multi_ur5_wuji_with_flange'


def public_kinematics():
    """Published robot geometry only; no runtime object transforms or oracle."""
    path=ROOT/'dex2bench_dataset/Robots_p/ur5+wuji/urdf/Multi_UR5_wuji_with_flange.urdf'
    result=[]
    for joint in ET.parse(path).getroot().findall('joint'):
        parent,child=joint.find('parent'),joint.find('child')
        if parent is None or child is None:continue
        origin=joint.find('origin');axis=joint.find('axis')
        result.append({'joint':joint.get('name'),'type':joint.get('type'),
            'parent':parent.get('link'),'child':child.get('link'),
            'origin_xyz_m':origin.get('xyz','0 0 0') if origin is not None else '0 0 0',
            'origin_rpy_rad':origin.get('rpy','0 0 0') if origin is not None else '0 0 0',
            'axis':axis.get('xyz') if axis is not None else None})
    return result


def jsonable(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


class Simulator:
    def __init__(self, task_id, seed, output, headless=True):
        self.out=Path(output);self.task_id=task_id
        self.task_spec=json.loads((ROOT/'tasks.json').read_text())[task_id]
        self.steps=0;self.physics_steps=0;self.done=False;self.last_evaluation={}
        self.policy_images={};self.writers={};self.logs=[];self.rig=None;self.app=None
        from environment.runtime.events import EventLog
        self.native_events=EventLog(self.out/'events/native-state.jsonl')
        self.robot_key=os.environ.get('WORLD_BENCH2DEX_ROBOT',ROBOT)
        self.profile=os.environ.get('WORLD_BENCH2DEX_PROFILE','smoke')
        if self.profile not in ('smoke','none'):raise ValueError('Supported profiles: smoke, none with original anchor')
        # Import the unmodified upstream module, including its own AppLauncher.
        # main() is not called: World supplies synchronous tool boundaries.
        os.chdir(SOURCE);sys.path.insert(0,str(SOURCE))
        import isaacsim
        experience=str(Path(isaacsim.__file__).parent/'apps/isaacsim.exp.full.kit')
        import isaaclab.app
        original_launcher=isaaclab.app.AppLauncher
        class CompatibleLauncher(original_launcher):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                from .compat import install
                install()
        isaaclab.app.AppLauncher=CompatibleLauncher
        old=sys.argv
        sys.argv=['run_policy.py','--task','scenes/'+self.task_spec['id']+'.yaml',
                  '--enable-rgb','--headless','--robot-key',self.robot_key,'--experience',experience]
        try:self.up=importlib.import_module('run_policy')
        finally:sys.argv=old
        self.app=self.up.simulation_app
        self.np=self.up.np;self.torch=self.up.torch
        self.task=self.up.load_yaml(str(SOURCE/'scenes'/f"{self.task_spec['id']}.yaml"))
        self.prompt=self.instructions()
        self.prompt+='\nEvaluation scene profile: '+self.profile+'. Public table dimensions are nominal; anchor-specific geometry and camera offsets are not exposed as extra observations.'
        if self.robot_key==ROBOT:
            self.prompt+='\nPublic robot URDF kinematics (model-local coordinates, NOT world/object poses; runtime names/limits remain authoritative):\n'+json.dumps(public_kinematics(),separators=(',',':'))

    def instructions(self):
        return f'''Bench2Dex original task: {self.task_spec['title']}
You control two fixed-base robot arms with independently articulated dexterous fingers, profile {self.robot_key}. No mobile base. Both arms and all active finger joints act simultaneously. The robot, worktable and task props are the original scene; no new grasp, automatic skill or EEF planner is supplied.
Scene: task objects are {', '.join(self.task['assets'])}. Public table specification: {json.dumps(self.task.get('table',{}))}. Observe current RGB to locate props, finger contact surfaces, handles, openings, keyboard keys or tool working ends; no object pose or success-stage oracle is available. Do not infer exact object dimensions from names.
Control: apply_action takes ABSOLUTE active joint targets in radians, not increments, torque or strength percentages. Index order and limits are in the runtime table. Every active arm and finger coordinate must be supplied; preserve measured or last desired targets for joints you do not intend to move. Zero requests joint angle zero, NOT hold. 0.01rad≈0.573deg;0.1rad≈5.73deg. Original actuators, gravity compensation, joint safety hooks and mimic expansion remain active. Commands need not be achieved immediately; use short segments near contacts. A hand is not a binary gripper: coordinate individual finger joints and thumb opposition. Do not assume mirrored joints have the same sign.
Observation: active joint positions plus original configured RGB cameras, including overhead where the pinned run_policy camera allowlist enables it. No depth/tactile sensor is claimed in this RGB evaluation profile. Private object poses, evaluator contacts and stage state are used only for scoring and never supplied to policy. Images include current+4 historical tool observations at interval2; camera videos are recorded every control step separately.
Code feedback: coding_control receives fresh qpos and step/time values at each20Hz tick. RGB is delivered to the model at tool boundaries, not as image arrays inside the restricted callback. Return to the model for a new visual correction when necessary; do not claim image-based20Hz closed-loop control from this callback alone.
Time: policy control20Hz (one step holds targets for3 physics ticks at approximately60Hz). Original success and safety metrics update at EVERY physics tick. A success inside a3-tick group stops immediately; exact physics time is recorded. Initialization settling and60-step homing precede the episode budget, as upstream. Public budget follows expert_time_step×1.5 rounded up to3 physics ticks, not run_policy's generic400-step CLI default.
Complete the exact task sequence and final stable conditions. Preserve native stable-success early stop; reaching the limit without stable success is unsuccessful. Reading observations or computing code does not advance physics. coding_control, when available, computes these SAME absolute joint targets from fresh allowed observations; no hidden simulator or scene access. Runtime Isaac6.0.1 is experimental compatibility, not a claim of official Isaac5.1 physics equivalence.'''

    def reset(self,seed):
        u=self.up
        from utils.seed_policy import task_base_seed, episode_seed
        self.seed=episode_seed(task_base_seed(seed,self.task_spec['upstream_task_id']),1)
        u.seed_everything(self.seed)
        cfg=u.sim_utils.SimulationCfg(dt=0.0166666,device='cuda:0',physx=u.sim_utils.PhysxCfg(
            enable_ccd=True,enable_stabilization=True,bounce_threshold_velocity=0.01,
            gpu_max_rigid_contact_count=2**23,gpu_max_rigid_patch_count=2**22))
        self.sim=u.sim_utils.SimulationContext(cfg);self.physics_dt=self.sim.get_physics_dt();self.dt=self.physics_dt*3
        self.sim.set_camera_view([0.,0.,2.20],[0.,0.,0.82])
        self.anchor=None;self.sample_dict={};generalization_cfg=None;sample=None
        if self.profile=='none':
            import h5py
            from build.generalization import resolve_sample_paths
            self.anchor=ROOT/'anchors'/(self.task_id+'.hdf5')
            with h5py.File(self.anchor,'r') as file:
                self.sample_dict=resolve_sample_paths(json.loads(file['meta/scene_generalization_sample'][()]),str(ROOT))
            u.validate_resolved_object_placement_keys(self.sample_dict,(x['id'] for x in self.task['objects']),required=True)
            gen_path=SOURCE/'configs/scene/generalization.yaml'
            raw=u.load_yaml(str(gen_path));overrides=self.task.get('generalization_overrides')
            if overrides:raw=u.merge_scene_generalization_overrides(raw,overrides)
            raw=u._apply_generalization_profile(raw,profile='none')
            if overrides:raw=u.merge_scene_generalization_overrides(raw,overrides)
            # none replays an exact anchor, never samples the texture catalogue.
            # Disable only unused directory enumeration so unrelated materials
            # need not be downloaded. The anchor's actual texture stays intact.
            raw.setdefault('appearance',{}).setdefault('table_surface',{})['asset_root']=''
            generalization_cfg=u.parse_scene_generalization_config(raw,config_path=str(gen_path))
            sample=u.dict_to_generalization_sample(self.sample_dict);sample.robot_key=self.robot_key
            self.sample_dict=u.scene_generalization_sample_to_dict(sample)
        self.runtime=u.build_scene(self.task,str(SOURCE/'scenes'),generalization_enabled=self.profile=='none',
            generalization_cfg=generalization_cfg,generalization_sample=sample,robot_key=self.robot_key,existing_robot_runtime=None)
        self.objects=self.runtime['interactive_objects']
        self.object_ids=sorted(self.runtime.get('asset_local_bbox',{})) or sorted(x for x in self.objects if x!='global_robot')
        self.hooks=list((self.runtime.get('robot_runtime') or {}).get('pre_step_hooks',[]))
        robot=u.classify_interactive_objects(self.objects).robot_articulation
        cc=u.load_collect_config(str(SOURCE/'configs/collect/default.yaml'),robot_key=self.robot_key)
        cc.cameras=[c for c in cc.cameras if c.camera_id in u._DEFAULT_INFERENCE_CAMERAS]
        self.rig=u.CameraRig(self.sim,cc.cameras,enable_rgb=True,enable_depth=False,
                            robot_articulation=robot,camera_generalization_sample=self.sample_dict)
        groups,self.controlled,_,_,self.hold=u.initialize_scene_runtime_state(sim=self.sim,
            physics_dt=self.physics_dt,interactive_objects=self.objects,
            object_prim_paths=self.runtime.get('object_prim_paths',{}),
            object_display_colors=self.runtime.get('object_display_colors',{}),collector=None,
            app_running_state_fn=lambda:self.app.is_running())
        self.robot=groups.robot_articulation
        self.rig.set_robot_articulation(self.robot);self.rig.initialize_after_reset()
        self.active=u.get_active_dof_info_for_runtime(self.robot_key,list(self.robot.data.joint_names))
        u._active_dof_info=self.active
        self.limits=u.read_joint_limits(self.robot)
        from benchmark.metric_tracker import MetricTracker
        height=self.sample_dict.get('spatial',{}).get('table_height',{})
        offset=float(height.get('height_offset_m',0.) or 0.) if height.get('enabled') else 0.
        self.tracker=MetricTracker(u._metrics_spec_for_episode(self.task['metrics'],self.runtime),
            dt=self.physics_dt,table_height_offset=offset,robot_key=self.robot_key)
        if not self.tracker.available:raise RuntimeError('Native evaluator unavailable')
        self.tracker.record_joint_limit_baseline('post_reset',robot_state=u.read_joint_state(self.robot),joint_limits=self.limits)
        self.contacts=None
        if self.runtime.get('contact_pairs'):
            from collector.contact_sensor_reader import ContactSensorReader
            self.contacts=ContactSensorReader(self.runtime['contact_pairs'],object_prim_paths=self.runtime.get('object_prim_paths',{}))
            if not self.contacts.available:raise RuntimeError('Native contact scoring required but unavailable')
        for _ in range(5):self.sim.render()
        u.move_controlled_articulations_home(self.controlled,self.hold,teleport=True)
        for _ in range(60):self._physics()
        self.tracker.record_joint_limit_baseline('post_home',robot_state=u.read_joint_state(self.robot),joint_limits=self.limits)
        names=self.active.active_joint_names;inds=self.active.active_indices
        low,high=self.limits['soft']
        self.action_metadata={'dim':len(names),'names':names,'lower':low[inds].astype(float).tolist(),
            'upper':high[inds].astype(float).tolist(),'terms':[{'dim':len(names),
            'resolved_joint_names':names,'scale':1.,'offset':0.,'config':{'class_type':'JointPositionAction'}}],
            'semantics':'absolute active joint radians; native mimic expansion; bounds=runtime soft joint limits',
            'robot_key':self.robot_key,'full_dof':self.active.full_dof}
        self.observation_metadata={'qpos':'active joints in action index order, radians',
            'cameras':list(self.rig.camera_ids),'excluded':['object_states','metric_stages','evaluator_contacts'],
            'generalization':self.profile,'anchor':self.anchor.name if self.anchor else None}
        (self.out/'native-scene.json').write_text(json.dumps(self.task,indent=2))
        (self.out/'scene-provenance.json').write_text(json.dumps({'profile':self.profile,
            'anchor':str(self.anchor) if self.anchor else None,'sample':self.sample_dict,
            'private_evaluator_file_not_exposed_to_policy':True},indent=2))
        self.native_events.write('initial_state',{'private_environment_record':True,
            'objects':self.up._get_object_states(self.objects,self.object_ids),
            'robot':self.up.read_joint_state(self.robot),'physics_step':0})
        self._capture()
        return self.observation()

    def _physics(self):
        for hook in self.hooks:hook()
        self.up.write_articulation_targets(self.controlled,self.hold)
        self.up._step_sim_with_mounted_camera_sync(self.sim,self.rig,self.objects,self.physics_dt,render=False)

    def _capture(self):
        before=self.sim.current_time
        self.up._render_camera_sample(self.sim,self.rig)
        # Kit110 can return an empty buffer until its render products have been
        # explicitly submitted. Render retries must never advance physics.
        import carb,omni.replicator.core as rep
        settings=carb.settings.get_settings();key='/app/player/playSimulations';previous=settings.get(key)
        settings.set(key,False)
        try:
            rep.orchestrator.step(delta_time=0.,pause_timeline=False)
            frames=self.up._capture_complete_camera_set(self.rig,self.sim,self.physics_dt)
        finally:settings.set(key,previous)
        if self.sim.current_time!=before:raise RuntimeError('Camera rendering advanced physics')
        self.policy_images={k:self.np.ascontiguousarray(v.rgb[:,:,:3]) for k,v in frames.items()}
        folder=self.out/'video';folder.mkdir(exist_ok=True)
        for name,img in self.policy_images.items():
            if name not in self.writers:
                from PIL import Image
                Image.fromarray(img).save(folder/(name+'-initial.png'))
                h,w=img.shape[:2];log=(folder/(name+'.ffmpeg.log')).open('w');self.logs.append(log)
                self.writers[name]=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo',
                    '-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(1/self.dt),'-i','-',
                    '-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+frag_keyframe+empty_moov',
                    str(folder/(name+'.mp4'))],stdin=subprocess.PIPE,stderr=log)
            self.writers[name].stdin.write(img.tobytes())

    def observation(self):
        return {'control_step':self.steps,'physics_steps':self.physics_steps,
            'elapsed_sim_seconds':self.physics_steps*self.physics_dt,
            'qpos':self.up.select_active(self.up.read_joint_state(self.robot)['qpos'],self.active).astype(float).tolist()}

    def step(self,action):
        from environment.benchmarks.native_project.control import validate_action
        if self.done:raise RuntimeError('Episode already terminated')
        raw=self.np.array(validate_action(action,self.action_metadata),dtype=self.np.float32)
        full=self.up.expand_to_full(raw,self.active)
        target=self.hold[id(self.robot)]
        target[0,:]=self.torch.as_tensor(full,device=target.device,dtype=target.dtype)
        command=self.up._joint_command_context(self.robot,raw,full)
        for _ in range(3):
            self._physics()
            states=self.up._get_object_states(self.objects,self.object_ids)
            if self.contacts:
                self.contacts.update(self.physics_dt)
                for obj,forces in self.contacts.read().items():
                    if obj in states:states[obj]['contact_forces']=forces
            robot_state=self.up.read_joint_state(self.robot)
            self.tracker.update(states,sim_step=self.physics_steps,dt=self.physics_dt,
                robot_state=robot_state,joint_limits=self.limits,joint_command=command)
            self.physics_steps+=1
            self.native_events.write('physics_step',{'private_environment_record':True,
                'physics_step':self.physics_steps,'control_step':self.steps+1,
                'objects':states,'robot':robot_state,'command':command,
                'stable_success':bool(self.tracker.success)})
            if self.tracker.success:
                self.done=True;break
        self.steps+=1
        if self.steps>=self.task_spec['steps']:self.done=True
        self.last_evaluation={'stable_success':bool(self.tracker.success),'physics_steps':self.physics_steps}
        self._capture()
        return self.observation()

    def result(self):
        success=bool(self.tracker.success)
        # A short probe, user cap or infrastructure interruption must not be
        # reported as exhaustion of the original benchmark budget.
        reason='stable_success' if success else ('max_steps' if self.steps>=self.task_spec['steps'] else 'external_stop')
        report=self.tracker.finalize(steps=self.physics_steps,terminated_reason=reason,
            policy_query_count=self.steps,policy_query_count_at_stable_success=self.steps if success else None,
            max_steps=self.task_spec['steps']*3,policy_stride=3,evaluation_protocol='reach_and_stop')
        return {'success':success,'native_metrics':jsonable(asdict(report)),
            'physics_steps':self.physics_steps,'actual_simulated_seconds':self.physics_steps*self.physics_dt,
            'robot_key':self.robot_key,'episode_seed':self.seed,'generalization_profile':self.profile,
            'anchor':str(self.anchor) if self.anchor else None,
            'evaluation_scope':'single official origin anchor / native none protocol; not full50episode score' if self.anchor else 'unanchored scene smoke test; not official none protocol'}

    def close(self):
        for writer in self.writers.values():writer.stdin.close()
        for writer in self.writers.values():
            if writer.wait(timeout=60):raise RuntimeError('Video encoding failed')
        for log in self.logs:log.close()
        if self.rig:self.rig.close()
        if self.app:self.app.close(skip_cleanup=True)


def create_sim(task_id,seed,output,headless=True):return Simulator(task_id,seed,output,headless)
def probe_action(task_id,sim):return sim.observation()['qpos']
