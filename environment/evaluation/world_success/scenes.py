"""World-owned scene overlays; the upstream checkout remains byte-identical."""
import json
import math
import os
import random
from pathlib import Path
from .profiles import get_profile, VERSION

# Geometry-dependent designs require an audited scene before becoming comparable.
# Never evaluate a new route against an unrelated native random-command scene.
READY = {'wheeledlab/visual','wheel_legged/wheel_legged_rough_terrain','omniisaacgymenvs/anymal_rough_terrain','steadytray/tray_balancing_walk','flamingo/wheel_legged_jump_balance','digit/digit_walk_hand_tracking','wheeledlab/mushr-drift','wheeledlab/f1tenth-drift','go2_push/quadruped_push_recovery', 'robot_lab/a1_front_leg_handstand',
         'wheeled_quadruped/rear_wheel_upright_balance',
         'wheel_legged/wheel_legged_upright_recovery',
         'omnidrones/drone_payload_hover', 'omnidrones/drone_inverted_pendulum_tracking',
         'volleybots/drone_volleyball_solo_juggle', 'ttrl/humanoid_table_tennis_return'}


def active(): return os.environ.get('WORLD_SCORING_PROFILE') == VERSION


def configure(benchmark, task, cfg, output=None):
    """Called before environment construction, only for explicit World profiles."""
    if not active(): return cfg
    p=get_profile(benchmark,task)
    if not p: return cfg
    key=benchmark+'/'+p['task']
    if key not in READY: raise NotImplementedError('World scene calibration pending: '+key)
    if benchmark=='omniisaacgymenvs':
        return cfg
    if benchmark in ('omnidrones','volleybots'):
        cfg.task.env.max_episode_length=p['steps']
        # OmegaConf.resolve materializes ${task.env}; the engine reads cfg.env,
        # so updating only the task subtree leaves the old native timeout active.
        cfg.env.max_episode_length=p['steps']
    elif benchmark=='ttrl':
        cfg.scene.max_episode_length_s=p['duration']
        # Native constructor increments serve count for initial launch; let the
        # evaluator stop after seven COMPLETED serves rather than seven launches.
        cfg.ball.max_serve_per_episode=8
    else:
        cfg.episode_length_s=p['duration']
        cfg.world_scoring_profile=VERSION
        if benchmark=='steadytray':
            from .steadytray_events import configure_pushes
            configure_pushes(cfg)
        if benchmark=='wheel_legged' and p['task']=='wheel_legged_rough_terrain':
            from .terrain import configure as configure_terrain
            configure_terrain(cfg)
        if benchmark=='wheeledlab' and task=='visual':
            from .visual_road import configure as configure_road
            configure_road(cfg,output)
        if benchmark=='wheeledlab' and task!='visual':
            from isaaclab.managers import EventTermCfg
            from .routes import reset_drift
            cfg.events.reset_root_state=EventTermCfg(func=reset_drift,mode='reset')
        if benchmark in ('go2_push','robot_lab','wheel_legged'):
            for name,value in list(vars(cfg.events).items()):
                if 'push' in name and value is not None: setattr(cfg.events,name,None)
        if benchmark=='go2_push':
            for name,value in list(vars(cfg.curriculum).items()):
                if 'push' in name:setattr(cfg.curriculum,name,None)
        if benchmark=='wheeled_quadruped':
            from .wheeled_balance import configure_push
            configure_push(cfg)
            from isaaclab.sensors import ContactSensorCfg
            cfg.scene.robot.spawn.activate_contact_sensors=True
            cfg.scene.contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=1,update_period=0.)
        # A1 is explicitly stationary in this new task. Go2 gets a goal observation;
        # old sampled locomotion reward is diagnostic, not the new success judge.
        if benchmark=='digit':
            cfg.commands.left_ee_pose.resampling_time_range=(1000.,1000.)
            cfg.commands.right_ee_pose.resampling_time_range=(1000.,1000.)
        if benchmark=='robot_lab':
            cmd=cfg.commands.base_velocity
            cmd.ranges.lin_vel_x=(0.,0.);cmd.ranges.lin_vel_y=(0.,0.)
            cmd.ranges.ang_vel_z=(0.,0.);cmd.heading_command=False
    return cfg


class JumpCounter:
    def __init__(self):
        self.completed=0;self.slot=-1;self.start_height=None;self.airborne=False;self.counted=False
    def update(self,t,z,airborne,supported):
        slot=1 if t>=10 else 0 if t>=3 else -1
        if slot<0:return
        if slot!=self.slot:
            self.slot=slot;self.start_height=z;self.airborne=False;self.counted=False
        if not self.counted and airborne and z-self.start_height>=.08:self.airborne=True
        if self.airborne and supported and not self.counted:
            self.completed+=1;self.counted=True


class Scene:
    def __init__(self,sim,profile,seed,enabled):
        self.sim,self.profile,self.enabled=sim,profile,enabled
        self.key=profile['benchmark']+'/'+profile['task']
        self.verified=enabled and self.key in READY
        self.failure=None;self.goal=None;self.origin=None;self.forward=(1.,0.);self.jumps=JumpCounter()
        self.route=None;self.corridor_half_width=None
        self.events=[];self.rng=random.Random(seed);self.directions={}
        self.seed=seed
        if enabled and not self.verified:raise NotImplementedError('World scene not ready: '+self.key)
        for t in [4,5,6,9,11,14]:
            theta=self.rng.uniform(-math.pi,math.pi);self.directions[t]=(math.cos(theta),math.sin(theta))
        self.road=None
        if profile['benchmark']=='wheeledlab' and profile['task']=='visual':
            from .visual_road import Road
            self.road=Road()
        self.track=None
        if profile['benchmark']=='wheeledlab' and profile['task'].endswith('-drift'):
            from .routes import Stadium
            self.track=Stadium(sim.dt)
        if enabled:self.install()

    def install(self):
        e=self.sim.env;b=self.profile['benchmark'];task=self.profile['task']
        if b=='omniisaacgymenvs':
            t=self.sim.task
            # Freeze one original rough-slope tile, retaining its native mesh.
            t.terrain_levels[:]=3;t.terrain_types[:]=3
            t.env_origins[:]=t.terrain_origins[3,3]
            t.update_terrain_level=lambda env_ids:None
            t.base_init_state[0]=-2.
            t.max_episode_length=self.profile['steps']+100
        if b=='omnidrones':
            if task=='drone_payload_hover':
                def push():
                    # Preserve original Normal sampling and asymmetric clamp_max;
                    # schedule two impulses at the first tick at/after 2/4.5s.
                    step=int(e.progress_buf[0].item())
                    import torch
                    mask=torch.zeros(e.num_envs,device=e.device)
                    if step in {math.ceil(2/self.sim.dt),math.ceil(4.5/self.sim.dt)}:
                        mask[0]=1
                        self.events.append({'tick':step,'type':'native_payload_force'})
                    forces=e.push_force_dist.sample((e.num_envs,)).clamp_max(e.push_force_dist.scale*3)*e.payload_masses*mask.unsqueeze(-1)
                    e.payload.apply_forces(forces)
                e._push_payload=push
            else:
                import types
                def trajectory(env,steps,env_ids=None,step_size=1.):
                    import torch
                    ids=... if env_ids is None else env_ids
                    ts=(env.progress_buf[ids].unsqueeze(1)+step_size*torch.arange(steps,device=env.device))*self.sim.dt
                    tau=(ts-.5).clamp(0.,8.)
                    p=env.origin.expand(ts.shape+(3,)).clone()
                    p[...,0]+=.30*torch.sin(2*math.pi*tau/8)
                    p[...,1]+=.20*torch.sin(4*math.pi*tau/8)
                    return p
                e._compute_traj=types.MethodType(trajectory,e)

    def initialize_contact(self, original_step):
        if not self.enabled or self.profile['benchmark']!='steadytray':return
        # Native reset explicitly leaves a 2cm object/tray gap. A whole-episode
        # support rule cannot start before physical contact has been initialized.
        # At most 0.5s initialization with native zero-residual joint targets, not
        # a policy rollout; never suppress native failure or teleport an object.
        sim=self.sim;e=sim.env
        from .readers import measure
        samples=[]
        for _ in range(round(.5/sim.dt)):
            if sim.done:raise RuntimeError('SteadyTray contact initialization hit native failure')
            original_step([0.]*sim.action_metadata['dim'])
            state=measure(sim,'steadytray',self.profile['task'],self)
            samples.append(dict(step=sim.steps,state=state))
            if state.get('object_supported') and state['object_tilt']<=math.radians(20):break
        Path(sim.output,'world-initialization-trace.json').write_text(json.dumps(samples,indent=2))
        if not state.get('object_supported') or state['object_tilt']>math.radians(20):
            raise RuntimeError('SteadyTray reset did not establish upright physical tray support; cannot score an always-supported episode')
        Path(sim.output,'world-initialization.json').write_text(json.dumps(dict(
            scored=False,physics_steps=sim.steps,seconds=sim.steps*sim.dt,
            action='zero native joint residual',post_state=state),indent=2))
        sim.steps=0;sim.reward_sum=0.;sim.reward_components={}
        e.episode_length_buf[:]=0
        # Clear evaluator accumulators; do not reset the physical robot/object.
        if hasattr(e,'_world_history'):del e._world_history
        sim.last_evaluation={}

    def after_reset(self):
        if not self.enabled:return
        e=self.sim.env;b=self.profile['benchmark']
        if b=='steadytray':
            from .steadytray_events import TrayPushes
            self.tray_pushes=TrayPushes(self.sim, e.cfg.seed, self.events)
        if b=='wheeled_quadruped':
            from .wheeled_balance import SinglePush
            self.balance_push=SinglePush(self.sim, e.cfg.seed, self.events)
        if b=='omniisaacgymenvs':
            from .readers import vector
            t=self.sim.task;self.origin=vector(t.base_pos[0]);self.forward=(1.,0.)
            from .anymal_timing import install_push_timing
            install_push_timing(self.sim, self.events)
            self.goal=[self.origin[0]+4.,self.origin[1]]
            from .routes import Gates
            self.route=Gates([((self.origin[0]+x,self.origin[1]),(1.,0.),.8) for x in (1.4,2.8)])
            self.corridor_half_width=.8
        if hasattr(e,'scene') and 'robot' in e.scene.keys():
            from .readers import vector
            d=e.scene['robot'].data
            self.origin=vector(d.root_pos_w[0])
            w,x,y,z=vector(d.root_quat_w[0]);yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            self.forward=(math.cos(yaw),math.sin(yaw))
            if b in ('go2_push','digit','steadytray'):
                distance={'go2_push':2.5,'digit':2.,'steadytray':3.}[b]
                self.goal=[self.origin[i]+distance*self.forward[i] for i in range(2)]
                self.mark_goal()
        if b=='wheel_legged' and self.profile['task']=='wheel_legged_rough_terrain':
            self.forward=(1.,0.);self.goal=[self.origin[0]+3.,self.origin[1]]
            from .routes import Gates
            self.route=Gates([((self.origin[0]+x,self.origin[1]),(1.,0.),.6) for x in (.75,1.85)])
            self.corridor_half_width=.6
            self.mark_goal()
        if b=='digit':
            self.install_hand_targets()
        if b=='flamingo':
            self.install_jump_commands()
        if self.track is not None:
            from .routes import draw_stadium
            draw_stadium(e.sim.stage)
        manifest={'version':VERSION,'benchmark':b,'task':self.profile['task'],
            'scene_verified':self.verified,'seed':self.seed,'origin':self.origin,'goal':self.goal,
            'upstream_source_modified':False,'step_limit':self.profile['steps'],'dt':self.sim.dt,
            'scene_profile':'RobotWorld-authored; not official benchmark equivalence',
            'contact_support_threshold_N':1.0,'minimum_filtered_tray_contact_N':.0001}
        Path(self.sim.output,'world-scene.json').write_text(json.dumps(manifest,indent=2))

    def install_jump_commands(self):
        e=self.sim.env
        term=e.command_manager.get_term('event')
        def compute(dt):
            t=float(e.episode_length_buf[0])*self.sim.dt
            start=10. if t>=10 else 3.
            elapsed=t-start
            term.event_command[:,0]=float(0<=elapsed<=1.2)
            term.event_command[:,1]=max(0.,elapsed) if elapsed<=1.2 else 0.
        term.compute=compute
        compute(0)
        for name in e.command_manager.active_terms:
            if name=='base_velocity':
                velocity=e.command_manager.get_term(name);original=velocity.compute
                def stationary(dt):
                    original(dt);velocity.command[:,:3]=0
                velocity.compute=stationary;velocity.command[:,:3]=0
        self.sim._observe(e.observation_manager.compute())

    def install_hand_targets(self):
        import torch
        from isaaclab.utils.math import subtract_frame_transforms
        e=self.sim.env;robot=e.scene['robot'];d=robot.data
        targets={}
        for side,phase in [('left',0.),('right',math.pi/2)]:
            term=e.command_manager.get_term(side+'_ee_pose')
            ids,_=robot.find_bodies(side+'_arm_wrist_yaw')
            p,q=subtract_frame_transforms(d.root_pos_w,d.root_quat_w,d.body_pos_w[:,ids[0]],d.body_quat_w[:,ids[0]])
            origin=torch.cat((p,q),dim=-1).clone()
            targets[side]=origin[0].detach().cpu().tolist()
            def command_update(dt,term=term,origin=origin,phase=phase):
                t=float(e.episode_length_buf[0])*self.sim.dt
                tau=max(0.,min(10.,t-2.))
                term.pose_command_b[:]=origin
                # Smooth bounded x displacement; starts and ends at the reachable
                # reset wrist pose. Right hand uses the approved phase shift.
                ramp=min(1.,max(0.,(t-1.5)/.5))
                term.pose_command_b[:,0]+=.05*math.sin(2*math.pi*tau/5+phase)*ramp
                term._update_metrics()
            term.compute=command_update;command_update(0.)
        Path(self.sim.output,'world-hand-targets.json').write_text(json.dumps(dict(
            root_relative_reference_poses=targets,axis='root x',amplitude_m=.05,period_s=5,
            phase_difference_rad=math.pi/2,calibration='reset FK pose; dynamic reachability still requires reference-control validation'),indent=2))
        self.sim._observe(e.observation_manager.compute())

    def mark_goal(self):
        from pxr import UsdGeom, Gf
        stage=self.sim.env.sim.stage
        for name,x,y,z,sx,sy,color in [('goal',*self.goal,float(self.sim.env.scene.env_origins[0,2])+.002,.5,.5,(.1,.8,.1))]:
            # Ground-level marker only; no collision or physical support.
            prim=UsdGeom.Cube.Define(stage,'/World/WorldSuccess/'+name)
            prim.GetSizeAttr().Set(1.)
            prim.AddTranslateOp().Set(Gf.Vec3d(x,y,z));prim.AddScaleOp().Set(Gf.Vec3f(sx,sy,.002))
            prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def before_step(self):
        if not self.enabled:return
        e=self.sim.env;b=self.profile['benchmark'];step=self.sim.steps;hz=self.profile['hz']
        if b=='steadytray':self.tray_pushes.before_step()
        if b=='wheeled_quadruped':self.balance_push.before_step()
        schedule={'robot_lab':([5],.3), 'wheel_legged':([4,9,14],.6)}
        if b=='wheel_legged' and self.profile['task']=='wheel_legged_rough_terrain':schedule[b]=([5,11],.4)
        if b in schedule:
            times,amount=schedule[b]
            for t in times:
                if step==round(t*hz):
                    robot=e.scene['robot'];v=robot.data.root_vel_w.clone()
                    dx,dy=self.directions[t];v[:,0]+=amount*dx;v[:,1]+=amount*dy
                    robot.write_root_velocity_to_sim(v)
                    if b=='wheel_legged':
                        # Native bookkeeping tensors are created by inference-mode steps.
                        import torch
                        with torch.inference_mode():e.start_recovery_attempt([0])
                    self.events.append(dict(type='world_velocity_increment',time_s=t,delta_v=[amount*dx,amount*dy,0.]))
        if b=='go2_push':
            import torch
            robot=e.scene['robot'];f=torch.zeros((1,1,3),device=e.device)
            for t in (6,14):
                if round(t*hz)<=step<round((t+.2)*hz):
                    f[0,0,0]=30*self.directions[t][0];f[0,0,1]=30*self.directions[t][1]
                    if step==round(t*hz):self.events.append(dict(type='world_force',time_s=t,newtons=30,duration_s=.2))
            robot.set_external_force_and_torque(f,torch.zeros_like(f),body_ids=[0],is_global=True)

    def spatial(self,pos,out):
        if self.route is not None:
            self.route.update(pos);out['route_complete']=self.route.complete
        if self.goal is not None:out['goal_distance']=math.hypot(pos[0]-self.goal[0],pos[1]-self.goal[1])
        if not self.enabled or self.origin is None:return
        dx,dy=pos[0]-self.origin[0],pos[1]-self.origin[1]
        if self.corridor_half_width is not None and abs(dy)>self.corridor_half_width:self.failure='world_corridor_exit'
        if self.profile['benchmark']=='go2_push':
            across=-dx*self.forward[1]+dy*self.forward[0]
            if abs(across)>.6:self.failure='world_corridor_exit'

    def public_observation(self):
        if not self.enabled or self.goal is None:return {}
        from .readers import vector
        if self.profile['benchmark']=='omniisaacgymenvs':
            p=vector(self.sim.task.base_pos[0]);q=self.sim.task.base_quat[0]
        else:
            d=self.sim.env.scene['robot'].data;p=vector(d.root_pos_w[0]);q=d.root_quat_w[0]
        w,x,y,z=vector(q);yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        dx,dy=self.goal[0]-p[0],self.goal[1]-p[1]
        return {'world_goal_relative_body_m':[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy],
                'goal_observation_condition':'World-added ideal relative localization; not original actor observation'}
