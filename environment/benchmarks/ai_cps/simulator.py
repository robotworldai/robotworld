"""Single-environment legacy OIGE execution order, original tasks and configurations."""
import importlib, json, sys
from pathlib import Path
from types import SimpleNamespace

TASKS={'22':('FrankaBallCatching','Franka_Ball_Catching'), '23':('FrankaBallBalancing','Franka_Ball_Balancing'),
       '24':('FrankaPegInHole','Franka_Peg_In_Hole'),'34':('FrankaPegInHole','Franka_Peg_In_Hole')}
class Simulator:
    def __init__(self,root,case,seed,output,noise=True):
        import numpy as np, torch
        from omegaconf import OmegaConf
        from isaacsim.core.api import World
        from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
        self.torch=torch;self.case=case;self.steps=0;self.done=False;self.camera=None;self.recovery=None;self.contact_force=0.;self.output=Path(output)
        self.task_name,module=TASKS[case];gym=Path(root)/'Gym_Envs';sys.path.insert(0,str(gym))
        for key,fn in {'eq':lambda x,y:str(x).lower()==str(y).lower(),'resolve_default':lambda default,arg:default if arg=='' else arg}.items():
            if not OmegaConf.has_resolver(key):OmegaConf.register_new_resolver(key,fn)
        cfg=OmegaConf.load(gym/'cfg/config.yaml');del cfg['defaults'];del cfg['wandb_name']
        cfg.task=OmegaConf.load(gym/'cfg/task'/f'{self.task_name}.yaml')
        cfg.num_envs=1;cfg.seed=seed;cfg.headless=True
        # Cameras are added externally without enabling training camera writers.
        resolved=OmegaConf.to_container(cfg,resolve=True)
        config=SimConfig(resolved)
        np.random.seed(seed);torch.manual_seed(seed)
        env=SimpleNamespace(_render=False)
        self.world=World(stage_units_in_meters=1.0,physics_dt=.0083,rendering_dt=.0166,backend='torch',device=config.config['sim_device'],sim_params=config.get_physics_params())
        env._world=self.world;env.world=self.world
        task_class=getattr(importlib.import_module('Tasks.'+module),self.task_name+'Task')
        class ObservedTask(task_class):
            def set_up_scene(task,scene):
                super().set_up_scene(scene)
                if case in ('24','34'):
                    from isaacsim.core.prims import RigidPrim
                    task.contact_sensor=RigidPrim('/World/envs/env_0/tool/tool/tool',name='world_contact',reset_xform_properties=False,
                      contact_filter_prim_paths_expr=['/World/envs/env_0/table/table/table_mesh'],track_contact_forces=True)
                    scene.add(task.contact_sensor)
            def post_reset(task):
                # Core 6 physics-ready callbacks bypass old initialize overrides.
                task._frankas.initialize(self.world.physics_sim_view)
                super().post_reset()
        self.task=ObservedTask(self.task_name,config,env)
        self.task.set_as_test()
        if noise:self.task.set_action_noise()
        if case=='22':initial=[np.random.rand()*.1-.05,np.random.rand()*.1-.05,1.,0.]
        elif case=='23':initial=[np.random.rand()*.3-.15,np.random.rand()*.3-.15]
        else:initial=[np.random.rand()*.2-.1,np.random.rand()*.2-.1]
        self.task.set_initial_test_value(np.array(initial))
        self.world.add_task(self.task);self.world.reset()
        self.physics_dt=self.world.get_physics_dt();self.control_dt=self.physics_dt*self.task.control_frequency_inv
        if abs(self.physics_dt-1/120)>1e-7:raise RuntimeError('Unexpected physics dt: '+str(self.physics_dt))
        self.task.reset()
        self.trace=[]
        self._step([0.]*9) # original VecEnv.reset performs a zero-action step
        self.steps=0
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'configuration.json').write_text(json.dumps({'case':case,'task':self.task_name,'seed':seed,'initial_test_value':initial,'action_noise':noise,'physics_dt_actual':self.physics_dt,'control_dt_actual':self.control_dt,'noise_std':.5 if noise else 0,'runtime':'isaac6.0.1-experimental','config':resolved},indent=2))
    def _step(self,action):
        t=self.torch
        requested=t.tensor([action],device=self.task.device if hasattr(self.task,'device') else self.task._device,dtype=t.float32)
        before=self.task.franka_dof_targets.clone()
        self.task.pre_physics_step(t.clamp(requested,-1,1))
        self.speed_limited=False
        if self.recovery and self.recovery.cancel_step is not None:
            # ID34 authored controller gate, applied AFTER upstream noise, never changing assets/physics.
            target=self.task.franka_dof_targets
            target[:,:7]=before[:,:7]+t.clamp(target[:,:7]-before[:,:7],-.00625,.00625)
            self.task._frankas.set_joint_position_targets(target)
            self.speed_limited=True
        self.target_delta=(self.task.franka_dof_targets-before)[0].cpu().tolist()
        forces=[]
        for _ in range(self.task.control_frequency_inv):
            self.world.step(render=False)
            if hasattr(self.task,'contact_sensor'):
                force=float(t.linalg.vector_norm(self.task.contact_sensor.get_contact_force_matrix(dt=self.physics_dt)).cpu())
                forces.append(force)
                if self.recovery:self.recovery.contact(force,self.steps+1)
        self.contact_force=max(forces,default=0.)
        self.contact_substeps=forces
        if self.recovery:self.recovery.tick(self.contact_force)
        obs,reward,done,extras=self.task.post_physics_step()
        self.trace.append(t.clamp(obs,-5,5)[0].cpu().tolist())
        self.done=bool(done[0]);self.steps+=1
        if self.camera:self.camera.capture()
        return self.observation()
    def step(self,action):
        if self.done:raise RuntimeError('Episode terminated; no autoreset allowed')
        return self._step(action)
    def observation(self):
        t=self.task
        def values(x):return x.detach().cpu().tolist()
        return {'control_step':self.steps,'dt':self.control_dt,'physics_time_s':float(self.world.current_time),'native_observation':self.trace[-1],
          'joint_positions_rad':values(t._frankas.get_joint_positions()[0,:7]),
          'finger_positions_m':values(t._frankas.get_joint_positions()[0,7:]),
          'joint_velocities_rad_s':values(t._frankas.get_joint_velocities()[0,:7]),
          'finger_velocities_m_s':values(t._frankas.get_joint_velocities()[0,7:]),
          'joint_targets_rad':values(t.franka_dof_targets[0,:7]),
          'finger_targets_m':values(t.franka_dof_targets[0,7:]),
          'joint_limits_rad':values(t._frankas.get_dof_limits()[0,:7]),
          'last_noisy_native_action':values(t.actions[0]),'episode_terminated':self.done,
          'peg_table_contact_peak_N':self.contact_force,'peg_table_contact_substeps_N':self.contact_substeps,
          'target_delta_rad':self.target_delta[:7],'finger_target_delta_m':self.target_delta[7:],'degraded_control':self.speed_limited,
          'recovery':self.recovery.observation() if self.recovery else None}
