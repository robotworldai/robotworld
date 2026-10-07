"""Single native AnymalTerrain episode; source and task dynamics untouched."""
from pathlib import Path
import json, math, hashlib
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'third_party/benchmarks/omniisaacgymenvs/checkout'
ASSETS=SOURCE.parent/'assets'


def preflight():
    manifest=SOURCE.parent/'asset-manifest.json'
    if not manifest.exists():raise RuntimeError('Missing official ANYmal asset closure: run prepare_assets.py')
    for row in json.loads(manifest.read_text()):
        p=ASSETS/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Missing/corrupt original asset: '+str(p))


def instructions(task_id):
    return '''Control ANYmal directly with12native joint-position offsets and native PD, no pretrained gait. Every action target is default_joint_angle+.5*action radians. Native torque is Kp80*(target-q)-Kd2*qdot, clipped to[-80,80]Nm. Native action has no finite clipping bound; send finite, physically sensible commands. Joint order and default angles are in action metadata. Zero action targets the configured nominal stance, not an automatic balance or locomotion policy. All joints move simultaneously.
Follow original commanded body linear/yaw velocity over native procedurally generated rough terrain and recover from native pushes. Standing still does not satisfy moving commands. Native push overwrites base horizontal velocity on its original750-control-step schedule; retain terrain, noise, reward and failure criteria. Native configuration nominally labels20s/1000steps at.02s, but the upstream VecEnv adds its own physics tick after task-internal4ticks: actual simulator elapsed time is measured and reported, not silently corrected. No manual reset, simulator-state access via files, teleport, task editing, or pretrained gait. Native measurements are noisy/scaled state, not pure vision. Observation188-vector order: body linear velocity3(scale2), body angular velocity3(scale.25), projected gravity3, command3(scales2,2,.25), joint positions12(scale1), joint velocities12(scale.05), terrain-relative heights140(clipped[-1,1]then*5), previous action12; native noise retained. Terrain heights are upstream policy sensors, not a added privileged map. coding_control may implement your own feedback per native action; review video is not a policy input. Stop only at native termination or budget. No native binary success exists; evaluate native return, tracking rewards, falls, time limit and whether a push occurred.'''


class Simulator:
    def __init__(self,task_id,seed,output,headless=True):
        if task_id!='T14':raise ValueError(task_id)
        preflight()
        import torch
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        # Import registers the exact upstream Hydra resolvers; no simulator startup.
        import omniisaacgymenvs.utils.hydra_cfg.hydra_utils
        from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
        from omniisaacgymenvs.utils.config_utils.path_utils import get_experience
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.torch=torch;self.task_id=task_id;self.policy_images={};self.writer=None
        with initialize_config_dir(config_dir=str(SOURCE/'omniisaacgymenvs/cfg'),version_base=None):
            cfg=compose(config_name='config',overrides=['task=AnymalTerrain','num_envs=1',f'seed={seed}',f'headless={str(headless).lower()}'])
        cfg_dict=OmegaConf.to_container(cfg,resolve=True)
        (self.output/'native-config.json').write_text(json.dumps(cfg_dict,indent=2))
        experience=get_experience(headless,False,True,True,cfg.kit_app)
        self.env=VecEnvRLGames(headless=headless,sim_device=cfg.device_id,enable_livestream=False,enable_viewport=True,experience=experience)
        from omni.isaac.core.utils.torch.maths import set_seed
        set_seed(seed,torch_deterministic=cfg.torch_deterministic)
        from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
        from omniisaacgymenvs.tasks.anymal_terrain import AnymalTerrainTask
        import omniisaacgymenvs.robots.articulations.anymal as asset_module
        # URI resolver only: original USD bytes and articulation settings unchanged.
        asset_module.get_assets_root_path=lambda: str(ASSETS)
        class SingleEpisode(AnymalTerrainTask):
            prevent_terminal_reset=False
            def reset_idx(self,env_ids):
                if not self.prevent_terminal_reset:
                    return super().reset_idx(env_ids)
        sim_config=SimConfig(cfg_dict)
        self.task=SingleEpisode(name=cfg_dict['task_name'],sim_config=sim_config,env=self.env)
        self.env.set_task(task=self.task,sim_params=sim_config.get_physics_params(),backend='torch',init_sim=True,
                          rendering_dt=sim_config.get_physics_params()['rendering_dt'])
        self.env._render=True  # Spectator recording only; no new policy sensor.
        self.dt=float(cfg.task.sim.dt)*(int(self.task.decimation)+int(self.task.control_frequency_inv))
        self.nominal_dt=float(self.task.dt)
        self.action_metadata={'dim':12,'names':list(self.task.dof_names),'lower':[None]*12,'upper':[None]*12,
          'default_joint_angles':self.task.default_dof_pos[0].detach().cpu().tolist(),
          'description':'Native unbounded finite normalized joint offsets; target=default+.5*action rad; PD80/2; torque clipped80Nm.'}
        self.prompt=instructions(task_id)

    def reset(self,seed):
        self.task.prevent_terminal_reset=False
        self.current=self.env.reset(seed=seed)
        self.task.prevent_terminal_reset=True
        self.steps=0;self.done=False;self.return_sum=0.;self.push_count=0
        self.start_time=float(self.env._world.current_time)
        self.start_native_progress=int(self.task.progress_buf[0].item())
        self.last_evaluation={'native_success':None}
        self._capture()
        return self.observation()

    def _capture(self):
        import imageio.v2 as imageio
        frame=self.env.render(mode='rgb_array')
        if frame is not None and frame.size:
            if self.writer is None:
                (self.output/'video').mkdir(exist_ok=True)
                self.writer=imageio.get_writer(str(self.output/'video/review.mp4'),fps=1/self.dt)
            self.writer.append_data(frame)

    def observation(self):
        vector=self.current['obs'][0].detach().cpu().tolist()
        if len(vector)!=188:raise RuntimeError('Native observation shape changed')
        return {'control_step':self.steps,'control_dt':self.dt,'native_policy_observation':vector}

    def step(self,action):
        if self.done:raise RuntimeError('Episode terminated')
        if len(action)!=12 or any(isinstance(x,bool) or not math.isfinite(x) for x in action):raise ValueError('12 finite joint offsets required')
        before=float(self.env._world.current_time)
        self.current,reward,done,extras=self.env.step(self.torch.tensor(action,device=self.task.rl_device).reshape(1,12))
        elapsed=float(self.env._world.current_time)-before
        if abs(elapsed-self.dt)>1e-5:raise RuntimeError(f'Unexpected native dt {elapsed}, expected{self.dt}')
        self.steps+=1;self.done=bool(done[0].item());self.return_sum+=float(reward[0].item())
        if hasattr(self.task,'world_push_count'):
            self.push_count=self.task.world_push_count
        elif self.task.common_step_counter%self.task.push_interval==0:self.push_count+=1
        self.last_evaluation={'native_success':None,'reward':float(reward[0].item()),'return':self.return_sum,
          'terminated':bool(self.task.has_fallen[0].item()),'truncated':bool(self.task.timeout_buf[0].item()),
          'native_progress':int(self.task.progress_buf[0].item()),'push_count':self.push_count,
          'actual_elapsed_seconds':float(self.env._world.current_time)-self.start_time}
        self._capture()
        return self.observation()

    def result(self):
        return {'steps':self.steps,'success':None,'native_success_available':False,'evaluation':self.last_evaluation,
           'actual_control_dt':self.dt,'native_config_control_dt':self.nominal_dt,
           'native_reset_progress_steps':self.start_native_progress,'observation_condition':'native_noisy_state',
           'terminal_autoreset_suppressed':True}

    def close(self):
        if self.writer:self.writer.close()
        self.env.close()


def create_sim(task_id,seed,output,headless=True):return Simulator(task_id,seed,output,headless)
