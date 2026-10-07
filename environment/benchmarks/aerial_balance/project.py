"""Native disturbance YAML with the original DirectRLEnv and evaluator."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]/'third_party/benchmarks/aerial_balance/checkout'
FIELDS = ['pb','vb','ab','theta','omega','alpha','drz','vrz','arz','pg','a_prev']


def configure(cfg, data, seed):
    """Same YAML-to-config mapping as upstream zero_action_policy_eval, no script side effects."""
    env = data.get('env',{})
    cfg.seed = seed
    cfg.task_name = data.get('task_name',cfg.task_name)
    cfg.interface_name = data.get('interface_name',cfg.interface_name)
    cfg.episode_length_s = float(env.get('episode_length_s',cfg.episode_length_s))
    cfg.scene.num_envs = 1
    cfg.sim.device = env.get('device',cfg.sim.device)
    if 'rope_length' in env:
        cfg.rope_length = float(env['rope_length'])
    for key in ('dt',):
        if key in data.get('sim',{}):
            setattr(cfg.sim,key,data['sim'][key])
    if 'decimation' in data.get('sim',{}):
        cfg.decimation = int(data['sim']['decimation'])
        cfg.sim.render_interval = cfg.decimation
    for section in ['target_position_task','trajectory_tracking_task','velocity_interface','position_interface',
                    'thrust_interface','robustness','target_position_evaluator','trajectory_tracking_evaluator']:
        values = data.get(section,{})
        for key,value in values.items():
            if value is not None:
                setattr(getattr(cfg,section),key,value)
    for section in ['target_position_evaluator','trajectory_tracking_evaluator']:
        getattr(cfg,section).episode_length_s = cfg.episode_length_s
        getattr(cfg,section).num_eval_episodes = 1
    return cfg


def instructions(task_id):
    if task_id!='T04':
        raise ValueError(task_id)
    return '''Control the original Aerial-Balance-Bench target-position task with template_eval_disturbance.yaml.
A vertically constrained drone pulls a rope attached to one end of a beam; a ball rolls along the beam. The opposite end receives a fresh OU velocity disturbance at every physics step. Adjust vertical motion to bring ball position pb to target pg. The robot/rope/beam joints, rotor model and velocity servo are original; no task-balancing policy is supplied.
Native action has ONE scalar: change in desired vertical velocity (m/s), clipped to +/-max_acc*step_dt = +/-0.0083333333. This is an INCREMENT accumulated by the velocity interface, not absolute velocity, normalized throttle or beam angle. Zero means no new increment, not stop. Native max_velocity=0 disables velocity clipping. Physics dt=1/180s;3 substeps per action, control60Hz;600 actions=10s. The original15-step action delay and OU disturbance remain enabled.
Observation is 11 original scalars: pb ball position m, vb velocity m/s, ab acceleration m/s2, theta beam angle rad, omega angular velocity rad/s, alpha angular acceleration rad/s2, drz drone vertical displacement m, vrz velocity, arz acceleration, pg target position, a_prev last applied action. No future disturbance samples or private evaluator flags are supplied. This is state-based, not pure vision. Consider coding_control to close the feedback loop at60Hz while accounting for the delay.
Original evaluator reports target error, convergence time, climbing time and success_rate. Target band is0.01m; the1s window applies to steady-state error, NOT a required1s success dwell. Original success can count a final target-band entry before10s and does not independently exclude all physical failure flags; those flags are reported separately. Do not invent stronger completion criteria. Continue until native end or budget.'''


class Simulator:
    def __init__(self, task_id, seed, output, headless=True):
        if task_id!='T04':
            raise ValueError(task_id)
        manifest = json.loads((SOURCE.parent/'asset-manifest.json').read_text())
        missing = [p for p in manifest.get('missing_author_payloads',[]) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError('Original USD has unresolved author-machine payloads; do not substitute assets: '+str(missing))
        from omni.isaac.lab.app import AppLauncher
        self.app = AppLauncher(headless=headless,enable_cameras=True).app
        import torch
        import yaml
        from environments.aerial_balance_env import AerialBalanceEnv, AerialBalanceEnvCfg
        self.torch = torch
        torch.manual_seed(seed)
        cfg_path = SOURCE/'environments/configs/template_eval_disturbance.yaml'
        data = yaml.safe_load(cfg_path.read_text())
        cfg = configure(AerialBalanceEnvCfg(),data,seed)
        self.output = Path(output)
        (self.output/'native-config.yaml').write_text(yaml.safe_dump(data))
        (self.output/'configuration.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
        class SingleEpisode(AerialBalanceEnv):
            capture_terminal = False
            def _reset_idx(self, ids):
                if not self.capture_terminal:
                    return super()._reset_idx(ids)
        self.env = SingleEpisode(cfg,render_mode='rgb_array')
        self.dt = self.env.step_dt
        limit = float(cfg.velocity_interface.max_acc*self.dt)
        self.action_metadata = {'dim':1,'names':['delta_vertical_velocity_mps'],
                                'lower':[-limit],'upper':[limit],
                                'description':'Native velocity increment,15-step delayed, not absolute velocity.'}
        self.policy_images = {}
        self.prompt = instructions(task_id)
        self.writer = None
        self.frames = 0

    def reset(self, seed):
        self.current, self.info = self.env.reset(seed=seed)
        self.env.capture_terminal = True
        self.steps = 0
        self.done = self.terminated = self.truncated = False
        self.reward_sum = 0.
        self.last_evaluation = {}
        self._capture()
        return self.observation()

    def observation(self):
        values = self.current['policy'][0].detach().cpu().tolist()
        if len(values)!=len(FIELDS):
            raise RuntimeError('Native observation layout changed')
        return {'control_step':self.steps,'dt':self.dt,'native_policy_terms':dict(zip(FIELDS,values)),
                'episode_terminated':self.done}

    def _capture(self):
        import imageio.v2 as imageio
        before = self.env.sim.current_time
        frame = self.env.render()
        if self.env.sim.current_time != before:
            raise RuntimeError('Review rendering advanced physics')
        if frame is None:
            raise RuntimeError('No review pixels from native renderer')
        if self.writer is None:
            (self.output/'video').mkdir(exist_ok=True)
            self.writer = imageio.get_writer(str(self.output/'video/review.mp4'),fps=1/self.dt)
        self.writer.append_data(frame)
        self.frames += 1

    def step(self, action):
        from environment.benchmarks.native_project.control import validate_action
        if self.done:
            raise RuntimeError('Episode ended')
        action = validate_action(action,self.action_metadata)
        with self.torch.inference_mode():
            self.current,reward,terminated,truncated,self.info = self.env.step(
                self.torch.tensor([action],device=self.env.device,dtype=self.torch.float32))
        self.steps += 1
        self.terminated,self.truncated = bool(terminated[0]),bool(truncated[0])
        self.done = self.terminated or self.truncated
        self.reward_sum += float(reward[0])
        metrics = {k:v.item() for k,v in self.info['benchmark'].items()}
        self.last_evaluation = {'reward':float(reward[0]),'native_benchmark':metrics,
                                'terminated':self.terminated,'truncated':self.truncated}
        self._capture()
        return self.observation()

    def result(self):
        metrics = {k:v.item() for k,v in self.env.evaluator.get_metrics().items()}
        return {'control_steps':self.steps,'control_dt':self.dt,'native_horizon':600,
                'native_episode_complete':self.done,'terminated':self.terminated,'truncated':self.truncated,
                'native_reward_sum':self.reward_sum,'native_metrics':metrics,
                'success':bool(metrics['success_rate']) if metrics['completed_episodes'] else None,
                'success_definition':'Original TargetPositionEvaluator success_rate; no additional dwell/failure exclusion.',
                'action_delay_steps':15,'external_disturbance_enabled':True,
                'time_semantics':'physics paused during LLM reasoning; original15-step action delay retained',
                'review_camera_exposed':False,'video_frames':self.frames}

    def close(self):
        if self.writer:
            self.writer.close()
        self.env.close()
        self.app.close()


def create_sim(task_id, seed, output, headless=True):
    return Simulator(task_id,seed,output,headless)
