"""Original SingleJuggleVolleyball, isolated experimental Isaac6 adapter."""
from pathlib import Path
import json
import math

SOURCE = Path(__file__).resolve().parents[3]/'third_party/benchmarks/volleybots/checkout'


def instructions(task_id):
    if task_id != 'T05-single':
        raise ValueError('This profile is only for SingleJuggleVolleyball; 1v1 still requires its own opponent/runtime')
    return '''Control one Iris quadrotor in the original SingleJuggleVolleyball task, NOT a 1v1 match. Keep contacting the ball upward while maintaining controlled flight near the native anchor. There is no opponent and no win-rate claim.
Scene: original court spans18x9m, with net-height reference2.43m. Anchor=[4.5,0,2]m; reset drone position is randomized by x/y±0.5m,z±0.2m around it, roll/pitch±0.1pi and yaw across a full turn. The ball begins2m above the drone; ball radius0.1m,mass0.005kg. These are public configuration constants, not a future-state oracle. It is the native Iris body-contact model; no new racket, automatic ball launcher or bounce assistance is added.
Action: four simultaneous Iris rotor inputs in[-1,1], original ordering. -1 requests zero thrust,+1 maximum,0 half maximum steady thrust, NOT hover. Native motor lag remains. No pretrained stabilizer or juggling policy is supplied. See the rotor action map; use small attitude corrections and measured feedback. Both apply_action and coding_control are available; code returns these same four commands each0.02s control step.
Public Iris model constants from its original YAML: nominal mass1.52kg, inertia diagonal[.0347563,.0458929,.0977]kg m^2; rotor arm lengths[.255539,.238537,.255539,.238537]m and angles[-.533708,2.565218,.533708,-2.565218]rad, spin directions[1,1,-1,-1]. Rotor maximum speed838rad/s, force coefficient8.54858e-6 and moment coefficient1.3677728816219314e-7. Native yaw torque uses negative spin direction. Rotor throttle updates by0.43*(target-current) each tick; thrust is quadratic in internal throttle. These are published model constants, not measured randomized properties or a supplied control law.
Observation:32native values ordered position_world3, quaternion_wxyz4, linear_velocity_world3, angular_velocity_world3, heading_world3, up_world3, drone_position-minus-anchor3, drone_position-minus-ball_position3, ball_linear_velocity_world3, repeated_normalized_episode_time4. Rotor throttle is NOT observed in this task. Z-up, metres and seconds. State is native simulator-derived actor feedback, not RGB-only. Review camera and backdrop are never supplied as policy evidence.
Scoring: original true-hit count and height/reward metrics, not invented binary success. A new true hit requires more than25steps since last hit; a contact within25steps is wrong_hit and terminates. With dt=.02 this is0.5s, despite an old source comment referring to.016s. Ball-height reward uses the native3.5m threshold. Native failures include drone z<.4 or>2.5m, drone abs(x)<.01m, ball z<.15 or>4.5m, ball out of court, and wrong_hit. Preserve all original conditions. Horizon800steps=16s; stop when the original env terminates. Thinking pauses physics. Native contact detection is real simulated contact, not visual proximity. Runtime is experimental Isaac Sim6.0.1; original task code is unchanged but physics equivalence is not claimed.'''


class Simulator:
    def __init__(self, task_id, seed, output, headless=True):
        self.prompt=instructions(task_id)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        from .compat_isaac6 import install, install_python, configure_report
        configure_report(self.output/'compatibility.json');install_python()
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from isaacsim import SimulationApp
        self.torch=torch
        self.app=SimulationApp({'headless':headless,'anti_aliasing':0})
        install(self.app,SOURCE)
        with initialize_config_dir(config_dir=str(SOURCE/'cfg'),version_base=None):
            cfg=compose(config_name='train',overrides=['task=SingleJuggleVolleyball',
                'task.env.num_envs=1','task.action_transform=null',f'seed={seed}',
                f'headless={str(headless).lower()}','wandb.mode=disabled'])
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        from environment.evaluation.world_success.scenes import configure
        configure('volleybots',task_id,cfg)
        (self.output/'native-config.yaml').write_text(OmegaConf.to_yaml(cfg))
        from volley_bots.envs.isaac_env import IsaacEnv
        self.env=IsaacEnv.REGISTRY[cfg.task.name](cfg,headless=headless)
        self.env.enable_render(True)
        self.dt=float(self.env.dt)*int(cfg.sim.substeps)
        self.policy_images={};self.writer=None;self.frames=0;self.review_wall_height=8.
        self.action_metadata={'dim':4,'names':['rotor_0','rotor_1','rotor_2','rotor_3'],
            'lower':[-1.]*4,'upper':[1.]*4,'description':'Original Iris raw rotor commands; no opponent or pretrained controller.'}
        self.observation_metadata={'condition':'native32Dactor','throttles_in_obs':False,'review_camera_exposed':False}
        names=[('position_world',3),('quaternion_wxyz',4),('linear_velocity_world',3),
               ('angular_velocity_world',3),('heading_world',3),('up_world',3),
               ('drone_minus_anchor',3),('drone_minus_ball',3),('ball_velocity_world',3),('time_encoding',4)]
        start=0;self.observation_metadata['layout']=[]
        for name,size in names:
            self.observation_metadata['layout'].append({'name':name,'start':start,'size':size})
            start+=size
        from .force_compat_isaac6 import NativeWrenchBatch
        self.wrenches=NativeWrenchBatch(self.env)
        original_step=self.env.sim.step
        def native_tick(render=True):
            before=self.env.sim.current_time
            self.wrenches.submit();original_step(render=False,update_fabric=True);self.wrenches.clear()
            elapsed=self.env.sim.current_time-before
            if abs(elapsed-self.env.sim.get_physics_dt())>1e-6:
                raise RuntimeError('Native physics tick changed')
            if render:self.env.sim.render()
        self.env.sim.step=native_tick

    def reset(self,seed):
        self.env.set_seed(seed);self.current=self.env.reset()
        self.steps=0;self.done=False;self.return_sum=0.;self.last_evaluation={}
        self._capture()
        return self.observation()

    def observation(self):
        v=self.current[('agents','observation')][0,0].detach().cpu().tolist()
        if len(v)!=32 or not all(math.isfinite(x) for x in v):
            raise ValueError('Expected finite32Dnative actor observation')
        return {'control_step':self.steps,'control_dt':self.dt,'native_policy_observation':v}

    def _capture(self):
        import carb
        settings=carb.settings.get_settings()
        key='/rtx/post/tonemap/filmIso'
        original=settings.get(key)
        if not isinstance(original,(int,float)) or original<=0:
            raise RuntimeError('Cannot audit spectator exposure without a valid original ISO')
        settings.set_float(key,float(original)/4.)
        try:
            return self._capture_exposed()
        finally:
            settings.set(key,original)

    def _capture_exposed(self):
        import imageio.v2 as imageio
        from environment.benchmarks.native_project.review_scene import aerial_review
        before=self.env.sim.current_time
        with aerial_review(self,other='ball'):
            minimum=32 if self.steps==0 else 4
            for i in range(minimum+12):
                self.env.sim.render()
                raw=self.env._rgb_annotator.get_data()
                if i>=minimum-1 and getattr(raw,'ndim',0)==3 and raw.size:
                    frame=self.env.render(mode='rgb_array')
                    break
            else:raise RuntimeError('No review frame')
            if self.env.sim.current_time!=before:raise RuntimeError('Render advanced physics')
            if self.writer is None:
                import carb
                (self.output/'review-exposure.json').write_text(json.dumps({
                    'spectator_only':True,'exposure_stops_relative_to_native':-2,
                    'film_iso_during_capture':carb.settings.get_settings().get('/rtx/post/tonemap/filmIso'),
                    'restored_after_capture':True},indent=2))
                (self.output/'video').mkdir(exist_ok=True)
                self.writer=imageio.get_writer(str(self.output/'video/review.mp4'),fps=1/self.dt)
                imageio.imwrite(self.output/'initial-review.png',frame)
            self.writer.append_data(frame);self.frames+=1

    def step(self,action):
        if self.done:raise RuntimeError('Native episode ended')
        from environment.benchmarks.native_project.control import validate_action
        action=validate_action(action,self.action_metadata)
        from tensordict import TensorDict
        a=self.torch.tensor(action,device=self.env.device,dtype=self.torch.float32).reshape(1,1,4)
        before=self.env.sim.current_time
        self.current=self.env.step(TensorDict({'agents':{'action':a}},batch_size=[1],device=self.env.device))['next']
        if abs(self.env.sim.current_time-before-self.dt)>1e-6:raise RuntimeError('Action dt changed')
        self.steps+=1;self.done=bool(self.current['done'].any().item())
        reward=float(self.current[('agents','reward')].sum().item());self.return_sum+=reward
        self.last_evaluation={'reward':reward,'return':self.return_sum,
            'terminated':bool(self.current['terminated'].any().item()),
            'truncated':bool(self.current['truncated'].any().item()),
            'stats':{str(k):v.detach().cpu().tolist() for k,v in self.current['stats'].items()},
            'true_hits':self.env.num_true_hits.detach().cpu().tolist()}
        self._capture();return self.observation()

    def result(self):
        return {'control_steps':self.steps,'control_dt':self.dt,'simulated_seconds':self.steps*self.dt,
            'success':None,'native_success_available':False,'evaluation':self.last_evaluation,
            'native_task':'SingleJuggleVolleyball','not_1v1':True,'video_frames':self.frames,
            'official_physics_equivalence':False,'review_camera_exposed':False}

    def close(self):
        if self.writer:self.writer.close()
        self.env.close();self.app.close(skip_cleanup=True)


def create_sim(task_id,seed,output,headless=True):
    return Simulator(task_id,seed,output,headless)


def probe_action(task_id,sim):
    return [.1,.1,.1,.1]  # Excitation only; not an author controller.
