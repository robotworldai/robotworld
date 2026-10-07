"""Native VolleyBots1v1: model controls player0; frozen independent opponent player1."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'third_party/benchmarks/volleybots/checkout'
OPPONENTS=SOURCE.parent/'opponents'


def preflight():
    manifest=OPPONENTS/'opponent.json'
    if not manifest.exists():
        raise RuntimeError('T05 requires a fixed independent opponent: opponents/opponent.json and its SHA256-pinned TorchScript weights are missing. Bundled checkpoints are3v3 policies or dedicated hierarchical skills; no compatible complete1v1 opponent is configured. Refusing a dummy/static opponent score.')
    spec=json.loads(manifest.read_text())
    weight=(OPPONENTS/spec['file']).resolve()
    if not weight.is_relative_to(OPPONENTS.resolve()):raise ValueError('Opponent weights must be in opponents/')
    if not weight.is_file() or hashlib.sha256(weight.read_bytes()).hexdigest()!=spec['sha256']:
        raise RuntimeError('Opponent weights missing or SHA256 mismatch')
    if spec.get('action_transform') is not None or spec.get('observation_dim')!=37 or spec.get('action_dim')!=4:
        raise ValueError('Opponent must accept native37-vector and emit4 raw rotor actions with action_transform=null')
    if not spec.get('provenance'):raise ValueError('Opponent provenance must be disclosed')
    return spec,weight


def instructions(task_id):
    return '''Play native VolleyBots1v1 as drone0. An independent SHA256-pinned frozen opponent controls drone1 from its OWN native observation at every physics-control tick; you never choose its action. You control all4raw Iris rotor actions[-1,1], action_transform=null; no automatic volley strategy, Serve/Attack skill, hover controller or pretrained action policy assists you. -1 targets zero thrust,+1 maximum; zero is not hover. Native actuator response and physical racket/ball/net collisions are unchanged. One control step=.02s, native horizon1000steps. Thinking pauses physics; coding_control can implement your feedback each step.
Native policy vector37 entries: position3, quaternionwxyz4, worldlinearvelocity3, worldangularvelocity3, heading3, up3, rotorstate4, position-minus-anchor3, opponent-minus-ownposition3, ownposition-minus-ball3, ballvelocity3, native turn encoding2. Units metres/radians/seconds,Z-up. Symmetric observation transformation remains native; you only see player0's original observation, not opponent internals or reward oracle. These are simulator-state measurements, not pure vision. Native court6mx3m, netheight2.43m/netwidth.76m, ballradius.1m/mass.005kg; initial anchors±1.5m alongX at2m altitude. Win/lose/draw/ground/net/out-of-bounds/fault detection is original. Successful tool execution is not a won game; final result uses native actor_0_wins/actor_1_wins and termination. Native time limit without winner is not victory. No resets,teleports,world edits,hidden state reads or shell-based physics control.'''


class Simulator:
    def __init__(self,task_id,seed,output,headless=True):
        if task_id!='T05':raise ValueError(task_id)
        self.opponent_spec,weight=preflight()
        import torch
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from volley_bots import init_simulation_app
        self.torch=torch;self.task_id=task_id;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.policy_images={};self.writer=None
        with initialize_config_dir(config_dir=str(SOURCE/'cfg'),version_base=None):
            cfg=compose(config_name='train',overrides=['task=Volleyball1v1','task.env.num_envs=1',
              'task.action_transform=null',f'seed={seed}',f'headless={str(headless).lower()}','wandb.mode=disabled'])
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        (self.output/'native-config.yaml').write_text(OmegaConf.to_yaml(cfg))
        (self.output/'opponent.json').write_text(json.dumps(self.opponent_spec,indent=2))
        self.app=init_simulation_app(cfg)
        from volley_bots.envs.isaac_env import IsaacEnv
        self.env=IsaacEnv.REGISTRY[cfg.task.name](cfg,headless=headless)
        self.env.enable_render(True)
        self.dt=float(cfg.sim.dt)*int(cfg.sim.substeps)
        self.opponent=torch.jit.load(str(weight),map_location=self.env.device).eval()
        self.action_metadata={'dim':4,'names':['rotor_0','rotor_1','rotor_2','rotor_3'],'lower':[-1]*4,'upper':[1]*4,
          'description':'Native player0 Iris4raw rotor actions, action_transform=null; player1 independent frozen opponent'}
        self.prompt=instructions(task_id)

    def reset(self,seed):
        self.env.set_seed(seed);self.current=self.env.reset();self.steps=0;self.done=False
        self.last_evaluation={'native_success':None};self.return_sum=0.
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
        vector=self.current[('agents','observation')][0,0].detach().cpu().tolist()
        if len(vector)!=37:raise RuntimeError('Native player0 observation shape changed')
        return {'control_step':self.steps,'control_dt':self.dt,'player':0,'native_policy_observation':vector}

    def step(self,action):
        if self.done:raise RuntimeError('Match ended')
        if len(action)!=4 or any(isinstance(x,bool) or not math.isfinite(x) or not -1<=x<=1 for x in action):raise ValueError('4rotors in[-1,1] required')
        from tensordict import TensorDict
        with self.torch.no_grad():
            # Only the opponent's allowed native observations enter its policy.
            opponent_action=self.opponent(self.current[('agents','observation')][:,1])
        if tuple(opponent_action.shape)!=(1,4) or not self.torch.isfinite(opponent_action).all() or (opponent_action.abs()>1).any():
            raise RuntimeError('Frozen opponent produced invalid raw rotor action')
        own=self.torch.tensor(action,device=self.env.device,dtype=self.torch.float32).reshape(1,4)
        actions=self.torch.stack([own,opponent_action],dim=1)
        self.current=self.env.step(TensorDict({'agents':{'action':actions}},batch_size=[1],device=self.env.device))['next']
        self.steps+=1;self.done=bool(self.current['done'].any().item())
        reward=float(self.current[('agents','reward')][0,0].sum().item());self.return_sum+=reward
        stats={str(k):v.detach().cpu().tolist() for k,v in self.current['stats'].items()}
        won=bool(self.current[('stats','actor_0_wins')].any().item())
        lost=bool(self.current[('stats','actor_1_wins')].any().item())
        self.last_evaluation={'reward':reward,'return':self.return_sum,'actor_0_wins':won,'actor_1_wins':lost,
          'native_success':won if self.done else None,'terminated':bool(self.current['terminated'].any().item()),
          'truncated':bool(self.current['truncated'].any().item()),'stats':stats,
          'opponent_action':opponent_action.detach().cpu().tolist()}
        self._capture()
        return self.observation()

    def result(self):
        return {'steps':self.steps,'success':self.last_evaluation.get('native_success'),
          'evaluation':self.last_evaluation,'opponent':self.opponent_spec,'observation_condition':'native_player0_state',
          'action_transform':None,'player':0}

    def close(self):
        if self.writer:self.writer.close()
        self.env.close();self.app.close()


def create_sim(task_id,seed,output,headless=True):return Simulator(task_id,seed,output,headless)
