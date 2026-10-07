"""Native Volleyball1v1 with an explicitly World-authored scripted opponent."""
from pathlib import Path
import json
import math
import os

SOURCE = Path(__file__).resolve().parents[3]/'third_party/benchmarks/volleybots/checkout'


def verify_opponent():
    import hashlib
    spec=json.loads((SOURCE.parent/'fixed-rally-protocol.json').read_text())
    for name,digest in spec['source_hashes'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Fixed opponent source hash mismatch: '+name)
    from .fixed_rally_opponent import manifest
    result=manifest()
    if result['version']!=spec['opponent']:
        raise RuntimeError('Fixed opponent version mismatch')
    return result


def instructions(task_id, player=None):
    if task_id != 'T05': raise ValueError(task_id)
    player=int(os.environ.get('WORLD_VOLLEY_PLAYER','1')) if player is None else player
    if player not in (0,1): raise ValueError('player must be 0 or 1')
    from .project import instructions as native_instructions
    prompt=native_instructions(task_id).replace('SHA256-pinned frozen opponent', 'source-hash-pinned World-authored fixed-rally-v1 opponent')
    if player==1:
        for a,b in [('player0','player1'),('drone0','drone1'),('actor_0_wins','actor_1_wins')]:
            prompt=prompt.replace(a,'__swap__').replace(b,a).replace('__swap__',b)
    prompt += f"""
Your native player index is {player}. Native player0 serves initially. In the standard player1 evaluation the fixed opponent serves; stabilize your own aircraft immediately while awaiting the incoming ball. Physics and the 1000-step budget start at the native reset, not after the first contact. The World-authored fixed opponent is a cooperative-return controller, NOT an official trained-policy baseline. Your objective remains winning the native match against that frozen opponent; surviving 20 seconds is a draw, not a win. Isaac Sim 6.0.1 compatibility is experimental, not claimed officially equivalent.
All 37 entries of the 37D observation are in your native symmetric frame: positive X is your own half, negative X the opponent's half. For player1, world X/Y and the quaternion are rotated 180 degrees about Z; Z is unchanged. The spectator frame is not your action frame. Ball position = observation[0:3] - observation[29:32]; ball velocity = observation[32:35]. Observation[35] > 0.5 means your turn. Throttle state indices are 19:23. The opponent's program/source, future actions and referee results are unavailable to your policy.
Vehicle: nominal mass 1.52 kg (base link 1.5 kg), body inertia diagonal [0.0347563,0.0458929,0.0977] kg*m^2. Gravity is -9.81 m/s^2 along Z. Rotor indices 0,1,2,3 have body XY arms [(+0.22,-0.13),(-0.20,+0.13),(+0.22,+0.13),(-0.20,-0.13)] m (nominal YAML values approximately); exact polar arms are [0.255539,0.238537,0.255539,0.238537] and angles [-0.533708,2.565218,0.533708,-2.565218] rad. All push along local +Z; rotor yaw reaction signs are [-1,-1,+1,+1]. Maximum thrust is approximately 6.00319 N each, maximum yaw moment 0.096051 N*m each. Motor state s=(observation[19+i]+1)/2 follows s += 0.43*(sqrt((u+1)/2)-s) each tick, thrust=6.00319*s*s.
For example, all four commands near +0.24 approximately balance nominal weight ONLY when level and at steady throttle. All -1 removes thrust; all +1 accelerates upward when level. Increasing front rotors 0/2 relative to rear 1/3 creates negative body-Y torque; it does not directly command an X position. Angular velocity observations are in your symmetric world frame; convert to body frame before using body-axis torque equations. Commands are absolute normalized actuator targets, not pose increments.
You may write your own attitude/position and ball-feedback controller using coding_control. It calls control(obs,memory) with the current observation dictionary each 0.02-second tick; read obs['native_policy_observation']. Use one control function and explicit loops or list comprehensions: generator expressions such as sum(x for x in values), imports, and nested helper functions are unsupported; use sum([x for x in values]) instead. One call allows up to 250 ticks, and apply_action repeats one four-rotor vector for 1..50 ticks. Persistent controller state within a call belongs in memory; across calls reconstruct from observations or explicitly initialize it. Each opposing action is independently recomputed at every tick. No balance or ball-return program is supplied on your side. The review video is not exposed to the policy.
"""
    if os.environ.get('WORLD_VOLLEY_CODING_CONTROL','0') != '1':
        prompt='\n'.join(line for line in prompt.splitlines() if not line.startswith('You may write your own attitude/position'))
        prompt=prompt.replace('coding_control can implement your feedback each step.', 'coding_control is disabled for this run.')
        prompt+='\nCoding control is disabled. Use apply_action with four rotor targets for 1..50 ticks; observe advances no physics. Offline calculations do not control the robot.'
    return prompt


class Simulator:
    def __init__(self, task_id, seed, output, headless=True, player=None, random_turn=False):
        self.player=int(os.environ.get("WORLD_VOLLEY_PLAYER","1")) if player is None else player
        if self.player not in (0,1): raise ValueError("player must be 0 or 1")
        self.prompt=instructions(task_id,self.player)
        self.opponent_spec=verify_opponent()
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
        import types,sys,importlib
        package=types.ModuleType('volley_bots.envs.competitive')
        package.__path__=[str(SOURCE/'volley_bots/envs/competitive')]
        sys.modules[package.__name__]=package
        importlib.import_module('volley_bots.envs.competitive.volleyball_1v1')
        with initialize_config_dir(config_dir=str(SOURCE/'cfg'),version_base=None):
            cfg=compose(config_name='train',overrides=['task=Volleyball1v1',
                'task.env.num_envs=1','task.action_transform=null',f'seed={seed}',
                f'headless={str(headless).lower()}','wandb.mode=disabled'])
        OmegaConf.resolve(cfg);OmegaConf.set_struct(cfg,False)
        cfg.task.random_turn=random_turn
        self.random_turn=random_turn

        (self.output/'native-config.yaml').write_text(OmegaConf.to_yaml(cfg))
        from volley_bots.envs.isaac_env import IsaacEnv
        self.env=IsaacEnv.REGISTRY[cfg.task.name](cfg,headless=headless)
        self.env.enable_render(True)
        self.dt=float(self.env.dt)*int(cfg.sim.substeps)
        self.policy_images={};self.writer=None;self.frames=0;self.review_wall_height=8.
        self.action_metadata={'dim':4,'names':['rotor_0','rotor_1','rotor_2','rotor_3'],
            'lower':[-1.]*4,'upper':[1.]*4,'description':f'Original player{self.player} Iris raw rotor commands; independent scripted player{1-self.player} opponent.'}
        self.observation_metadata={'condition':'native37Dactor','throttles_in_obs':True,'review_camera_exposed':False}
        names=[('position_world',3),('quaternion_wxyz',4),('linear_velocity_world',3),
               ('angular_velocity_world',3),('heading_world',3),('up_world',3),
               ('rotor_state',4),('drone_minus_anchor',3),('opponent_minus_drone',3),('drone_minus_ball',3),('ball_velocity_world',3),('turn',2)]
        start=0;self.observation_metadata['layout']=[]
        for name,size in names:
            self.observation_metadata['layout'].append({'name':name,'start':start,'size':size})
            start+=size
        from .fixed_rally_opponent import FixedRallyOpponent
        self.controllers=[FixedRallyOpponent(self.env.device,{}) for _ in range(2)]
        self.opponent=self.controllers[1-self.player]
        (self.output/'opponent.json').write_text(json.dumps(self.opponent_spec,indent=2))
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
        self.seed=seed
        self.env.set_seed(seed);self.current=self.env.reset()
        from isaacsim.core.simulation_manager import SimulationManager
        before=self.env.sim.current_time
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
        if before!=self.env.sim.current_time:raise RuntimeError('Reset synchronization advanced physics')
        for controller in self.controllers:
            controller.hold=None;controller.was_turn=False
        self.steps=0;self.done=False;self.return_sum=0.;self.last_evaluation={}
        self._capture()
        return self.observation()

    def observation(self):
        v=self.current[('agents','observation')][0,self.player].detach().cpu().tolist()
        if len(v)!=37 or not all(math.isfinite(x) for x in v):
            raise ValueError('Expected finite37Dnative actor observation')
        return {'control_step':self.steps,'control_dt':self.dt,'player':self.player,'native_policy_observation':v}

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
        from contextlib import nullcontext
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[7.,-9.,5.5],target=[0.,0.,1.7])
        before=self.env.sim.current_time
        with nullcontext():
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
        own=self.torch.tensor(action,device=self.env.device,dtype=self.torch.float32).reshape(1,4)
        opponent=self.opponent(self.current[('agents','observation')][:,1-self.player])
        if tuple(opponent.shape)!=(1,4) or not self.torch.isfinite(opponent).all() or (opponent.abs()>1).any():
            raise RuntimeError('Fixed opponent produced invalid raw rotor action')
        pair=[own,opponent] if self.player==0 else [opponent,own]
        a=self.torch.stack(pair,dim=1)
        before=self.env.sim.current_time
        self.current=self.env.step(TensorDict({'agents':{'action':a}},batch_size=[1],device=self.env.device))['next']
        if abs(self.env.sim.current_time-before-self.dt)>1e-6:raise RuntimeError('Action dt changed')
        self.steps+=1;self.done=bool(self.current['done'].any().item())
        reward=float(self.current[('agents','reward')][0,self.player].sum().item());self.return_sum+=reward
        self.last_evaluation={'reward':reward,'return':self.return_sum,
            'terminated':bool(self.current['terminated'].any().item()),
            'truncated':bool(self.current['truncated'].any().item()),
            'stats':{str(k):v.detach().cpu().tolist() for k,v in self.current['stats'].items()},
            'opponent_action':opponent.detach().cpu().tolist(),
            'actor_0_wins':bool(self.current[('stats','actor_0_wins')].any()),
            'actor_1_wins':bool(self.current[('stats','actor_1_wins')].any()),
            'native_success': bool(self.current[('stats',f'actor_{self.player}_wins')].any()) if self.done else None}
        with (self.output/'both-player-actions.jsonl').open('a') as f:
            f.write(json.dumps({'step':self.steps,'player0':pair[0].tolist(),'player1':pair[1].tolist()})+'\n')
        self._capture();return self.observation()

    def result(self):
        stats=self.last_evaluation.get('stats',{})
        labels=['wrong_turn','non_racket_contact','crossed_half_court','ball_ground_own_half','ball_out','ball_net']
        faults={str(player):[label for i,label in enumerate(labels,1) if stats.get(f'drone_{player}_case_{i}',[[0.]])[0][0]] for player in (0,1)}
        metrics={'hits_total':stats.get('num_hits',[[0.]])[0][0],
                 'hits_controlled':stats.get(f'num_hits_drone_{self.player}',[[0.]])[0][0],
                 'hits_opponent':stats.get(f'num_hits_drone_{1-self.player}',[[0.]])[0][0]}
        return {'control_steps':self.steps,'control_dt':self.dt,'simulated_seconds':self.steps*self.dt,
            'success':self.last_evaluation.get('native_success'),'native_success_available':True,
            'outcome':('win' if self.last_evaluation.get(f'actor_{self.player}_wins') else 'loss' if self.last_evaluation.get(f'actor_{1-self.player}_wins') else 'draw_or_no_winner') if self.done else 'unfinished','evaluation':self.last_evaluation,
            'protocol':'volleybots-fixed-rally-receive-v2' if self.player==1 else 'volleybots-fixed-rally-v1','protocol_parameters':{'seed':self.seed,'controlled_player':self.player,'random_turn':self.random_turn,'native_step_limit':1000},
            'success_definition':'Native terminal actor_<controlled_player>_wins; survival, draws and successful tool calls are not wins',
            'native_metrics':metrics,'faults':faults,
            'controlled_player':self.player,'native_task':'Volleyball1v1','opponent':self.opponent_spec,'opponent_variant':self.opponent_spec['version'],'video_frames':self.frames,
            'official_physics_equivalence':False,'review_camera_exposed':False}

    def close(self):
        if self.writer:self.writer.close()
        self.env.close();self.app.close(skip_cleanup=True)


def create_sim(task_id,seed,output,headless=True):
    return Simulator(task_id,seed,output,headless)


def probe_action(task_id,sim):
    return sim.controllers[sim.player](sim.current[('agents','observation')][:,sim.player])[0].tolist()


def reference_action(sim):
    return sim.controllers[sim.player](sim.current[('agents','observation')][:,sim.player])[0].tolist()
