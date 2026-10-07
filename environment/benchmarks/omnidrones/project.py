"""Untouched OmniDrones TorchRL environments, raw Hummingbird rotor controls."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/benchmarks/omnidrones/checkout'
TASKS = {'T16': 'Payload/PayloadHover', 'T17': 'InvPendulum/InvPendulumTrack'}


def validate_action(action):
    if not isinstance(action, (list, tuple)) or len(action) != 4:
        raise ValueError('Four rotor commands are required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not -1 <= v <= 1 for v in action):
        raise ValueError('Rotor commands must be finite numbers in [-1,1]')
    return list(action)


def observation_layout(task_id):
    # Exact concatenation in upstream _compute_state_and_obs, get_state has 23 entries.
    drone = ['quaternion_wxyz', 'world_linear_velocity_xyz', 'world_angular_velocity_xyz',
             'heading_world_xyz', 'up_world_xyz', 'normalized_rotor_throttle_4']
    if task_id == 'T16':
        names = [('drone_minus_payload_xyz', 3), ('target_minus_payload_xyz', 3)]
    else:
        names = [('drone_minus_payload_xyz', 3)]
    names += list(zip(drone, [4, 3, 3, 3, 3, 4]))
    if task_id == 'T17':
        names.append(('six_future_reference_minus_payload_xyz', 18))
    names += [('payload_world_velocity_linear_angular', 6), ('time_encoding', 4)]
    offset = 0
    result = []
    for name, size in names:
        result.append({'name': name, 'start': offset, 'size': size})
        offset += size
    return result


def instructions(task_id):
    common = '''You directly control a Hummingbird quadrotor through FOUR normalized rotor commands in native order. action_transform=null; no Lee controller, hover policy, automatic attitude stabilizer or trajectory tracker runs for you. All four commands are simultaneous, each in [-1,1]: -1 targets zero thrust, +1 maximum thrust. Zero is half maximum steady-state thrust, NOT a neutral/hover command. Native actuator first-order throttle response remains active. Body rotor angles [0,pi/2,pi,-pi/2], arm length .17m, spin directions [-1,+1,-1,+1]. Native nominal mass .716kg, inertia diagonal [.007,.007,.012] kg m^2, max rotor speed838 rad/s, thrust coefficient8.54858e-6. Additional payload mass is randomized at reset. All are public model constants, not hidden episode parameters.
Native policy measurements are simulator-state observations (not pure vision). World frame Z-up, quaternion wxyz, angular velocity rad/s, position metres. The observation vector is provided unchanged with a field layout. Normalized rotor throttle is the measured internal motor state. The native config requests .016s, but original Core converts 1/dt to integer62Hz: actual control step is 1/62s (approximately .016129s), with one physics substep. Thinking or writing code pauses physics; coding_control may react at every native control step. Use control(obs,memory) to return bounded action commands; no task controller is supplied. A completed tool call is not task success; continue until native termination or budget. No reset, teleport, physics edits, hidden task state, or file-based simulation control. Native rewards/termination and randomized forces remain untouched. No native binary success score exists: surviving the horizon alone does not demonstrate objective completion.'''
    if task_id == 'T16':
        return common + '''\nTask: stabilize a 1m suspended payload at the original target and suppress swinging under native random payload pushes. Native push selection is independent Bernoulli each step with probability1-exp(-dt/2), not a periodic force every2s. Force standard deviation=[1,1,.5]/dt and the exact original clamping rule remain active. Drone or payload height below.2m, or NaN, terminates. Native horizon500steps is approximately8.0645s of actual physics. Reward combines payload position, uprightness, yaw spin, effort and configured smoothness. Observe payload motion as well as drone attitude.'''
    return common + '''\nTask: keep a1m inverted pendulum upright while its tip follows the native reference trajectory. Six future tip-reference relative positions are sampled at offsets0,5,10,15,20,25 control steps. No extra wind has been added. Native horizon600steps is approximately9.6774s of actual physics; drone altitude<.2m, upward rod component<.2, tip tracking error>.8m, or NaN terminates. Reward uses tip tracking distance and effort; reports tracking-error statistics. Stabilizing only the aircraft does not solve this coupled task.'''


class Simulator:
    def __init__(self, task_id, seed, output, headless=True):
        if task_id not in TASKS:
            raise ValueError(task_id)
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        self.torch = torch
        self.task_id = self.case = task_id
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        with initialize_config_dir(config_dir=str(SOURCE / 'cfg'), version_base=None):
            cfg = compose(config_name='train', overrides=[f'task={TASKS[task_id]}',
                'task.env.num_envs=1', 'task.action_transform=null', f'headless={str(headless).lower()}',
                f'seed={seed}', 'wandb.mode=disabled'])
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        from environment.evaluation.world_success.scenes import configure
        configure('omnidrones',task_id,cfg)
        (self.output / 'native-config.yaml').write_text(OmegaConf.to_yaml(cfg))
        self.app = init_simulation_app(cfg)
        from omni_drones.envs.isaac_env import IsaacEnv
        self.env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=headless)
        self.env.enable_render(True)
        self.configured_physics_dt = float(cfg.sim.dt)
        # Native Core4.1 converts 1/cfg_dt to integer Hz: .016 -> 62 Hz.
        # Report the actual engine dt used by the original env, not cfg rounding.
        self.dt = float(self.env.dt) * int(cfg.sim.substeps)
        self.native_steps = int(cfg.env.max_episode_length)
        self.action_metadata = {'dim': 4, 'names': ['rotor_0','rotor_1','rotor_2','rotor_3'],
            'lower': [-1.0]*4, 'upper': [1.0]*4, 'description': 'Native raw normalized rotor thrust commands; action_transform=null; no stabilizer.'}
        self.policy_images = {}  # Native tasks have no policy camera: spectator render never exposed.
        self.layout = observation_layout(task_id)
        self.writer = None
        self.prompt = instructions(task_id)

    def reset(self, seed):
        self.env.set_seed(seed)
        self.current = self.env.reset()
        self.steps = 0
        self.done = False
        self.last_evaluation = {'native_success': None, 'terminated': False, 'truncated': False}
        self.return_sum = 0.0
        self._capture()
        return self.observation()

    def _capture(self):
        import imageio.v2 as imageio
        frame = self.env.render(mode='rgb_array')
        if frame is not None and frame.size:
            if self.writer is None:
                (self.output / 'video').mkdir(exist_ok=True)
                self.writer = imageio.get_writer(str(self.output/'video/review.mp4'), fps=1/self.dt)
            self.writer.append_data(frame)

    def observation(self):
        vector = self.current[('agents','observation')][0,0].detach().cpu().tolist()
        if len(vector) != sum(x['size'] for x in self.layout):
            raise RuntimeError('Native observation shape differs from audited field layout')
        return {'control_step': self.steps, 'control_dt': self.dt, 'native_policy_observation': vector,
                'native_policy_layout': self.layout}

    def step(self, action):
        if self.done:
            raise RuntimeError('Native episode has ended')
        from tensordict import TensorDict
        a = self.torch.tensor(validate_action(action), device=self.env.device, dtype=self.torch.float32).reshape(1,1,4)
        transition = self.env.step(TensorDict({'agents': {'action': a}}, batch_size=[1], device=self.env.device))
        self.current = transition['next']
        self.steps += 1
        reward = float(self.current[('agents','reward')].sum().item())
        terminated = bool(self.current['terminated'].any().item())
        truncated = bool(self.current['truncated'].any().item())
        self.done = terminated or truncated
        self.return_sum += reward
        stats = {str(k): v.detach().cpu().tolist() for k,v in self.current['stats'].items()}
        self.last_evaluation = {'native_success': None, 'reward': reward, 'return': self.return_sum,
            'terminated': terminated, 'truncated': truncated, 'stats': stats}
        self._capture()
        return self.observation()

    def result(self):
        return {'task': self.task_id, 'steps': self.steps, 'control_dt': self.dt, 'success': None,
                'configured_physics_dt': self.configured_physics_dt,
                'native_success_available': False, 'evaluation': self.last_evaluation,
                'action_transform': None, 'observation_condition': 'native_state',
                'runtime': 'Isaac Sim4.1.0 native; compatibility not assumed'}

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.env.close()
        self.app.close()


def create_sim(task_id, seed, output, headless=True):
    return Simulator(task_id, seed, output, headless)
