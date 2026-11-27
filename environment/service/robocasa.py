"""Native RoboCasa adapter. No agent imports, no simulator-state mutation shortcuts."""
import base64
import io
import json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
from environment.benchmarks.robocasa.control import CAMERAS, state_of, tool_specs, action_for


class RoboCasaAdapter:
    def __init__(self, task, seed, output, horizon=None, *, seed_protocol="legacy-task-index",
                 reset_seed=None, task_set="all_tasks", num_trials=1, episode_index=0):
        import gymnasium as gym
        import robocasa  # registers Gym environments
        import imageio.v2 as imageio
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        try:
            self.official_horizon = get_task_horizon(task)
        except ValueError:
            self.official_horizon = None
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError('Explicit control-step budget must be a positive integer')
        self.horizon = horizon if horizon is not None else self.official_horizon
        if self.horizon is None:
            raise ValueError('No registered official horizon; supply an explicit World evaluation budget')
        self.horizon_source = 'explicit_world_budget' if horizon is not None else 'upstream_dataset_registry'
        self.task, self.seed = task, seed
        from environment.benchmarks.robocasa.seeding import seed_config
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        self.seed_config = seed_config(task, seed, list(TASK_SET_REGISTRY[task_set]),
            protocol=seed_protocol, reset_seed=reset_seed, num_trials=num_trials,
            episode_index=episode_index, task_set=task_set)
        self.env = gym.make('robocasa/' + task, split='pretrain', seed=seed)
        self.obs, _ = self.env.reset(seed=self.seed_config['reset_seed'])
        self.gripper = 0.0
        self.step_index = 0
        self.history = deque(maxlen=9)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.events = (output/'native-steps.jsonl').open('a', buffering=1)
        self.video = imageio.get_writer(str(output/'video.mp4'), fps=self.env.unwrapped.env.control_freq,
                                        codec='libx264', quality=8)
        self.capture()
        (output/'evaluation-config.json').write_text(json.dumps({**self.describe(), 'seed':seed, 'split':'pretrain'},indent=2))

    def capture(self):
        frames = [np.asarray(self.obs[k]).copy() for k in CAMERAS]
        self.video.append_data(np.concatenate(frames, axis=1))
        self.history.append((self.step_index, frames))

    def describe(self):
        from environment.benchmarks.robocasa.instructions import BASE
        from environment.benchmarks.action_contracts import action_contract
        from environment.benchmarks.operating_brief import operating_brief
        return {'benchmark': 'robocasa', 'task': self.task, 'tools': tool_specs(),
                'control_frequency_hz': self.env.unwrapped.env.control_freq,
                'action_contract': action_contract('robocasa'),
                'operating_brief': operating_brief('robocasa'),
                'control_instructions': BASE.replace('every two feedback rounds','every two control steps'),
                'native_horizon': self.horizon, 'official_horizon': self.official_horizon,
                'horizon_source': self.horizon_source, 'initialization': self.seed_config,
                'observation_history': {'previous_frames':4, 'interval_control_steps':2},
                'instruction': self.obs['annotation.human.task_description']}

    def observe(self):
        images = []
        for step, frames in reversed(list(self.history)[::-2][:5]):
            for camera, frame in zip(CAMERAS, frames):
                b = io.BytesIO(); Image.fromarray(frame).save(b, format='JPEG', quality=90)
                images.append({'camera':camera, 'control_step':step, 'mime_type':'image/jpeg',
                               'base64':base64.b64encode(b.getvalue()).decode()})
        return {'instruction':self.obs['annotation.human.task_description'],
                'state':state_of(self.obs), 'images':images}

    def prepare(self, commands):
        result, gripper = [], self.gripper
        for cmd in commands:
            if not isinstance(cmd, dict) or set(cmd) != {'tool','arguments'}:
                raise ValueError('Each command requires tool and arguments')
            action, count, gripper = action_for(cmd['tool'], cmd['arguments'], gripper)
            result.append((action, count))
        return result

    def step(self, action):
        from robocasa.utils.env_utils import convert_action
        self.obs, _, terminated, truncated, info = self.env.step(convert_action(action))
        self.gripper = float(action[6]); self.step_index += 1
        self.capture()
        self.events.write(json.dumps({'control_step':self.step_index, 'action':action.tolist(),
            'state':state_of(self.obs), 'success':bool(info.get('success',False)),
            'terminated':bool(terminated), 'truncated':bool(truncated)})+'\n')
        return {'success':bool(info.get('success', False)),
                'terminated':bool(terminated), 'truncated':bool(truncated)}

    def close(self):
        self.video.close()
        self.events.close()
        self.env.close()
