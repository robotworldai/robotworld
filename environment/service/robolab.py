"""RoboLab episode adapter using the unchanged native environment and evaluator.

Imported only after Isaac's AppLauncher and task registrations are ready. No
model client, privileged state exporter or simulator state setter is used.
"""
import base64
import copy
import io
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from environment.benchmarks.robolab.control import STATE_KEYS, action_for, specs

CAMERAS = ('over_shoulder_left_camera', 'wrist_cam')


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def native_outcome(env):
    """Mirror run_episode: raw term flags can be transient initial artifacts."""
    if not env.all_terminated:
        return {'success': False, 'terminated': False, 'truncated': False}
    results = env.get_env_results()
    if len(results) != 1 or results[0]['env_id'] != 0 or type(results[0]['success']) is not bool:
        raise RuntimeError('RoboLab terminal episode has no authoritative native result')
    success = results[0]['success']
    return {'success': success, 'terminated': success, 'truncated': not success}


class RoboLabAdapter:
    def __init__(self, args, output):
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.utils.video_utils import VideoWriter
        from environment.benchmarks.robolab.rollout_config import configure_episode

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        set_output_dir(str(self.output))
        tasks = get_envs(task=args.task)
        if len(tasks) != 1:
            raise ValueError('Exactly one registered native RoboLab task is required')
        self.task = tasks[0]
        self.mode = args.control_mode
        self.env, self.cfg = create_env(self.task, device=args.device, num_envs=1,
            instruction_type=args.instruction_type, policy='robotworld_episode_service',
            renderer=args.renderer, rendering_mode=args.rendering_type)
        self.budget = configure_episode(self.env, args.world_step_cap, args.world_seed)
        # Official run_episode resets twice, before setting recorder indices.
        self.obs, _ = self.env.reset()
        self.obs, _ = self.env.reset()
        self.horizon = int(self.env.max_episode_length)
        recorder = self.env.recorder_manager
        if recorder is not None and hasattr(recorder, 'set_hdf5_file'):
            recorder.set_hdf5_file('run_0.hdf5')
            recorder.set_episode_index(0, env_ids=[0])
        self.step_index = 0
        self.gripper = 0
        self.history = deque(maxlen=9)
        self.events = (self.output/'native-steps.jsonl').open('a', buffering=1)
        # One video frame for every actual control step, not sampled prompt history.
        self.video = VideoWriter(str(self.output/'video.mp4'), 1.0/self.env.step_dt)
        self.closed = False
        self.capture()
        (self.output/'rollout-budget.json').write_text(json.dumps(self.budget, indent=2))
        (self.output/'evaluation-config.json').write_text(json.dumps({
            **self.describe(), 'arguments': vars(args), 'native_budget': self.budget,
            'success_source': 'RobolabEnv.get_env_results after all_terminated',
            'reset_policy': 'official double reset; native initial-artifact reset retained'}, indent=2))

    def state(self):
        return {key: array(self.obs['proprio_obs'][key])[0].tolist() for key in STATE_KEYS}

    def capture(self):
        from robolab.core.observations.observation_utils import unpack_image_obs
        frames = {key: array(self.obs['image_obs'][key])[0, :, :, :3].copy() for key in CAMERAS}
        self.history.append((self.step_index, frames))
        frame = unpack_image_obs(self.obs, scale=0.5, env_id=0).get('combined_image')
        if frame is None:
            raise RuntimeError('Native RoboLab camera recording is unavailable')
        self.video.write(frame)

    def describe(self):
        from environment.benchmarks.robolab.instructions import BASE
        from environment.benchmarks.action_contracts import action_contract
        from environment.benchmarks.operating_brief import operating_brief
        return {'benchmark': 'robolab', 'task': self.task, 'instruction': self.cfg.instruction,
                'tools': specs(self.mode), 'controller': self.mode,
                'action_contract': action_contract('robolab', self.mode),
                'operating_brief': operating_brief('robolab'),
                'control_instructions': BASE.replace('every two tool-feedback rounds', 'every two control steps'),
                'control_frequency_hz': 1.0/self.env.step_dt, 'native_horizon': self.budget['native_steps'],
                'effective_horizon': self.horizon,
                'observation_history': {'previous_frames': 4, 'interval_control_steps': 2}}

    def observe(self):
        images = []
        for step, frames in reversed(list(self.history)[::-2][:5]):
            for camera, frame in frames.items():
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format='JPEG', quality=90)
                images.append({'camera': camera, 'control_step': step, 'mime_type': 'image/jpeg',
                               'base64': base64.b64encode(buffer.getvalue()).decode()})
        return {'instruction': self.cfg.instruction, 'state': self.state(), 'images': images}

    def prepare(self, commands):
        prepared, grip = [], self.gripper
        for command in commands:
            if not isinstance(command, dict) or set(command) != {'tool', 'arguments'}:
                raise ValueError('Each command requires tool and arguments')
            _, repeat, grip = action_for(self.mode, command['tool'], command['arguments'], self.state(), grip)
            # Validate the whole chunk now, but resolve omitted absolute targets at
            # the start of each command from the then-current measured pose.
            prepared.append(({'command': copy.deepcopy(command), 'resolved': None}, repeat))
        return prepared

    def step(self, action):
        import torch
        from robolab.core.logging.results import get_all_env_subtask_infos
        if action['resolved'] is None:
            command = action['command']
            resolved, _, grip = action_for(self.mode, command['tool'], command['arguments'], self.state(), self.gripper)
            action['resolved'] = resolved
            self.gripper = grip
        native = action['resolved']
        self.obs, _, term, trunc, _ = self.env.step(torch.as_tensor(native, device=self.env.device).unsqueeze(0))
        self.step_index += 1
        outcome = native_outcome(self.env)
        self.capture()
        self.events.write(json.dumps({'control_step': self.step_index, 'action': native.tolist(),
            'state': self.state(), 'raw_terminated': bool(array(term)[0]),
            'raw_truncated': bool(array(trunc)[0]), 'native_results': self.env.get_env_results(),
            'subtask_info': get_all_env_subtask_infos(self.env), **outcome}, default=str)+'\n')
        return outcome

    def close(self):
        if self.closed:
            return
        self.closed = True
        (self.output/'native-results.json').write_text(json.dumps(self.env.get_env_results(), indent=2))
        try:
            self.video.release()
        finally:
            self.events.close()
            self.env.close()
