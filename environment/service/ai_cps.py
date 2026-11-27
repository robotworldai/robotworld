"""AI-CPS episode adapter: original joint action and authoritative scorer."""
import base64
import io
import json
from collections import deque
from pathlib import Path

from environment.benchmarks.ai_cps.control import specs, validate_action


class AICPSAdapter:
    def __init__(self, sim, case, output, horizon, source, *, state_only=False):
        if case not in {'22', '23', '24'}:
            raise ValueError('Only selected AI-CPS tasks 22, 23 and 24 are supported')
        from environment.benchmarks.ai_cps.policy import instructions
        self.sim, self.case, self.output = sim, case, Path(output)
        self.horizon, self.source = horizon, source
        self.state_only = state_only
        self.instruction = instructions(case, sim.task_name, coding_control_enabled=False)
        self.tools = [t for t in specs(case, False) if t['name'] != 'observe']
        self.history = deque(maxlen=9)
        self.closed = False
        self.final_result = None
        self.log = (self.output / 'native-steps.jsonl').open('a', buffering=1)
        self.capture()
        (self.output / 'evaluation-config.json').write_text(json.dumps(self.describe(), indent=2))

    def describe(self):
        return dict(benchmark='ai_cps', task=self.case, tools=self.tools, instruction=self.instruction,
                    native_horizon=self.horizon, control_frequency_hz=1 / self.sim.control_dt,
                    observation_modalities=['state'] if self.state_only else ['state', 'images'],
                    scoring_profile='peg-xy-z-window-v2' if self.case == '24' else 'native',
                    coding_control_enabled=False, observation_history={'previous_frames': 4, 'interval_control_steps': 2})

    def capture(self):
        if self.state_only:
            return
        from PIL import Image
        image = getattr(self.sim.camera, 'image', None)
        if image is None:
            raise RuntimeError('Missing AI-CPS policy camera')
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format='JPEG', quality=85)
        self.history.append(dict(camera='camera', control_step=self.sim.steps,
                                 mime_type='image/jpeg', base64=base64.b64encode(buf.getvalue()).decode()))

    def observe(self):
        return {'instruction': self.instruction, 'state': self.sim.observation(),
                'images': list(reversed(list(self.history)[::-2][:5]))}

    def prepare(self, commands):
        prepared = []
        for command in commands:
            if not isinstance(command, dict) or set(command) != {'tool', 'arguments'} or command['tool'] != 'move_joints':
                raise ValueError('Only move_joints is enabled; use observe endpoint for observations')
            args = command['arguments']
            if not isinstance(args, dict) or set(args) != {'note', 'steps', 'arm_action'} or not isinstance(args['note'], str):
                raise ValueError('Expected note, steps, arm_action')
            if type(args['steps']) is not int or not 1 <= args['steps'] <= 50:
                raise ValueError('steps must be an integer in 1..50')
            prepared.append((validate_action(args['arm_action']), args['steps']))
        return prepared

    def result(self):
        if self.final_result is None:
            from environment.benchmarks.ai_cps.scoring import score
            self.final_result = score(self.sim.task_name, self.sim.trace, self.sim.done,
                                      self.source, verify_insertion=self.case == '24')
            self.final_result.update(case=self.case, policy='service', control_steps=self.sim.steps,
                                     coding_control_enabled=False, video_frames=0 if self.state_only else self.sim.camera.frames)
        return self.final_result

    def step(self, action):
        self.sim.step(action)
        self.capture()
        success = self.result().get('success') is True if self.sim.done else False
        feedback = dict(success=success, terminated=bool(self.sim.done and not success), truncated=False)
        self.log.write(json.dumps(dict(control_step=self.sim.steps, action=action, **feedback)) + '\n')
        return feedback

    def close(self):
        if self.closed:
            return
        (self.output / 'result.json').write_text(json.dumps(self.result(), indent=2, allow_nan=False))
        (self.output / 'native_trace.json').write_text(json.dumps(self.sim.trace))
        if not self.state_only:
            self.sim.camera.close()
        self.log.close()
        self.closed = True
