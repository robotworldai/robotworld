"""WheeledLab episode adapter; never exposes review cameras or scoring state."""
import base64
import io
import json
from collections import deque
from pathlib import Path

from environment.benchmarks.wheeledlab.control import action_guide, specs, validate_action


class WheeledLabAdapter:
    def __init__(self, sim, output, horizon):
        from environment.benchmarks.wheeledlab.catalog import CASES, MODEL_TITLES
        from environment.benchmarks.wheeledlab.policy import BASE, TASK
        from environment.benchmarks.action_contracts import describe_tools
        from environment.benchmarks.operating_brief import operating_brief
        self.sim, self.output, self.horizon = sim, Path(output), horizon
        family = CASES[sim.case]['family']
        custom = family == 'custom'
        self.observation_modalities = ['state'] if family in ('drift', 'elevation') else ['state', 'images']
        reverse = getattr(sim, 'spec', {}).get('reverse_enabled', False)
        guide = action_guide(sim.dt, reverse)
        self.tools = describe_tools([t for t in specs(False, custom, reverse) if t['name'] != 'observe'], guide)
        if custom:
            from environment.benchmarks.wheeledlab.custom import instructions
            base = instructions(sim)
        else:
            base = BASE + '\n' + TASK[CASES[sim.case]['family']]
        if getattr(sim, 'world_success_scene', None) and sim.world_success_scene.enabled:
            from environment.evaluation.world_success.runtime import instruction
            base += '\n' + instruction(sim.world_success_monitor.profile)
        self.instruction = base + '\n' + operating_brief('wheeledlab') + '\n' + guide + '\nTask: ' + MODEL_TITLES[sim.case] + '\nOnly drive and read-only observations are enabled; coding_control is disabled.'
        self.history = deque(maxlen=9)
        self.closed = False
        self.log = (self.output / 'native-steps.jsonl').open('a', buffering=1)
        self.capture()
        (self.output / 'evaluation-config.json').write_text(json.dumps(self.describe(), indent=2))

    def describe(self):
        return dict(benchmark='wheeledlab', task=self.sim.case, tools=self.tools, instruction=self.instruction,
                    observation_modalities=self.observation_modalities,
                    native_horizon=self.horizon, control_frequency_hz=1 / self.sim.dt,
                    coding_control_enabled=False, observation_history={'previous_frames': 4, 'interval_control_steps': 2})

    def capture(self):
        from PIL import Image
        images = []
        # Only the original policy camera(s); never sim.camera (external review view).
        sources = [('front' if getattr(self.sim, 'policy_sensor', None) else 'native_camera', self.sim.policy_image)]
        rear = getattr(self.sim, 'rear_sensor', None)
        if rear is not None:
            sources.append(('rear_not_mirrored', rear.image))
        for name, image in sources:
            if image is not None:
                buf = io.BytesIO()
                Image.fromarray(image).save(buf, format='PNG')
                images.append(dict(camera=name, control_step=self.sim.steps, mime_type='image/png',
                                   base64=base64.b64encode(buf.getvalue()).decode()))
        self.history.append(images)

    def observe(self):
        return dict(instruction=self.instruction, state=self.sim.observation(),
                    images=[image for packet in reversed(list(self.history)[::-2][:5]) for image in packet])

    def prepare(self, commands):
        prepared = []
        for command in commands:
            if not isinstance(command, dict) or set(command) != {'tool', 'arguments'} or command['tool'] != 'drive':
                raise ValueError('Only drive is enabled; query observations through observe endpoint')
            args = command['arguments']
            if not isinstance(args, dict) or set(args) != {'note', 'steps', 'action'} or not isinstance(args['note'], str):
                raise ValueError('Expected note, steps, action')
            if type(args['steps']) is not int or not 1 <= args['steps'] <= 50:
                raise ValueError('steps must be integer in 1..50')
            prepared.append((validate_action(args['action']), args['steps']))
        return prepared

    def step(self, action):
        self.sim.step(action)
        self.capture()
        result = self.sim.result()
        success = result.get('success') is True
        feedback = dict(success=success, terminated=bool(self.sim.done and not success and not self.sim.truncated),
                        truncated=bool(self.sim.done and not success and self.sim.truncated))
        self.log.write(json.dumps(dict(control_step=self.sim.steps, action=action, **feedback)) + '\n')
        return feedback

    def close(self):
        if self.closed:
            return
        result = self.sim.result()
        result.update(policy='service', coding_control_enabled=False, requested_steps=self.horizon)
        (self.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        (self.output / 'final-observation.json').write_text(json.dumps(self.sim.observation(), indent=2, allow_nan=False))
        for name in ('camera', 'policy_sensor', 'rear_sensor'):
            sensor = getattr(self.sim, name, None)
            if sensor is not None:
                sensor.close()
        self.log.close()
        self.closed = True
