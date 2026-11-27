"""Harness-independent native-project Episode adapter.

Receives an already constructed, reset simulator with its existing scoring wrapper.
Only native actor observations and explicitly exposed sensor images cross the API.
"""
import base64
from collections import deque
import io
import json
from pathlib import Path

from environment.benchmarks.native_project.control import specs, validate_action
from environment.benchmarks.native_project.action_prompt import build_action_guide


class NativeProjectAdapter:
    def __init__(self, sim, benchmark, task, instructions, horizon, output):
        from environment.evaluation.task_names import canonical, routing_key
        self.sim, self.benchmark, self.task = sim, benchmark, canonical(benchmark, task)
        # Kit may terminate the process in app.close(). Keep the service alive
        # through Harbor artifact collection; Docker owns final process shutdown.
        sim._world_defer_app_shutdown = True
        from environment.benchmarks.operating_brief import operating_brief
        from environment.benchmarks.native_project.instructions import BASE
        base = '\n'.join(line for line in BASE.splitlines() if 'coding_control' not in line)
        base = base.replace('sampled every two tool observations', 'sampled every two control steps').replace('history does NOT mean every second physical step', 'history uses control steps, not physics substeps')
        self.instructions = base + '\n' + instructions + '\n' + operating_brief('native_project', routing_key(benchmark, task)) + '\nHarbor interaction profile: code control is disabled. Use the listed apply_action tool directly in the native interface; only use execute if that generic interface is listed. observe is read-only. If a World evaluation profile is specified, its stated goal and success conditions supersede the upstream task description.'
        self.horizon = horizon
        self.output = Path(output)
        self.history = deque(maxlen=9)
        self.closed = False
        self.guide = build_action_guide(task, sim.action_metadata, sim.dt, coding=False)
        self.capture()
        (self.output/'evaluation-config.json').write_text(json.dumps(self.describe(),indent=2))

    def capture(self):
        images = []
        if getattr(self.sim, 'policy_images', None):
            from PIL import Image
            for camera, pixels in self.sim.policy_images.items():
                data = io.BytesIO()
                Image.fromarray(pixels).save(data, format='PNG')
                images.append({'camera':camera, 'control_step':self.sim.steps, 'mime_type':'image/png',
                               'base64':base64.b64encode(data.getvalue()).decode()})
        self.history.append((self.sim.steps, images))

    def describe(self):
        metadata = getattr(self.sim, 'observation_metadata', {})
        # A configured camera is still required if the initial capture is empty.
        has_camera = (metadata.get('cameras') or hasattr(getattr(self.sim, 'project', None), 'policy_images')
                      or getattr(self.sim, 'policy_images', None))
        return {'benchmark':self.benchmark, 'task':self.task, 'native_horizon':self.horizon,
                'control_frequency_hz':1 / self.sim.dt, 'instruction':self.instructions,
                'action_contract':self.guide, 'action_metadata':self.sim.action_metadata,
                'observation_metadata':metadata,
                'observation_modalities':['state', 'images'] if has_camera else ['state'],
                'observation_history':{'previous_frames':4,'interval_control_steps':2},
                'tools':[t for t in specs(self.sim.action_metadata, coding=False, action_guide=self.guide) if t['name'] != 'observe']}

    def observe(self):
        # Repeated reads do not push fake frames into the physical history window.
        images = [image for _, frame in reversed(list(self.history)[::-2][:5]) for image in frame]
        return {'instruction':self.instructions, 'state':self.sim.observation(), 'images':images}

    def prepare(self, commands):
        prepared = []
        for command in commands:
            if not isinstance(command, dict) or set(command) != {'tool','arguments'} or command['tool'] != 'apply_action':
                raise ValueError('Native project accepts only apply_action commands')
            args = command['arguments']
            if not isinstance(args, dict) or set(args) != {'action','steps','note'} or not isinstance(args['note'],str):
                raise ValueError('Expected action, steps, note')
            if type(args['steps']) is not int or not 1 <= args['steps'] <= 50:
                raise ValueError('steps must be an integer in [1,50]')
            prepared.append((validate_action(args['action'], self.sim.action_metadata), args['steps']))
        return prepared

    def step(self, action):
        self.sim.step(action)
        self.capture()
        result = self.sim.result()  # Existing World wrapper is authoritative when configured.
        success = result.get('success')
        if success is not None and type(success) is not bool:
            raise ValueError('Native project success must be boolean or explicitly unavailable')
        self._save_result(result)
        done = bool(self.sim.done)
        # The World wrapper can set sim.done at its own horizon while the native
        # simulator is still running. Episode records this as budget exhaustion.
        ended_at_world_budget = (self.sim.steps == self.horizon
                                 and result.get('stop_reason') == 'world_evaluation_end'
                                 and result.get('native_episode_complete') is False)
        native_evaluation = result.get('evaluation') or {}
        truncated = bool(getattr(self.sim, 'truncated', native_evaluation.get('truncated', False)))
        feedback = {'success':success is True, 'terminated':done and not truncated and not ended_at_world_budget,
                    'truncated':truncated}
        with (self.output/'native-steps.jsonl').open('a') as stream:
            stream.write(json.dumps({'control_step':self.sim.steps, 'action':action, **feedback})+'\n')
        return feedback

    def _save_result(self, result):
        result = {**result, 'benchmark':self.benchmark, 'task':self.task,
                  'control_steps':self.sim.steps, 'requested_steps':self.horizon,
                  'policy':'harbor-episode-service', 'coding_control_enabled':False}
        (self.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))

    def close(self):
        # Flush video before Harbor collects it; integration skips a second close.
        if not self.closed:
            self._save_result(self.sim.result())
            self.sim.close()
            self.sim._world_service_closed = True
            self.closed = True
