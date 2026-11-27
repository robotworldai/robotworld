"""BEHAVIOR public observations and original evaluator action/metric path."""
import base64
import copy
import io
import json
from collections import deque
from pathlib import Path

from environment.benchmarks.behavior_1k.control import action_for, tool_specs, validate_tool


class BehaviorAdapter:
    def __init__(self, evaluator, policy, output, task, horizon):
        from omnigibson.eval.evaluator import create_video_writer
        from environment.benchmarks.action_contracts import action_contract
        from environment.benchmarks.operating_brief import operating_brief
        from environment.benchmarks.behavior_1k.motion_prompt import MOTION_INSTRUCTIONS
        self.evaluator, self.policy = evaluator, policy
        self.output, self.task, self.horizon = Path(output), task, horizon
        self.steps = 0
        self.closed = False
        self.state = evaluator.instance_eval_states[0]
        self.history = deque(maxlen=9)
        self.instruction = ('Control R1Pro using the original absolute-pose IK tools. Use only the supplied onboard RGB, depth and robot-base proprioception. '
            'No global localization, hidden object poses, success predicates or evaluator state are available. '
            'Actions run serially; use move_robot for simultaneous arm/base/trunk/gripper motion. '
            'Omitted arm and trunk targets hold measured poses at the START of each command; the resulting target is held for that command. '
            'Gripper targets persist. Base velocity is in body coordinates and defaults to zero for each command. '
            'No collision-free planner is supplied. Use small increments and short segments around contact; inspect feedback before continuing. '
            'Current plus four previous frames at interval two control steps are supplied; videos record every step. '
            'Raw onboard linear-depth NPY arrays are exported with observations for offline shell calculation; '
            'use local_observation_files.sensors for their agent-container paths. Depth values are in metres. '
            'A tool completion does not mean task completion. Continue until environment termination or the step budget. '
            'coding_control is disabled in this service profile.\n' + action_contract('behavior_1k') + '\n' +
            operating_brief('behavior_1k') + '\n' + MOTION_INSTRUCTIONS + '\nTask: ' + policy.task_instruction)
        self.log = (self.output / 'native-steps.jsonl').open('a', buffering=1)
        video = self.output / 'videos' / (task + '.mp4')
        video.parent.mkdir(parents=True, exist_ok=True)
        evaluator._set_video_writer(self.state, create_video_writer(fpath=str(video), resolution=(448,672), rate=30))
        evaluator._write_video(self.state)
        self.ingest()
        existing = json.loads((self.output / 'evaluation-config.json').read_text())
        (self.output / 'evaluation-config.json').write_text(json.dumps({**existing, **self.describe()}, indent=2))

    def describe(self):
        return dict(benchmark='behavior_1k', task=self.task, tools=tool_specs(self.policy.profile),
                    instruction=self.instruction, native_horizon=self.horizon, control_frequency_hz=30,
                    coding_control_enabled=False, observation_history={'previous_frames':4,'interval_control_steps':2})

    def ingest(self):
        import numpy as np
        from PIL import Image
        self.policy.step = self.steps
        self.policy._ingest(self.evaluator._batch_obs())  # original explicit sensor allowlist
        frames = []
        folder = self.output / 'observations' / str(self.steps)
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in self.policy.images.items():
            label = name
            if name.endswith('depth_linear'):
                depth = np.squeeze(value).astype(np.float32)
                np.save(folder / (name + '.npy'), depth)
                pixels = (np.clip(np.nan_to_num(depth, nan=3., posinf=3., neginf=0.),0,3)/3*255).astype(np.uint8)
                image = Image.fromarray(pixels).convert('RGB')
                label += ': black=0m, white=3m or farther; invalid rendered white'
            else:
                if value.dtype != np.uint8:
                    raise ValueError('Expected upstream RGB uint8')
                image = Image.fromarray(value[...,:3])
            image.save(folder / (name + '.png'))
            buffer = io.BytesIO()
            image.save(buffer,format='JPEG',quality=95)
            frames.append(dict(camera=label,control_step=self.steps,mime_type='image/jpeg',base64=base64.b64encode(buffer.getvalue()).decode()))
        (folder / 'state.json').write_text(json.dumps({k:v.tolist() for k,v in self.policy.state.items()}))
        self.history.append(frames)

    def observe(self):
        # Export only the original sensor allowlist, not arbitrary simulator files.
        sensor_files = []
        for name in self.policy.images:
            if name.endswith('depth_linear'):
                path = self.output / 'observations' / str(self.steps) / (name + '.npy')
                sensor_files.append(dict(sensor=name, control_step=self.steps,
                    mime_type='application/x-npy', units='metres',
                    base64=base64.b64encode(path.read_bytes()).decode()))
        return dict(instruction=self.instruction,state={k:v.tolist() for k,v in self.policy.state.items()},
                    sensor_files=sensor_files,
                    images=[image for packet in reversed(list(self.history)[::-2][:5]) for image in packet])

    def prepare(self, commands):
        prepared = []
        for command in commands:
            if not isinstance(command,dict) or set(command)!={'tool','arguments'}:
                raise ValueError('Each command requires tool and arguments')
            args = validate_tool(command['tool'],command['arguments'],self.policy.profile)
            # Validate all targets now; fill omitted pose dimensions at each command's first actual step.
            action_for(args['targets'],self.policy.state,self.policy.profile,self.policy.grippers)
            prepared.append(({'targets':copy.deepcopy(args['targets']),'action':None},args['steps']))
        return prepared

    def step(self, command):
        import torch
        if command['action'] is None:
            command['action'], self.policy.grippers = action_for(command['targets'],self.policy.state,self.policy.profile,self.policy.grippers)
        action = command['action']
        terminated,truncated,_ = self.evaluator._apply_actions(torch.as_tensor(action,dtype=torch.float32).unsqueeze(0),[0])
        self.steps += 1
        self.evaluator._write_video(self.state)
        self.ingest()
        feedback = dict(success=bool(self.state.env_accessor.success),terminated=bool(terminated[0]),truncated=bool(truncated[0]))
        self.log.write(json.dumps(dict(control_step=self.steps,action=action.tolist(),**feedback))+'\n')
        return feedback

    def close(self):
        if self.closed:
            return
        self.evaluator._set_video_writer(self.state,None)
        result = dict(task=self.task,control_steps=self.steps,success=bool(self.state.env_accessor.success),policy='service',coding_control_enabled=False)
        for metric in self.state.metrics:
            result.update(metric.aggregate())
        (self.output / 'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
        self.log.close()
        self.closed = True
