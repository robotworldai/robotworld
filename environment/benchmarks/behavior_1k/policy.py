"""Codex policy for the official single-environment BatchedEvaluator.

Only forward(obs) receives observations. bind() receives controller layout, never
an environment handle. Simulator stepping and scoring remain with the evaluator.
"""
from environment.benchmarks.action_contracts import action_contract, describe_tools

import base64
import io
import json
import os
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.runtime.events import EventLog
from environment.runtime.episode_recovery import TurnRecovery
from environment.runtime.image_history import publish_image_window
from environment.benchmarks.operating_brief import operating_brief
from .control import action_for, split_proprio, tool_specs, validate_tool
from .motion_prompt import MOTION_INSTRUCTIONS


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class WorldPolicy:
    def __init__(self, output_dir, manifest, task_name, max_actions=100, timeout=1800, model=None, model_catalog=None, task_instruction=None, step_budget=None, coding_control_enabled=False):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest, self.task = manifest, task_name
        self.task_instruction = task_instruction or task_name.replace("_", " ")
        self.max_actions, self.timeout, self.model = max_actions, timeout, model
        self.step_budget = step_budget
        self.coding_control_enabled = coding_control_enabled
        self.overrides = [f'model_catalog_json={json.dumps(str(model_catalog))}'] if model_catalog else []
        self.profile = None
        self.session = None
        self.round = -1

    def bind(self, profile):
        self.profile = profile
        (self.output / 'robot-profile.json').write_text(json.dumps(profile, indent=2))

    def reset(self):
        if self.session:
            raise RuntimeError('Finish the previous rollout before resetting WorldPolicy')
        if self.profile is None:
            raise RuntimeError('Robot profile must be bound before reset')
        self.round += 1
        self.run_dir = self.output / f'rollout-{self.round:03d}'
        self.run_dir.mkdir(exist_ok=False)
        self.tools = EventLog(self.run_dir / 'events/tools.jsonl')
        self.env_events = EventLog(self.run_dir / 'events/environment.jsonl')
        self.step = 0
        self.last_action = None
        self.observation = 0
        self.history = deque(maxlen=9)
        self.pending = None
        self.grippers = {'left': 1., 'right': 1.}
        self.calls = 0
        self.program_counter = 0
        self.stopped = None
        self.seen = set()
        self.result = {'runtime': 'source-built-codex-app-server', 'official_success': None, 'events': []}
        self.deadline = time.monotonic() + self.timeout
        self.recovery = TurnRecovery(self.deadline,self.env_events)
        self.last_receipt = None
        self.latest_content = None

    def _ingest(self, obs):
        key = self.profile['proprio_key']
        vec = array(obs[key])
        if vec.ndim != 2 or vec.shape[0] != 1:
            raise ValueError('WorldPolicy currently supports exactly one logical environment')
        self.state = split_proprio(vec[0], self.profile['proprio_fields'])
        self.images = {}
        # Explicit allowlist. Drop task observation/object states/global poses/extrinsics.
        for role, prefix in self.profile['camera_keys'].items():
            for mode in ('rgb', 'depth_linear'):
                value = array(obs[prefix + '::' + mode])
                if value.shape[0] != 1:
                    raise ValueError('Camera batch mismatch')
                self.images[f'{role}_{mode}'] = value[0]
        self.env_events.write('observation', {'env_step': self.step, 'state': self.state})

    @remember_packet(lambda self:self.step)
    def _content(self):
        frame = self.run_dir / 'frames' / str(self.observation)
        frame.mkdir(parents=True, exist_ok=True)
        state = {k: v.tolist() for k, v in self.state.items()}
        text = f'Task: {self.task_instruction}\nEnv step: {self.step}; tool calls remaining: {self.max_actions-self.calls}\n'
        if self.step_budget is not None:
            text += f'Simulator step budget: {self.step_budget}; nominal steps remaining: {max(0,self.step_budget-self.step)}.\n'
        text += 'Robot-base proprioception (metres, XYZW quaternions, radians):\n' + json.dumps(state)
        if self.pending and self.pending.get('kind') != 'program':
            targets = self.pending['arguments']['targets']
            errors = {}
            for side in ('left', 'right'):
                errors[side] = {axis: float(self.state[f'eef_{side}_pos'][i] - targets[f'{side}_{axis}'])
                                for i, axis in enumerate(('x', 'y', 'z')) if f'{side}_{axis}' in targets}
            text += '\nEEF target residuals (metres): ' + json.dumps(errors)
        (frame / 'observation.txt').write_text(text)
        parts = []
        for name, value in self.images.items():
            if name.endswith('depth_linear'):
                depth = np.squeeze(value).astype(np.float32)
                np.save(frame / f'{name}.npy', depth)
                pixels = (np.clip(np.nan_to_num(depth, nan=3., posinf=3., neginf=0.), 0, 3) / 3 * 255).astype(np.uint8)
                label = f'{name}: linear depth visualization; black=0m, white=3m or farther; invalid rendered white.'
                rgb = Image.fromarray(pixels).convert('RGB')
            else:
                pixels = value[..., :3]
                if pixels.dtype != np.uint8:
                    raise ValueError('Expected upstream RGB uint8')
                rgb = Image.fromarray(pixels)
                label = name
            rgb.save(frame / f'{name}.png')
            buffer = io.BytesIO(); rgb.save(buffer, format='JPEG', quality=95)
            parts.extend([{'type': 'inputText', 'text': label},
                          {'type': 'inputImage', 'imageUrl': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()}])
        self.history.append((self.observation, self.step, parts))
        selected = {self.observation - 2*i for i in range(5)}
        content = [{'type': 'inputText', 'text': text}]
        included = []
        for idx, step, images in self.history:
            if idx in selected:
                content.append({'type': 'inputText', 'text': f'{"CURRENT" if idx == self.observation else "HISTORY"} observation {idx}; env step {step}'})
                content.extend(images)
                included.append({'observation': idx, 'env_step': step,
                                 'role': 'CURRENT' if idx == self.observation else 'HISTORY'})
        (frame / 'history.json').write_text(json.dumps(included))
        exported = os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if exported:
            publish_image_window(Path(exported).parent,content,included)
            # Agent-visible input contains only the observation allowlist above.
            target = Path(exported) / self.run_dir.name / str(self.observation)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(frame, target)
        self.observation += 1
        self.latest_content = content
        return content

    def _start(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'):
            raise RuntimeError('BEHAVIOR agent requires the isolated host relay; use containers/behavior_1k/run.py')
        self.session = CodexSession(self.manifest, self.run_dir, timeout=min(self.timeout,180), config_overrides=self.overrides)
        from environment.runtime.nonaction_budget import attach
        attach(self.session,self.run_dir,lambda:self.step)
        self.session.__enter__()
        instructions = ('You control an R1Pro mobile dual-arm robot through benchmark-specific move_robot, move_base, move_arms, set_grippers and (when available) move_torso tools. Complete the named household task. '
            'Use move_robot to combine both arm poses, base velocities, gripper openings and available trunk targets in one simultaneous bounded action. '
            'Each control step applies all components together and counts once toward the budget; separate tool calls execute sequentially, not concurrently. '
            'Combined gripper changes begin alongside motion, not after arm arrival. Split approach and closure when needed. '
            'Trunk targets are absolute joint positions in proprioception order, radians, within the tool runtime limits; omitted trunk holds its measured target. '
            'Use only onboard RGB, depth and proprioception. EEF positions and XYZW quaternions are relative to the robot articulation root, '
            'not world coordinates. Read the measured starting poses. Base local +x forward, +y left, +z up; base velocities in m/s and yaw rad/s. '
            'Unspecified arms hold their observed pose during the action; gripper targets persist (0 closed, 1 open). '
            'Base velocities default to zero on each call. steps repeats the absolute IK target for 1..30 control steps. '
            'IK does not plan collision-free paths: use small segments and inspect pose residuals, contact and object motion. '
            'Observe retention after lifting, transporting and rotating. Verify placement after releasing. '
            'For multi-room tasks, maintain a concise plan and remember observed landmarks, explored routes, carried objects and unfinished goals across tool calls. '
            'Discover rooms from onboard observations; no ground-truth map or global localization is supplied. '
            'Before translating the base, inspect free space with RGB-D; use small turns to inspect other directions and do not assume unseen routes are clear. '
            'Current images plus up to four historical observations at interval two are labeled; they are not simultaneous views. '
            'The simulator pauses while you reason. Persist while the environment remains active and execution budgets remain. '
            'A failed grasp, IK residual, uncertain pose or blocked approach is feedback for the next attempt, not a reason to stop. '
            'Inspect the observations and residuals, revise the plan, change the approach, base/torso pose, arm or subgoal, and try again. '
            'Use computation and workspace memory when useful; avoid repeating the same failed command without a change. '
            'Do not give up or end the task with a final answer. Only the evaluator determines task completion; '
            'if the scene appears complete, verify it through observations and continue checking remaining requirements until the environment ends.')
        instructions += '\nTrunk joint order (matching trunk_qpos, radians): '+str(self.profile.get('trunk_joint_names', 'as observed'))+'; position bounds: '+str(self.profile.get('trunk_limits'))
        tools = tool_specs(self.profile)
        if self.coding_control_enabled:
            from .code_control import coding_spec, INSTRUCTIONS
            tools.extend(describe_tools([coding_spec()], action_contract('behavior_1k')))
            instructions += '\n'+INSTRUCTIONS
        else:
            instructions += '\nNo coding_control tool is enabled for this rollout; offline code cannot step the environment.'
        instructions += '\n'+action_contract('behavior_1k')+'\n'+MOTION_INSTRUCTIONS
        params = {'cwd': '/workspace', 'approvalPolicy': 'never', 'sandbox': 'danger-full-access', 'baseInstructions': instructions+'\n'+operating_brief('behavior_1k'),
                  'developerInstructions': 'Use shell, code execution and files in /workspace for computation, planning, maps and persistent memory. '
                      'Allowed sensor exports are read-only in /observations; each numbered directory includes RGB PNGs, raw depth NPYs and proprioception. '
                      'Use only supplied task/control information and onboard observations as episode evidence. '
                      'All robot actions and new observations must pass through the provided benchmark tools. '
                      'Never bypass that interface, access hidden simulator/evaluator state, modify scenes, reset an episode or determine official scores yourself. '
                      'The runtime is externally isolated; tool capability is not permission to access additional benchmark data.',
                  'dynamicTools': tools, 'ephemeral': True,
                  'config': {'features.shell_tool': True, 'features.multi_agent': False, 'features.code_mode': True,
                             'features.apps': False, 'features.plugins': False, 'web_search': 'disabled'}}
        if self.model: params['model'] = self.model
        (self.run_dir / 'prompt.json').write_text(json.dumps(params,indent=2))
        self.thread = self.session.rpc('thread/start',params)['thread']['id']
        self._begin_turn(self._content())

    def _begin_turn(self, content):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Agent wall timeout before turn start')
        initial = [{'type':'image','url':x['imageUrl']} if x['type']=='inputImage'
                   else {'type':'text','text':x['text'],'text_elements':[]} for x in content]
        self.turn = self.session.rpc('turn/start',{'threadId':self.thread,'input':initial},
                                     timeout=remaining)['turn']['id']

    def _reply(self, message, response):
        self.tools.write('tool_completed', {'call_id':message['params']['callId'],'response':response})
        self.session.send({'id':message['id'],'result':response})

    def _complete_pending(self, reason='segment_complete', ok=True, error=None):
        if self.pending is None: return
        item = self.pending
        report = {'executed_steps':item['executed'], 'requested_steps':item['steps'],
                  'reason':reason, 'task_success_asserted':False}
        if error: report['error'] = error
        if item.get('kind') == 'program':
            report.update(program_id=item['program_id'], final_memory=item['worker'].memory)
            item['worker'].close()
            self.env_events.write('program_finished', report)
        targets = item['worker'].targets if item.get('kind') == 'program' else item['arguments']['targets']
        report['eef_target_residuals_m'] = {side: {axis:float(self.state[f'eef_{side}_pos'][i]-targets[f'{side}_{axis}'])
            for i,axis in enumerate('xyz') if f'{side}_{axis}' in targets} for side in ('left','right')}
        # Keep the native trace and close the worker even if the agent socket has gone away.
        self.result['events'].append({'call_id':item['message']['params']['callId'],
            'tool':item['message']['params']['tool'], 'arguments':item['arguments'], **report})
        self.pending = None
        self._save()
        content = [{'type':'inputText','text':json.dumps(report)}]
        content.extend(self._content())
        self.last_receipt = {'call_id':item['message']['params']['callId'],
                             'report':report,'control_step':self.step}
        self.tools.write('action_receipt',self.last_receipt)
        self._reply(item['message'], {'success':ok,'contentItems':content})

    def _accept_motion(self, message):
        call = message['params']
        if call['tool'] == 'coding_control':
            if not self.coding_control_enabled: raise ValueError('coding_control is disabled')
            from .code_control import FeedbackController, validate_program
            args = validate_program(call['arguments'])
            self.program_counter += 1
            program_id = f'program-{self.program_counter:04d}'
            folder = self.run_dir/'programs'; folder.mkdir(exist_ok=True)
            (folder/(program_id+'.py')).write_text(args['code'])
            (folder/(program_id+'.json')).write_text(json.dumps(args,indent=2))
            worker = FeedbackController(args, self.state, self.profile)
            self.pending = {'kind':'program','message':message,'arguments':args,'steps':args['max_steps'],
                            'executed':0,'worker':worker,'program_id':program_id}
            self.env_events.write('program_started',{'program_id':program_id, **args})
        else:
            args = validate_tool(call['tool'],call['arguments'],self.profile)
            action, grip = action_for(args['targets'],self.state,self.profile,self.grippers)
            self.grippers = grip
            self.pending = {'kind':'motion','message':message,'arguments':args,'action':action,
                            'steps':args['steps'],'executed':0}
        self.calls += 1

    def _program_action(self):
        item = self.pending
        obs = None
        try:
            obs = item['worker'].observation(self.state,self.images,self.grippers,
                                             self.step,item['executed'],self.step_budget)
            action, grippers, command = item['worker'].tick(obs,self.state,self.grippers)
        except (ValueError,TypeError,KeyError,OSError) as error:
            self.env_events.write('program_error',{'program_id':item['program_id'], 'observation':obs,'error':str(error)})
            self._complete_pending('program_error',False,str(error))
            return None
        self.env_events.write('program_tick',{'program_id':item['program_id'], 'observation':obs,
                              'command':command,'memory':item['worker'].memory,
                              'validated_native_action':action})
        if action is None:
            self._complete_pending('program_done')
            return None
        self.grippers = grippers
        return action

    def _save(self):
        self.result.update(action_calls=self.calls,env_steps=self.step,termination=self.stopped,
                           coding_control_enabled=self.coding_control_enabled)
        (self.run_dir/'episode.json').write_text(json.dumps(self.result,indent=2))

    def _hold(self):
        return action_for({'base_vx':0},self.state,self.profile,self.grippers)[0]

    def forward(self, obs):
        import torch
        completed = self.last_action is not None
        if completed: self.step += 1
        self._ingest(obs)
        if completed:
            self.env_events.write('action_completed', {'env_step':self.step,'state':self.state})
            self.last_action = None
            if self.pending:
                self.pending['executed'] += 1
                if self.pending['executed'] >= self.pending['steps']:
                    self._complete_pending()
        if self.session is None and not self.stopped: self._start()
        while True:
            if time.monotonic() >= self.deadline:
                self.stopped = 'timeout'
            if self.stopped:
                self._save()
                raise RuntimeError('Agent stopped before native episode end: '+self.stopped)
            if self.pending:
                action = self._program_action() if self.pending.get('kind') == 'program' else self.pending['action']
                if action is None: continue  # done/error pauses for LLM; no hidden hold action
                self.last_action = action.copy()
                self.env_events.write('action_requested',{'env_step_before':self.step,'action':action,
                    'program_id':self.pending.get('program_id'),'policy_stopped':self.stopped})
                self._save()
                return torch.from_numpy(action[None,:])
            if self.calls >= self.max_actions:
                self.stopped = 'action_budget'; continue
            try: message = self.session.receive(self.deadline-time.monotonic())
            except TimeoutError:
                self.stopped = 'timeout'; continue
            method = message.get('method')
            if method == 'item/tool/call':
                call = message['params']; self.tools.write('tool_requested',call)
                if call.get('threadId') != self.thread or call.get('turnId') != self.turn:
                    raise RuntimeError('Cross-session tool request')
                if call['callId'] in self.seen:
                    raise RuntimeError('Duplicate motion request; abort rather than reexecute')
                self.seen.add(call['callId'])
                try:
                    self._accept_motion(message)
                except (ValueError,TypeError,KeyError,OSError) as error:
                    self._reply(message,{'success':False,'contentItems':[{'type':'inputText','text':str(error)}]})
            elif method == 'turn/completed':
                event = message['params']
                if event.get('threadId') != self.thread or event['turn']['id'] != self.turn:
                    raise RuntimeError('Cross-session turn completion')
                status = event['turn']['status']
                if status == 'completed':
                    self.tools.write('agent_turn_continued',{'turn_id':self.turn,'env_step':self.step})
                    self._begin_turn(continue_packet(self,self.step,self._content))
                elif status == 'failed' and self.pending is None and self.last_action is None and self.recovery.wait(
                        event['turn'].get('error'),thread=self.thread,turn=self.turn,control_step=self.step):
                    note = ('The previous model turn failed at the model service. '
                            'The same simulator is paused at control step '+str(self.step)+'. '
                            'No actions were replayed and the original budget remains. '
                            'Continue from the current state. Last completed tool receipt: '
                            +json.dumps(self.last_receipt))
                    self._begin_turn([{'type':'inputText','text':note}]+self.latest_content)
                    self.env_events.write('turn_recovery_started',{'thread_id':self.thread,
                                          'turn_id':self.turn,'control_step':self.step})
                else:
                    self.stopped = 'agent_'+status
                    self.result['agent_turn_error'] = event['turn'].get('error')
            elif method and 'id' in message:
                self.session.send({'id':message['id'],'error':{'code':-32601,'message':'Unsupported client RPC; approvals cannot expand the external isolation boundary'}})

    def finish(self, final_obs=None, official_result=None):
        try:
            if final_obs is not None and getattr(self,'last_action',None) is not None:
                self.step+=1; self._ingest(final_obs)
                self.env_events.write('action_completed',{'env_step':self.step,'state':self.state})
                if self.pending: self.pending['executed']+=1
            self.result['official_result']=official_result
            self.result['official_success']=official_result.get('success') if official_result else None
            self.stopped=self.stopped or ('environment_end' if official_result else 'infrastructure_error')
            self._save()
            if self.pending:
                try:
                    self._complete_pending('environment_end' if official_result is not None else 'infrastructure_error',
                                           official_result is not None)
                except (OSError,TimeoutError) as error:
                    self.result['terminal_reply_error']=str(error); self._save()
            if self.session and hasattr(self,'turn'):
                self.session.request('turn/interrupt',{'threadId':self.thread,'turnId':self.turn})
        finally:
            if self.pending and self.pending.get('kind') == 'program':
                self.pending['worker'].close(); self.pending = None
            if self.session: self.session.__exit__(None,None,None); self.session=None
            self.last_action=None
