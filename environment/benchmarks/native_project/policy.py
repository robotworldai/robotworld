"""Source-built Codex dynamic tools, shared across isolated native projects."""
import base64
from collections import deque
import io
import json
import os
from pathlib import Path
import shutil
import time
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.runtime.events import EventLog
from environment.runtime.image_history import INSTRUCTIONS, publish_image_window
from environment.runtime.episode_recovery import TurnRecovery
from environment.benchmarks.operating_brief import operating_brief
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
from .control import specs, validate_action
from .action_prompt import build_action_guide


class PolicyInputError(ValueError):
    """Invalid tool arguments, distinct from an environment failure."""


class EnvironmentExecutionError(RuntimeError):
    """The environment may have advanced; never retry as a policy input error."""

BASE = '''You are the robot's policy, running through the locally source-built Codex app-server.
Use only dynamic robot tools to control the local simulator. No reset, teleportation, scene/physics edits, private evaluator queries or file/network bypass. Shell is for calculations in /workspace. Only allowlisted observations are in /observations. No pretrained task controller is supplied unless the project explicitly declares one.
apply_action executes the whole native action vector simultaneously, with its original scaling, actuators and controller. A completed tool call is not task success. observe costs no physics steps. The step budget counts native env.step calls, not tokens, physics substeps or tool calls. Physics pauses during LLM reasoning, shell computation and between callbacks: this is NOT a measurement of real-time inference latency.
Use short action segments or coding_control for fast feedback. Define control(obs,memory), returning {"action":[values]} or {"done":true} to return to reasoning without another physics step. JSON memory persists within one call. Arithmetic, math, loops, conditionals and lists/dicts are available; no NumPy, arbitrary imports, simulator handles or file access. Limits: 16000 source characters, 100000 operations, 4096 collection elements, 2s per callback. Invalid code takes no physics step unless earlier ticks of that call already executed.
Only the original actor policy group is supplied. It may include simulator-derived state, commands or a task's native predictor; consult this task's explicit observation contract. Never claim pure visual control if state is included. Critic/privileged evaluator observations are excluded. Third-person review video is never a policy input. Native sensor images, if any, are labelled separately.
At tool boundaries you receive current plus at most four historical observations sampled every two tool observations. Each includes its actual control_step; history does NOT mean every second physical step. Callback feedback is current at every control step. Review videos record every control step independently of prompt history.
Continue attempting until native termination or the allowed budget. A final text answer does not stop an active episode. Preserve the original task objective; do not optimize an invented success rule or call survival success where upstream reports only rewards/metrics.
'''


class Agent:
    def __init__(self, sim, out, manifest, model, steps, timeout, instructions, coding=True, task_id=None):
        self.sim, self.out, self.manifest = sim, Path(out), manifest
        self.model, self.steps, self.coding = model, steps, coding
        self.instructions = instructions + ("\n"+operating_brief("native_project",task_id) if task_id else "")
        self.action_guide = build_action_guide(task_id, sim.action_metadata, sim.dt, coding)
        self.deadline = time.monotonic() + timeout
        self.events = EventLog(self.out/'events/environment.jsonl')
        self.tools = EventLog(self.out/'events/tools.jsonl')
        self.history = deque(maxlen=9)
        self.seq = self.program = 0
        self.seen = set()
        self.last_receipt = None
        self.latest_content = None
        self.recovery = TurnRecovery(self.deadline, self.events)
        self.tool_schema = specs(sim.action_metadata, coding, self.action_guide)

    @remember_packet(lambda self:self.sim.steps)
    def content(self):
        from PIL import Image
        state = dict(self.sim.observation(), episode_steps_remaining=max(0, self.steps-self.sim.steps))
        folder = self.out/'observations'/str(self.seq)
        folder.mkdir(parents=True)
        (folder/'state.json').write_text(json.dumps(state, ensure_ascii=False, allow_nan=False))
        frames = []
        for i, (label, pixels) in enumerate(self.sim.policy_images.items()):
            im = Image.fromarray(pixels)
            im.save(folder/f'sensor-{i}.png')
            b = io.BytesIO()
            im.save(b, format='PNG')
            frames.append((label, base64.b64encode(b.getvalue()).decode()))
        export = os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export:
            shutil.copytree(folder, Path(export)/str(self.seq))
        self.history.append((self.seq, state, frames))
        parts, rounds = [], []
        selected = {self.seq-2*i for i in range(5)}
        for seq, obs, images in self.history:
            if seq in selected:
                role = 'CURRENT' if seq == self.seq else 'HISTORY'
                step = self.steps-obs['episode_steps_remaining']
                rounds.append(dict(observation=seq, env_step=step, role=role))
                parts.append({'type':'inputText', 'text':f'{role} observation {seq}; env step {step}'})
                parts.append({'type':'inputText', 'text':f'{"CURRENT" if seq == self.seq else "HISTORY"} observation={seq}\n'+json.dumps(obs)})
                for label, data in images:
                    parts.extend([{'type':'inputText', 'text':'Native sensor: '+label},
                                  {'type':'inputImage', 'imageUrl':'data:image/png;base64,'+data}])
        publish_image_window(self.out, parts, rounds)
        self.seq += 1
        self.events.write('observation', {'state':state, 'content':parts})
        self.latest_content = parts
        return parts

    def execute(self, name, args):
        keys = {'observe': {'note'}, 'apply_action': {'note','action','steps'},
                'coding_control': {'note','code','max_steps'}}
        if name not in {t['name'] for t in self.tool_schema} or not isinstance(args,dict) or set(args)!=keys[name] or not isinstance(args['note'],str):
            raise PolicyInputError('Invalid tool or arguments')
        if name == 'observe':
            return {'executed_steps':0}
        count = args.get('steps', args.get('max_steps'))
        if type(count) is not int or not 1<=count<=(50 if name=='apply_action' else 250):
            raise PolicyInputError('Invalid action duration')
        count = min(count, self.steps-self.sim.steps)
        program = None
        executed, reason = 0, 'segment_complete'
        try:
            if name == 'apply_action':
                action = validate_action(args['action'], self.sim.action_metadata)
            else:
                code = args['code']
                if not isinstance(code,str) or not 1<=len(code)<=16000:
                    raise ValueError('Invalid program')
                self.program += 1
                folder = self.out/'programs'
                folder.mkdir(exist_ok=True)
                (folder/f'{self.program:04d}.py').write_text(code)
                program = ControllerProgram(code)
            for i in range(count):
                if self.sim.done:
                    reason = 'native_termination'
                    break
                if time.monotonic() > self.deadline:
                    raise TimeoutError('Agent wall timeout')
                try:
                    before = self.sim.observation()
                    callback = getattr(self.sim, 'controller_observation', self.sim.observation)() if program else None
                except Exception as e:
                    raise EnvironmentExecutionError('Environment observation failed') from e
                if program:
                    state = dict(callback, program_step=i, episode_steps_remaining=self.steps-self.sim.steps)
                    response = program.request({'obs':state})
                    command = response['action']
                    self.events.write('program_tick', {'program_id':self.program, 'observation':state,
                                                      'command':command, 'memory':response['memory']})
                    if command == {'done':True} and type(command['done']) is bool:
                        reason = 'program_done'
                        break
                    if not isinstance(command,dict) or set(command)!={'action'}:
                        raise ValueError('Return action vector or done')
                    action = validate_action(command['action'], self.sim.action_metadata)
                step_before = self.sim.steps
                try:
                    after = self.sim.step(action)
                except Exception as e:
                    self.events.write('environment_step_error', {
                        'requested_action':action, 'before':before,
                        'control_step_before':step_before, 'control_step_after':self.sim.steps,
                        'completed_steps_in_call':executed, 'advance_uncertain':True,
                        'error_type':type(e).__name__, 'error':str(e)})
                    raise EnvironmentExecutionError('Native environment step failed; episode aborted') from e
                executed += 1
                self.events.write('environment_step', {'requested_action':action, 'before':before,
                                                       'after':after, 'evaluation':self.sim.last_evaluation})
            return {'executed_steps':executed, 'reason':reason}
        except (ValueError,TypeError,KeyError,BrokenPipeError) as e:
            return {'executed_steps':executed, 'reason':'program_or_action_error', 'error':str(e)}
        finally:
            if program:
                program.close()

    def start_turn(self, parts):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Agent wall timeout')
        inputs = [{'type':'image','url':p['imageUrl']} if p['type']=='inputImage'
                  else {'type':'text','text':p['text'],'text_elements':[]} for p in parts]
        self.turn = self.session.rpc('turn/start', {'threadId':self.thread, 'input':inputs},
                                     timeout=remaining)['turn']['id']

    def build_prompt(self):
        base = BASE if self.coding else '\n'.join(line for line in BASE.splitlines() if 'coding_control' not in line)
        return {'model':self.model, 'cwd':'/workspace','approvalPolicy':'never',
                  'sandbox':'danger-full-access','ephemeral':True,
                  'baseInstructions':base+'\n'+INSTRUCTIONS+'\n'+self.instructions+'\n'+self.action_guide+'\nNative action metadata:\n'+json.dumps(self.sim.action_metadata)
                    +'\nNative observation layout:\n'+json.dumps(getattr(self.sim,'observation_metadata',{})),
                  'developerInstructions':'Only dynamic robot tools control physics. Shell is for calculations. Continue until the episode terminates or budget ends.',
                  'dynamicTools':self.tool_schema, 'config':{'features.shell_tool':True,'features.code_mode':True,
                  'features.multi_agent':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}}

    def run(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'):
            raise RuntimeError('Use the isolated local source-Codex launcher')
        prompt = self.build_prompt()
        (self.out/'action-guide.md').write_text(self.action_guide)
        (self.out/'prompt.json').write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
        with CodexSession(self.manifest, self.out) as self.session:
            from environment.runtime.nonaction_budget import attach
            attach(self.session,self.out,lambda:self.sim.steps)
            self.thread = self.session.rpc('thread/start',prompt)['thread']['id']
            self.start_turn(self.content())
            while self.sim.steps < self.steps and not self.sim.done:
                remaining = self.deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError('Agent wall timeout')
                message = self.session.receive(remaining)
                method = message.get('method')
                if method == 'item/tool/call':
                    call = message['params']
                    if call.get('threadId')!=self.thread or call.get('turnId')!=self.turn or call['callId'] in self.seen:
                        raise RuntimeError('Duplicate/cross-session tool call')
                    self.seen.add(call['callId'])
                    self.tools.write('tool_requested',call)
                    try:
                        report = self.execute(call['tool'],call['arguments'])
                    except PolicyInputError as e:
                        report = {'error':str(e),'executed_steps':0}
                    response = {'success':'error' not in report,
                                'contentItems':[{'type':'inputText','text':json.dumps(report)}]+self.content()}
                    # Persist completion before delivery; uncertain send failures still abort.
                    self.last_receipt = {'call_id':call['callId'], 'tool':call['tool'],
                                         'report':report, 'control_step':self.sim.steps}
                    self.tools.write('action_receipt', self.last_receipt)
                    self.session.send({'id':message['id'],'result':response})
                    self.tools.write('tool_completed',{'call_id':call['callId'],'response':response})
                elif method == 'turn/completed':
                    turn = message['params']['turn']
                    if message['params'].get('threadId') != self.thread or turn['id'] != self.turn:
                        raise RuntimeError('Cross-session turn completion')
                    if turn['status']!='completed':
                        if turn['status'] != 'failed' or not self.recovery.wait(
                                turn.get('error'), thread=self.thread, turn=self.turn,
                                control_step=self.sim.steps):
                            raise RuntimeError('Codex turn failed')
                        note = ('The previous model turn failed at the model service. '
                                'The same simulator is still paused at control step '
                                f'{self.sim.steps}; no actions were replayed. Continue from the '
                                'current state, not from reset. Do not repeat a completed action '
                                'merely to recover its reply. Last completed tool receipt: '
                                + json.dumps(self.last_receipt))
                        # Physics has not moved: reuse the exact packet, preserving history rounds.
                        self.start_turn([{'type':'inputText','text':note}]+self.latest_content)
                        self.events.write('turn_recovery_started', dict(
                            thread_id=self.thread, turn_id=self.turn, control_step=self.sim.steps))
                        continue
                    self.start_turn(continue_packet(self,self.sim.steps,self.content))
                elif method and 'id' in message:
                    self.session.send({'id':message['id'],'error':{'code':-32601,'message':'Unsupported RPC'}})
