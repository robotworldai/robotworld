"""Synchronous source-Codex decisions at control boundaries of run_trial."""
import base64
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.runtime.events import EventLog
from environment.runtime.image_history import ObservationHistory
from environment.benchmarks.operating_brief import operating_brief
from .control import specs, validate, compose_action
from .coding import ControllerProgram

CODING_INSTRUCTIONS = '''
Feedback-program tool: coding_control
Use coding_control when a segment benefits from fresh feedback every control step, such as keeping a posture while shifting weight, landing a foot, reacting to tilt/contact, or coordinating a kick. You write the feedback law and stopping conditions; the tool does not supply a walking or balancing skill. For this diagnostic, try coding_control for the initial balance assessment and subsequent motion segments, and revise your code from the returned evidence. move_joints/review_action remain available for simple actions.

Execution contract:
1. Submit note, max_steps (1..500), and code defining control(obs, memory). Each call starts with empty memory. The runner calls control exactly once before each 0.02s action, with FRESH numeric observations; no LLM round trip occurs between ticks. Observation acquisition and Python execution pause simulation; this is simulated-time feedback, not proof of real-time hardware latency.
2. obs contains the same allowed numeric state as the LLM, plus dt=0.02, program_step (starts at 0), program_time_s, program_steps_remaining, episode_steps_remaining, and joint_limits_rad. Current ball/goal vectors, COM velocity, foot contact flags, joint positions/velocities and body orientation can drive your feedback. No images, privileged scene objects, simulator handle, or success oracle are supplied to the program. In hybrid mode, obs additionally contains the newly computed original policy action/targets for THIS tick.
3. DIRECT: return {"joint_positions": {name: absolute_angle_rad}}. All joints are simultaneous. Omitted joints preserve the previous program targets; at the first tick they latch current measured angles. This differs from separate move_joints calls, which latch measured angles anew. Use initial_joint_reference_rad or last_requested_joint_targets_rad as explicit nominal references when appropriate. HYBRID: return {"decision":"accept", "joint_offsets":{}} or {"decision":"modify", "joint_offsets":{name:delta_rad}}. Every offset applies to the current native action, never a stale proposal. Existing mechanical limits and offset bounds are enforced each tick.
4. Return {"done": True} to stop WITHOUT applying an action on that tick. This only ends the program segment, not the task. A segment also ends at max_steps or the remaining episode budget. The LLM then receives fresh images/state and a summary of actual steps, final memory and any error. Previously executed actions cannot be rolled back. An invalid return/angle, runtime exception, excessive computation or memory stops that program; it does not step the simulator with the invalid action.
5. You may store gait phase, filtered errors, nominal angles and counters in memory, which must remain JSON-serializable. It persists only in this call, not between coding_control calls. Recreate or explicitly carry useful state into your next program. Code and every per-tick returned command/memory are logged.

Supported Python subset:
- Define control(obs, memory), optional helper functions with positional parameters, constants and optional import math. Use assignments, if/elif/else, for/while, break/continue, return, arithmetic/comparisons, list/dict comprehensions, indexing and slicing. This is an AST interpreter, not unrestricted Python.
- Builtins: abs, min, max, sum, len, range, enumerate, zip, int, float, bool, round, list, dict, tuple, sorted, all, any. dict methods: get/copy/items/keys/values/update/setdefault/pop; list methods: copy/append/pop. math offers pi/e/tau and sin/cos/tan, asin/acos/atan/atan2, sqrt/exp/log, floor/ceil/fabs/hypot/degrees/radians/isfinite/copysign. Function arguments are positional; no decorators/default arguments, classes, lambdas, generators, try/except, f-strings or arbitrary attributes/imports. No numpy, open, eval, exec or simulator access.
- Code limit 16000 characters; collection limit 4096 elements; 100000 interpreted operations per callback; result+memory <=64KiB. Each callback has a 2s wall timeout. These are resource ceilings, not a promise of a 500Hz Python loop. The existing PD and optional ankle assistance run independently at 500Hz.

Design guidance:
- Begin with a short posture/feedback test. Read actual tilt/contact/COM velocity, bound corrections and limit target changes between ticks. Gradually increase motion after observing stable support. Define a stop condition when your controller leaves its intended posture range; return to the LLM for revision. Stopping pauses simulation, not a physical emergency brake on hardware.
- Interpolation supplies smooth commands but does not by itself stabilize a floating-base robot. Closed-loop feedback also does not guarantee balance. A desired foot lift may require shifting support first; both-feet COM assistance can oppose a single-support maneuver. Inspect actual assistance and do not accumulate its ankle offsets into your nominal targets.
- You can write an entire phase controller for a short subtask: initialize memory, shift weight, lift/swing/place a foot, check contact, settle, then done. The phase transitions and all joint commands are YOUR code; there is no hidden walk/kick implementation. Evaluate actual forward progress while upright, not movement caused by falling.

Minimal DIRECT example (a standing probe, not a gait):
def control(obs, memory):
    if obs["pelvis_height_m"] < 0.55:
        return {"done": True}
    if "nominal" not in memory:
        memory["nominal"] = obs["initial_joint_reference_rad"].copy()
    q = memory["nominal"].copy()
    memory["ticks"] = memory.get("ticks", 0) + 1
    return {"joint_positions": q}
'''

BASE = '''You control a simulated Unitree G1 humanoid kicking a ball toward the marked goal.
Use only the robot tools to advance physics. No resets, scene edits, simulator access or external solutions.
Direct joint targets are absolute radians; hybrid joint offsets are relative radians. The root is floating: joint PD commands do not guarantee balance or collision avoidance. No walking/EEF planner exists here.
Control runs at 50 Hz (0.02s); steps consume simulator control steps, not LLM turns. Physics pauses while you reason. Current + four historical observations at interval two refer to TOOL boundaries; video is recorded every control step separately.
The observed ball/goal vectors and proprioception come from the upstream state-based evaluation; these are simulated state measurements, not vision estimates. Camera images are added for observation. Do not infer success from a tool response. Only the upstream goal-crossing predicate decides success. No changes to initial states, task goals or physics are permitted.
HYBRID mode: inspect the displayed original ONNX proposal, accept it or add small named joint target offsets with review_action. The original policy recomputes at every control step within a segment. You supervise the segment, not every unseen future action. Preserve balance; start by observing the competent original controller and modify when observations warrant it.
DIRECT mode: you supply joint positions yourself, with no learned action proposal or learned stabilization. Omitted joints hold the measured angle at segment start. An optional classical ankle balance assist is described in the current state; do not assume it is active. Use short feedback segments, account for gravity and coordinate hip/knee/ankle/arm motion. Work toward the ball and kick toward the goal while retaining balance.
Robot and conventions:
- The G1 has 29 actuated joints, all controllable simultaneously: each leg has hip pitch/roll/yaw, knee, ankle pitch/roll (6); waist yaw/roll/pitch (3); each arm has shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (7). Joint names and mechanical limits are in the tool schema. There is no actuated root translation or heading teleport.
- Pelvis-local x is forward, y left, z up. Ball, goal, COM and foot vectors use the CURRENT pelvis orientation, which tilts with the robot. Quaternion uses w,x,y,z. Joint angles are radians, positions metres, velocities rad/s or m/s as named. Joint velocities follow the same 29-name order shown in joint_positions. Positive rotations follow each model joint axis; test small motions rather than assume mirror-image signs for the two legs.
- Physics dt=0.002s with 10 physical substeps per 0.02s control step. A tool request of 50 steps holds a DIRECT target for 1 simulated second; it is not 50 independently designed poses. All named joints move together through torque-limited upstream PD. No automatic interpolation or whole-body trajectory planning is supplied.
- move_joints sets absolute joint targets; omitted joints latch their current measured angle at the start of that request. To keep a consistent multi-segment reference, explicitly include the joints you intend to hold. review_action offsets are added to a newly recomputed native action each step, not to a single frozen action. accept uses empty joint_offsets.
Feedback and balance:
- Read pelvis height/orientation, angular velocity, joint angles/velocities, COM position/velocity, foot ankle positions and foot-ground contact booleans. An ankle-link position is not the sole contact point. Contact booleans do not measure available friction, pressure or support force. Images show two cameras of ONE scene.
- If balance_assist=ankle-com, a classical feedback controller runs at every physical substep. It adjusts BOTH ankle pitch/roll targets using COM displacement/velocity relative to the midpoint of the ankles, clips to joint limits, and uses the original PD torque limits. It helps double-support standing but is NOT a walking policy, footstep planner, single-support guarantee or fall recovery controller. It turns off at low pelvis height or severe tilt. Returned balance_ankle_offsets_rad records its latest applied corrections. Your requested ankle targets may consequently differ from executed targets.
- initial_joint_reference_rad is the measured initial posture, not a learned gait. last_requested_joint_targets_rad is your previous DIRECT request before ankle feedback. When preserving a posture with assistance, reuse explicit nominal targets rather than repeatedly copying measured ankle angles: measured angles already include assistance and copying them as new nominal targets can accumulate corrections. Balance feedback uses the initial heading axes and is intended for small motions near upright.
- First establish a stable stance and check for drift. For walking, consider small weight shifts, short swing-foot motions, landing and settling phases; coordinate hip/knee/ankle and torso/arms rather than moving a single hip aggressively. Use short segments during support changes and ball contact. For kicking, inspect ball direction/range and supporting foot before backswing, strike and follow-through. These are strategies to test, not provided skills or guaranteed solutions.
Code-assisted control:
- You may write and run Python in /workspace to calculate kinematics approximations, smooth joint waypoint sequences, coordinated gait phases or kick profiles using allowed observations. Check installed packages; python3 and the standard library are a fallback. Save/inspect the resulting named joint dictionaries and issue robot tools to execute them.
- For example generate small interpolated multi-joint waypoints using a smoothstep phase, or submit a feedback controller through coding_control to update targets from new observations every control step. Offline shell code alone does not move the robot. Do not read simulator internals, connect to its process, edit the environment, set root state or replace observations through shell/files. Only robot tools advance physics. A coding_control program can return done between control steps; ordinary move_joints/review_action segments run for their requested duration.
- In an extended 1000-step diagnostic, assess whether you can stand and make controlled forward progress before a strike. Continue attempting the soccer objective within the remaining budget. Do not treat incidental displacement while falling as walking, or tool completion as a goal.
Keep trying while the episode remains active. A final text answer does not stop the episode. Report what you actually observe; do not claim goals or recovery without evidence.'''
DEV = 'Shell/code are for calculations and notes in /workspace only. /observations contains allowed observation exports. Robot tools are the only way to move or get fresh simulator observations. No hidden evaluator access.'


class AgentPolicy:
    def __init__(self, original, env, cfg, mode, model, manifest, output, timeout):
        from mujoco_soccer.constants import JOINT_NAMES, DEFAULT_JOINT_POS_ARRAY, ACTION_SCALE_ARRAY
        from mujoco_soccer.policy import parse_metadata_array
        self.coding_control_enabled = True
        self.original, self.env, self.cfg = original, env, cfg
        self.mode, self.model, self.manifest = mode, model, manifest
        self.out = Path(output); self.out.mkdir(parents=True, exist_ok=True)
        self.names = list(JOINT_NAMES)
        self.initial_joint_reference = dict(zip(self.names,env.joint_pos.tolist()))
        self.limits = env.model.jnt_range[[env._joint_id(n) for n in self.names]].copy()
        self.metadata = original.metadata
        self.obs_dim = original.obs_dim
        self.default = parse_metadata_array(self.metadata, 'default_joint_pos', DEFAULT_JOINT_POS_ARRAY)
        self.scale = parse_metadata_array(self.metadata, 'action_scale', ACTION_SCALE_ARRAY)
        self.events = EventLog(self.out/'events/environment.jsonl')
        self.tools = EventLog(self.out/'events/tools.jsonl')
        self.session = None; self.pending = None; self.step_count = 0; self.remaining = 0
        self.history = ObservationHistory(); self.seq = 0; self.seen = set()
        self.deadline = time.monotonic()+timeout
        self.capture = None; self.proposal = None; self.decision = None
        self.program=None;self.program_id=None;self.program_tick=0;self.program_memory={}
        self.program_counter=0;self.program_limit=0;self.program_requested=0
        self.reset()

    def reset(self):
        self.original.reset()
        self.ref_joint_pos = self.original.ref_joint_pos.copy()
        self.ref_joint_vel = self.original.ref_joint_vel.copy()
        self.ref_ang_vel = self.original.ref_ang_vel.copy()

    def state(self):
        from mujoco_soccer.math_utils import quat_apply_inverse
        from .balance import balance_state
        env = self.env
        goal = env.data.mocap_pos[env.goal_mocap_id]
        return {**balance_state(env), 'control_step': self.step_count, 'time_s': float(env.data.time),
            'initial_joint_reference_rad':self.initial_joint_reference,
            'last_requested_joint_targets_rad':dict(zip(self.names,self.targets.tolist())) if self.mode=='direct' and hasattr(self,'targets') else None,
            'joint_positions': dict(zip(self.names, env.joint_pos.tolist())),
            'joint_velocities': env.joint_vel.tolist(), 'pelvis_quaternion_wxyz': env.pelvis_quat.tolist(),
            'pelvis_height_m': float(env.pelvis_pos[2]), 'base_angular_velocity': env.base_ang_vel.tolist(),
            'ball_in_pelvis_frame_m': quat_apply_inverse(env.pelvis_quat, env.ball_pos-env.pelvis_pos).tolist(),
            'goal_in_pelvis_frame_m': quat_apply_inverse(env.pelvis_quat, goal-env.pelvis_pos).tolist()}

    @remember_packet(lambda self:self.step_count)
    def content(self):
        from mujoco_soccer.constants import ISAACLAB_TO_MUJOCO_REINDEX
        state = self.state()
        if self.mode == 'hybrid' and self.proposal is not None:
            state['original_policy_action_isaaclab_order'] = self.proposal.action.tolist()
            target = self.default + self.scale*self.proposal.action[ISAACLAB_TO_MUJOCO_REINDEX]
            state['original_proposed_joint_targets_rad'] = dict(zip(self.names, target.tolist()))
        text = f'Mode={self.mode}; budget={round(self.cfg.sim_time/self.cfg.control_dt)} control steps. '+json.dumps(state)
        if self.capture.scenery == 'training-pitch':
            text += ' The training pitch, white goal frame/net, fence and trees are visual decoration only, without collisions. The original colored goal markers and supplied goal vector define the scored goal; the turf lines are context, not extra task constraints.'
        imgs = self.capture.images()
        folder = self.out/'observations'/str(self.seq); folder.mkdir(parents=True)
        (folder/'state.json').write_text(json.dumps(state, indent=2))
        for key, img in imgs.items(): Image.fromarray(img).save(folder/(key+'.png'))
        export = os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export: shutil.copytree(folder, Path(export)/str(self.seq), dirs_exist_ok=True)
        image_parts = []
        for key, img in imgs.items():
            b=io.BytesIO(); Image.fromarray(img).save(b,format='JPEG',quality=85)
            image_parts += [{'type':'inputText','text':key},{'type':'inputImage','imageUrl':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()}]
        parts = [{'type':'inputText', 'text':text}] + self.history.append(self.seq, self.step_count, image_parts)
        # Publish the exact packet before Codex can request the next model response.
        window = self.out/'image-window.json'
        temporary = window.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.history.snapshot()))
        temporary.replace(window)
        self.seq += 1
        return parts

    def code_observation(self):
        from mujoco_soccer.constants import ISAACLAB_TO_MUJOCO_REINDEX
        state=self.state()
        state.update(dt=self.cfg.control_dt,program_step=self.program_tick,
            program_time_s=self.program_tick*self.cfg.control_dt,
            program_steps_remaining=self.remaining,
            episode_steps_remaining=max(0,round(self.cfg.sim_time/self.cfg.control_dt)-self.step_count),
            joint_limits_rad=dict(zip(self.names,self.limits.tolist())))
        if self.mode=='hybrid':
            state['original_policy_action_isaaclab_order']=self.proposal.action.tolist()
            state['original_proposed_joint_targets_rad']=dict(zip(self.names,(self.default+self.scale*self.proposal.action[ISAACLAB_TO_MUJOCO_REINDEX]).tolist()))
        return state

    def finish_program(self,reason,ok=True):
        if self.program:self.program.close();self.program=None
        report={'program_id':self.program_id,'reason':reason,'executed_steps':self.program_tick,
            'requested_steps':self.program_requested,'effective_limit':self.program_limit,'final_memory':self.program_memory,
            'task_success_asserted':False}
        self.events.write('program_finished',report)
        self.reply(ok,[{'type':'inputText','text':json.dumps(report)}]+self.content())
        self.program_id=None

    def begin_program(self,args):
        if not self.coding_control_enabled:raise ValueError('coding_control is disabled for this run')
        if not isinstance(args,dict) or set(args)!={'note','max_steps','code'} or not isinstance(args['note'],str):raise ValueError('coding_control requires note, max_steps, code')
        if type(args['max_steps']) is not int or not 1<=args['max_steps']<=500:raise ValueError('max_steps must be integer 1..500')
        if not isinstance(args['code'],str) or not 1<=len(args['code'])<=16000:raise ValueError('code must contain 1..16000 characters')
        self.program_counter+=1;self.program_id=f'program-{self.program_counter:03d}'
        folder=self.out/'programs';folder.mkdir(exist_ok=True)
        (folder/(self.program_id+'.py')).write_text(args['code'])
        (folder/(self.program_id+'.json')).write_text(json.dumps(args,indent=2))
        self.program_tick=0;self.program_memory={};self.program_requested=args['max_steps']
        self.program_limit=min(args['max_steps'],max(0,round(self.cfg.sim_time/self.cfg.control_dt)-self.step_count))
        self.remaining=self.program_limit
        self.events.write('program_started',{'program_id':self.program_id,'code':args['code'],'max_steps':args['max_steps'],'effective_limit':self.remaining})
        self.program=ControllerProgram(args['code'])
        self.targets=self.env.joint_pos.copy() if self.mode=='direct' else np.zeros(len(self.names))

    def turn_start(self, parts):
        inp=[{'type':'image','url':p['imageUrl']} if p['type']=='inputImage' else {'type':'text','text':p['text'],'text_elements':[]} for p in parts]
        self.turn=self.session.rpc('turn/start',{'threadId':self.thread,'input':inp})['turn']['id']

    def start(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'): raise RuntimeError('Use isolated Docker launcher')
        from .control import action_guide
        guide=action_guide(self.names,self.limits,self.mode,self.cfg.control_dt)
        (self.out/'action-guide.md').write_text(guide)
        self.session=CodexSession(self.manifest,self.out); self.session.__enter__()
        from environment.runtime.nonaction_budget import attach
        attach(self.session,self.out,lambda:self.step_count)
        enabled = self.coding_control_enabled
        base = BASE if enabled else '\n'.join(line for line in BASE.splitlines() if 'coding_control' not in line)
        tools = specs(self.names,self.limits,self.mode)
        if not enabled:tools = [tool for tool in tools if tool['name'] != 'coding_control']
        from environment.benchmarks.action_contracts import describe_tools
        tools = describe_tools(tools, guide)
        coding = CODING_INSTRUCTIONS if enabled else 'coding_control is disabled. Use the declared joint/action tools. Offline shell calculations cannot advance the simulator.'
        params={'model':self.model,'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
            'baseInstructions':base+'\n'+operating_brief('humanoid_soccer')+'\n'+guide+'\n'+coding+'\n'+self.history.instructions, 'developerInstructions':DEV+'\n'+getattr(self,'controller_notes',''), 'dynamicTools':tools,
            'config':{'features.shell_tool':True,'features.code_mode':True,'features.multi_agent':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}}
        (self.out/'prompt.json').write_text(json.dumps(params,indent=2))
        self.thread=self.session.rpc('thread/start',params)['thread']['id'];self.turn_start(self.content())

    def reply(self, ok, parts):
        result={'success':ok,'contentItems':parts}
        self.session.send({'id':self.pending['id'],'result':result})
        self.tools.write('tool_completed',{'call_id':self.pending['params']['callId'],'response':result})
        self.pending=None

    def step(self, obs, time_step, motion):
        from mujoco_soccer.constants import ISAACLAB_TO_MUJOCO_REINDEX, MUJOCO_TO_ISAACLAB_REINDEX
        from mujoco_soccer.policy import PolicyStep, _motion_reference_at
        # Exactly one recurrent inference per physics control step; retries never advance it twice.
        self.proposal = self.original.step(obs,time_step,motion) if self.mode=='hybrid' else None
        if self.session is None:self.start()
        if self.pending and self.remaining==0:
            if self.program_id:self.finish_program('max_steps reached')
            else:self.reply(True,[{'type':'inputText','text':'Requested segment completed; task success is not asserted.'}]+self.content())
        while True:
            while self.pending is None:
                remaining=self.deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Agent wall budget exhausted')
                msg=self.session.receive(remaining);method=msg.get('method')
                if method=='item/tool/call':
                    call=msg['params'];self.tools.write('tool_requested',call)
                    if call.get('threadId')!=self.thread or call.get('turnId')!=self.turn or call['callId'] in self.seen:raise RuntimeError('Duplicate/cross-session call')
                    self.seen.add(call['callId']);self.pending=msg
                    try:
                        if call['tool']=='coding_control':self.begin_program(call['arguments'])
                        else:self.remaining,self.targets=validate(call['tool'],call['arguments'],self.names,self.limits,self.env.joint_pos,self.mode)
                    except (ValueError,TypeError,KeyError) as e:
                        if self.program_id:self.finish_program(str(e),False)
                        else:self.reply(False,[{'type':'inputText','text':str(e)}])
                        continue
                    self.decision=call['arguments'].get('decision','direct')
                elif method=='turn/completed':
                    if msg['params']['turn']['status']!='completed':raise RuntimeError('Agent turn failed')
                    self.turn_start(continue_packet(self,self.step_count,self.content))
                elif method and 'id' in msg:self.session.send({'id':msg['id'],'error':{'code':-32601,'message':'Unsupported RPC'}})
            if self.program:
              try:
                  code_obs=self.code_observation();started=time.monotonic()
                  result=self.program.request({'obs':code_obs})
                  command=result['action'];self.program_memory=result['memory']
                  self.events.write('program_tick',{'program_id':self.program_id,'program_step':self.program_tick,
                      'control_step':self.step_count,'observation':code_obs,
                      'callback_wall_ms':(time.monotonic()-started)*1000,
                      'command':command,'memory':self.program_memory})
                  if not isinstance(command,dict):raise ValueError('control must return an action dict or {"done":true}')
                  if command=={'done':True} and type(command.get('done')) is bool:
                      self.finish_program('program returned done');continue
                  name='move_joints' if self.mode=='direct' else 'review_action'
                  expected={'joint_positions'} if self.mode=='direct' else {'decision','joint_offsets'}
                  if set(command)!=expected:raise ValueError('Invalid controller return keys; expected '+str(expected)+' or {"done":true}')
                  _,self.targets=validate(name,{'note':'coding_control','steps':1,**command},self.names,self.limits,self.targets,self.mode)
                  self.decision=command.get('decision','direct')
              except (ValueError,TypeError,KeyError,BrokenPipeError) as e:
                  self.finish_program(str(e),False);continue
            break
        if self.mode=='hybrid':
            refs=(self.proposal.ref_joint_pos,self.proposal.ref_joint_vel,self.proposal.ref_ang_vel)
        else:
            refs=_motion_reference_at(motion,time_step)
        action=compose_action(self.mode,self.decision,None if self.proposal is None else self.proposal.action,
            self.targets,self.default,self.scale,self.limits,ISAACLAB_TO_MUJOCO_REINDEX,MUJOCO_TO_ISAACLAB_REINDEX)
        self.ref_joint_pos,self.ref_joint_vel,self.ref_ang_vel=refs
        self.events.write('action_requested',{'control_step':self.step_count,'decision':self.decision,
            'program_id':self.program_id,
            'original_action':None if self.proposal is None else self.proposal.action,'executed_action':action,
            'action_stage':'input to PD; optional balance assist subsequently adjusts ankle targets',
            'command_targets_or_offsets':self.targets,'state':self.state()})
        return PolicyStep(action,*refs)

    def completed_step(self):
        self.step_count+=1;self.remaining-=1
        if self.program:self.program_tick+=1
        self.events.write('action_completed',{'control_step':self.step_count,'state':self.state(),
            'last_physics_substep_pd_targets':getattr(self.env,'last_pd_targets',None),
            'qpos':self.env.data.qpos.copy(),'qvel':self.env.data.qvel.copy(),'actuator_torque':self.env.data.ctrl.copy()})

    def close(self, complete):
        if self.session:
            try:
                if self.pending:
                    if self.program_id:self.finish_program('episode ended',complete)
                    else:self.reply(complete,[{'type':'inputText','text':f'Episode ended. Executed control steps={self.step_count}; official_complete={complete}. Unexecuted requested steps={max(0,self.remaining)}.'}]+self.content())
            finally:
                if self.program:self.program.close();self.program=None
                self.session.__exit__(None,None,None);self.session=None
