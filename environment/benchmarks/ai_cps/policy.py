"""Source-Codex dynamic tools; physics only advances through native actions."""
from environment.benchmarks.action_contracts import action_contract
import base64, io, json, os, shutil, time
from collections import deque
from pathlib import Path
from PIL import Image
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.runtime.events import EventLog
from environment.runtime.image_history import INSTRUCTIONS, publish_image_window
from environment.benchmarks.operating_brief import operating_brief
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
from .control import specs,validate_action

BASE='''You control one fixed-base Franka Panda in an AI-CPS manipulation task. Use robot tools to advance physics; no resets, teleports, edits to scene/physics, hidden evaluator queries or simulator access via shell/files. The task starts with its tool in the hand. All seven arm joints can move together; the two finger targets are controlled by the ORIGINAL task and are not independently controllable. There is no native Cartesian/EEF or inverse-kinematics tool.
Actions and coordinates:
- move_joints arm_action is seven dimensionless increments in panda_joint1..7 order, each in [-1,1], NOT absolute radians or velocities. The task adds action * 7.5 * (1/60) = action * 0.125 rad to the PREVIOUS joint target on every tick, then clamps to the mechanical limits. Repeating an action keeps accumulating target changes. A zero request holds the old target before noise; it does not latch current angles or brake momentum.
- Original evaluation adds independent Gaussian action noise with std 0.5 AFTER request clipping, before target construction. Thus zero commands can move. Inspect measured positions, previous target and actual noisy action. Do not assume a planned trajectory was executed exactly.
- The task config requests physics dt=0.0083 seconds. Core quantizes this to120 physics steps/second: two substeps per action, actual control dt~1/60 second (60Hz), recorded in obs.dt. The native action multiplier also uses1/60. A 300-step episode is roughly five simulated seconds. LLM reasoning pauses physics. Each tool step and each coding callback consumes this SAME episode budget; observations and calculations do not.
- Numeric observations are simulator measurements, as in the upstream state-based benchmark. World coordinates: metres, z up, quaternion w,x,y,z; arm joint radians and rad/s; the two prismatic fingers are separately reported in metres and m/s. Native observation first 9 entries: joint positions scaled to [-1,1] using limits; next 9: measured joint velocity multiplied by 0.1 for ball tasks or 0.02 for peg tasks. The native array is clipped to [-5,5]. Ball tasks entries18:21 ball xyz,21:24 tool-centre minus ball xyz,24:27 ball velocity xyz. Peg task18:21 tool xyz,21:25 tool quaternion,25:28 tool-minus-hole xyz. Check the named joint limits and measured state before acting.
- Catch: intercept the moving ball with the held catching tool, then retain it. Balance: keep the ball near the centre of the held tray. Peg: align the held tool to the table hole and insert gently. Peg fingers open automatically when the original 3D release criterion is met, otherwise they stay at0.015; do not assume contact means insertion.
Observations and evidence:
- Images show ONE scene from an external camera. Each response includes current plus up to four prior tool-boundary observations at interval2; this is not an eight-physics-step history. Their actual control-step indices are labeled. Video separately records EVERY control step at simulated-time rate.
- The upstream score uses XY distance in fixed time windows, not a physical grasp/insertion certificate. Tool completion is not task success. Do not claim success from a returned command; continue until the environment horizon. Native termination is authoritative. A final text answer cannot stop an active episode.
Feedback coding:
- Use coding_control for fast feedback (intercepting, stabilizing a ball or reacting to contact). Define control(obs,memory). The runner calls it once before EACH action using fresh numeric state; return {"arm_action":[a1,a2,a3,a4,a5,a6,a7]}. Return {"done":true} to stop a segment without the next step. This does not finish the task.
- obs includes the named measurements above, dt, program_step and episode_steps_remaining. memory is a mutable JSON dictionary, initially empty and persistent within this one call. Archive useful constants in the next program if needed. max_steps caps actual actions by the remaining budget. Invalid output/runtime error stops the program and reports any actions already executed; there is no rollback.
- Supported bounded Python subset: def functions (positional parameters), arithmetic, comparisons, if/elif/else, for/while, break/continue, lists/dicts/comprehensions/indexing/slicing, optional import math. Builtins abs,min,max,sum,len,range,enumerate,zip,int,float,bool,round,list,dict,tuple,sorted,all,any. Dict get/copy/items/keys/values/update/setdefault/pop; list copy/append/pop. math sin/cos/tan/asin/acos/atan/atan2/sqrt/exp/log/floor/ceil/fabs/hypot/degrees/radians/isfinite/copysign and pi/e/tau. No numpy, files, eval, exec, arbitrary imports, classes, lambdas, generators, decorators or simulator handles. Limits:16000 code characters,4096 collection entries,100000 interpreted operations per callback,2s callback deadline.
- Consider short low-amplitude probes to estimate how joint increments move the tool; use observed error/velocity for bounded corrections. You can compute offline in /workspace, but ONLY dynamic robot tools cause motion. Code receives no extra privileged scene or reward/success oracle. No supplied controller solves the task for you.
'''
RECOVERY='''ID34 is an authored contact-recovery diagnostic built on the ORIGINAL peg task. Contact sensor reports ONLY peg/table pair force, excluding the normal gripper grasp. A force >=10N on two consecutive physics substeps triggers an anomaly and immediately returns control at the end of that action tick. The old segment is discarded; call cancel_action to acknowledge and latch degraded control before further movement. No synthetic collision is injected. Degraded mode caps every arm target increment, AFTER upstream action noise, to +/-0.00625rad/tick; inertia means actual velocity need not be equally small. Inspect measured velocity and contact, retreat or re-align gently. Recovery requires cancellation followed by >=10 degraded steps and the LAST >=10 consecutive steps below5N; no anomaly means NOT COVERED, not success. Recovered contact handling and native peg success are scored separately. Merely saying cancelled or completed is insufficient.'''

JOINT_ONLY='''Direct joint-tool control (feedback-code tool masked):
- Use observe and move_joints. No per-tick user-program execution tool is available. You decide seven normalized joint increments and a duration of 1..50 control steps for each move_joints call. They are repeated unchanged during that segment; a longer segment is NOT interpolation to a fixed pose.
- Receive fresh numeric state and camera history after each tool call, then choose the next segment. For rapidly moving balls, short segments let you revise the action before the interception opportunity passes. All seven joints can move in the same call.
- You may use shell for offline calculations and notes, but shell must not advance physics, call the simulator, or bypass the dynamic-tool interface. Robot motion is exclusively through move_joints (and cancel_action acknowledgement in ID34). Reasoning and calculations pause physics.
- Account for the original action noise and inertia. Use measured joint positions, velocities and last targets; zero increments do not brake momentum or remove noise. There is no supplied balancing controller, EEF planner or automatic task solver.
- For catching, predict a reachable future ball/tool intersection from measured position, velocity and gravity; use the verified robot-root/world transform, keep the opening suitably oriented and check containment after contact. Return to observation and revise commands as needed. Do not substitute XY proximity for physical capture.
'''

def instructions(case,task_name,coding_control_enabled=True):
    base=BASE if coding_control_enabled else BASE.split('Feedback coding:\n',1)[0]+JOINT_ONLY
    if not coding_control_enabled:
        base=base.replace('Each tool step and each coding callback consumes this SAME episode budget','Each executed move_joints tick consumes this SAME episode budget')
    if case=='34':base+=RECOVERY
    if case=='24':
        base+='\nCase 24 evaluation (peg-xy-z-window-v2): throughout trace samples 250 through 299, keep tool-to-hole XY distance <= 0.1 m and the tool reference point world z between 0.38 and 0.40 m, inclusive. This reference point is not the peg bottom. Reaching z=0.43 m only brings the bottom to the surface. Maintain the conditions through the final window, not just momentarily. No additional orientation-angle threshold is scored. The original XY STL result is recorded separately.\n'
    if case=='22':
        catch=(Path(__file__).parent/'prompts/catch.md').read_text()
        if not coding_control_enabled:
            scene=catch.split('Control and feedback workflow',1)[0]
            evaluation=catch.split('Evaluation and termination\n',1)[1]
            catch=scene+'Evaluation and termination\n'+evaluation
            catch=catch.replace('every callback','every observation').replace('each control tick','after each tool call').replace('code callbacks do.','executed move_joints ticks do.')
        base+='\n'+catch
    return base+'\n'+action_contract('ai_cps')+'\n'+operating_brief('ai_cps')+'\nCurrent task: '+task_name

class Agent:
    def __init__(self,sim,output,manifest,model,steps,timeout,coding_control_enabled=True):
        self.sim=sim;self.out=Path(output);self.steps=steps;self.manifest=manifest;self.model=model
        self.coding_control_enabled=coding_control_enabled
        self.events=EventLog(self.out/'events/environment.jsonl');self.tools=EventLog(self.out/'events/tools.jsonl')
        self.history=deque(maxlen=9);self.seq=0;self.program=0;self.deadline=time.monotonic()+timeout;self.seen=set()
    @remember_packet(lambda self:self.sim.steps)
    def content(self):
        state=self.sim.observation();state['episode_steps_remaining']=max(0,self.steps-self.sim.steps)
        folder=self.out/'observations'/str(self.seq);folder.mkdir(parents=True,exist_ok=True)
        (folder/'state.json').write_text(json.dumps(state,indent=2));img=Image.fromarray(self.sim.camera.image);img.save(folder/'camera.png')
        export=os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export:shutil.copytree(folder,Path(export)/str(self.seq),dirs_exist_ok=True)
        b=io.BytesIO();img.save(b,format='JPEG',quality=85)
        self.history.append((self.seq,self.sim.steps,base64.b64encode(b.getvalue()).decode()))
        parts=[{'type':'inputText','text':json.dumps(state)}];rounds=[]
        for seq,step,data in self.history:
            if seq in {self.seq-2*i for i in range(5)}:
                role='CURRENT' if seq==self.seq else 'HISTORY'
                rounds.append(dict(observation=seq,env_step=step,role=role))
                parts.extend([{'type':'inputText','text':f'{role} observation {seq}; env step {step}'},{'type':'inputImage','imageUrl':'data:image/jpeg;base64,'+data}])
        publish_image_window(self.out,parts,rounds)
        self.seq+=1;self.events.write('observation',{'state':state,'content':parts});return parts
    def start_turn(self,parts):
        inp=[{'type':'image','url':x['imageUrl']} if x['type']=='inputImage' else {'type':'text','text':x['text'],'text_elements':[]} for x in parts]
        self.turn=self.session.rpc('turn/start',{'threadId':self.thread,'input':inp})['turn']['id']
    def execute(self,name,args):
        if name=='coding_control' and not getattr(self,'coding_control_enabled',True):raise ValueError('coding_control is disabled for this run')
        if not isinstance(args,dict) or not isinstance(args.get('note'),str):raise ValueError('note required')
        allowed={'observe':{'note'},'cancel_action':{'note'},'move_joints':{'note','steps','arm_action'},'coding_control':{'note','max_steps','code'}}
        if name not in allowed or set(args)!=allowed[name]:raise ValueError('Unknown tool or invalid arguments')
        if name=='observe':return {'executed_steps':0}
        if name=='cancel_action':
            if not self.sim.recovery:raise ValueError('Only ID34 supports cancellation')
            self.sim.recovery.cancel(self.sim.steps);self.events.write('action_cancelled',self.sim.recovery.observation());return self.sim.recovery.observation()
        if self.sim.recovery and self.sim.recovery.pending:raise ValueError('Contact interrupted the segment; cancel_action required before continuing')
        count=args.get('steps',args.get('max_steps'))
        if type(count) is not int or not 1<=count<=(50 if name=='move_joints' else 300):raise ValueError('Invalid step count')
        count=min(count,self.steps-self.sim.steps);program=None;executed=0;reason='segment_complete';memory={}
        try:
            if name=='coding_control':
                code=args['code']
                if not isinstance(code,str) or not 1<=len(code)<=16000:raise ValueError('Invalid code')
                self.program+=1;folder=self.out/'programs';folder.mkdir(exist_ok=True)
                (folder/f'{self.program:04d}.py').write_text(code);program=ControllerProgram(code)
            else:action=validate_action(args['arm_action'])
            for tick in range(count):
                if time.monotonic()>self.deadline:raise TimeoutError('Wall timeout')
                if self.sim.done:reason='native_termination';break
                before=self.sim.observation()
                if program:
                    obs={**before,'program_step':tick,'episode_steps_remaining':self.steps-self.sim.steps}
                    reply=program.request({'obs':obs});command=reply['action'];memory=reply['memory']
                    self.events.write('program_tick',{'program_id':self.program,'observation':obs,'command':command,'memory':memory})
                    if command=={'done':True} and type(command['done']) is bool:reason='program_done';break
                    if not isinstance(command,dict) or set(command)!={'arm_action'}:raise ValueError('Return arm_action or done')
                    action=validate_action(command['arm_action'])
                state=self.sim.step(action);executed+=1
                self.events.write('environment_step',{'requested_action':action,'before':before,'after':state})
                if self.sim.recovery and self.sim.recovery.pending:reason='real_contact_interrupted';break
            return {'executed_steps':executed,'reason':reason,'final_memory':memory,'task_success_asserted':False}
        except (ValueError,TypeError,KeyError,BrokenPipeError) as e:
            return {'executed_steps':executed,'error':str(e),'reason':'program_or_action_error','final_memory':memory}
        finally:
            if program:program.close()
    def run(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'):raise RuntimeError('Use isolated Docker launcher')
        params={'model':self.model,'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
          'baseInstructions':instructions(self.sim.case,self.sim.task_name,self.coding_control_enabled)+'\n'+INSTRUCTIONS,
          'developerInstructions':'Shell is for calculations/notes in /workspace. Only dynamic robot tools may control physics. /observations exports allowed observations.',
          'dynamicTools':specs(self.sim.case,self.coding_control_enabled),'config':{'features.shell_tool':True,'features.code_mode':True,'features.multi_agent':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}}
        (self.out/'prompt.json').write_text(json.dumps(params,indent=2))
        with CodexSession(self.manifest,self.out) as self.session:
            from environment.runtime.nonaction_budget import attach
            attach(self.session,self.out,lambda:self.sim.steps)
            self.thread=self.session.rpc('thread/start',params)['thread']['id'];self.start_turn(self.content())
            while self.sim.steps<self.steps and not self.sim.done:
                remaining=self.deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Agent wall budget exhausted')
                message=self.session.receive(remaining);method=message.get('method')
                if method=='item/tool/call':
                    call=message['params']
                    if call.get('threadId')!=self.thread or call.get('turnId')!=self.turn or call['callId'] in self.seen:raise RuntimeError('Duplicate/cross-session call')
                    self.seen.add(call['callId']);self.tools.write('tool_requested',call)
                    try:report=self.execute(call['tool'],call['arguments'])
                    except (ValueError,TypeError,KeyError) as e:report={'error':str(e),'executed_steps':0}
                    result={'success':'error' not in report,'contentItems':[{'type':'inputText','text':json.dumps(report)}]+self.content()}
                    self.session.send({'id':message['id'],'result':result});self.tools.write('tool_completed',{'call_id':call['callId'],'response':result})
                elif method=='turn/completed':
                    if message['params']['turn']['status']!='completed':raise RuntimeError('Agent turn failed')
                    self.start_turn(continue_packet(self,self.sim.steps,self.content))
                elif method and 'id' in message:self.session.send({'id':message['id'],'error':{'code':-32601,'message':'Unsupported RPC'}})
