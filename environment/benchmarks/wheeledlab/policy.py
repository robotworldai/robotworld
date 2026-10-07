"""Independent source-Codex app-server controls unchanged WheeledLab native actions."""
import base64,io,json,os,shutil,time
from collections import deque
from pathlib import Path
from PIL import Image
from environment.runtime.events import EventLog
from environment.runtime.image_history import publish_image_window
from environment.benchmarks.operating_brief import operating_brief
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram
from .catalog import CASES, MODEL_TITLES
from .control import specs,validate_action,action_guide

BASE='''You drive one RC car in the original WheeledLab task using native robot tools. No resets, teleportation, scene/physics edits, hidden evaluation queries, or simulator access through shell/files. Only dynamic driving tools advance simulation. Shell is for calculations in /workspace; assume python3 and standard-library math, not NumPy. Do not use shell/network to bypass robot tools.
Vehicle interface: native action=[normalized_speed,normalized_steering], each in[-1,1]. The original action term clips, scales by[3.0,0.488] with zero offset, then clamps negative speed to zero. No reverse is available. Speed is a wheel-velocity target in m/s, not force throttle or guaranteed actual vehicle velocity. Steering is a central command converted through the ORIGINAL tan-steering logic; do not assume wheel joint angles equal0.488*action. Wheel radius0.05m; MuSHR wheelbase0.325m, width0.2m; F1Tenth wheelbase0.365m, width0.284m. Original friction, drives, suspension and randomization remain active. No automatic stabilizer or route planner is supplied.
Drive holds the same native action for requested steps; all driven wheels and steering are updated simultaneously by upstream. Zero speed does not instantaneously freeze the car. Use observed motion and short segments near boundaries. Observe costs no physics step. LLM reasoning and shell computation pause physics.
Observations: native_policy_terms are the upstream policy group AFTER configured corruption/augmentation, with original term names, scales and ordering. World xyz is metres, Z up; root_euler_xyz is radians; base velocities are in the vehicle frame. Pose/velocity corruption may be active. Do not interpret noisy pose as exact localization. Term 'last_action' is normalized, not physical speed. No ground-truth reward, terminal predicate internals, traversability label map or hidden goal coordinates are exposed. Only original policy measurements are available. The observer video is for review and is not a policy input.
Visual task: the image is the upstream cropped/augmented normalized grayscale observation rendered for you,40x80 pixels. It is not an overhead map. At tool boundaries you receive current and up to four previous observations at interval2; history selection counts tool observations, and each state also reports its actual control_step; spacing is not fixed simulation time. Nonvisual tasks return corresponding history of numeric policy observations instead of invented images.
Evaluation: maintain native task objectives until the first native termination/time limit; final text does not end an active episode. Tool success means execution, not task success. The upstream repository exposes training rewards and termination terms, not one standardized binary simulator score for all tasks. Do not invent success. Do not alter task/time horizon to get a passing result.
'''
TASK={
 'drift':'''Objective: drive the stadium track with controlled drift while staying in the native track band. The original scene is a plane; absence of painted track lines does not remove the numerical track boundary. The straights run along world Y between y=-0.8 and y=+0.8; their x positions are around +/-0.8m. Corner centres are (0,+0.8) and (0,-0.8). Corner inner radius0.3m and outer radius2.0m. Native drift rewards combine motion/track/side-slip terms. The native out_of_bounds predicate can terminate before5seconds. MuSHR is RWD, F1Tenth is4WD. Use native position/yaw/body velocities to track the course; do not simply steer at full lock indefinitely. Physics200Hz, actions50Hz, four physics ticks per action,250action horizon.''',
 'elevation':'''Objective: traverse the original elevated terrain toward the task's generated goal while avoiding low height, getting stuck and rollover. Native height map is26x26 and included as elevation_map; original world_height_map processing is retained. Goal_relative_xyz is exactly the original task term: it is produced by the upstream goal-command processing, not a promised new world-frame waypoint API. Do not query hidden goal state. Physics100Hz, actions10Hz, ten physics ticks per action,200action horizon. Commands may resample on their native schedule. Native at_goal is reported separately by the evaluator after the episode.''',
 'visual':'''Objective: follow the traversable light-colored path in the original procedurally generated black/white map, using the native onboard camera and body velocities. No global map or ground-truth traversability labels are provided. Preserve forward motion while staying on the path. Original visual rewards score traversability and motion; out_range terminates at map bounds. Physics50Hz, actions5Hz, ten physics ticks per action,50action horizon. Image augmentation is original, not a rendering error to remove.''',
}
CODING='''Use coding_control for feedback at the task's control frequency. Define control(obs,memory) returning {"action":[speed,steering]}, or {"done":true} to return to model deliberation without taking another action. You may use arithmetic, math, if/loops, lists/dicts; no NumPy, files, eval/exec, arbitrary imports, classes, lambdas or simulator handles. Memory is JSON and persists only within a call. Limits16000chars,100000operations,4096collection items,2s per callback. All callbacks consume the same environment step budget. Visual callback state does not contain pixel arrays; obtain images through normal tool responses. Bound commands, handle noisy feedback and stop a segment if your assumptions break.'''

class Agent:
    def __init__(self,sim,out,manifest,model,steps,timeout,coding=True):
        self.sim=sim;self.out=Path(out);self.manifest=manifest;self.model=model;self.steps=steps;self.coding=coding
        self.deadline=time.monotonic()+timeout;self.events=EventLog(self.out/'events/environment.jsonl');self.tools=EventLog(self.out/'events/tools.jsonl')
        self.history=deque(maxlen=9);self.seq=0;self.program=0;self.seen=set()
        self.tool_schema=specs(coding,CASES.get(getattr(sim,'case',None),{}).get('family')=='custom',getattr(sim,'spec',{}).get('reverse_enabled',False))
    @remember_packet(lambda self:self.sim.steps)
    def content(self):
        state=self.sim.observation();state['episode_steps_remaining']=max(0,self.steps-self.sim.steps)
        folder=self.out/'observations'/str(self.seq);folder.mkdir(parents=True,exist_ok=True);(folder/'state.json').write_text(json.dumps(state,indent=2))
        frame=[]
        if self.sim.policy_image is not None:
            im=Image.fromarray(self.sim.policy_image);im.save(folder/('front-camera.png' if hasattr(self.sim,'policy_sensor') else 'native-camera.png'));b=io.BytesIO();im.save(b,format='PNG');frame.append(('FRONT onboard' if hasattr(self.sim,'policy_sensor') else 'native camera',base64.b64encode(b.getvalue()).decode()))
        rear=getattr(self.sim,'rear_sensor',None)
        if rear is not None and rear.image is not None:
            im=Image.fromarray(rear.image);im.save(folder/'rear-camera.png');b=io.BytesIO();im.save(b,format='PNG');frame.append(('REAR onboard (not mirrored)',base64.b64encode(b.getvalue()).decode()))
        export=os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export:shutil.copytree(folder,Path(export)/str(self.seq),dirs_exist_ok=True)
        self.history.append((self.seq,state,frame));parts=[];rounds=[]
        for seq,obs,img in self.history:
            if seq in {self.seq-2*i for i in range(5)}:
                rounds.append(dict(observation=seq,env_step=obs['control_step'],role='CURRENT' if seq==self.seq else 'HISTORY'))
                parts.append({'type':'inputText','text':f'{rounds[-1]["role"]} observation {seq}; env step {obs["control_step"]}'})
                parts.append({'type':'inputText','text':f'{"CURRENT" if seq==self.seq else "HISTORY"} observation={seq}\n'+json.dumps(obs)})
                for label,pixels in img:
                    parts.append({'type':'inputText','text':label});parts.append({'type':'inputImage','imageUrl':'data:image/png;base64,'+pixels})
        publish_image_window(Path(export).parent if export else self.out,parts,rounds)
        self.seq+=1;self.events.write('observation',{'state':state,'content':parts});return parts
    def execute(self,name,args):
        if name not in {t['name'] for t in self.tool_schema}:raise ValueError('Tool unavailable in this run')
        keys={'observe':{'note'},'drive':{'note','action','steps'},'coding_control':{'note','code','max_steps'}}
        if not isinstance(args,dict) or set(args)!=keys[name] or not isinstance(args['note'],str):raise ValueError('Invalid tool arguments')
        if name=='observe':return {'executed_steps':0}
        count=args.get('steps',args.get('max_steps'))
        if type(count)is not int or not 1<=count<=(50 if name=='drive' else 250):raise ValueError('Invalid duration')
        count=min(count,self.steps-self.sim.steps);program=None;executed=0;memory={};reason='segment_complete'
        try:
            if name=='drive':action=validate_action(args['action'])
            else:
                code=args['code']
                if not isinstance(code,str) or not 1<=len(code)<=16000:raise ValueError('Invalid code')
                self.program+=1;folder=self.out/'programs';folder.mkdir(exist_ok=True);(folder/f'{self.program:04d}.py').write_text(code)
                program=ControllerProgram(code)
            for i in range(count):
                if self.sim.done:reason='native_termination';break
                if time.monotonic()>self.deadline:raise TimeoutError('Wall timeout')
                before=self.sim.observation()
                if program:
                    callback_obs=getattr(self.sim,'controller_observation',self.sim.observation)()
                    state={**callback_obs,'program_step':i,'episode_steps_remaining':self.steps-self.sim.steps}
                    result=program.request({'obs':state});command=result['action'];memory=result['memory']
                    self.events.write('program_tick',{'program_id':self.program,'observation':state,'command':command,'memory':memory})
                    if command=={'done':True} and type(command['done'])is bool:reason='program_done';break
                    if not isinstance(command,dict) or set(command)!={'action'}:raise ValueError('Return action or done')
                    action=validate_action(command['action'])
                after=self.sim.step(action);executed+=1
                self.events.write('environment_step',{'requested_action':action,'before':before,'after':after,'evaluation':self.sim.last_evaluation})
            return {'executed_steps':executed,'reason':reason,'final_memory':memory,'task_success_asserted':False}
        except (ValueError,TypeError,KeyError,BrokenPipeError) as e:return {'executed_steps':executed,'error':str(e),'reason':'program_or_action_error'}
        finally:
            if program:program.close()
    def start_turn(self,parts):
        data=[{'type':'image','url':x['imageUrl']} if x['type']=='inputImage' else {'type':'text','text':x['text'],'text_elements':[]} for x in parts]
        self.turn=self.session.rpc('turn/start',{'threadId':self.thread,'input':data})['turn']['id']
    def run(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'):raise RuntimeError('Use isolated launcher')
        if CASES[self.sim.case]['family']=='custom':
            from .custom import instructions
            base=instructions(self.sim)
        else:base=BASE+'\n'+TASK[CASES[self.sim.case]['family']]
        coding_prompt=CODING
        if CASES[self.sim.case]['family']=='custom':
            coding_prompt=CODING.replace('Visual callback state does not contain pixel arrays; obtain images through normal tool responses.',
                "Each callback receives obs['front_camera'], a dictionary with width=48,height=27 and pixels. Read the flat RGB8 array from obs['front_camera']['pixels'], index=(y*48+x)*3+channel, values0..255. These are downsampled actual onboard camera pixels, not road labels or a map. Full640x360 front RGB is provided to the model at tool boundaries. Camera/encoder/IMU measurements are all the available feedback; infer routes and hazards yourself.")
        guide=action_guide(self.sim.dt,getattr(self.sim,'spec',{}).get('reverse_enabled',False))
        from environment.benchmarks.action_contracts import describe_tools
        self.tool_schema=describe_tools(self.tool_schema, guide)
        (self.out/'action-guide.md').write_text(guide)
        if getattr(self.sim,'world_success_scene',None) and self.sim.world_success_scene.enabled:
            from environment.evaluation.world_success.runtime import instruction
            base+='\n\n'+instruction(self.sim.world_success_monitor.profile)
        prompt={'model':self.model,'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
          'baseInstructions':base+'\n'+operating_brief('wheeledlab')+'\n'+guide+'\n'+(coding_prompt if self.coding else 'Only observe and drive are enabled; no per-tick program execution tool is available.')+'\nTask: '+MODEL_TITLES[self.sim.case],
          'developerInstructions':'Use shell only for calculations in /workspace. Only dynamic tools control physics. /observations contains allowed policy observations.',
          'dynamicTools':self.tool_schema,'config':{'features.shell_tool':True,'features.code_mode':True,'features.multi_agent':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}}
        (self.out/'prompt.json').write_text(json.dumps(prompt,indent=2))
        with CodexSession(self.manifest,self.out) as self.session:
            from environment.runtime.nonaction_budget import attach
            attach(self.session,self.out,lambda:self.sim.steps)
            self.thread=self.session.rpc('thread/start',prompt)['thread']['id'];self.start_turn(self.content())
            while self.sim.steps<self.steps and not self.sim.done:
                remaining=self.deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Agent budget exhausted')
                message=self.session.receive(remaining);method=message.get('method')
                if method=='item/tool/call':
                    c=message['params']
                    if c.get('threadId')!=self.thread or c.get('turnId')!=self.turn or c['callId']in self.seen:raise RuntimeError('Duplicate/cross-session tool call')
                    self.seen.add(c['callId']);self.tools.write('tool_requested',c)
                    try:report=self.execute(c['tool'],c['arguments'])
                    except (ValueError,TypeError,KeyError) as e:report={'error':str(e),'executed_steps':0}
                    result={'success':'error'not in report,'contentItems':[{'type':'inputText','text':json.dumps(report)}]+self.content()}
                    self.session.send({'id':message['id'],'result':result});self.tools.write('tool_completed',{'call_id':c['callId'],'response':result})
                elif method=='turn/completed':
                    if message['params']['turn']['status']!='completed':raise RuntimeError('Agent turn failed')
                    self.start_turn(continue_packet(self,self.sim.steps,self.content))
                elif method and 'id'in message:self.session.send({'id':message['id'],'error':{'code':-32601,'message':'Unsupported RPC'}})
