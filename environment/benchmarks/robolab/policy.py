"""Source Codex InferenceClient plugged into the unchanged official evaluator."""
from environment.benchmarks.action_contracts import action_contract
import base64,io,json,os,time,shutil
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
from robolab.eval import InferenceClient
from environment.runtime.codex_session import CodexSession
from environment.runtime.continuation import remember_packet, continue_packet
from environment.runtime.events import EventLog
from environment.runtime.image_history import publish_image_window
from environment.benchmarks.operating_brief import operating_brief
from .control import specs,action_for,STATE_KEYS

BASE='''You control the fixed-base DROID robot in RoboLab. Complete the exact upstream task instruction using supplied RGB and measured proprioception only. The robot has one Franka arm and a Robotiq gripper; it has no mobile base, torso or head control. All positions are in metres relative to the robot articulation root; all quaternions are WXYZ. ee_pos/ee_quat describe the gripper base_link flange, not the fingertips. eef_pos/eef_quat describe a differently oriented reporting frame, NOT the IK target frame. Read the active controller profile and tool descriptions: joint_position uses seven absolute joint angles in radians; absolute_ik uses base_link absolute position and WXYZ orientation; relative_ik uses root-axis translation and rotation-vector input with upstream scale 0.5 on every repeated step. Do not confuse these modes or other benchmarks' conventions. gripper_close=0 opens and 1 closes. Use move_robot to combine arm and gripper in the same native action; closure begins alongside movement, not on arrival. Separate tools are sequential. A tool segment costs steps once; the native rate is 15 Hz. IK is not a collision-free planner; shorten segments near contact, inspect measured motion and verify grasp retention after lifting, moving and rotating. Verify release and placement visually; successful tool execution never proves task success. Maintain a concise plan and workspace memory, revise grasp/approach/subgoal after failure, and continue while the episode is active. Current images plus up to four historical observations sampled every two tool-feedback rounds are labeled; these are not simultaneous. The simulator pauses during model reasoning. Only the upstream evaluator decides success; no resets, hidden object state, privileged cameras, score queries or simulator access. An ordinary final answer does not stop the episode.'''
DEV='Shell/code and /workspace are for local calculations and notes. Read-only /observations contains only allowed RGB and proprioception. All movement and fresh observations must use the robot tools. Never read or change simulator/evaluator files or seek external task solutions.'

def array(v):return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)

class CodexClient(InferenceClient):
    def __init__(self,manifest,output,mode='joint_position',timeout=7200,max_calls=5000):
        super().__init__();self.manifest=manifest;self.output=Path(output);self.mode=mode
        self.timeout=timeout;self.max_calls=max_calls;self.session=None
    # ABC hooks are unused because infer is explicitly overridden.
    def _extract_observation(self,*args,**kwargs):raise NotImplementedError
    def _pack_request(self,*args):raise NotImplementedError
    def _query_server(self,*args):raise NotImplementedError
    def _unpack_response(self,*args):raise NotImplementedError
    def begin_episode(self,episode_idx):
        super().begin_episode(episode_idx)
        self.out=self.output/f'episode-{episode_idx:03d}';self.out.mkdir(parents=True,exist_ok=False)
        self.events=EventLog(self.out/'events/environment.jsonl');self.tools=EventLog(self.out/'events/tools.jsonl')
        self.step=0;self.calls=0;self.seq=0;self.pending=None;self.remaining=0;self.grip=0
        self.last=None;self.history=deque(maxlen=9);self.seen=set();self.deadline=time.monotonic()+self.timeout
    @remember_packet(lambda self:self.step)
    def content(self):
        frame=self.out/'frames'/str(self.seq);frame.mkdir(parents=True)
        text=f'Task: {self.instruction}\nController: {self.mode}; control step: {self.step}; tool calls remaining: {self.max_calls-self.calls}\nMeasured state: {json.dumps(self.state)}'
        (frame/'observation.txt').write_text(text)
        self.history.append((self.seq,self.step,{k:v.copy() for k,v in self.images.items()}))
        selected=[entry for entry in self.history if entry[0] in {self.seq-2*i for i in range(5)}]
        parts=[{'type':'inputText','text':text}]
        rounds=[]
        for idx,step,images in selected:
            role='CURRENT' if idx==self.seq else 'HISTORY'
            rounds.append(dict(role=role,observation=idx,env_step=step))
            parts.append({'type':'inputText','text':f'{role} observation {idx}; env step {step}'})
            for key,img in images.items():
                b=io.BytesIO();Image.fromarray(img).save(b,format='JPEG',quality=90)
                parts.extend([{'type':'inputText','text':key},{'type':'inputImage','imageUrl':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()}])
        for key,img in self.images.items():Image.fromarray(img).save(frame/(key+'.png'))
        (frame/'history.json').write_text(json.dumps([{'observation':i,'control_step':s} for i,s,_ in selected]))
        export=os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export:
            shutil.copytree(frame,Path(export)/self.out.parent.name/self.out.name/str(self.seq),dirs_exist_ok=True)
            publish_image_window(Path(export).parent,parts,rounds)
        self.seq+=1;return parts
    def turn_start(self,parts):
        inp=[{'type':'image','url':p['imageUrl']} if p['type']=='inputImage' else {'type':'text','text':p['text'],'text_elements':[]} for p in parts]
        self.turn=self.session.rpc('turn/start',{'threadId':self.thread,'input':inp})['turn']['id']
    def start(self):
        if not os.environ.get('WORLD_CODEX_SOCKET'):raise RuntimeError('Use the isolated Docker launcher')
        self.session=CodexSession(self.manifest,self.out);self.session.__enter__()
        from environment.runtime.nonaction_budget import attach
        attach(self.session,self.out,lambda:self.step)
        params={'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
                'baseInstructions':BASE+'\n'+action_contract('robolab', self.mode)+'\n'+operating_brief('robolab'),'developerInstructions':DEV,'dynamicTools':specs(self.mode),
                'config':{'features.shell_tool':True,'features.code_mode':True,'features.multi_agent':False,'features.apps':False,'features.plugins':False,'web_search':'disabled'}}
        (self.out/'prompt.json').write_text(json.dumps(params,indent=2));self.thread=self.session.rpc('thread/start',params)['thread']['id'];self.turn_start(self.content())
    def reply(self,message,success,content):
        response={'success':success,'contentItems':content};self.session.send({'id':message['id'],'result':response})
        self.tools.write('tool_completed',{'call_id':message['params']['callId'],'response':response})
    def infer(self,obs,instruction,*,env_id=0):
        if env_id!=0:raise ValueError('Only one logical environment supported')
        self.instruction=instruction
        self.state={k:array(obs['proprio_obs'][k])[0].tolist() for k in STATE_KEYS}
        self.images={k:array(obs['image_obs'][k])[0,:,:,:3] for k in ('over_shoulder_left_camera','wrist_cam')}
        if self.last is not None:
            self.step+=1;self.events.write('action_completed',{'env_step':self.step,'action':self.last,'state':self.state});self.remaining-=1;self.last=None
        self.events.write('observation',{'env_step':self.step,'state':self.state})
        if self.session is None:self.start()
        if self.pending and self.remaining==0:
            self.reply(self.pending,True,[{'type':'inputText','text':'Segment executed. Inspect observations; this does not prove task success.'}]+self.content());self.pending=None
        while self.pending is None:
            if self.calls>=self.max_calls or time.monotonic()>=self.deadline:raise TimeoutError('World policy budget exhausted; not an official task failure')
            m=self.session.receive(self.deadline-time.monotonic());method=m.get('method')
            if method=='item/tool/call':
                c=m['params'];self.tools.write('tool_requested',c)
                if c.get('threadId')!=self.thread or c.get('turnId')!=self.turn or c['callId'] in self.seen:raise RuntimeError('Cross-session/duplicate call')
                self.seen.add(c['callId'])
                try:action,n,grip=action_for(self.mode,c['tool'],c['arguments'],self.state,self.grip)
                except (ValueError,KeyError,TypeError) as e:
                    self.reply(m,False,[{'type':'inputText','text':str(e)}]);continue
                self.action=action;self.remaining=n;self.grip=grip;self.pending=m;self.calls+=1
            elif method=='turn/completed':
                p=m['params']
                if p.get('threadId')!=self.thread or p['turn']['id']!=self.turn or p['turn']['status']!='completed':raise RuntimeError('Agent turn failed')
                self.turn_start(continue_packet(self,self.step,self.content))
            elif method and 'id' in m:self.session.send({'id':m['id'],'error':{'code':-32601,'message':'Unsupported RPC'}})
        self.last=self.action.copy();self.events.write('action_requested',{'env_step_before':self.step,'action':self.last})
        return {'action':self.last,'viz':None}
    def reset(self,*,env_id=None):
        if self.session:
            # Official reset hook supplies no terminal sensor data: never invent it.
            self.events.write('episode_client_closed',{'confirmed_observed_steps':self.step,'last_action_terminal_feedback_unavailable':self.last is not None})
            try:
                if self.pending:self.reply(self.pending,False,[{'type':'inputText','text':'Evaluator closed the episode. No final sensor observation or success score was supplied to this client; terminal execution count is not asserted.'}])
            finally:self.session.__exit__(None,None,None);self.session=None
        super().reset(env_id=env_id)
