"""Codex tool loop over public RoboCasa Gym observations, via existing source-verified relay."""
from environment.benchmarks.action_contracts import action_contract
import base64,io,json,time,os
from collections import deque
from pathlib import Path
from PIL import Image
from environment.runtime.codex_session import CodexSession
from environment.runtime.events import EventLog
from environment.runtime.image_history import publish_image_window
from environment.benchmarks.operating_brief import operating_brief
from .control import CAMERAS,state_of,tool_specs,action_for
from environment.runtime.nonaction_budget import attach, NonActionBudgetExceeded
from environment.runtime.continuation import remember_packet, continue_packet

BASE='''You control the standard RoboCasa365 PandaOmron robot using benchmark-specific robot tools. Complete the exact task instruction from the environment. Use only supplied RGB and proprioception. There is no depth sensor enabled in this standard evaluation. EEF pose observations are relative to the robot base; quaternions are XYZW. Actions are the native normalized 12D controller inputs, NOT the absolute IK targets of other benchmarks. move_eef supplies base-frame OSC increments: normalized translation scales to up to 0.05m and rotation to 0.5rad per step. Repeating a delta repeats motion; use small magnitudes and short segments near objects. Base and torso commands are normalized, not metres or m/s. Gripper close=1/open=0 persists. Unspecified motion inputs are zero. control_mode=0 uses achieved arm pose; 1 follows desired pose when base moves. Inspect collisions, grasp retention after lifting, and placement after opening. Navigate from camera observations and remember routes; do not assume unseen space is clear. Current plus up to four historical observations sampled every two feedback rounds are labeled by control step. The simulator waits while you reason or execute code. Persist after failures by changing approach, pose, grasp or subgoal; do not give up. Task success is decided only by the evaluator. Use workspace notes and calculations when helpful. No hidden scene state, external task solutions, simulator access, resets or score queries are allowed.'''
DEV='Use shell/code only for local computation and memory in /workspace. Read-only /observations contains RGB PNG and measured robot state. All motion and new observations must use robot tools. Do not access simulator/evaluator files or bypass the interface. A final answer does not terminate the episode.'

class Policy:
    def __init__(self,manifest,out,timeout=7200,max_actions=5000):
        self.out=Path(out);self.session=CodexSession(manifest,self.out)
        self.tools=EventLog(self.out/'events/tools.jsonl');self.deadline=time.monotonic()+timeout
        self.calls=0;self.max_actions=max_actions;self.gripper=0.;self.seen=set();self.history=deque(maxlen=9)
        self.pending=None;self.obs_index=0;self.step=0;self.stop=None
        self.interaction_budget=attach(self.session,self.out,lambda:self.step)
    def ingest(self,obs,step):
        self.obs=obs;self.step=step
        self.history.append((step,{k:obs[k].copy() for k in CAMERAS}))
    @remember_packet(lambda self:self.step)
    def content(self):
        directory=self.out/'frames'/str(self.obs_index);directory.mkdir(parents=True,exist_ok=True)
        text=f"Task: {self.obs['annotation.human.task_description']}\nControl step: {self.step}\nMeasured state: {json.dumps(state_of(self.obs))}"
        (directory/'observation.txt').write_text(text)
        selected=list(self.history)[::-2][:5][::-1];parts=[{'type':'inputText','text':text}]
        rounds=[]
        for step,images in selected:
            role='CURRENT' if step==self.step else 'HISTORY'
            rounds.append(dict(role=role,observation=step,env_step=step))
            parts.append({'type':'inputText','text':f'{role} observation {step}; env step {step}'})
            for k,img in images.items():
                b=io.BytesIO();Image.fromarray(img).save(b,format='JPEG',quality=90)
                parts.extend([{'type':'inputText','text':k},{'type':'inputImage','imageUrl':'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()}])
        for k in CAMERAS:Image.fromarray(self.obs[k]).save(directory/(k.removeprefix('video.')+'.png'))
        (directory/'history.json').write_text(json.dumps([x[0] for x in selected]))
        export=os.environ.get('WORLD_AGENT_OBSERVATIONS')
        if export:
            import shutil
            shutil.copytree(directory,Path(export)/str(self.obs_index),dirs_exist_ok=True)
            publish_image_window(Path(export).parent,parts,rounds)
            # Preserve RoboCasa's existing control-step sampling semantics.
            window=Path(export).parent/'image-window.json'
            metadata=json.loads(window.read_text())
            metadata['policy']='robocasa-current-plus-4-control-step-history-stride-2-v1'
            metadata['history_unit']='control_steps'
            temporary=window.with_suffix('.tmp');temporary.write_text(json.dumps(metadata));temporary.replace(window)
        self.obs_index+=1
        return parts
    def begin_turn(self,content):
        converted=[{'type':'image','url':x['imageUrl']} if x['type']=='inputImage' else {'type':'text','text':x['text'],'text_elements':[]} for x in content]
        self.turn=self.session.rpc('turn/start',{'threadId':self.thread,'input':converted})['turn']['id']
    def __enter__(self):
        self.session.__enter__()
        return self
    def start(self):
        params={'cwd':'/workspace','approvalPolicy':'never','sandbox':'danger-full-access','ephemeral':True,
          'baseInstructions':BASE+'\n'+action_contract('robocasa')+'\n'+operating_brief('robocasa'),'developerInstructions':DEV,'dynamicTools':tool_specs(),
          'config':{'features.shell_tool':True,'features.code_mode':True,'features.multi_agent':False,
                    'features.apps':False,'features.plugins':False,'web_search':'disabled'}}
        (self.out/'prompt.json').write_text(json.dumps(params,indent=2))
        self.thread=self.session.rpc('thread/start',params)['thread']['id'];self.begin_turn(self.content())
    def reply(self,message,success,content):
        response={'success':success,'contentItems':content}
        self.session.send({'id':message['id'],'result':response})
        self.tools.write('tool_completed',{'call_id':message['params']['callId'],'response':response})
    def decide(self):
        while self.calls<self.max_actions and time.monotonic()<self.deadline:
            try:m=self.session.receive(self.deadline-time.monotonic())
            except NonActionBudgetExceeded as end:
                self.stop=end.snapshot['stop_reason'];return None
            method=m.get('method')
            if method=='item/tool/call':
                c=m['params'];self.tools.write('tool_requested',c)
                if c.get('threadId')!=self.thread or c.get('turnId')!=self.turn or c['callId'] in self.seen:raise RuntimeError('Cross-session or duplicate tool call')
                self.seen.add(c['callId'])
                try:action,n,grip=action_for(c['tool'],c['arguments'],self.gripper)
                except (ValueError,TypeError,KeyError) as e:
                    self.reply(m,False,[{'type':'inputText','text':str(e)}]);continue
                self.gripper=grip;self.calls+=1;self.pending=m
                return action,n
            elif method=='turn/completed':
                p=m['params']
                if p.get('threadId')!=self.thread or p['turn']['id']!=self.turn:raise RuntimeError('Wrong turn')
                if p['turn']['status']!='completed':raise RuntimeError('Agent turn failed or interrupted')
                self.tools.write('agent_turn_continued',{'step':self.step})
                self.begin_turn(continue_packet(self,self.step,self.content))
            elif method and 'id' in m:self.session.send({'id':m['id'],'error':{'code':-32601,'message':'Unsupported client RPC'}})
        self.stop='action_budget' if self.calls>=self.max_actions else 'wall_timeout'
        return None
    def complete(self,n,terminal=False):
        self.reply(self.pending,True,[{'type':'inputText','text':f'Executed {n} control steps. Episode ended: {terminal}. Execution is not proof of goal achievement.'}]+self.content())
        self.pending=None
    def __exit__(self,*args):
        self.session.__exit__(*args)
