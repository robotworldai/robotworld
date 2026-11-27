"""Opt-in feedback programs over the same public observation/action interface."""
import copy
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram

SPEC = {'type':'function','name':'coding_control','description':
    'Run restricted Python control(obs, memory). obs is the public observe payload without images, '
    'plus program_step. Return {"tool": native_tool_name, "arguments": {...}} or {"done": true}. '
    'Use exactly the native units, frames and bounds from describe. Each return must prepare one '
    'native action with steps=1 where applicable; planner macros may consume several physical steps. '
    'Fresh public feedback is supplied after each action. No files, network, simulator handles, '
    'imports except math, resets or scoring access. memory persists within this call. '
    'done ends this segment, not the episode. max_steps limits callbacks, not guaranteed physical ticks.',
    'inputSchema':{'type':'object','additionalProperties':False,'required':['code','max_steps','note'],
    'properties':{'code':{'type':'string','minLength':1,'maxLength':16000},
                  'max_steps':{'type':'integer','minimum':1,'maximum':600},'note':{'type':'string'}}}}

class Program:
    def __init__(self, arguments):
        if not isinstance(arguments,dict) or set(arguments)!={'code','max_steps','note'}:
            raise ValueError('coding_control requires code, max_steps, note')
        if type(arguments['max_steps']) is not int or not 1<=arguments['max_steps']<=600:
            raise ValueError('max_steps must be 1..600')
        if not isinstance(arguments['note'],str) or not isinstance(arguments['code'],str) or not 1<=len(arguments['code'])<=16000:
            raise ValueError('Invalid code or note')
        self.arguments=arguments
        self.worker=None
        self.index=0
    def close(self):
        if self.worker is not None:
            self.worker.close(); self.worker=None

class FeedbackAdapter:
    def __init__(self, adapter, journal):
        self.adapter=adapter; self.journal=journal
    def describe(self):
        spec=copy.deepcopy(self.adapter.describe())
        # The legacy adapter describes its direct-only profile; the outer service
        # adds a distinct, explicitly documented command-return feedback interface.
        spec['coding_control_enabled']=True
        spec['code_control']=True
        spec['feedback_control_profile']='harbor-command-return-v1'
        spec['tools']=[t for t in spec['tools'] if t['name']!='coding_control']+[copy.deepcopy(SPEC)]
        spec['feedback_control_note']='The outer Harbor service enables coding_control as documented here, even when the native adapter describes its direct-only profile. Legacy targets/action-return code must be rewritten to return a native tool/arguments command.'
        return spec
    def observe(self): return self.adapter.observe()
    def prepare(self, commands):
        result=[]
        for cmd in commands:
            if isinstance(cmd,dict) and cmd.get('tool')=='coding_control':
                if set(cmd)!={'tool','arguments'}: raise ValueError('Expected tool and arguments')
                p=Program(cmd['arguments']); result.append((p,p.arguments['max_steps']))
            else: result.extend(self.adapter.prepare([cmd]))
        return result
    def step(self, action):
        if not isinstance(action,Program): return self.adapter.step(action)
        try:
            if action.worker is None: action.worker=ControllerProgram(action.arguments['code'])
            obs=copy.deepcopy(self.adapter.observe()); obs.pop('images',None)
            obs['program_step']=action.index
            reply=action.worker.request({'obs':obs})
            command=reply['action']
            self.journal('feedback_callback',{'program_step':action.index,'observation':obs,'reply':reply})
            if command=={'done':True}:
                return {'segment_done':True,'executed_steps':0}
            if not isinstance(command,dict) or set(command)!={'tool','arguments'} or command['tool']=='coding_control':
                raise ValueError('Return a native tool/arguments command, or done')
            prepared=self.adapter.prepare([command])
            if len(prepared)!=1 or prepared[0][1]!=1:
                raise ValueError('Feedback callback must return exactly one action with one repetition')
        except Exception as exc:
            # No physics was stepped on this callback; keep the episode usable.
            return {'segment_done':True,'executed_steps':0,'details':{'program_error':str(exc)}}
        action.index+=1
        return self.adapter.step(prepared[0][0])
    def close(self): return self.adapter.close()
