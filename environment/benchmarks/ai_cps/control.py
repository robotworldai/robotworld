"""Native joint increments and explicitly authored contact recovery bookkeeping."""
from environment.benchmarks.action_contracts import action_contract, describe_tools
import math

def validate_action(value):
    if not isinstance(value,list) or len(value)!=7 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1 for x in value):
        raise ValueError('arm_action must contain seven finite numbers in [-1,1], panda_joint1..7')
    return value+[0.,0.]

def specs(case, coding_control_enabled=True):
    def tool(name,desc,props):
        return {'type':'function','name':name,'description':desc,'inputSchema':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}
    note={'type':'string'}
    result=[tool('observe','Read current state and camera history without stepping physics.',{'note':note}),
      tool('move_joints','Apply seven native normalized joint increments simultaneously for 1..50 control steps. Each tick adds action*0.125 rad to the previous target before joint-limit clamp; fingers remain task-controlled. Noise is added by upstream. Zero holds the last target before noise, not measured pose. No EEF planner.',{'note':note,'steps':{'type':'integer','minimum':1,'maximum':50},'arm_action':{'type':'array','items':{'type':'number','minimum':-1,'maximum':1},'minItems':7,'maxItems':7}}),
      tool('coding_control','Run def control(obs,memory) once per ~1/60s control tick with fresh numeric state. Return {"arm_action":[seven normalized increments]}, or {"done":true} before the next action. Supports bounded Python arithmetic, math, lists/dicts, if/loops; no files, imports beyond math, simulator access or resets. memory persists within this call. Invalid code/action stops without another step.',{'note':note,'max_steps':{'type':'integer','minimum':1,'maximum':300},'code':{'type':'string','minLength':1,'maxLength':16000}})]
    if case=='34':result.append(tool('cancel_action','Acknowledge the real contact interruption, discard the old segment and latch degraded control for the rest of the episode. No physics step or teleport. Subsequent noisy joint TARGET changes are limited to 0.00625 rad/tick; actual velocity/inertia may differ. Does not declare recovery success.',{'note':note}))
    result = describe_tools(result, action_contract('ai_cps'))
    return result if coding_control_enabled else [item for item in result if item['name']!='coding_control']

class ContactRecovery:
    """ID34 v1: 10N peg/table pair force at 2 consecutive physics substeps.
    Recovery requires explicit cancellation, 10 degraded control steps, and
    10 consecutive control steps below 5N (peak across their physics substeps).
    """
    threshold=10.;release=5.;required=10
    def __init__(self):
        self.high=0;self.trigger_step=None;self.cancel_step=None;self.low=0;self.degraded_steps=0;self.pending=False;self.recovered=False
    def contact(self,force,step):
        if not math.isfinite(force) or force<0:raise ValueError('Invalid contact measurement')
        self.high=self.high+1 if force>=self.threshold else 0
        if self.high>=2 and self.trigger_step is None:self.trigger_step=step;self.pending=True
    def cancel(self,step):
        if self.trigger_step is None:raise ValueError('No measured anomaly to cancel')
        if self.cancel_step is None:self.cancel_step=step
        self.pending=False
    def tick(self,peak):
        if self.cancel_step is not None:
            self.degraded_steps+=1;self.low=self.low+1 if peak<self.release else 0
            self.recovered=self.low>=self.required and self.degraded_steps>=self.required
    def observation(self):
        return {'trigger_step':self.trigger_step,'cancel_step':self.cancel_step,'contact_pending':self.pending,'degraded_control':self.cancel_step is not None}
    def result(self):
        return {'status':'not_covered' if self.trigger_step is None else ('recovered' if self.recovered else 'failed'),
                'trigger_step':self.trigger_step,'cancel_step':self.cancel_step,'degraded_steps':self.degraded_steps,
                'success':None if self.trigger_step is None else self.recovered,'contact_pending':self.pending}
