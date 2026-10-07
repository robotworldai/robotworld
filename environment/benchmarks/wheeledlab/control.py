"""Native two-dimensional driving controls; no hidden planner or vehicle controller."""
import math


def action_guide(dt, reverse=False):
    negative = 'u=-1/-0.1 requests -3/-0.3 m/s (reverse).' if reverse else 'Any negative u is clamped to zero target speed; this task cannot reverse.'
    return f'''ACTION CONTRACT — WheeledLab
action[0]=normalized speed; action[1]=normalized central steering; both finite in[-1,1].
Speed target=3*u m/s: u=0/0.1/0.5/1 requests0/0.3/1.5/3m/s. {negative}
Steering parameter=0.488*u radians: u=0/0.1/-0.1/1 gives0/0.0488/-0.0488/0.488rad before the original RC-car tan conversion: both front steering-joint targets=tan(0.488*u) rad, e.g. u=0.1 gives about0.048839rad and u=1 about0.53082rad. Zero requests straight steering. In the published upright MuSHR/F1Tenth mounting, positive steering rotates about vehicle +Z (left turn when driving forward); negative steers right. Reversing reverses the nominal yaw response for the same steering input; tire slip can change the achieved yaw. Vehicle +X is forward, +Y left, +Z up, not camera image axes.
Example action=[0.1,0] requests0.3m/s wheel-ground speed and straight steering; at dt=0.02 and steps=10 the ideal no-slip travel is6cm, not a guaranteed displacement. With wheel radius0.05m, the rear-wheel target is6rad/s in MuSHR; F1Tenth distributes wheel speeds through its native geometry. Both entries are targets, not acceleration/torque percentages or guaranteed motion. A zero speed target does not teleport velocity to zero. Tire slip and native drives remain active.
One control step={float(dt):.8g}s; steps=10 holds ONE command for{10*float(dt):.8g}s, not ten increasing commands. coding_control, if enabled, may update both entries every tick.
These numerical examples explain units and gain, not safe speeds or a supplied driving controller.
'''

def validate_action(value):
    if not isinstance(value,list) or len(value)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1 for x in value):raise ValueError('action must contain two finite numbers in [-1,1]: speed, steering')
    return value

def specs(coding=True,onboard=False,reverse=False):
    def tool(name,description,properties):return {'type':'function','name':name,'description':description,'inputSchema':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}
    note={'type':'string'}
    tools=[tool('observe','Read current native policy observation without advancing physics.',{'note':note}),
           tool('drive','Apply the native normalized [speed,steering] action for 1..50 control steps. Positive speed scales to 3m/s, negative speed is clamped to zero (no reverse). Steering scales by0.488 before upstream tan steering mapping. This commands wheel speed, not guaranteed car displacement or braking.',{'note':note,'action':{'type':'array','items':{'type':'number','minimum':-1,'maximum':1},'minItems':2,'maxItems':2},'steps':{'type':'integer','minimum':1,'maximum':50}})]
    if coding:tools.append(tool('coding_control','Run control(obs,memory) with fresh native policy feedback each tick. Return {"action":[speed,steering]} or {"done":true}. Bounded Python arithmetic/math only, no files, imports except math, simulator handles or resets; memory lasts within this call.',{'note':note,'code':{'type':'string','minLength':1,'maxLength':16000},'max_steps':{'type':'integer','minimum':1,'maximum':250}}))
    if onboard:
        tools[0]['description']='Read front onboard RGB plus wheel/steering encoders and IMU, without advancing physics. No map, world pose, waypoint, progress or gate-clearance oracle.'
        if coding:
            tools[-1]['description']='Run control(obs,memory) every control tick using only onboard48x27RGB pixels, wheel/steering encoders and IMU. Infer road/hazards yourself; no world pose, navigation hints, gate state or private evaluator. Return {"action":[speed,steering]} or {"done":true}. Bounded Python arithmetic/math only, no files, simulator handles or resets; memory lasts within this call.'
    if reverse:
        tools[1]['description']=tools[1]['description'].replace('negative speed is clamped to zero (no reverse)','negative speed commands reverse wheel motion (signed speed scale3m/s)')
        if coding:tools[-1]['description']+=' This task supports signed reverse speed and rear_camera pixels; no parking planner is supplied.'
    return tools
