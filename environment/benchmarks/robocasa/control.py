"""RoboCasa365 native normalized action contract; no simulator state access."""
from environment.benchmarks.action_contracts import action_contract, describe_tools
import numpy as np

CAMERAS=('video.robot0_agentview_left','video.robot0_agentview_right','video.robot0_eye_in_hand')
STATE_KEYS=('state.end_effector_position_relative','state.end_effector_rotation_relative',
            'state.gripper_qpos','state.base_position','state.base_rotation')
FIELDS={'eef_delta':(0,6),'base_motion':(7,10)}
DESCRIPTIONS={
 'move_eef':'Move the Panda hand with normalized base-frame OSC pose increments [dx,dy,dz,rx,ry,rz]. Values [-1,1] map to up to 0.05m translation and 0.5rad rotation per control step. These are NOT absolute poses. Repeating steps repeats the increment, not a fixed target. Omitted base/torso motion is zero.',
 'move_base':'Move the mobile base using 3 normalized controller inputs [-1,1]; these are NOT physical m/s. Base x/y translation and yaw; enables base-following mode. Gripper target persists.',
 'move_torso':'Apply a normalized torso joint-position increment [-1,1], not an absolute height. Uses base-following mode; arm delta is zero.',
 'set_gripper':'Set binary gripper state: close=1, open=0. Target persists; no arm/base/torso motion requested.',
 'move_robot':'Coordinate native RoboCasa365 hand, base, torso and gripper commands. EEF is normalized delta OSC, base and torso are normalized inputs. control_mode=0 updates arm goal from achieved pose; 1 from desired pose for base following.'}

def tool_specs():
    props={
      'eef_delta':{'type':'array','items':{'type':'number','minimum':-1,'maximum':1},'minItems':6,'maxItems':6},
      'base_motion':{'type':'array','items':{'type':'number','minimum':-1,'maximum':1},'minItems':3,'maxItems':3},
      'torso':{'type':'number','minimum':-1,'maximum':1},
      'gripper_close':{'type':'integer','enum':[0,1]},
      'control_mode':{'type':'integer','enum':[0,1]}}
    allowed={'move_eef':['eef_delta'],'move_base':['base_motion'],'move_torso':['torso'],
             'set_gripper':['gripper_close'],'move_robot':list(props)}
    result = [{'type':'function','name':name,'description':DESCRIPTIONS[name],
       'inputSchema':{'type':'object','additionalProperties':False,'properties':{
         'note':{'type':'string'},'steps':{'type':'integer','minimum':1,'maximum':30},
         'targets':{'type':'object','additionalProperties':False,'properties':{k:props[k] for k in keys},
                    'required':keys if name!='move_robot' else []}},'required':['note','steps','targets']}}
       for name,keys in allowed.items()]
    return describe_tools(result, action_contract('robocasa'))

def action_for(name,args,gripper):
    spec=next((s for s in tool_specs() if s['name']==name),None)
    if spec is None:raise ValueError('Unknown robot tool')
    if not isinstance(args,dict) or set(args)!={'note','steps','targets'} or not isinstance(args['note'],str):raise ValueError('Expected note, steps, targets')
    if type(args['steps']) is not int or not 1<=args['steps']<=30:raise ValueError('steps must be 1..30')
    t=args['targets'];ts=spec['inputSchema']['properties']['targets']
    if not isinstance(t,dict) or not t or set(t)-set(ts['properties']) or set(ts['required'])-set(t):raise ValueError('Invalid targets for tool')
    a=np.zeros(12,dtype=np.float32);a[6]=gripper;a[11]=int(name in ('move_base','move_torso'))
    for k,v in t.items():
        if k in FIELDS:
            lo,hi=FIELDS[k];vals=np.asarray(v,dtype=float)
            if vals.shape!=(hi-lo,) or not np.isfinite(vals).all() or (np.abs(vals)>1).any():raise ValueError('Invalid normalized vector')
            a[lo:hi]=vals
        elif k=='torso':
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or abs(v)>1:raise ValueError('Invalid torso input')
            a[10]=v
        else:
            if type(v) is not int or v not in (0,1):raise ValueError('gripper_close / control_mode must be 0 or 1')
            a[6 if k=='gripper_close' else 11]=v
    return a,args['steps'],float(a[6])

def state_of(obs):return {k:np.asarray(obs[k]).tolist() for k in STATE_KEYS}
