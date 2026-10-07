"""Native upstream DROID action profiles; no simulator state access."""
from environment.benchmarks.action_contracts import action_contract, describe_tools
import numpy as np

MODES=('joint_position','absolute_ik','relative_ik')
STATE_KEYS=('arm_joint_pos','gripper_pos','ee_pos','ee_quat','eef_pos','eef_quat')

def specs(mode):
    if mode not in MODES:raise ValueError('Unknown controller profile')
    def vec(n,description):return {'type':'array','items':{'type':'number'},'minItems':n,'maxItems':n,'description':description}
    arm=({'joint_positions':vec(7,'Absolute panda_joint1..7 positions, radians.')} if mode=='joint_position' else
         {'position':vec(3,'Absolute base_link position relative to robot root, metres.'),'quaternion_wxyz':vec(4,'Absolute base_link orientation in robot-root coordinates; WXYZ, not XYZW.')} if mode=='absolute_ik' else
         {'delta_pose':vec(6,'Native relative IK input [dx,dy,dz,rx,ry,rz], robot-root axes, rotation vector. Upstream scale=0.5; repeats accumulate each control step.')})
    gripper={'gripper_close':{'type':'integer','enum':[0,1],'description':'0 open, 1 closed; persists.'}}
    name='move_joints' if mode=='joint_position' else 'move_eef'
    groups=[(name,arm),('set_gripper',gripper),('move_robot',dict(arm,**gripper))]
    result=[]
    for name,fields in groups:
        result.append({'type':'function','name':name,'description':f'Native DROID {mode} control. No base, torso or head controller exists in this profile. '
            'steps=1..30 at 15 Hz. move_robot applies arm and gripper simultaneously, including closing immediately; split approach and grasp when necessary. '
            'Omitted absolute pose/joints hold measured values; omitted relative delta is zero; gripper target persists. IK does not plan collision-free paths.',
            'inputSchema':{'type':'object','additionalProperties':False,'required':['note','steps','targets'],'properties':{
                'note':{'type':'string'},'steps':{'type':'integer','minimum':1,'maximum':30},
                'targets':{'type':'object','additionalProperties':False,'minProperties':1,'properties':fields}}}})
    return describe_tools(result, action_contract('robolab', mode))

def action_for(mode,name,args,state,gripper):
    tools={s['name']:s for s in specs(mode)}
    if name not in tools:raise ValueError('Unsupported tool for this controller')
    if not isinstance(args,dict) or set(args)!={'note','steps','targets'} or not isinstance(args['note'],str):raise ValueError('Expected note,steps,targets')
    if type(args['steps']) is not int or not 1<=args['steps']<=30:raise ValueError('steps must be 1..30')
    targets=args['targets'];fields=tools[name]['inputSchema']['properties']['targets']['properties']
    if not isinstance(targets,dict) or not targets or set(targets)-set(fields):raise ValueError('Invalid target fields')
    for k,v in targets.items():
        if k=='gripper_close':
            if type(v) is not int or v not in (0,1):raise ValueError('gripper_close must be 0 or 1')
        elif not isinstance(v,list) or len(v)!=fields[k]['minItems'] or any(type(x) not in (int,float) or not np.isfinite(x) for x in v):raise ValueError('Invalid vector: '+k)
    grip=targets.get('gripper_close',gripper)
    if mode=='joint_position':arm=np.asarray(targets.get('joint_positions',state['arm_joint_pos']),dtype=float)
    elif mode=='absolute_ik':
        q=np.asarray(targets.get('quaternion_wxyz',state['ee_quat']),dtype=float)
        if np.linalg.norm(q)<1e-8:raise ValueError('Zero quaternion')
        arm=np.r_[targets.get('position',state['ee_pos']),q/np.linalg.norm(q)]
    else:arm=np.asarray(targets.get('delta_pose',[0]*6),dtype=float)
    action=np.r_[arm,grip].astype(np.float32)
    if not np.isfinite(action).all():raise ValueError('Nonfinite native action')
    return action,args['steps'],grip
