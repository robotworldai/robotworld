"""R1Pro proprioception-to-controller mapping. No simulator state queries."""
from environment.benchmarks.action_contracts import action_contract, describe_tools
import math
import numpy as np


def axis_angle(quat):
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError('Expected finite nonzero XYZW quaternion')
    q = q / norm
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    return np.zeros(3) if s < 1e-8 else q[:3] * (2 * math.atan2(s, q[3]) / s)


def split_proprio(vector, fields):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if not np.isfinite(vector).all() or len(vector) != sum(fields.values()):
        raise ValueError('Proprioception shape or values differ from the robot profile')
    out, index = {}, 0
    for name, size in fields.items():
        out[name] = vector[index:index + size]
        index += size
    return out


def action_for(targets, state, profile, grippers):
    allowed = {'base_vx', 'base_vy', 'base_wz', 'trunk_qpos'}
    for side in ('left', 'right'):
        allowed.update(f'{side}_{key}' for key in ('x', 'y', 'z', 'quat_xyzw', 'gripper'))
    if not isinstance(targets, dict) or not targets or set(targets) - allowed:
        raise ValueError('Unknown or empty robot targets')
    action = np.zeros(profile['action_dim'], dtype=np.float32)
    next_grippers = dict(grippers)
    indices = profile['controller_indices']
    for side in ('left', 'right'):
        pos = state[f'eef_{side}_pos'].copy()
        for i, key in enumerate(('x', 'y', 'z')):
            if f'{side}_{key}' in targets:
                pos[i] = float(targets[f'{side}_{key}'])
        if not np.isfinite(pos).all() or np.any(np.abs(pos) > 2):
            raise ValueError('EEF position must be finite and within +/-2m of robot base')
        ori = axis_angle(targets.get(f'{side}_quat_xyzw', state[f'eef_{side}_quat']))
        action[indices[f'arm_{side}']] = np.r_[pos, ori]
        if f'{side}_gripper' in targets:
            opening = float(targets[f'{side}_gripper'])
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError('Gripper opening must be in [0,1]')
            next_grippers[side] = opening
        action[indices[f'gripper_{side}']] = 2 * next_grippers[side] - 1
    base = np.array([float(targets.get(k, 0)) for k in ('base_vx', 'base_vy', 'base_wz')])
    if not np.isfinite(base).all() or np.any(np.abs(base) > [.3, .3, .5]):
        raise ValueError('Base command exceeds 0.3m/s or 0.5rad/s')
    action[indices['base']] = base
    if indices.get('trunk'):
        trunk = np.asarray(targets.get('trunk_qpos', state['trunk_qpos']), dtype=float)
        bounds = profile.get('trunk_limits')
        if trunk.shape != state['trunk_qpos'].shape or not np.isfinite(trunk).all():
            raise ValueError('Invalid trunk joint target')
        if 'trunk_qpos' in targets and (bounds is None or np.any(trunk < np.array(bounds[0])) or np.any(trunk > np.array(bounds[1]))):
            raise ValueError('Trunk target outside measured joint limits')
        action[indices['trunk']] = trunk
    elif 'trunk_qpos' in targets:
        raise ValueError('This robot has no independent trunk controller')
    return action, next_grippers


def tool_specs(profile=None):
    def tool(name, description, props):
        return {'type':'function','name':name,'description':description,
                'inputSchema':{'type':'object','additionalProperties':False,'required':['note','targets','steps'],
                    'properties':{'note':{'type':'string'},'targets':{'type':'object','additionalProperties':False,'properties':props,'minProperties':1},
                                  'steps':{'type':'integer','minimum':1,'maximum':30}}}}
    arms={}
    for side in ('left','right'):
        for axis in ('x','y','z'):
            arms[f'{side}_{axis}']={'type':'number','minimum':-2,'maximum':2,'description':f'Absolute {axis} position in robot articulation-root frame, metres.'}
        arms[f'{side}_quat_xyzw']={'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4,
                                 'description':'Absolute end-effector orientation relative to robot articulation root, quaternion XYZW.'}
    result=[tool('move_arms','Move one or both R1Pro end effectors with absolute-pose IK. Base stationary; omitted arm holds its observed pose; gripper commands persist. No collision-free planning. steps is 1..30 at 30Hz.',arms),
            tool('move_base','Drive the holonomic base in local body axes: +x forward, +y left, positive yaw counterclockwise. Arms hold robot-relative EEF targets and grippers persist. Velocities apply only during this bounded segment; no teleport or global localization.',
                 {'base_vx':{'type':'number','minimum':-.3,'maximum':.3,'description':'Forward velocity m/s; omitted=0.'},
                  'base_vy':{'type':'number','minimum':-.3,'maximum':.3,'description':'Leftward velocity m/s; omitted=0.'},
                  'base_wz':{'type':'number','minimum':-.5,'maximum':.5,'description':'Yaw velocity rad/s; omitted=0.'}}),
            tool('set_grippers','Set left/right gripper opening: 0 closed, 1 open. EEFs hold measured robot-relative poses and base stops. Opening targets persist until changed; verify actual object retention.',
                 {f'{side}_gripper':{'type':'number','minimum':0,'maximum':1} for side in ('left','right')})]
    if profile and profile['controller_indices'].get('trunk'):
        n=len(profile['controller_indices']['trunk'])
        result.append(tool('move_torso','Command absolute trunk joint positions in the proprioception order (radians). Joint names: '+str(profile.get('trunk_joint_names', 'see proprioception order'))+'. Limits: '+str(profile.get('trunk_limits'))+'. Arms use IK to hold their robot-relative EEF poses; use small increments and inspect tracking.',
                           {'trunk_qpos':{'type':'array','items':{'type':'number'},'minItems':n,'maxItems':n,'description':'Full absolute joint vector, radians, in this order: '+str(profile.get('trunk_joint_names', 'proprioception order'))}}))
    combined = {}
    for spec in result:
        combined.update(spec['inputSchema']['properties']['targets']['properties'])
    result.append(tool('move_robot',
        'Coordinate both EEF absolute IK poses, local base velocities, gripper openings and (when available) absolute trunk joint positions in ONE native action per control step. '
        'All specified components execute simultaneously for steps=1..30 at 30Hz; steps counts once, not per component. '
        'EEF positions are metres relative to the robot articulation root, orientations are XYZW quaternions; these are NOT RoboCasa normalized increments. '
        'Base +x forward/+y left in m/s, yaw rad/s; limits 0.3/0.3/0.5. Omitted base velocities are zero. '
        'Omitted EEF dimensions and trunk joints hold their measured targets; gripper openings persist (0 closed, 1 open). '
        'Gripper changes start alongside arm motion, NOT after arrival; split approach and closure when needed. '
        'Arm targets move with the robot root during base motion, not fixed world poses. '
        'Trunk targets are a complete vector in proprioception order, radians, within runtime limits '+str(profile.get('trunk_limits') if profile else None)+'. '
        'No collision-free planning or guaranteed tracking; use short segments and inspect feedback. Use this tool instead of parallel motion-tool calls.', combined))
    return describe_tools(result, action_contract('behavior_1k'))


def validate_tool(name, arguments, profile):
    specs={x['name']:x for x in tool_specs(profile)}
    if name not in specs:raise ValueError('Unsupported benchmark tool')
    if not isinstance(arguments,dict) or set(arguments)!={'note','targets','steps'} or not isinstance(arguments['note'],str):
        raise ValueError('Expected note, targets, steps')
    targets=arguments['targets'];allowed=specs[name]['inputSchema']['properties']['targets']['properties']
    if not isinstance(targets,dict) or not targets or set(targets)-set(allowed):raise ValueError('Targets do not belong to this tool')
    if type(arguments['steps']) is not int or not 1<=arguments['steps']<=30:raise ValueError('steps must be 1..30')
    for key, value in targets.items():
        rule = allowed[key]
        values = value if rule['type'] == 'array' else [value]
        if rule['type'] == 'array' and (not isinstance(value, list) or not rule['minItems'] <= len(value) <= rule['maxItems']):
            raise ValueError('Invalid target array: '+key)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Expected finite numeric target: '+key)
        if rule['type'] == 'number' and not rule.get('minimum', -math.inf) <= value <= rule.get('maximum', math.inf):
            raise ValueError('Target outside bounds: '+key)
        if key.endswith('_quat_xyzw'):
            axis_angle(value)
        if key == 'trunk_qpos':
            bounds = profile.get('trunk_limits')
            if bounds is None or np.any(np.asarray(value) < bounds[0]) or np.any(np.asarray(value) > bounds[1]):
                raise ValueError('Trunk target outside measured joint limits')
    return arguments
