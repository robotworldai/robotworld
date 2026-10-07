"""Readable action contracts derived from runtime metadata, never simulator truth.

This describes the existing command channel; it does not change action processing.
Joint indices/scales/offsets come from the instantiated action manager, not regex order.
"""
import math
import re


def _number(value):
    return f'{float(value):.8g}'


def _vector(value, size):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, (int, float)):
        return [float(value)] * size
    if isinstance(value, list) and len(value) == size and all(isinstance(x, (int, float)) for x in value):
        return value
    raise ValueError('Cannot describe action scaling without resolved runtime values')


def _range(metadata, index):
    low, high = metadata['lower'][index], metadata['upper'][index]
    return f'[{_number(low) if low is not None else "-inf"}, {_number(high) if high is not None else "+inf"}]'


def _affine_rows(metadata):
    """Only recognized native joint terms; reject incomplete joint maps."""
    rows = []
    offset = 0
    for term in metadata.get('terms', []):
        size = term['dim']
        cfg = term['config']
        kind = str(cfg.get('class_type', '')).rsplit(':', 1)[-1]
        if kind not in ('JointPositionAction', 'JointVelocityAction'):
            return None
        names = term.get('resolved_joint_names')
        if not isinstance(names, list) or len(names) != size:
            raise ValueError('Resolved joint order is required for the action prompt')
        scales, offsets = _vector(term['scale'], size), _vector(term['offset'], size)
        for local, name in enumerate(names):
            clips = [bounds for pattern, bounds in (cfg.get('clip') or {}).items() if re.fullmatch(pattern, name)]
            if len(clips) > 1:
                raise ValueError('Ambiguous native processed-target clipping')
            rows.append((offset + local, name, 'rad' if kind == 'JointPositionAction' else 'rad/s',
                         scales[local], offsets[local], clips[0] if clips else None))
        offset += size
    if rows and offset != metadata['dim']:
        raise ValueError('Action prompt dimension differs from native action manager')
    return rows or None


def build_action_guide(task_id, metadata, dt, coding=True):
    dim = metadata['dim']
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('A measured positive native control timestep is required')
    lines = [f'ACTION CONTRACT — {task_id or "native task"}',
             f'Use exactly {dim} finite scalars in action[0] through action[{dim-1}], in the order below. All channels act simultaneously.',
             f'One control step = {_number(dt)} s. Holding steps=10 lasts {_number(10*dt)} simulated seconds; steps=50 lasts {_number(50*dt)} s, unless terminated earlier.',
             'apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.',
             'Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.',
             'Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.',
             'Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.']
    lines.append('coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.' if coding else
                 'coding_control is disabled in this run; only observe and apply_action are available.')

    if task_id == 'T03':
        lines += ['Robot-root coordinates, metres; quaternion order WXYZ. The seven pose entries form ONE absolute EEF pose, not seven joints or increments.',
                  '| Index | Meaning | Scale / zero / sign |', '|---|---|---|',
                  '| 0,1,2 | x,y,z | Absolute robot-root position; +0.01 changes a target coordinate by +1 cm. Repeating it does not add another centimetre. |',
                  '| 3,4,5,6 | qw,qx,qy,qz | Unit quaternion; identity is [1,0,0,0]. [0,0,0,0] is invalid. Quaternion magnitude is not rotation strength. |',
                  '| 7 | gripper | >0 opens to 0.015 m per finger; <=0 closes to 0. No proportional strength or speed is encoded here. |',
                  'For example a +10 degree root-Z rotation has quaternion [0.9961947,0,0,0.08715574]; this is an orientation example, not a command to execute.',
                  'Do not use an all-zero action to hold still. Preserve a valid desired pose and an explicit gripper choice. Native damped IK resolves arm joints.']
    elif task_id in ('T07', 'T08'):
        lines += ['All six inputs are in [-1,1]. Positive input increases the corresponding virtual target; these are NOT six raw joint angles.',
                  '| Index | Native control | Target formula | u=-1 / 0 / +1 |', '|---|---|---|---|']
        names = metadata['control_names']
        for i, name in enumerate(names):
            j = i % 3
            formula, values = [('angle=0.35*u rad', '-0.35 / 0 / +0.35 rad'),
                               ('length=clip(0.237+0.06*u,0.18,0.30) m', '0.18 / 0.237 / 0.297 m'),
                               ('wheel_velocity=24*u rad/s', '-24 / 0 / +24 rad/s')][j]
            lines.append(f'| {i} | {name} | {formula} | {values} |')
        lines += ['A change of +0.1 changes the unclipped angle target by +0.035 rad, leg length by +0.006 m, or wheel speed by +2.4 rad/s, respectively.',
                  'Virtual leg angle is measured from the native vertical reference, not world yaw: theta0=atan2(end_y,end_x)-pi/2 in the native leg plane. Increasing length requests extension along that virtual leg, not a guaranteed base-height change. Zero requests virtual angle 0, length 0.237 m and wheel speed 0. It does not supply automatic balance. Original VMC converts these targets into motor torques.']
    elif task_id == 'T04':
        lines += [f'action[0] = delta_vertical_velocity_mps, allowed {_range(metadata,0)} m/s per control step.',
                  'This is an INCREMENT of desired upward velocity, not absolute velocity or throttle. +0.001 adds +0.001 m/s to the interface target each applied step; -0.001 subtracts it.',
                  'Holding +0.001 for 10 applied steps adds +0.01 m/s in total. Original 15-control-step delay remains; the effect is delayed, not immediate.',
                  'Zero adds no new increment; it does not reset the accumulated target or stop motion. Configured max_velocity=0 disables velocity clipping, not motion.']
    elif task_id in ('T05', 'T05-single', 'T16', 'T17'):
        lines += ['| Index | Control | Input limits |', '|---|---|---|']
        for i, name in enumerate(metadata['names']):
            lines.append(f'| {i} | {name} | {_range(metadata,i)} |')
        lines += ['u=-1 requests zero steady-state rotor thrust; u=0 requests 50% of that rotor\'s maximum steady-state thrust; u=+1 requests 100%. u=-0.5/+0.5 requests 25%/75%. Zero is NOT hover or stop.',
                  'Original motor mapping: desired internal throttle s*=sqrt(clip((u+1)/2,0,1)); actual thrust=KF*s^2. Motor lag means a command is not achieved instantly.',
                  'The normalized rotor-throttle observation is v=2*s-1; its current thrust fraction is ((v+1)/2)^2, NOT (v+1)/2.',
                  'Each rotor thrust acts along its own local +Z. Tilting the aircraft changes its world force direction. Differential rotor inputs create moments; collective input alone cannot correct tilt. No hover/balance controller is supplied.']
        if task_id in ('T16', 'T17'):
            maximum = 8.54858e-6 * 838**2
            lines += [f'Hummingbird nominal KF=8.54858e-6*838^2={_number(maximum)} N per rotor. Steady u=0 gives {_number(maximum/2)} N per rotor; +0.1 changes steady thrust by {_number(maximum*.05)} N. These are nominal model constants, not hidden randomized episode parameters.',
                      'Original throttle update is s <- s + 0.43*(s*-s) each control tick in these fixed tasks. Rotor body angles [0,pi/2,pi,-pi/2]; spin directions [-1,+1,-1,+1]; native yaw reaction moment uses -direction.']
    else:
        rows = _affine_rows(metadata)
        if task_id == 'T02':
            rows = [(i, name, 'rad', metadata['scale'], metadata['default_joint_position'][i], None)
                    for i, name in enumerate(metadata['names'])]
            lines.append('T02 clips raw u to [-100,100] BEFORE scaling; valid tool inputs must already lie within these bounds. This wide input range is not a recommended motion.')
        elif task_id == 'T14':
            rows = [(i, name, 'rad', .5, metadata['default_joint_angles'][i], None)
                    for i, name in enumerate(metadata['names'])]
            lines.append('Original ANYmal PD: torque=clip(80*(target-q)-2*qdot,-80,+80) Nm. This torque limit is not an action-input bound.')
        if rows:
            lines += ['For each channel u=action[i], target=offset+scale*u, followed by any listed native processed-target clip.',
                      '| i | Exact runtime joint | Unit | Input limits | offset | scale | Target at u=0 | Target at u=+0.1 | Target at u=+1 | Processed-target clip |',
                      '|---|---|---|---|---|---|---|---|---|---|']
            for i, name, unit, scale, offset, clip in rows:
                def target(u):
                    value = offset + scale*u
                    return _number(min(max(value,clip[0]),clip[1]) if clip else value)
                limits = f'[{_number(clip[0])},{_number(clip[1])}] {unit}' if clip else 'none configured'
                lines.append(f'| {i} | {name} | {unit} | {_range(metadata,i)} | {_number(offset)} | {_number(scale)} | {target(0)} | {target(.1)} | {target(1)} | {limits} |')
            lines += ['For a negative change du, the unclipped target changes by scale*du. A zero position input requests the listed offset, not the current measured joint angle. A zero-offset velocity input requests zero speed, not instantaneous braking.',
                      'To request an absolute joint target q with nonzero scale, use u=(q-offset)/scale within allowed bounds, not u=q unless offset=0 and scale=1. Example offset=0.2,scale=0.5,u=0.1 requests q=0.25rad on every repeated step, not measured_q+0.05 each time. To hold the measured pose, first undo any observation normalization/default subtraction; do not copy relative/scaled observations directly into the action vector.',
                      'For position channels, radians convert to degrees by *57.2957795: 0.025 rad is about 1.43 degrees; 0.05 rad about 2.86 degrees.']
            if task_id == 'T15':
                lines += ['FLAMINGO EXCEPTION: left/right leg position targets are MOTOR-SPACE angles. Gear ratio=-1.5: q_motor=-1.5*q_physical. Thus motor target +0.1 rad corresponds to static physical target about -0.0666667 rad, subject to the original dynamics and limits. Other position channels are physical joint radians; wheel channels are rad/s.']
        else:
            lines.append('Consult the task-specific native action metadata for this unrecognized action type; do not infer a normalized range or a physical gain.')
    return '\n'.join(lines) + '\n'
