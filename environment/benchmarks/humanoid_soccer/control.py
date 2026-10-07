"""Expose real policy/PD actions; never alter simulator state or task goals."""
import numpy as np


def action_guide(names, limits, mode, dt):
    lines = [f'ACTION CONTRACT — HumanoidSoccer {mode}',
             f'All named joints act simultaneously. One control tick={float(dt):.8g}s; steps=10 holds targets or supervision offsets for{10*float(dt):.8g}s.',
             'Positive values increase the named joint coordinate along its authored axis; mirrored joints need not have mirrored signs. Values are radians, not motor torque or percentage strength.',
             '0.01rad=0.573deg, 0.1rad=5.73deg, 0.25rad=14.32deg, 1rad=57.30deg. These are conversions, not recommended motions.',
             'DIRECT uses absolute joint_positions: 0 requests the zero joint angle; 0.1 requests +0.1rad, not current+0.1. Repeat calls do not accumulate targets. Omitted joints latch measured angles at each move_joints start; within coding_control they keep the previous program target.',
             'HYBRID uses joint_offsets in[-0.25,+0.25]rad relative to each freshly recomputed original-policy target. +0.1 adds +0.1rad; zero leaves that joint unmodified, not at zero angle. accept uses an empty offset dictionary. No cumulative offset integration.',
             'Only the tools for the selected mode above are available. Native PD and torque limits govern achieved motion; optional ankle assistance remains explicitly reported.',
             '| Coordinate order for numeric observations | Named control key | Physical joint bounds (rad) |',
             '|---|---|---|']
    for i,(name,(lo,hi)) in enumerate(zip(names,limits)):
        lines.append(f'| {i} | {name} | [{float(lo):.8g},{float(hi):.8g}] |')
    lines.append('Use these names as dictionary keys; the joint list is not a new vector tool. Bounds do not guarantee collision-free or balanced configurations.')
    return '\n'.join(lines)+'\n'


def coding_spec(mode):
    output = ('Return {"joint_positions": {joint_name: absolute_radians}}. Omitted joints retain the previous program target; on the first tick they latch measured positions.'
              if mode=='direct' else 'Return {"decision":"accept", "joint_offsets":{}} or {"decision":"modify", "joint_offsets":{joint_name: offset_radians}}. Offsets apply to the CURRENT original-policy proposal, within +/-0.25 rad.')
    return {'type':'function','name':'coding_control',
        'description': 'Execute a bounded feedback controller written as def control(obs, memory). The runner calls it once per 0.02s control step with FRESH allowed numeric robot observations, before applying one action. Use this for feedback corrections, gait phases, coordinated trajectories or a short subtask. '+output+
        ' Return {"done": true} to stop BEFORE the next action and return fresh observations to the LLM. memory is a mutable dict retained only within this call. max_steps counts actual simulator control steps, shares the episode budget, and is capped by remaining episode steps. This does not call the LLM at every tick. The existing PD/optional balance assist still runs at 500 Hz. Code is interpreted in a restricted Python subset with no file/network/process/simulator access; it cannot reset, teleport, edit physics, or declare task success. Invalid code/action stops the program with an error and reports any steps already executed.',
        'inputSchema':{'type':'object','additionalProperties':False,
            'required':['note','max_steps','code'],
            'properties':{
                'note':{'type':'string','description':'What this controller segment should accomplish and which feedback it uses.'},
                'max_steps':{'type':'integer','minimum':1,'maximum':500,'description':'Maximum 50 Hz action steps for this call (500 = 10 simulated seconds). Early done/errors consume only executed steps.'},
                'code':{'type':'string','minLength':1,'maxLength':16000,'description':'Define control(obs, memory). Supports helper functions, if/for/while, dict/list operations and comprehensions, arithmetic, and import math. No numpy, exec/eval, file access, classes, generators or arbitrary imports. See system instructions for obs keys, supported methods and return contract.'}}}}


def compose_action(mode, decision, proposal, targets, default, scale, limits, to_mujoco, to_isaac):
    """Convert named MJCF targets to the upstream ISAACLAB-ordered native action."""
    if mode == 'hybrid' and decision == 'accept':
        return proposal.copy()  # Preserve even upstream out-of-limit proposals exactly.
    if mode == 'hybrid':
        targets = np.clip(default + scale * proposal[to_mujoco] + targets, limits[:, 0], limits[:, 1])
    return ((targets - default) / scale)[to_isaac].astype(np.float32)


def specs(names, limits, mode):
    common = {'note': {'type': 'string'}, 'steps': {'type': 'integer', 'minimum': 1, 'maximum': 50}}
    result = []
    if mode == 'hybrid':
        return [{'type': 'function', 'name': 'review_action',
            'description': 'Review the displayed original PAiD proposal. decision=accept keeps native policy actions; modify adds named joint target offsets in radians. During the requested 1..50 steps the original recurrent policy recomputes actions at 50 Hz and the same offset is applied to each new proposal. This is chunk-level supervision, not individual review of unseen future actions. Joint limits clamp modified targets. No resets or scene edits.',
            'inputSchema': {'type': 'object', 'additionalProperties': False,
                'required': ['note', 'steps', 'decision', 'joint_offsets'],
                'properties': {**common, 'decision': {'type': 'string', 'enum': ['accept', 'modify']},
                    'joint_offsets': {'type': 'object', 'additionalProperties': False,
                        'properties': {name: {'type': 'number', 'minimum': -0.25, 'maximum': 0.25} for name in names}}}}},coding_spec(mode)]
    fields = {name: {'type': 'number', 'minimum': float(lo), 'maximum': float(hi)} for name, (lo, hi) in zip(names, limits)}
    result.append({'type': 'function', 'name': 'move_joints',
        'description': 'Apply absolute G1 joint position targets in radians through the ORIGINAL PD torque controller. All named joints move simultaneously. Omitted joints hold their measured position at command start. Optional ankle balance feedback is reported in observations and can adjust ankle targets; it is not a walking or collision planner. Open-loop commands can cause a fall. 1..50 steps at 50 Hz. Robot is floating-base, not a fixed arm.',
        'inputSchema': {'type': 'object', 'additionalProperties': False, 'required': ['note', 'steps', 'joint_positions'],
            'properties': {**common, 'joint_positions': {'type': 'object', 'additionalProperties': False, 'minProperties': 1, 'properties': fields}}}})
    return result+[coding_spec(mode)]


def validate(name, args, names, limits, measured, mode):
    if name not in ({'review_action'} if mode=='hybrid' else {'move_joints'}):
        raise ValueError('Unknown or unavailable controller tool')
    required = {'note', 'steps'} | ({'joint_positions'} if name == 'move_joints' else {'decision', 'joint_offsets'})
    if not isinstance(args, dict) or set(args) != required or not isinstance(args.get('note'), str):
        raise ValueError('Invalid arguments')
    if type(args['steps']) is not int or not 1 <= args['steps'] <= 50:
        raise ValueError('steps must be an integer in 1..50')
    target = np.asarray(measured, dtype=float).copy()
    if name == 'move_joints':
        updates = args['joint_positions']
        if not isinstance(updates, dict) or not updates or set(updates) - set(names):
            raise ValueError('Use nonempty named joint_positions from the tool schema')
        for key, value in updates.items():
            i = names.index(key)
            if type(value) not in (int, float) or not np.isfinite(value) or not limits[i][0] <= value <= limits[i][1]:
                raise ValueError('Invalid joint target: ' + key)
            target[i] = value
    else:
        offsets = args['joint_offsets']
        if args['decision'] not in ('accept', 'modify') or not isinstance(offsets, dict) or set(offsets) - set(names):
            raise ValueError('Invalid review decision or offsets')
        if args['decision'] == 'accept' and offsets or args['decision'] == 'modify' and not offsets:
            raise ValueError('accept requires empty offsets; modify requires named offsets')
        target = np.zeros(len(names))
        for key, value in offsets.items():
            if type(value) not in (int, float) or not np.isfinite(value) or abs(value) > .25:
                raise ValueError('Offsets must be finite radians in [-0.25, 0.25]')
            target[names.index(key)] = value
    return args['steps'], target
