"""Bounded per-observation feedback controller over the existing R1Pro action interface."""
import math
import numpy as np
from .control import action_for, validate_tool
from environment.benchmarks.humanoid_soccer.coding import ControllerProgram

MAX_PROGRAM_STEPS = 600


def coding_spec():
    return {'type':'function','name':'coding_control',
        'description':'Run a bounded Python-subset control(obs, memory) callback once per NEW 30Hz observation. '
          'Return {"targets": {...}} using the SAME target keys, units and bounds as move_robot, or {"done": true} '
          'to end this segment before another action. Both arms, base, trunk and grippers act simultaneously. '
          'No LLM round trip occurs between ticks. Omitted EEF/trunk targets retain the last requested program targets, '
          'initially measured poses; omitted base velocities are zero each tick; gripper openings persist. '
          'Native evaluator alone steps physics and decides completion; the program cannot reset, teleport, '
          'inspect hidden state, change scoring, use files/network, or declare task success. Fresh proprioception '
          'and a sparse grid of current onboard depth are supplied; no RGB recognition or map oracle is built in. '
          'Follow the system motion discipline: generate gradual references from measured feedback; '
          'this tool does not automatically smooth or slow down large targets.',
        'inputSchema':{'type':'object','additionalProperties':False,'required':['note','max_steps','code'],
          'properties':{'note':{'type':'string'},
                        'max_steps':{'type':'integer','minimum':1,'maximum':MAX_PROGRAM_STEPS,
                                     'description':'Maximum callback action ticks, at 30Hz; 600 = 20 simulated seconds. Native episode termination may stop sooner.'},
                        'code':{'type':'string','minLength':1,'maxLength':16000,
                                'description':'Define control(obs, memory). Return named move_robot targets or exactly {"done": True}. See system instructions for observation keys and permitted Python.'}}}}


INSTRUCTIONS = '''
BEHAVIOR feedback-program tool: coding_control
For a phase that benefits from frequent feedback, write control(obs, memory) and submit note, max_steps (1..600), code. The host invokes it once for each NEW official policy observation (30Hz), and sends the returned action through the same move_robot controller mapping. No LLM inference runs between callbacks. This can track EEF targets, coordinate base/arms/trunk, wait for measured convergence or react to nearby depth changes. It is not a prebuilt navigation/grasp skill, and it does not bypass action validation. Physics pauses during computation; this is not a real-time hardware claim.

Callback input:
- obs['proprio']: the same named measured arrays as the LLM's robot-base proprioception, including eef_left_pos/eef_right_pos and eef_left_quat/eef_right_quat (XYZW). No global pose, hidden object identities/poses, contacts not already observed, goals, scores or simulator handles.
- obs['depth']: per onboard camera role, {source_shape, rows, columns, values_m}. values_m is a 12x16-or-smaller grid sampled from the CURRENT depth_linear image. rows/columns give the original pixel indices; metres, null for nonpositive/nonfinite samples. This is sparse depth, not object recognition, a clearance guarantee or a collision detector. Unsampled small obstacles can be missed. No RGB pixels are passed into the callback; obtain scene semantics from the LLM's normal images.
- obs['dt']=1/30; env_step counts previously observed completed actions; program_step starts at 0; nominal_episode_steps_remaining is a number or null. The native evaluator owns the actual episode end, including its original counter convention.
- obs['last_targets']: previous requested EEF/trunk targets; initially latched measured poses. obs['gripper_openings']: persistent commanded openings (0 closed, 1 open). obs['trunk_limits']: measured joint bounds if available.

Return exactly {'targets': {...}} or {'done': True}. Target keys, bounds and conventions are IDENTICAL to move_robot: left/right_x/y/z in root-frame metres, left/right_quat_xyzw, base_vx/base_vy in m/s (limits +/-0.3), base_wz in rad/s (+/-0.5), left/right_gripper in [0,1], and trunk_qpos as a full joint vector when available. All specified components act simultaneously for one tick, not a repeated 30-step segment. Omitted EEF/trunk coordinates retain the prior program target; root-frame targets move with the robot base. Base commands default to zero EACH tick; grippers persist across ticks and calls. For multi-tick corrections, explicitly use measured error; repeatedly adding a delta to last_targets would integrate movement. Return done before another action to request fresh images/LLM reasoning; it ends only this tool call, never declares task success.

Each call starts a fresh JSON memory dict. Use memory for phases, initial measurements and counters. Bound changes, check measured convergence, and return to the LLM when visual interpretation is needed. Use a short initial test before a long segment; max_steps=600 is a ceiling, not a recommended duration for every motion. A worker error/invalid target stops the segment with no action on that failing tick; previous ticks cannot be undone. Normal native success/failure/timeout stops the segment immediately through the evaluator. Every program, callback observation, return value and resulting action is recorded.

Supported subset: def functions with positional arguments; assignments, if/elif/else, for/while, break/continue, comprehensions, lists/dicts, indexing/slicing, arithmetic and comparisons; optional import math. Builtins abs/min/max/sum/len/range/enumerate/zip/int/float/bool/round/list/dict/tuple/sorted/all/any; ordinary allowed list/dict methods such as get/copy/items/keys/values/update/setdefault/pop/append. No numpy, arbitrary imports, open, eval/exec, processes, sockets or simulator objects. Limits: 16000 source characters, 100000 interpreted operations per callback, 4096 collection items, 64KiB output plus memory, 2s callback timeout. Large raw images are not supplied.

Example: hold measured EEF/trunk poses for 10 feedback ticks (not a task-solving controller):
def control(obs, memory):
    if obs['program_step'] >= 10:
        return {'done': True}
    return {'targets': {'base_vx': 0.0}}
'''


def validate_program(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {'note','max_steps','code'}:
        raise ValueError('coding_control requires note, max_steps, code')
    if not isinstance(arguments['note'], str): raise ValueError('note must be a string')
    if type(arguments['max_steps']) is not int or not 1 <= arguments['max_steps'] <= MAX_PROGRAM_STEPS:
        raise ValueError('max_steps must be 1..600')
    if not isinstance(arguments['code'], str) or not 1 <= len(arguments['code']) <= 16000:
        raise ValueError('code must contain 1..16000 characters')
    return arguments


def depth_grid(images):
    grids = {}
    for key, image in images.items():
        if not key.endswith('_depth_linear'): continue
        depth = np.squeeze(np.asarray(image))
        if depth.ndim != 2: raise ValueError('Expected 2D onboard linear depth')
        rows = np.linspace(0, depth.shape[0]-1, min(12, depth.shape[0]), dtype=int)
        cols = np.linspace(0, depth.shape[1]-1, min(16, depth.shape[1]), dtype=int)
        sampled = depth[np.ix_(rows, cols)]
        grids[key[:-len('_depth_linear')]] = {'source_shape':list(depth.shape), 'rows':rows.tolist(),
            'columns':cols.tolist(), 'values_m':[[float(v) if math.isfinite(v) and v > 0 else None for v in row] for row in sampled]}
    return grids


class FeedbackController:
    def __init__(self, arguments, state, profile, worker_factory=ControllerProgram):
        validate_program(arguments)
        self.profile = profile
        self.targets = {}
        for side in ('left','right'):
            self.targets.update({f'{side}_{axis}':float(state[f'eef_{side}_pos'][i]) for i,axis in enumerate('xyz')})
            self.targets[f'{side}_quat_xyzw'] = state[f'eef_{side}_quat'].tolist()
        if profile['controller_indices'].get('trunk'):
            self.targets['trunk_qpos'] = state['trunk_qpos'].tolist()
        self.memory = {}; self.worker = worker_factory(arguments['code'])

    def observation(self, state, images, grippers, env_step, program_step, step_budget):
        return {'proprio':{k:v.tolist() for k,v in state.items()}, 'depth':depth_grid(images),
                'dt':1/30, 'env_step':env_step, 'program_step':program_step,
                'nominal_episode_steps_remaining':max(0,step_budget-env_step) if step_budget is not None else None,
                'last_targets':dict(self.targets), 'gripper_openings':dict(grippers),
                'trunk_limits':self.profile.get('trunk_limits')}

    def tick(self, observation, state, grippers):
        reply = self.worker.request({'obs':observation})
        command = reply['action']; self.memory = reply['memory']
        if command == {'done':True} and type(command['done']) is bool:
            return None, grippers, command
        if not isinstance(command, dict) or set(command) != {'targets'}:
            raise ValueError('Program must return targets or exactly {done: True}')
        validate_tool('move_robot', {'note':'program tick','targets':command['targets'],'steps':1}, self.profile)
        targets = {**self.targets, **command['targets']}
        action, next_grippers = action_for(targets, state, self.profile, grippers)
        # Commit persistent references only after the complete command passes validation.
        self.targets = {k:v for k,v in targets.items() if k not in {'base_vx','base_vy','base_wz','left_gripper','right_gripper'}}
        return action, next_grippers, command

    def close(self): self.worker.close()
