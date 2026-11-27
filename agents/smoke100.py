"""Public-action 100-control-step Harbor infrastructure probe (not a policy)."""
import argparse
import json
import math
import random
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def has_reading(value):
    if isinstance(value, dict):
        return any(has_reading(item) for item in value.values())
    if isinstance(value, list):
        return any(has_reading(item) for item in value)
    return value is not None and value != ''


def command_for(description, observation, steps, *, origin=None, call_index=0):
    names = {t['name'] for t in description['tools']}
    args = {'note': 'Non-solving infrastructure smoke probe', 'steps': steps}
    if 'apply_action' in names:
        meta = description['action_metadata']
        args['action'] = [max(lo if lo is not None else -float('inf'), min(hi if hi is not None else float('inf'), 0.0)) for lo, hi in zip(meta['lower'], meta['upper'])]
        name = 'apply_action'
    elif 'drive' in names:
        name = 'drive'; args['action'] = [0, 0]
    elif 'set_grippers' in names:
        name = 'set_grippers'; args['targets'] = {'left_gripper': 1, 'right_gripper': 1}
    elif 'set_gripper' in names:
        name = 'set_gripper'; args['targets'] = {'gripper_close': 0}
    elif 'move_eef' in names and description['benchmark'] == 'robodojo':
        name = 'move_eef'; args.pop('steps')
        text = '\n'.join((origin or observation)['observation']['feedback_text'])
        keys = ('left_x', 'left_y', 'left_z', 'left_pitch_deg', 'left_roll_deg', 'left_yaw_deg', 'left_gripper')
        args['targets'] = {key: float(re.findall(r'^'+key+r'=([-+0-9.eE]+)$', text, re.M)[-1]) for key in keys}
        # A planner may legitimately return zero waypoints for an exact hold.
        # Exercise stepping with a bounded +/-5 mm probe about the initial pose,
        # never an accumulating offset or a task-solving reference trajectory.
        args['targets']['left_z'] += 0.005 if call_index % 2 == 0 else -0.005
    elif 'move_joints' in names:
        name = 'move_joints'
        if description['benchmark'] == 'humanoid_soccer':
            joints = observation['observation']['state']['joint_positions']
            key = next(iter(joints)); args['joint_positions'] = {key: joints[key]}
        else:
            args['arm_action'] = [0]*7
    else:
        raise ValueError('No reviewed probe action for '+description['benchmark'])
    return {'tool': name, 'arguments': args}


def random_command(description, observation, steps, origin, call_index, rng):
    command = command_for(description, observation, steps, origin=origin, call_index=call_index)
    tool = command['tool']
    args = command['arguments']
    args['note'] = 'Seeded random infrastructure smoke probe'
    if tool == 'apply_action' and description['benchmark'] == 'reflexbench':
        position = [center + rng.uniform(-0.02, 0.02) for center in (0.45, 0.0, 0.40)]
        rotation = [center + rng.uniform(-0.02, 0.02) for center in (0.0, 1.0, 0.0, 0.0)]
        norm = math.sqrt(sum(value * value for value in rotation))
        args['action'] = position + [value / norm for value in rotation] + [1.0]
    elif tool == 'apply_action':
        bounds = description['action_metadata']
        args['action'] = [
            rng.uniform(max(center - 0.05, lo) if lo is not None else center - 0.05,
                        min(center + 0.05, hi) if hi is not None else center + 0.05)
            for center, lo, hi in zip(args['action'], bounds['lower'], bounds['upper'])
        ]
    elif tool == 'drive':
        args['action'] = [rng.uniform(0.02, 0.12), rng.uniform(-0.12, 0.12)]
    elif tool == 'set_grippers':
        args['targets'] = {name: rng.uniform(0.6, 1.0) for name in args['targets']}
    elif tool == 'set_gripper':
        args['targets']['gripper_close'] = rng.randrange(2)
    elif tool == 'move_eef':
        direction = 1 if call_index % 2 == 0 else -1
        center = args['targets']['left_z'] - direction * 0.005
        args['targets']['left_z'] = center + direction * rng.uniform(0.005, 0.008)
    elif 'arm_action' in args:
        args['arm_action'] = [rng.uniform(-0.05, 0.05) for _ in args['arm_action']]
    else:
        props = next(t for t in description['tools'] if t['name'] == tool)['inputSchema']['properties']
        joints = props['joint_positions']['properties']
        measured = observation['observation']['state']['joint_positions']
        joint = rng.choice(sorted(joints.keys() & measured.keys()))
        limits = joints[joint]
        target = measured[joint] + rng.uniform(-0.03, 0.03)
        args['joint_positions'] = {joint: max(limits['minimum'], min(limits['maximum'], target))}
    return command


def run(output, step_budget=100, seed=None):
    from environment.clients.episode import EpisodeClient
    output.mkdir(parents=True, exist_ok=True)
    summary = {'classification': 'infrastructure_probe_not_model_evaluation', 'requested_control_steps': step_budget, 'passed': False}
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        summary['random_seed'] = seed
    client = EpisodeClient()
    try:
        description = client.request('describe')
        summary['description'] = description
        limit = description['native_horizon'] if step_budget == 'native' else int(step_budget)
        if limit < 1: raise ValueError('Step budget must be positive')
        summary['resolved_control_steps'] = limit
        observation = client.request('observe')
        origin = observation
        require(observation == client.request('observe'), 'Query changed observation')
        packet = observation['observation']
        if description.get('observation_modalities') == ['state']:
            state = packet.get('state')
            require(isinstance(state, dict), 'Initial state observation missing')
            # Native adapters wrap actor readings with clock and action metadata.
            fields = state.keys() & {'native_policy_terms', 'native_policy_observation', 'native_actor_history'}
            if not fields:
                fields = state.keys() - {'episode_steps_remaining', 'control_step', 'dt', 'control_dt',
                                         'last_requested_action', 'episode_terminated'}
            require(any(has_reading(state[key]) for key in fields), 'Initial state observation missing')
        else:
            require(packet.get('images'), 'Initial camera images missing')
        initial = observation['control_step']; summary['initial_step'] = initial
        stalled = 0; count = 0
        with (output/'receipts.jsonl').open('w') as trace:
            while True:
                state = client.request('status')
                advanced = state['control_step']-initial
                if state['terminal'] or advanced >= limit:
                    break
                steps = min(10, limit-advanced, state['horizon']-state['control_step'])
                action = (random_command(description, observation, steps, origin, count, rng)
                          if rng is not None else command_for(description, observation, steps, origin=origin, call_index=count))
                request = {'request_id': 'smoke100-'+str(count), 'episode_id': state['episode_id'], 'observation_id': state['observation_id'], 'commands': [action]}
                receipt = client.request('execute', request)
                trace.write(json.dumps({'request': request, 'receipt': receipt})+'\n'); trace.flush()
                require(not receipt['execution_uncertain'], 'Uncertain execution')
                require(client.request('execute', request) == receipt, 'Duplicate changed receipt')
                require(client.request('receipts/'+request['request_id']) == receipt, 'Receipt lookup mismatch')
                observation = client.request('observe')
                require(observation['control_step'] == receipt['control_step'], 'Query stepped simulation')
                require(observation == client.request('observe'), 'Query changed observation')
                stalled = stalled+1 if receipt['executed_steps'] == 0 else 0
                require(stalled < 3, 'Three actions made no physical progress')
                count += 1
        reason = state.get('termination')
        ended_at_horizon = reason == 'budget_exhausted' and state['control_step'] == state['horizon']
        early_native = (0 < advanced < limit and state['terminal']
                        and (ended_at_horizon or reason in {'success', 'native_failure', 'native_truncation'}))
        summary.update(final_status=state, actual_control_steps=advanced, action_calls=count,
                       early_native_termination=early_native)
        if advanced < limit and not early_native:
            raise AssertionError(f'Probe stopped after {advanced} of {limit} requested control steps ({reason})')
        summary.update(passed=True, query_did_not_step=True, duplicate_did_not_step=True)
    except Exception as exc:
        summary['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        (output/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', default='100', help='Positive control steps, or native for the task horizon')
    parser.add_argument('--seed', type=int, help='Sample a reproducible bounded action sequence')
    args = parser.parse_args(); run(args.output, args.steps, args.seed)
else:
    import os
    import shlex
    from agents.robotworld import RobotWorldProbeAgent, ROOT, REMOTE
    class RobotWorldSmokeAgent(RobotWorldProbeAgent):
        @staticmethod
        def name(): return 'robotworld-smoke100'
        async def install(self, environment):
            await super().install(environment)
            await environment.upload_file(ROOT/'agents/smoke100.py', REMOTE+'/agents/smoke100.py')
        async def run(self, instruction, environment, context):
            context.metadata = {'classification': 'infrastructure_probe_not_model_evaluation', 'execution': 'inside-agent-container'}
            await self.exec_as_agent(environment, command='python -m agents.smoke100 --output '+shlex.quote(str(self.environment_logs_dir)+'/robotworld'), env={'PYTHONPATH': REMOTE}, timeout_sec=3600)

    class RobotWorldNativeBudgetProbeAgent(RobotWorldSmokeAgent):
        @staticmethod
        def name(): return 'robotworld-native-budget-probe'
        async def run(self, instruction, environment, context):
            context.metadata = {'classification': 'infrastructure_probe_not_model_evaluation', 'execution': 'inside-agent-container'}
            await self.exec_as_agent(environment, command='python -m agents.smoke100 --steps native --output '+shlex.quote(str(self.environment_logs_dir)+'/robotworld'), env={'PYTHONPATH': REMOTE}, timeout_sec=43200)

    class RobotWorldRandomProbeAgent(RobotWorldSmokeAgent):
        @staticmethod
        def name(): return 'robotworld-random-probe'

        async def run(self, instruction, environment, context):
            seed = int(os.environ.get('WORLD_RANDOM_PROBE_SEED', '20261124'))
            context.metadata = {'classification': 'infrastructure_probe_not_model_evaluation', 'random_seed': seed}
            output = shlex.quote(str(self.environment_logs_dir)+'/robotworld')
            result = await self.exec_as_agent(environment, command=f'python -m agents.smoke100 --seed {seed} --output {output}',
                                              env={'PYTHONPATH': REMOTE}, timeout_sec=3600)
            if result.return_code != 0:
                raise RuntimeError(f'Random probe exited with code {result.return_code}')
