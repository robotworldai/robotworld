"""Serialized, bounded execution. Only adapters may advance the simulator."""
import copy
from collections import deque
import hashlib
import json
import threading
from typing import Protocol
from environment.protocol import VERSION


class Adapter(Protocol):
    def describe(self) -> dict: ...
    def observe(self) -> dict: ...
    def prepare(self, commands: list) -> list: ...
    def step(self, action) -> dict: ...
    def close(self): ...


class Episode:
    def __init__(self, adapter: Adapter, horizon: int, episode_id: str, journal=None, control_step_limit=None):
        if type(horizon) is not int or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if control_step_limit is not None and (type(control_step_limit) is not int or control_step_limit < 1):
            raise ValueError('control_step_limit must be a positive integer or None')
        self.control_step_limit = control_step_limit
        self.stop_reason = None
        self.adapter, self.horizon, self.episode_id = adapter, horizon, episode_id
        self.steps = 0
        self.reason = None
        self.success = False
        self.receipts = {}
        self.lock = threading.RLock()
        self.journal = journal
        # Most legacy policies sampled tool feedback rounds, not physics frames.
        # Keep this at the episode observation boundary: coding callbacks still
        # read current adapter state directly and cannot pollute model history.
        benchmark = adapter.describe().get('benchmark')
        self.state_history_enabled = benchmark in {'wheeledlab','bench2dex','digit','flamingo','go2_push','omnidrones','omniisaacgymenvs','reflexbench','robot_lab','steadytray','ttrl','volleybots','wheel_legged','wheeled_quadruped'}
        self.feedback_history = deque(maxlen=9) if benchmark in {
            'robolab','behavior_1k','wheeledlab','ai_cps','humanoid_soccer',
            'bench2dex','digit','flamingo','go2_push','omnidrones','omniisaacgymenvs',
            'reflexbench','robot_lab','steadytray','ttrl','volleybots',
            'wheel_legged','wheeled_quadruped'} else None

    def status(self):
        with self.lock:
            result = dict(protocol=VERSION, episode_id=self.episode_id, control_step=self.steps,
                        observation_id=f'{self.episode_id}:{self.steps}', horizon=self.horizon,
                        terminal=self.reason is not None, termination=self.reason, success=self.success)
            if self.control_step_limit is not None:
                result.update(control_step_limit=self.control_step_limit,
                              effective_control_step_limit=min(self.horizon, self.control_step_limit),
                              control_step_limit_overshoot=max(0, self.steps-self.control_step_limit),
                              stop_reason=self.stop_reason)
            return result

    def observe(self):
        with self.lock:
            observation = copy.deepcopy(self.adapter.observe())
            if self.state_history_enabled and isinstance(observation.get('state'), dict):
                observation['state']['episode_steps_remaining'] = max(0, self.horizon-self.steps)
            if self.feedback_history is not None:
                if not self.feedback_history or self.feedback_history[-1][0] != self.steps:
                    images = observation.get('images', [])
                    current_step = max((x['control_step'] for x in images), default=self.steps)
                    current = [x for x in images if x['control_step'] == current_step]
                    self.feedback_history.append((self.steps, current, copy.deepcopy(observation.get('state'))))
                observation['images'] = [copy.deepcopy(image)
                    for _, images, _ in reversed(list(self.feedback_history)[::-2][:5]) for image in images]
                if self.state_history_enabled:
                    observation['history_states'] = [dict(control_step=step, state=state)
                        for step, _, state in reversed(list(self.feedback_history)[::-2][:5])]
            return {**self.status(), 'observation': observation}

    def describe(self):
        return {'protocol': VERSION, 'execution': 'serial; stop on native terminal or budget',
                'chunk': 'ordered commands; steps repeats one native action',
                **self.adapter.describe(),
                **({'observation_history': {'previous_frames':4,
                    'interval_tool_feedback_rounds':2,
                    'repeated_queries_advance_history':False}}
                   if self.feedback_history is not None else {}),
                'execution_budget': {
                    'native_horizon': self.horizon,
                    'control_step_limit': self.control_step_limit,
                    'effective_control_step_limit': min(self.horizon, self.control_step_limit or self.horizon),
                    'instruction': 'Stop when the service reports terminal. An optional evaluation cap stops execution before the next action; it does not change the native horizon or success checker. Atomic planner macros may finish beyond this cap; any overshoot is recorded.'}}

    def execute(self, request):
        # In-process retries are idempotent. A process crash invalidates the episode;
        # callers must not replay uncertain commands against a replacement episode.
        with self.lock:
            if not isinstance(request, dict) or set(request) != {'request_id','episode_id','observation_id','commands'}:
                raise ValueError('Expected request_id, episode_id, observation_id, commands')
            rid = request['request_id']
            if not isinstance(rid, str) or not 1 <= len(rid) <= 128:
                raise ValueError('Invalid request_id')
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if rid in self.receipts:
                old_hash, receipt = self.receipts[rid]
                if old_hash != fingerprint:
                    raise ValueError('request_id reused with different payload')
                return copy.deepcopy(receipt)
            if request['episode_id'] != self.episode_id:
                raise ValueError('Wrong episode_id')
            if request['observation_id'] != self.status()['observation_id']:
                raise ValueError('Stale observation_id; query before acting')
            if self.reason:
                raise ValueError('Episode is terminal')
            commands = request['commands']
            if not isinstance(commands, list) or not 1 <= len(commands) <= 256:
                raise ValueError('Expected 1..256 commands (transport safety limit)')
            prepared = self.adapter.prepare(commands)  # validate whole chunk before mutation
            if self.journal:
                self.journal('execution_requested', request)
            counts = [0] * len(prepared)
            error = None
            feedback_details = []
            try:
                for i, (action, repeat) in enumerate(prepared):
                    for _ in range(repeat):
                        feedback = self.adapter.step(action)
                        if feedback.get('segment_done'):
                            if 'details' in feedback:
                                feedback_details.append({'command_index':i,'details':feedback['details']})
                            break
                        delta = feedback.get('executed_steps', 1)
                        if type(delta) is not int or delta < 0:
                            raise ValueError('Adapter reported invalid native step count')
                        self.steps += delta
                        counts[i] += delta
                        if self.steps > self.horizon:
                            raise RuntimeError('Adapter exceeded native step budget')
                        if 'details' in feedback:
                            feedback_details.append({'command_index':i,'details':feedback['details']})
                        self.success = bool(feedback['success'])
                        if self.success:
                            self.reason = 'success'
                        elif feedback['terminated']:
                            self.reason = 'native_failure'
                        elif feedback['truncated']:
                            self.reason = 'native_truncation'
                        elif self.steps >= self.horizon:
                            self.reason = 'budget_exhausted'
                        elif self.control_step_limit is not None and self.steps >= self.control_step_limit:
                            self.reason = 'agent_stopped'
                            self.stop_reason = 'control_step_limit'
                        if self.reason:
                            break
                    if self.reason:
                        break
            except Exception as exc:
                # A failing native step might have mutated state: never retry it.
                self.reason = 'infrastructure_error'
                error = type(exc).__name__
            finally:
                from environment.service.feedback import Program
                for action, _ in prepared:
                    if isinstance(action, Program): action.close()
            receipt = {**self.status(), 'request_id': rid, 'executed_steps': sum(counts),
                       'executed_steps_per_command': counts, 'error': error,
                       'execution_uncertain': error is not None, 'feedback': feedback_details}
            self.receipts[rid] = fingerprint, copy.deepcopy(receipt)
            if self.journal:
                self.journal('execution_completed', receipt)
            return receipt

    def receipt(self, request_id):
        with self.lock:
            return copy.deepcopy(self.receipts[request_id][1])
