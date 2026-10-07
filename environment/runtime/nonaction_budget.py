"""Versioned policy interaction budget, independent of physics and API retries."""
import json
import os
from pathlib import Path

VERSION = 'nonaction-20-150-v1'
CALIBRATED = {f'astra-nonaction-15-{n}-v1': (15, n) for n in (30, 60, 120)}


def limits(protocol):
    if protocol in ('off', 'observe', VERSION):
        return 20, 150
    if protocol in CALIBRATED:
        return CALIBRATED[protocol]
    raise ValueError('Unknown non-action interaction protocol')


def instructions(budget):
    return INSTRUCTIONS.replace('20 consecutive or 150 total',
        f'{budget.consecutive_limit} consecutive or {budget.total_limit} total')

REASONS = {'nonaction_consecutive_limit', 'nonaction_total_limit'}
INSTRUCTIONS = ('A final answer does not end the episode. Non-action interaction budget: '
                '20 consecutive or 150 total calls/empty turns. Auxiliary tool calls and '
                'robot calls executing zero control steps each cost one. A normal turn '
                'with no executed control steps and no counted calls costs one. '
                'Confirmed control steps reset only the consecutive counter. API retries '
                'do not consume this budget. The evaluator ends the episode at either limit.')


def mode():
    value=os.environ.get('WORLD_NONACTION_PROTOCOL','off')
    limits(value)
    return value


def enabled():
    """Whether stopping semantics, not just telemetry, are enabled."""
    return mode() in (VERSION, *CALIBRATED)


class NonActionBudgetExceeded(Exception):
    def __init__(self, snapshot):
        self.snapshot = snapshot
        super().__init__(snapshot['stop_reason'])


class NonActionBudget:
    def __init__(self, output, step, consecutive_limit=20, total_limit=150, *, observe_only=False):
        self.output, self.step = Path(output), step
        self.observe_only = observe_only
        self.consecutive_limit, self.total_limit = consecutive_limit, total_limit
        self.total = self.consecutive = 0
        self.reason = None
        self.first_threshold = None
        self.consecutive_peak = 0
        self.counts = {}
        self.terminal_events = 0
        self.started_turns = set()
        self.completed_turns = set()
        self.last_step = step()
        self.turn = None
        self.turn_counted = False
        self.turn_progress = False
        self.seen = set()
        self.robot_calls = {}
        self.write()

    def snapshot(self):
        return dict(version=getattr(self, 'protocol', VERSION), mode='observe' if self.observe_only else 'enforce',
                    accounting_version='observable-events-v2', counts_complete=False,
                    limitations=['Some invalid built-in calls and final empty stdin polls emit no event',
                                 'Terminal interactions lack reliable per-invocation IDs'],
                    missing_turn_end_count=len(self.started_turns - self.completed_turns),
                    consecutive_peak=self.consecutive_peak, counts_by_kind=dict(self.counts),
                    first_threshold=self.first_threshold,
                    consecutive_limit=self.consecutive_limit,
                    total_limit=self.total_limit, total=self.total, consecutive=self.consecutive,
                    control_steps=self.last_step, stop_reason=self.reason)

    def write(self):
        self.output.mkdir(parents=True, exist_ok=True)
        target = self.output/'interaction-budget.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.snapshot(), indent=2)+'\n')
        temporary.replace(target)

    def progress(self):
        value = self.step()
        if value < self.last_step:
            raise RuntimeError('Control step regressed within one episode')
        if value > self.last_step:
            self.consecutive = 0
            self.turn_progress = True
            self.last_step = value
            self.write()

    def count(self, key):
        if key in self.seen:
            return
        self.seen.add(key)
        self.total += 1
        self.consecutive += 1
        self.consecutive_peak = max(self.consecutive_peak, self.consecutive)
        self.counts[key[0]] = self.counts.get(key[0], 0) + 1
        self.turn_counted = True
        if self.first_threshold is None:
            reason = None
            if self.consecutive >= self.consecutive_limit:
                reason = 'nonaction_consecutive_limit'
            elif self.total >= self.total_limit:
                reason = 'nonaction_total_limit'
            if reason:
                self.first_threshold = dict(reason=reason, total=self.total,
                                            consecutive=self.consecutive,
                                            control_steps=self.last_step, turn_id=self.turn)
                if not self.observe_only:
                    self.reason = reason
        self.write()

    def observe(self, message):
        self.progress()
        method, params = message.get('method'), message.get('params', {})
        if method == 'turn/started':
            turn = params['turn']['id']
            self.start_turn(turn)
        elif method == 'item/tool/call':
            self.robot_calls.setdefault(message['id'], (params['callId'], self.step()))
        elif method in ('item/started', 'item/completed'):
            item = params.get('item', {})
            if item.get('type') in ('commandExecution', 'imageView'):
                self.count(('aux', params.get('turnId', self.turn), item['id']))
        elif method == 'item/commandExecution/terminalInteraction':
            # Identical polls can be distinct calls; do not deduplicate by payload.
            self.terminal_events += 1
            self.count(('stdin', params.get('turnId', self.turn), self.terminal_events))
        elif method == 'turn/completed':
            self.completed_turns.add(params['turn']['id'])
            if (params['turn']['status'] == 'completed' and
                    not self.turn_progress and not self.turn_counted):
                self.count(('empty_turn', params['turn']['id']))
            self.write()

    def start_turn(self, turn):
        self.started_turns.add(turn)
        if turn != self.turn:
            self.turn = turn
            self.turn_counted = self.turn_progress = False
        self.write()

    def reply(self, message):
        call = self.robot_calls.pop(message.get('id'), None)
        if call is not None and ('result' in message or 'error' in message):
            self.progress()
            if self.step() == call[1]:
                self.count(('robot_zero', call[0]))


def attach(session, output, step):
    """Attach on a newly constructed policy session; never mutate live processes."""
    selected = mode()
    if selected == 'off':
        return None
    consecutive, total = limits(selected)
    budget = NonActionBudget(output, step, consecutive, total, observe_only=selected == 'observe')
    budget.protocol = selected if selected in CALIBRATED else VERSION
    budget.write()
    session.nonaction_budget = budget
    return budget


def stop_session(session):
    budget = getattr(session, 'nonaction_budget', None)
    if budget is None or budget.observe_only or budget.reason is None:
        return
    if getattr(session, '_budget_interrupted', False):
        return
    session._budget_interrupted = True
    if getattr(session, '_budget_thread', None) and budget.turn:
        session.request('turn/interrupt', {'threadId':session._budget_thread, 'turnId':budget.turn})


def check(session):
    budget = getattr(session, 'nonaction_budget', None)
    if budget is not None:
        budget.progress()
        if not budget.observe_only and budget.reason and not budget.robot_calls:
            stop_session(session)
            raise NonActionBudgetExceeded(budget.snapshot())


def run_bounded(agent):
    try:
        agent.run()
        return None
    except NonActionBudgetExceeded as stop:
        return stop.snapshot


def apply_result(result, snapshot):
    if snapshot is None:
        return result
    result['interaction_budget'] = snapshot
    if snapshot.get('mode') == 'observe':
        return result
    result['stop_reason'] = snapshot['stop_reason']
    # Missing full-horizon evidence is not invented; zero is an interaction-budget
    # outcome, not a claim that the native checker completed its whole window.
    result['native_success'] = result.get('success')
    world=result.get('world_evaluation')
    if result.get('scoring_profile')=='world-state-v1':
        result['success']=bool(isinstance(world,dict) and world.get('valid') and world.get('world_success') is True)
    else:
        result['success'] = result.get('success') is True
    return result
