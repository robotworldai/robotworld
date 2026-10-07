"""Deterministic predicate fixtures; never claimed as physically achievable rollouts."""
import json
import math
from pathlib import Path
from .monitor import Monitor
from .profiles import PROFILES


def passing_state(profile):
    state = {}
    for rule in profile['checks']:
        for name, op, value in rule['conditions']:
            if type(value) is bool or op == '==': state[name] = value
            elif op in ('<', '<='): state[name] = min(state.get(name, math.inf), value - .001)
            else: state[name] = max(state.get(name, -math.inf), value + .001)
    return state


def run_trace(profile, state, change=None, *, verified=True):
    monitor = Monitor(profile)
    for step in range(profile['steps'] + 1):
        row = dict(state)
        flags = change(step, row) if change else {}
        monitor.update(step, row, **(flags or {}))
    return monitor.report(scene_verified=verified)


def validate():
    results = []
    for key, profile in PROFILES.items():
        state = passing_state(profile)
        cases = []
        def record(name, result, expected):
            actual = result['world_success']
            if actual is not expected: raise AssertionError((key, name, expected, result))
            cases.append(dict(name=name, expected=expected, actual=actual, passed=True))
        record('all_conditions_for_full_duration', run_trace(profile, state), True)
        record('unverified_scene_is_unscored', run_trace(profile, state, verified=False), None)
        record('native_failure_on_success_tick_wins', run_trace(profile, state,
               lambda i,s: {'native_failure': i==profile['steps']}), False)
        record('scene_failure_latches', run_trace(profile, state,
               lambda i,s: {'scene_failure': 'out_of_bounds' if i==1 else None}), False)
        for rule in profile['checks']:
            for name, op, threshold in rule['conditions']:
                bad = not threshold if type(threshold) is bool else (
                    threshold+.001 if op in ('<','<=','==') else threshold-.001)
                # One violation in an always window must remain latched. For hold
                # or final, place it at the last tick; successful history cannot hide it.
                at = round(sum(rule['window'])/2*profile['hz']) if rule['operator']=='always' else profile['steps']
                def change(i,s):
                    if i == at: s[name] = bad
                record(rule['id']+':single_violation:'+name, run_trace(profile,state,change), False)
                def missing(i,s):
                    if i == at: s.pop(name)
                record(rule['id']+':missing:'+name, run_trace(profile,state,missing), None)
                if type(threshold) is not bool:
                    def nan(i,s):
                        if i == at: s[name] = float('nan')
                    record(rule['id']+':nan:'+name, run_trace(profile,state,nan), None)
        short = Monitor(profile)
        short.update(0,state)
        record('reset_alone_cannot_pass', short.report(scene_verified=True), None)
        results.append(dict(task=key,checks=len(profile['checks']),tests=cases,
            evidence_type='synthetic_state_trace', physical_rollout=False))
    return dict(kind='predicate_unit_validation', all_passed=True, tasks=len(results),
        tests=sum(len(r['tests']) for r in results), results=results,
        limitation='Proves predicate logic and temporal boundaries, not scene integration, reachability or policy performance.')


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
