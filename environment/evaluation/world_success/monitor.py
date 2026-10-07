"""Fail-closed temporal checker, evaluated at every control tick including reset.

Always/hold claims refer to sampled simulator state, not an unobserved continuous
physics trajectory. Missing ticks and missing/nonfinite fields invalidate scoring.
Native failures dominate successes even at the final tick. A short run cannot pass.
"""
import math
import operator

OPS = {'<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge, '==': operator.eq}


class Monitor:
    def __init__(self, profile):
        self.profile = profile
        self.last_step = -1
        self.records = {r['id']: dict(passed=False, since=None, first_violation=None,
                        samples=0, latest=None, window_started=False) for r in profile['checks']}
        self.invalid = []
        self.failures = []
        self.last_state = {}

    def update(self, step, state, *, native_failure=False, scene_failure=None):
        if type(step) is not int or step != self.last_step + 1:
            raise ValueError(f'Expected contiguous tick {self.last_step + 1}, got {step}')
        self.last_step = step
        t = step / self.profile['hz']
        self.last_state = dict(state)
        if native_failure and 'native_failure' not in self.failures: self.failures.append('native_failure')
        if scene_failure and str(scene_failure) not in self.failures: self.failures.append(str(scene_failure))
        for rule in self.profile['checks']:
            row = self.records[rule['id']]
            values, ok = {}, True
            for field, op, limit in rule['conditions']:
                value = state.get(field)
                valid = (type(value) is bool if type(limit) is bool else
                         type(value) in (int, float) and math.isfinite(value))
                if not valid:
                    reason = 'missing_or_invalid:' + field
                    if reason not in self.invalid: self.invalid.append(reason)
                passed = valid and OPS[op](value, limit)
                values[field] = {'value': value if valid else None, 'operator': op, 'threshold': limit, 'passed': passed}
                ok = ok and passed
            row['latest'] = values
            if rule['operator'] == 'always':
                lo, hi = rule['window']
                # Bracket the requested window even when endpoints are off-grid.
                active = math.floor(lo*self.profile['hz']) <= step <= math.ceil(hi*self.profile['hz'])
                if active:
                    row['window_started'] = True
                    row['samples'] += 1
                    if not ok and row['first_violation'] is None: row['first_violation'] = t
                row['passed'] = row['window_started'] and step >= math.ceil(hi*self.profile['hz']) and row['first_violation'] is None
            elif rule['operator'] == 'hold':
                row['samples'] += 1
                if ok:
                    if row['since'] is None: row['since'] = t
                else:
                    row['since'] = None
                    if row['first_violation'] is None: row['first_violation'] = t
                row['passed'] = ok and t - row['since'] >= rule['seconds'] - 1e-10
            else:
                row['samples'] += 1
                row['passed'] = ok
        return self.report()

    def report(self, *, scene_verified=False, event_complete=False):
        early_success = (self.profile.get('early_success', False) and
                         all(r['passed'] for r in self.records.values()))
        complete = self.last_step >= self.profile['steps'] or (self.profile['event_end'] and event_complete) or early_success
        # Failures are valid negative outcomes; unverified scenes or incomplete
        # traces have no comparable binary outcome, never silently counted as 0.
        valid = scene_verified and not self.invalid and (complete or bool(self.failures))
        success = (not self.failures and complete and all(r['passed'] for r in self.records.values())) if valid else None
        return dict(version=self.profile['version'], world_success=success, valid=valid,
                    scene_verified=scene_verified, complete=complete,
                    control_steps=self.last_step, elapsed_seconds=max(0,self.last_step)/self.profile['hz'],
                    expected_steps=self.profile['steps'], failures=list(self.failures), invalid=list(self.invalid),
                    checks={k:dict(v) for k,v in self.records.items()},
                    sampling='every control tick, including reset; no continuous-physics guarantee')
