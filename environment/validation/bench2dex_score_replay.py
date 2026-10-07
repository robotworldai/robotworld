"""Replay native success/stage predicates from every saved physics-step state.

No Isaac import, policy or simulator. Joint-limit and robot safety metrics are
deliberately outside this audit because the old records lack full joint limits.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]
FIELDS = ('instant_success', 'stable_success', 'ever_instant_success',
          'at_end_success_observed', 'first_success_step', 'stable_success_step',
          'success_hold_s', 'terminal_success_rate', 'stage_completion_rate',
          'normalized_progress_score', 'chain_depth', 'current_stage_completion_rate',
          'current_normalized_progress_score', 'current_chain_depth',
          'latched_stage_completion_rate', 'latched_normalized_progress_score',
          'latched_chain_depth', 'stage_completion', 'current_stage_completion',
          'stage_first_completion_step')


def audit(run):
    from benchmark.metric_tracker import MetricTracker
    result = json.loads((run / 'result.json').read_text())
    task = json.loads((run / 'native-scene.json').read_text())
    sample = json.loads((run / 'scene-provenance.json').read_text())['sample']
    height = sample.get('spatial', {}).get('table_height', {})
    offset = float(height.get('height_offset_m', 0.) or 0.) if height.get('enabled') else 0.
    dt = result['actual_simulated_seconds'] / result['physics_steps']
    tracker = MetricTracker(task['metrics'], dt=dt, table_height_offset=offset, robot_key=result['robot_key'])
    count = 0
    disagreements = []
    with (run / 'events/native-state.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['kind'] != 'physics_step':
                continue
            state = event['payload']
            count += 1
            if state['physics_step'] != count:
                raise ValueError('Missing/reordered physics-step record')
            tracker.update(state['objects'], sim_step=count - 1, dt=dt)
            if bool(tracker.success) != state['stable_success']:
                disagreements.append(count)
    native = result['native_metrics']
    replay = asdict(tracker.finalize(steps=count, terminated_reason='stable_success' if tracker.success else 'max_steps',
                                    policy_query_count=result['control_steps'],
                                    policy_query_count_at_stable_success=result['control_steps'] if tracker.success else None,
                                    max_steps=result['requested_steps'] * 3, policy_stride=3,
                                    evaluation_protocol='reach_and_stop'))
    comparisons = {field: replay[field] == native[field] for field in FIELDS}
    comparisons['success'] = bool(tracker.success) == result['success']
    comparisons['physics_steps'] = count == result['physics_steps']
    return {'run': str(run.resolve()), 'case': result['task'], 'checked_physics_steps': count,
            'passed': not disagreements and all(comparisons.values()),
            'step_success_disagreements': disagreements, 'comparisons': comparisons,
            'scope': 'Exact original terminal and stage scoring from saved simulator states. '
                     'Safety/grasp/tool metrics not compared; no physical success demonstration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(WORLD / 'third_party/benchmarks/bench2dex/checkout'))
    rows = [audit(run) for run in args.runs]
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(row['passed'] for row in rows) else 1)


if __name__ == '__main__':
    main()
