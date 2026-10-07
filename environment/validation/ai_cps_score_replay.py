"""Recompute saved native STL scores/contact recovery without starting a simulator.

Run inside the AI-CPS dependency image (CPU only). No model, environment reset,
or scene mutation. This checks score propagation, not physical task completion.
"""
import argparse
import json
from pathlib import Path

from environment.benchmarks.ai_cps.scoring import score
from environment.benchmarks.ai_cps.control import ContactRecovery

WORLD = Path(__file__).resolve().parents[2]


def audit(run):
    original = json.loads((run / 'result.json').read_text())
    case = original['case']
    name = {'22': 'FrankaBallCatching', '23': 'FrankaBallBalancing',
            '24': 'FrankaPegInHole', '34': 'FrankaPegInHole'}[case]
    raw = json.loads((run / 'native_trace.json').read_text())
    replay = score(name, raw, original['native_episode_complete'],
                   WORLD / 'third_party/benchmarks/ai_cps/checkout')
    comparisons = {key: original.get(key) == value for key, value in replay.items()}
    recovery = None
    if case == '34':
        state = ContactRecovery()
        for line in (run / 'events/environment.jsonl').open():
            event = json.loads(line)
            payload = event['payload']
            if event['kind'] == 'action_cancelled':
                state.cancel(payload['cancel_step'])
            elif event['kind'] == 'environment_step':
                after = payload['after']
                for force in after['peg_table_contact_substeps_N']:
                    state.contact(force, after['control_step'])
                state.tick(after['peg_table_contact_peak_N'])
        recovery = state.result()
        comparisons['authored_contact_recovery'] = recovery == original['authored_contact_recovery']
    return {'run': str(run), 'case': case, 'passed': all(comparisons.values()),
            'comparisons': comparisons, 'recomputed': replay, 'contact_recovery': recovery,
            'scope': 'Recorded trace re-evaluated by unchanged upstream RTAMT monitor; '
                     'authored contact state machine replayed separately. '
                     'No independent certification of physical capture/insertion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    rows = [audit(run) for run in args.runs]
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(row['passed'] for row in rows) else 1)


if __name__ == '__main__':
    main()
