"""Replay archived public tool calls through the Harbor episode executor.

Run in the RoboCasa image with the repository on PYTHONPATH. Reference folders
contain run.json, tools.json and environment.json (JSON arrays of logged events).
This is a deterministic migration diagnostic, not a model evaluation. It never
injects simulator state or changes a success predicate.
"""
import argparse
import json
from pathlib import Path


def replay(reference, output, tolerance):
    import numpy as np
    from environment.benchmarks.robocasa.control import state_of
    from environment.service.robocasa import RoboCasaAdapter
    from environment.service.episode import Episode

    run = json.loads((reference / 'run.json').read_text())
    if run['benchmark'] != 'robocasa':
        raise ValueError('Expected a RoboCasa reference')
    output.mkdir(parents=True, exist_ok=False)
    history = json.loads((reference / 'environment.json').read_text())
    states = [r['payload']['state'] for r in history if r['kind'] == 'observation']
    actions = [r['payload']['action'] for r in history if r['kind'] == 'action_requested']
    requests = [r['payload'] for r in json.loads((reference / 'tools.json').read_text())
                if r['kind'] == 'tool_requested']
    adapter = RoboCasaAdapter(run['task'], run['seed'], output, horizon=run['step_limit'])
    episode = Episode(adapter, adapter.horizon, 'reference-replay')
    rejected = []
    try:
        initial_error = max(float(np.max(np.abs(np.asarray(state_of(adapter.obs)[k]) - states[0][k])))
                            for k in states[0])
        with (output / 'receipts.jsonl').open('w') as log:
            for index, request in enumerate(requests):
                if episode.status()['terminal']:
                    break
                if request['tool'] == 'observe':
                    continue
                try:
                    receipt = episode.execute(dict(request_id=f'reference-{index}',
                        episode_id=episode.episode_id, observation_id=episode.status()['observation_id'],
                        commands=[dict(tool=request['tool'], arguments=request['arguments'])]))
                    log.write(json.dumps(receipt) + '\n')
                except ValueError as exc:
                    rejected.append(dict(index=index, error=str(exc)))
        final = episode.status()
        initialization = adapter.seed_config
    finally:
        adapter.close()
    steps = [json.loads(line) for line in (output / 'native-steps.jsonl').read_text().splitlines()]
    comparable = len(steps) == len(actions) == len(states) - 1
    state_error, action_error = initial_error, 0.0
    if comparable:
        for index, step in enumerate(steps):
            action_error = max(action_error, float(np.max(np.abs(np.asarray(step['action']) - actions[index]))))
            state_error = max(state_error, *(float(np.max(np.abs(np.asarray(step['state'][k]) - states[index+1][k])))
                                            for k in states[index+1]))
    expected = run['result']
    passed = (comparable and state_error <= tolerance and action_error <= tolerance
              and final['success'] == expected['success']
              and final['control_step'] == expected['control_steps'])
    result = dict(diagnostic='historical-tool-replay', task=run['task'], passed=passed,
        initialization=initialization, expected_steps=expected['control_steps'],
        expected_success=expected['success'], actual=final, comparable_trace_lengths=comparable,
        initial_state_error=initial_error, max_recorded_state_error=state_error,
        max_native_action_error=action_error, tolerance=tolerance, rejected_calls=rejected,
        scope='Recorded robot state, native actions, step count and success; not complete hidden simulator state or model behaviour.')
    (output / 'parity.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    if not 0 <= args.tolerance < float('inf'):
        parser.error('tolerance must be finite and nonnegative')
    result = replay(args.reference, args.output, args.tolerance)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
