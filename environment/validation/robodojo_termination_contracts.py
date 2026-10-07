"""Run the original episode-end method on synthetic reward/count boundary inputs."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

WORLD = Path(__file__).resolve().parents[2]


def main():
    source = WORLD/'third_party/benchmarks/RoboDojo/src/eval_client/eval_env.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'is_episode_end')
    namespace = {'deepcopy':deepcopy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    native = namespace['is_episode_end']
    rows = []
    cases = [
        ('below_limit_incomplete', 799, 0., True, False, False, True, False),
        ('reward_exact_0999_not_success', 799, .999, True, False, False, True, False),
        ('reward_above_0999_success', 799, .9991, True, False, True, True, False),
        ('exact_limit_incomplete_failure', 800, 0., True, False, True, False, True),
        ('exact_limit_final_goal_success', 800, 1., True, False, True, True, True),
        ('native_failure_before_limit', 10, 0., False, False, True, False, True),
        ('already_ended_preserved', 10, 0., True, True, True, True, False),
    ]
    for label, count, reward, initial_success, initial_end, expected_end, expected_success, expected_final in cases:
        calls = []
        frames = []
        def get_reward(final_check):
            calls.append(final_check)
            return [reward]
        env = SimpleNamespace(num_envs=1, take_action_cnt=[count], step_lim=800,
                              success=[initial_success], end_flag=[initial_end],
                              reward_manager=SimpleNamespace(get_reward=get_reward),
                              get_obs_batch=lambda **kwargs:frames.append(kwargs))
        ended = native(env)
        expected_frames = 1 if expected_end != initial_end else 0
        actual = {'ended':ended, 'success':env.success[0], 'final_check':calls[-1], 'last_frame_calls':len(frames)}
        expected = {'ended':expected_end, 'success':expected_success, 'final_check':expected_final, 'last_frame_calls':expected_frames}
        rows.append({'fixture':label, 'actual':actual, 'expected':expected, 'passed':actual == expected})
    result = {'scope':'Untouched native is_episode_end function; synthetic reward leaves and frame sink, no physical proof.',
              'source':str(source.relative_to(WORLD)), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'tests':rows, 'total':len(rows), 'passed':sum(row['passed'] for row in rows)}
    output = WORLD/'reports/validation/2026-09-29/robodojo-termination-contracts.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'total':result['total'], 'passed':result['passed']}))
    raise SystemExit(0 if result['total']==result['passed'] else 1)


if __name__ == '__main__':
    main()
