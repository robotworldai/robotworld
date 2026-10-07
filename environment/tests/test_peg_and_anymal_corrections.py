from types import SimpleNamespace

from environment.benchmarks.ai_cps.scoring import peg_insertion_check
from environment.evaluation.world_success.anymal_timing import install_push_timing


def trace(z=.39, x=0.):
    rows = [[0.] * 28 for _ in range(299)]
    for row in rows:
        row[20] = z
        row[25] = x
    return rows


def test_peg_rejects_xy_only_above_hole():
    assert peg_insertion_check(trace(z=.5), True)['success'] is False


def test_peg_requires_whole_original_window():
    rows = trace()
    assert peg_insertion_check(rows, True)['success'] is True
    rows[260][25] = .101
    assert peg_insertion_check(rows, True)['success'] is False


def test_peg_short_nonfinite_and_incomplete_are_unscored():
    assert peg_insertion_check(trace()[:298], True)['success'] is None
    assert peg_insertion_check(trace(), False)['success'] is None
    rows = trace(); rows[260][20] = float('nan')
    assert peg_insertion_check(rows, True)['valid'] is False


def test_peg_inclusive_independent_bounds_and_reset_sample_ignored():
    for z in (.38, .40):
        assert peg_insertion_check(trace(z=z, x=.1), True)['success'] is True
    for z in (.37999, .40001, .43):
        assert peg_insertion_check(trace(z=z), True)['success'] is False
    assert peg_insertion_check(trace(x=.10001), True)['success'] is False
    rows = trace(); rows[0][20] = .8
    assert peg_insertion_check(rows, True)['success'] is True


def test_anymal_uses_measured_seconds_and_records_real_pushes():
    calls = []
    clock = SimpleNamespace(current_time=100.)
    task = SimpleNamespace(push_robots=lambda: calls.append(clock.current_time), push_interval=750)
    sim = SimpleNamespace(task=task, env=SimpleNamespace(_world=clock), start_time=100., steps=0)
    events = []
    install_push_timing(sim, events)
    for step in range(1, 801):
        clock.current_time = 100. + step * .025
        sim.steps = step - 1
        task.push_robots()
    assert calls == [115.]
    assert task.world_push_count == 1
    assert events[0]['control_step'] == 600
    assert events[0]['actual_time_s'] == 15.
    assert events[0]['scheduled_time_s'] == 15.


def test_anymal_reset_does_not_double_wrap_or_inherit_old_deadline():
    calls = []
    clock = SimpleNamespace(current_time=0.)
    task = SimpleNamespace(push_robots=lambda: calls.append(clock.current_time))
    sim = SimpleNamespace(task=task, env=SimpleNamespace(_world=clock), start_time=0., steps=0)
    install_push_timing(sim, [])
    clock.current_time = 15.; task.push_robots()
    sim.start_time = 20.; clock.current_time = 20.
    events = []; install_push_timing(sim, events)
    clock.current_time = 34.99; task.push_robots()
    assert task.world_push_count == 0
    clock.current_time = 35.01; task.push_robots()
    assert calls == [15., 35.01]
    assert len(events) == 1


def test_old_peg_result_is_rescored_without_rewriting_artifacts(tmp_path):
    import json
    from environment.evaluation.rollout_results import extract
    original = json.dumps({'case': '24', 'success': True, 'native_episode_complete': True})
    (tmp_path/'result.json').write_text(original)
    (tmp_path/'native_trace.json').write_text(json.dumps(trace(z=.5)))
    result = extract('ai_cps', tmp_path)
    assert result['native_success'] is True
    assert result['success'] is False
    assert result['scoring_profile'] == 'peg-xy-z-window-v2'
    assert (tmp_path/'result.json').read_text() == original


def test_peg_z_violation_inside_hold_window_fails():
    rows = trace(); rows[270][20] = .41
    assert peg_insertion_check(rows, True)['success'] is False


def test_peg_xy_is_radial_and_pose_has_no_new_threshold():
    rows = trace(x=.08)
    for row in rows:
        row[26] = .08
    assert peg_insertion_check(rows, True)['success'] is False
    rows = trace(x=.02)
    for row in rows:
        row[21:25] = [0., 1., 0., 0.]
    assert peg_insertion_check(rows, True)['success'] is True
