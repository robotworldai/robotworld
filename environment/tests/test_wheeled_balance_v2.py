from types import SimpleNamespace as NS
import math

import pytest

from environment.evaluation.world_success.profiles import get_profile
from environment.evaluation.world_success.monitor import Monitor
from environment.evaluation.world_success.scenes import Scene
from environment.evaluation.world_success.runtime import instruction
from environment.evaluation.world_success.wheeled_balance import SinglePush, configure_push


def state(stable=True):
    return dict(rear_wheels_supported=stable, front_wheels_clear=True,
                other_body_support=False, base_height_error=.08, tilt=math.radians(15))


@pytest.mark.parametrize('bad_step,success', [(849, True), (850, False), (999, False), (1000, False), (None, True)])
def test_final_three_seconds(bad_step, success):
    m = Monitor(get_profile('wheeled_quadruped', 'rear_wheel_upright_balance'))
    for step in range(1001):
        m.update(step, state(step >= 850 and step != bad_step))
    report = m.report(scene_verified=True)
    assert report['world_success'] is success
    assert report['version'] == 'rear-wheel-balance-v2'


def test_early_fall_still_fails_and_short_run_cannot_pass():
    p = get_profile('wheeled_quadruped', 'rear_wheel_upright_balance')
    for fall in (False, True):
        m = Monitor(p)
        for step in range(65):
            m.update(step, state(), native_failure=fall and step == 64)
        assert m.report(scene_verified=True)['world_success'] is (False if fall else None)


def test_leaving_practice_circle_is_not_failure():
    p = get_profile('wheeled_quadruped', 'rear_wheel_upright_balance')
    scene = Scene(NS(env=NS()), p, 7, True)
    scene.origin = [0., 0., 0.]
    scene.spatial([5., 5., 0.], {})
    assert scene.failure is None
    prompt = instruction(p)
    assert 'final 17-20 seconds' in prompt
    assert 'For the final 10 seconds' not in prompt


def test_single_seeded_native_callback_preserves_rng_and_parameters():
    torch = pytest.importorskip('torch')
    calls = []
    def native(env, ids, velocity_range):
        calls.append((ids.tolist(), velocity_range, torch.rand(2).tolist()))
    params = {'velocity_range': {'x': (-.5, .5), 'y': (-.5, .5)}}
    term = NS(func=native, mode='interval', params=params)
    cfg = NS(events=NS(push_robot=term), seed=7)
    configure_push(cfg)
    assert cfg.events.push_robot is None
    assert cfg.world_balance_push.func is native
    sim = NS(dt=.02, steps=0, env=NS(cfg=cfg, device='cpu', num_envs=1))
    events = []
    push = SinglePush(sim, 7, events)
    assert 1 <= push.step <= 500
    assert push.step == SinglePush(sim, 7, []).step
    rng = torch.random.get_rng_state().clone()
    for step in range(1000):
        sim.steps = step
        push.before_step()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(calls) == 1 and calls[0][:2] == ([0], params['velocity_range'])
    assert events[-1]['time_s'] <= 10
    sim.steps = 0
    second = SinglePush(sim, 7, [])
    torch.rand(20)
    sim.steps = second.step
    second.before_step()
    assert calls[0] == calls[1]
