from types import SimpleNamespace as NS
import pytest
from environment.evaluation.world_success.profiles import get_profile
from environment.evaluation.world_success.monitor import Monitor
from environment.evaluation.world_success.steadytray_events import configure_pushes, TrayPushes


def state(**changes):
    return dict(goal_distance=.25, speed=.1, object_supported=True,
                disturbances_complete=True, **changes)


def test_early_success_requires_both_pushes_then_two_elapsed_seconds():
    m = Monitor(get_profile('steadytray', 'tray_balancing_walk'))
    for i in range(401):
        s = state(); s['disturbances_complete'] = i >= 300
        # Earlier wobble/contact loss does not latch a World failure.
        s['object_supported'] = i >= 290
        s['object_tilt'] = 1.
        m.update(i, s)
        if i < 400: assert m.report(scene_verified=True)['world_success'] is None
    assert m.report(scene_verified=True)['world_success'] is True
    m.update(401, state(), native_failure=True)
    assert m.report(scene_verified=True)['world_success'] is False


def test_lost_support_resets_timer_and_missing_push_fails_at_deadline():
    p = get_profile('steadytray', 'tray_balancing_walk')
    m = Monitor(p)
    for i in range(151):
        s = state(); s['object_supported'] = i != 100
        m.update(i, s)
    assert m.report(scene_verified=True)['world_success'] is None
    m = Monitor(p)
    for i in range(1001):
        s = state(); s['disturbances_complete'] = False
        m.update(i, s)
    assert m.report(scene_verified=True)['world_success'] is False


def test_original_callbacks_once_seeded_and_no_global_rng_change():
    torch = pytest.importorskip('torch')
    calls = []
    def native(env, ids, target):
        calls.append((target, torch.rand(2).tolist()))
    cfg = NS(events=NS(**{name: NS(func=native, mode='interval', params={'target': name})
                         for name in ('push_object', 'push_robot')}))
    configure_pushes(cfg)
    assert cfg.events.push_robot is None and cfg.events.push_object is None
    sim = NS(dt=.02, steps=0, env=NS(cfg=cfg, device='cpu', num_envs=1))
    events = []; p = TrayPushes(sim, 7, events)
    assert 1 <= p.schedule['push_object'] <= 250
    assert 251 <= p.schedule['push_robot'] <= 500
    rng = torch.random.get_rng_state().clone()
    for i in range(1000):
        sim.steps = i; p.before_step()
    assert p.complete and len(calls) == 2
    assert torch.equal(rng, torch.random.get_rng_state())
    sim.steps = 0; q = TrayPushes(sim, 7, [])
    assert p.schedule == q.schedule
    torch.rand(12)
    for i in range(1000):
        sim.steps = i; q.before_step()
    assert calls[:2] == calls[2:]


@pytest.mark.parametrize('fall', [False, True])
def test_runtime_ends_early_but_native_failure_wins(tmp_path, monkeypatch, fall):
    from environment.evaluation.world_success import runtime
    class Scene:
        def __init__(self, *args):
            self.enabled=True; self.verified=True; self.goal=None
            self.failure=None; self.events=[]
        def initialize_contact(self, step): pass
        def after_reset(self): pass
        def before_step(self): pass
        def public_observation(self): return {}
    sim=NS(dt=.02, output=tmp_path, steps=0, done=False, terminated=False, last_evaluation={})
    sim.reset=lambda: None
    def step(action):
        sim.steps+=1
        if fall and sim.steps==400:sim.done=True;sim.terminated=True
    sim.step=step;sim.observation=lambda: {'actor': [1]}
    sim.result=lambda: {'success': None, 'reward': 7}
    def measure(*args):
        s=state();s['disturbances_complete']=sim.steps>=300
        return s
    monkeypatch.setattr(runtime,'Scene',Scene)
    monkeypatch.setattr(runtime,'measure',measure)
    runtime.attach(sim,'steadytray','tray_balancing_walk',7,'world-state-v1')
    sim.reset()
    for i in range(400):
        assert not sim.done
        sim.step([0])
    assert sim.done
    result=sim.result()
    assert result['success'] is (not fall)
    assert result['stop_reason']==('native_termination' if fall else 'world_success')
    assert result['reward']==7 and result['world_evaluation']['control_steps']==400
    assert sim.observation()=={'actor': [1]}
