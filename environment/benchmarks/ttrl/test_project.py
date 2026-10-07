from .project import ServeMetrics, actor_layout, instructions, verify_assets, termination_checks
from . import project, project_isaac6
import json
from types import SimpleNamespace
from .compat import install_global_wrench_bridge


def test_private_native_bounds_keep_strict_original_thresholds():
    assert not any(termination_checks([-1.35, 1.1, .50]).values())
    assert termination_checks([-1.6, 0., .49])['base_below_0_50m']
    assert termination_checks([-1.34, 0., .7])['base_x_above_minus1_35m']


def test_inverse_rotation_preserves_wxyz_and_batch_dimensions():
    import pytest
    torch = pytest.importorskip('torch')
    from .compat import quat_apply_inverse
    q = torch.tensor([2 ** -.5, 0., 0., 2 ** -.5]).expand(2, 3, 4)
    v = torch.tensor([1., 0., 0.]).expand(2, 3, 3)
    actual = quat_apply_inverse(q, v)
    expected = torch.tensor([0., -1., 0.]).expand_as(v)
    assert actual.shape == v.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_world_aerodynamic_wrench_keeps_frame_on_each_physics_write():
    class Tensor:
        def view(self, *shape):
            return self

    class Body:
        def __init__(self):
            self.calls = []
            self.local_writes = 0
            self._ALL_INDICES = [0]
            self.root_physx_view = SimpleNamespace(apply_forces_and_torques_at_position=lambda **kw: self.calls.append(kw))

        def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None):
            self._external_force_b, self._external_torque_b = forces, torques
            self.has_external_wrench = True

        def write_data_to_sim(self):
            self.local_writes += 1

    install_global_wrench_bridge(Body)
    body = Body()
    force, torque = Tensor(), Tensor()
    body.set_external_force_and_torque(force, torque, is_global=True)
    body.write_data_to_sim()
    body.write_data_to_sim()
    assert len(body.calls) == 2
    assert all(call['is_global'] is True and call['force_data'] is force and call['torque_data'] is torque for call in body.calls)
    body.set_external_force_and_torque(force, torque)
    body.write_data_to_sim()
    assert body.local_writes == 1
    assert len(body.calls) == 2


def test_experimental_profile_preserves_native_identity_actions_and_metrics():
    profile = json.loads((project.ROOT/'runtime-profiles/isaac6.json').read_text())
    assert not {'key', 'repository', 'commit', 'tasks'}.intersection(profile)
    assert profile['runtime_source']['commit'] == project_isaac6.LAB_COMMIT
    assert project_isaac6.Simulator.step is project.Simulator.step
    assert project_isaac6.Simulator.result is project.Simulator.result
    assert project_isaac6.Simulator.observation is project.Simulator.observation
    assert project_isaac6.Simulator.reset is project.Simulator.reset


def test_two_complete_warmup_serves_then_latched_hit_and_return():
    score=ServeMetrics()
    score.update(True,True,True)
    score.update(True,False,True)
    assert score.report()['scored_serves']==0
    score.update(True,False,False)
    score.update(False,True,False)
    score.update(False,False,True)
    report=score.report()
    assert report['finished_serves']==3
    assert report['hits']==report['valid_returns']==1
    assert report['hit_rate']==report['valid_return_rate']==1


def test_partial_serve_not_scored_and_flags_do_not_leak_across_serves():
    score=ServeMetrics()
    score.update(False,False,True)
    score.update(False,False,True)
    score.update(True,True,True)
    score.update(False,False,True)
    score.update(True,True,False)
    report=score.report()
    assert report['scored_serves']==2
    assert report['hit_rate']==report['valid_return_rate']==.5
    assert report['partial_serve_scored'] is False


def test_actor_layout_is_81_by_5_with_no_future_truth_slots():
    layout,dim=actor_layout(21,21)
    assert dim==81
    assert dim*5==405
    assert next(row for row in layout if row['name']=='ball_prediction')=={'name':'ball_prediction','start':75,'stop':78}
    assert next(row for row in layout if row['name']=='delayed_perception')['start']==69
    assert all('future' not in row['name'] for row in layout)
    assert 'DISABLED' in instructions('T02')


def test_pinned_asset_manifest_is_complete_and_matches_files():
    verify_assets()
