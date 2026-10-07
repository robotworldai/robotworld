import json
from dataclasses import dataclass, field
from pathlib import Path
from . import project
from .compat import fixed_improved_friction
import pytest
from types import SimpleNamespace


def test_fork_field_guard_supports_default_factories():
    @dataclass
    class Termination:
        track_only: bool = field(default_factory=lambda: False)
        track_only_delay: float = field(default_factory=lambda: 0.)
    @dataclass
    class Observation:
        delay_min_lag: int = field(default_factory=lambda: 0)
        delay_max_lag: int = field(default_factory=lambda: 0)
    assert not hasattr(Termination, 'track_only')
    evidence = project.verify_fork_terms(Termination, Observation, Path(__file__).parent)
    assert 'track_only_delay' in evidence['Termination']['fields']


def test_removed_friction_flag_only_accepts_preservable_true_mode():
    assert fixed_improved_friction(True) is None
    with pytest.raises(RuntimeError):
        fixed_improved_friction(False)


def test_native_history_shape_metadata_is_json_serializable():
    class NativeDimension:
        # Shape scalars in the native manager may be numpy.int64, rather than
        # JSON-native Python ints. Simulate its integer conversion protocol.
        def __int__(self): return 3
    class Tensor:
        shape = (2, 3)
        def __getitem__(self, index): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return [[0., 1., 2.], [3., 4., 5.]]
    manager = SimpleNamespace(
        active_terms={name: ['body'] for name in project.policy_groups},
        group_obs_term_dim={name: [(NativeDimension(),)] for name in project.policy_groups})
    observations = SimpleNamespace(**{name: SimpleNamespace(history_length=2, flatten_history_dim=False)
                                     for name in project.policy_groups})
    env = SimpleNamespace(observation_manager=manager, cfg=SimpleNamespace(observations=observations))
    result = project.policy_observation(env, {name: Tensor() for name in project.policy_groups})
    restored = json.loads(json.dumps(result))
    assert restored['encoder']['terms'][0]['shape'] == [3]
    assert restored['policy']['values'] == [[0., 1., 2.], [3., 4., 5.]]


def test_kit6_missing_asset_root_is_resolved_without_changing_task_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(project, 'ROOT', tmp_path)
    path = tmp_path / 'assets/Isaac/Environments/example.hdr'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fixture')
    cfg = {'light': {'texture_file': 'None/Isaac/Environments/example.hdr'}, 'mass': 4.2}
    project.localize(cfg)
    assert cfg == {'light': {'texture_file': str(path)}, 'mass': 4.2}


def test_transient_object_failure_is_latched_without_done():
    history = {}
    project.latch_failures(history, {"object_fallen": True, "time_out": False})
    project.latch_failures(history, {"object_fallen": False, "time_out": True})
    assert history == {"object_fallen": True}


def test_author_fork_and_full_action_boundary():
    config = json.loads((project.ROOT / "project.json").read_text())
    assert config["runtime_source"]["commit"] == project.FORK_COMMIT
    assert all("isaaclab_checkout" in p for p in config["runtime_python_paths"])
    assert project.policy_groups == ["policy", "encoder"]
    assert config["tasks"]["T01"]["steps"] == 1000


def test_original_thresholds_present():
    text = (project.ROOT / "checkout/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py").read_text()
    assert "track_only_delay=1.0" in text
    assert '"minimum_height": 0.7' in text
    assert '"limit_angle": 0.7' in text
