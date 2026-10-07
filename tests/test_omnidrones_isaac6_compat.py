import importlib.util
import json
import sys

import pytest

from environment.benchmarks.omnidrones import compat_isaac6 as compat
from environment.benchmarks.omnidrones.project import SOURCE


def test_ground_physics_difference_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv('WORLD_OMNIDRONES_ALLOW_FIXED_PATCH_FRICTION', raising=False)
    assert compat.fixed_improved_friction(True) is True
    with pytest.raises(RuntimeError, match='explicit experimental'):
        compat.fixed_improved_friction(False)
    monkeypatch.setenv('WORLD_OMNIDRONES_ALLOW_FIXED_PATCH_FRICTION', '1')
    monkeypatch.setattr(compat, '_report_path', None)
    compat.configure_report(tmp_path / 'compatibility.json')
    assert compat.fixed_improved_friction(False)
    record = json.loads((tmp_path / 'compatibility.json').read_text())
    assert record['official_physics_equivalence'] is False
    difference = record['physics_differences'][0]
    assert difference['native_requested'] is False
    assert difference['effective'] is True
    assert difference['configurable_in_new_engine'] is False


def test_dataclass_default_conversion_keeps_original_values(monkeypatch):
    loader = compat._ConfigDefaults(SOURCE)
    spec = loader.find_spec('omni_drones.robots.config')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    loader.exec_module(module)
    first, second = module.RobotCfg(), module.RobotCfg()
    assert first.rigid_props.linear_damping == second.rigid_props.linear_damping == .2
    assert first.articulation_props.solver_position_iteration_count == 4
    assert first.rigid_props is not second.rigid_props


def test_model_instructions_disclose_unavoidable_physics_difference():
    from environment.benchmarks.omnidrones.project_isaac6 import instructions
    for task in ('T16', 'T17'):
        prompt = instructions(task)
        assert 'action_transform=null' in prompt
        assert 'friction=False' in prompt and 'effective=True' in prompt
        assert 'official physics equivalence' in prompt
