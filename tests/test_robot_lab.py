from pathlib import Path
import ast,json,importlib.util
from types import SimpleNamespace
from environment.benchmarks.robot_lab import project
W=Path(__file__).resolve().parents[1];P=W/'third_party/benchmarks/robot_lab'

def test_source_horizon_overrides_material_summary():
    path=P/'checkout/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/rough_env_cfg.py'
    source=path.read_text()
    assert 'self.episode_length_s = 10.0' in source
    assert 'handstand_type = "back"' in source
    assert json.loads((P/'project.json').read_text())['tasks']['T11']['steps']==500
    spec=importlib.util.spec_from_file_location('t11assets',P/'prepare_assets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert len(m.verify()['files'])==26

def test_config_not_retimed_or_actions_replaced(monkeypatch,tmp_path):
    marker=object();cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=4096),episode_length_s=10.,actions=marker,events=marker)
    monkeypatch.setattr(project.importlib,'import_module',lambda _:SimpleNamespace(UnitreeA1HandStandFlatEnvCfg=lambda:cfg))
    r=project.build('T11',3,tmp_path)
    assert r.events is marker and r.actions is marker and r.episode_length_s==10
    assert '10s=500steps' in project.instruction('T11')
