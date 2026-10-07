from pathlib import Path
import ast,json,importlib.util
from types import SimpleNamespace
from environment.benchmarks.wheeled_quadruped import project
W=Path(__file__).resolve().parents[1];P=W/'third_party/benchmarks/wheeled_quadruped'

def test_native_balance_config_and_assets():
    spec=importlib.util.spec_from_file_location('verify_t09',P/'prepare_assets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r=m.verify();assert len(r['files'])==1 and r['files'][0]['bytes']>10_000_000
    p=json.loads((P/'project.json').read_text());assert p['tasks']['T09']['steps']==1000
    assert p['tasks']['T09']['id']=='Wheeled-Quadruped-Balance-v0'

def test_build_not_play_and_preserves_events(monkeypatch,tmp_path):
    original=object();cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=4096),events=original)
    monkeypatch.setattr(project.importlib,'import_module',lambda _:SimpleNamespace(WheeledQuadrupedBalanceEnvCfg=lambda:cfg))
    got=project.build('T09',4,tmp_path)
    assert got.events is original and got.scene.num_envs==1 and got.seed==4
    assert 'privileged critic' in project.instruction('T09')
    assert 'scale' not in vars(got)
