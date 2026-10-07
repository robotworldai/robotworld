from pathlib import Path
import json
from types import SimpleNamespace
from environment.benchmarks.go2_push import project
W=Path(__file__).resolve().parents[1];P=W/'third_party/benchmarks/go2_push'

def test_assets_and_task_definition_pinned():
    manifest=json.loads((P/'asset-manifest.json').read_text())
    assert manifest['complete']
    assert {row['path'] for row in manifest['files']} == {
        'Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd',
        'Isaac/IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd',
        'Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
    }
    assert manifest['engine_builtins']==['OmniPBR.mdl']
    assert manifest['task_dependency']['commit']=='90b79bb2d44feb8d833f260f2bf37da3487180ba'
    cfg=(P/'checkout/src/isaaclab_go2_pushrecovery/env_cfg.py').read_text()
    assert '_IMPULSE_FORCE_START: float = 30.0' in cfg and '(6.0, 10.0)' in cfg

def test_build_keeps_native_events(monkeypatch,tmp_path):
    marker=object();cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=4096,robot=SimpleNamespace(spawn=SimpleNamespace())),events=marker,curriculum=marker)
    monkeypatch.setattr(project,'SOURCE',P/'checkout')
    monkeypatch.setattr(project.importlib,'import_module',lambda _:SimpleNamespace(UnitreeGo2PushRecoveryEnvCfg=lambda:cfg))
    got=project.build('T10',2,tmp_path)
    assert got.events is marker and got.curriculum is marker and got.seed==2
    assert got.scene.robot.spawn.usd_path.endswith('Go2/go2.usd')
    assert 'starts30N, not120N' in project.instruction('T10')
