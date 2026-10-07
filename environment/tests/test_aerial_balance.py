import json
from pathlib import Path
from types import SimpleNamespace as NS
from environment.benchmarks.aerial_balance.project import configure, instructions


def test_native_disturbance_delay_and_timing_config():
    cfg=NS(seed=0,task_name='target_position',interface_name='velocity',episode_length_s=10,
           decimation=3,sim=NS(dt=1/180,device='cuda:0'),scene=NS(num_envs=500))
    sections=['target_position_task','trajectory_tracking_task','velocity_interface','position_interface',
              'thrust_interface','robustness','target_position_evaluator','trajectory_tracking_evaluator']
    for key in sections:setattr(cfg,key,NS())
    data={'env':{'episode_length_s':10},'robustness':{'external_disturbance_enabled':True,
          'action_delay_enabled':True,'delay_step':15},'velocity_interface':{'max_acc':.5,'max_velocity':0.}}
    configure(cfg,data,7)
    assert cfg.scene.num_envs==1 and cfg.seed==7
    assert cfg.robustness.delay_step==15 and cfg.robustness.external_disturbance_enabled
    assert cfg.decimation*cfg.sim.dt==1/60
    assert cfg.velocity_interface.max_velocity==0
    assert 'NOT a required1s success dwell' in instructions('T04')


def test_usd_asset_is_real_and_pinned():
    import hashlib
    root=Path(__file__).resolve().parents[2]/'third_party/benchmarks/aerial_balance'
    manifest=json.loads((root/'asset-manifest.json').read_text())
    usd=[r for r in manifest['files'] if r['path'].endswith('.usd')]
    assert usd
    for row in usd:
        data=(root/'checkout'/row['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert not data.startswith(b'version https://git-lfs')
