from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,sys
import numpy as np
from environment.benchmarks.flamingo import project
W=Path(__file__).resolve().parents[1];P=W/'third_party/benchmarks/flamingo'

class Tensor:
    def __init__(self,v):self.array=np.asarray(v)
    @property
    def shape(self):return self.array.shape
    def clone(self):return Tensor(self.array.copy())
    def __getitem__(self,i):return Tensor(self.array[i])
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.array.tolist()
    def item(self):return self.array.item()

def test_original_actor_history_excludes_critic(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cat=lambda tensors,dim:Tensor(np.concatenate([t.array for t in tensors],axis=dim))))
    monkeypatch.setattr(project,'SOURCE',P/'checkout')
    env=SimpleNamespace(episode_length_buf=Tensor([0]))
    first={'stack_policy':Tensor([[1.,2.]]),'none_stack_policy':Tensor([[3.,4.]]),'stack_critic':Tensor([[999.]])}
    out=project.policy_observation(env,first)
    assert out['actor_vector']==[1,2,1,2,1,2,3,4]
    env.episode_length_buf=Tensor([1]);first['stack_policy']=Tensor([[5,6]])
    second=project.policy_observation(env,first)
    assert second['actor_vector']==[5,6,1,2,1,2,3,4]
    assert '999' not in str(second)

def test_unmodified_zip_assets_and_correct_rev():
    spec=importlib.util.spec_from_file_location('flamingo_assets',P/'prepare_assets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert len(m.verify()['files'])==2
    manifest=json.loads((P/'asset-manifest.json').read_text())
    assert len(manifest['extracted_files'])==5
    assert sum(r['bytes'] for r in manifest['extracted_files'])==44821051
    assert 'rev01_5_2' in project.instruction('T15')
    assert 'gear_ratio=-1.5' in project.instruction('T15')

def test_original_config_not_play(monkeypatch,tmp_path):
    marker=object();cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=4096,robot=SimpleNamespace(spawn=SimpleNamespace())),events=marker,observations=marker)
    monkeypatch.setattr(project,'SOURCE',P/'checkout')
    monkeypatch.setattr(project.importlib,'import_module',lambda _:SimpleNamespace(FlamingoFlatEnvCfg=lambda:cfg))
    got=project.build('T15',2,tmp_path)
    assert got.events is marker and got.observations is marker and got.seed==2
    assert 'checkout' not in got.scene.robot.spawn.usd_path
