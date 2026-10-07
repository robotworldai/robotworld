import json
import pytest
from environment.benchmarks.volleybots import project as volley
from environment.benchmarks.omniisaacgymenvs import project as anymal

def test_volley_missing_opponent_fails_before_simulation(tmp_path,monkeypatch):
    monkeypatch.setattr(volley,'OPPONENTS',tmp_path)
    with pytest.raises(RuntimeError,match='fixed independent opponent'):volley.preflight()

def test_volley_untrusted_missing_hash_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(volley,'OPPONENTS',tmp_path)
    (tmp_path/'opponent.json').write_text(json.dumps({'file':'a.pt','sha256':'bad'}))
    (tmp_path/'a.pt').write_bytes(b'not a model')
    with pytest.raises(RuntimeError,match='SHA256'):volley.preflight()

def test_anymal_asset_manifest_available():
    anymal.preflight()

def test_interfaces_disclose_original_semantics():
    assert 'no finite clipping bound' in anymal.instructions('T14')
    assert 'player0' in volley.instructions('T05')
    assert 'no automatic volley strategy' in volley.instructions('T05')

def test_volley_synchronous_actions_and_observation_boundary(monkeypatch):
    """Exercise actual adapter step with an in-memory Torch/TensorDict seam."""
    import contextlib,sys,types
    import numpy as np
    class Tensor:
        def __init__(self,x):self.x=np.asarray(x)
        @property
        def shape(self):return self.x.shape
        def __getitem__(self,k):return Tensor(self.x[k])
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return self.x.tolist()
        def any(self):return Tensor(self.x.any())
        def all(self):return bool(self.x.all())
        def sum(self):return Tensor(self.x.sum())
        def item(self):return self.x.item()
        def reshape(self,*shape):return Tensor(self.x.reshape(*shape))
        def abs(self):return Tensor(abs(self.x))
        def __gt__(self,x):return Tensor(self.x>x)
        def __bool__(self):return bool(self.x.item())
    fake_torch=types.SimpleNamespace(no_grad=contextlib.nullcontext,float32='float32',
        tensor=lambda x,**kw:Tensor(x),isfinite=lambda x:Tensor(np.isfinite(x.x)),
        stack=lambda x,dim:Tensor(np.stack([v.x for v in x],axis=dim)))
    monkeypatch.setitem(sys.modules,'tensordict',types.SimpleNamespace(TensorDict=lambda d,**kw:d))
    sim=object.__new__(volley.Simulator)
    sim.torch=fake_torch;sim.done=False;sim.steps=0;sim.return_sum=0.;sim.dt=.02
    sim.current={('agents','observation'):Tensor([[list(range(37)),list(range(100,137))]])}
    calls=[]
    def opponent(obs):
        calls.append(('opponent',obs.tolist()))
        return Tensor([[.1,.2,.3,.4]])
    sim.opponent=opponent
    nxt={('agents','observation'):Tensor([[list(range(37)),list(range(100,137))]]),
         ('agents','reward'):Tensor([[[2.],[-2.]]]),'done':Tensor([[True]]),
         ('stats','actor_0_wins'):Tensor([[1]]),('stats','actor_1_wins'):Tensor([[0]]),
         'terminated':Tensor([[True]]),'truncated':Tensor([[False]]),'stats':{'private':Tensor([[999]])}}
    def native_step(td):
        calls.append(('native_step',td['agents']['action'].tolist()))
        return {'next':nxt}
    sim.env=types.SimpleNamespace(device='cpu',step=native_step)
    sim._capture=lambda:None
    obs=sim.step([-.1,-.2,-.3,-.4])
    assert calls==[('opponent',[list(range(100,137))]),
                   ('native_step',[[[-.1,-.2,-.3,-.4],[.1,.2,.3,.4]]])]
    assert sim.steps==1 and sim.done
    assert sim.last_evaluation['native_success'] is True
    assert obs['native_policy_observation']==list(range(37))
    assert set(obs)=={'control_step','control_dt','player','native_policy_observation'}
    with pytest.raises(RuntimeError,match='Match ended'):sim.step([0]*4)
    assert len(calls)==2
