"""Verify force packets for two aircraft cannot overwrite each other."""
import sys
import types
import pytest
torch=pytest.importorskip('torch')
from environment.benchmarks.volleybots.force_compat_isaac6 import NativeWrenchBatch

@pytest.mark.parametrize('count',[1,2])
def test_two_aircraft_routing(count,monkeypatch):
    names=['base','rotor'];paths=[[f'/d{i}/{n}' for n in names] for i in range(count)]
    recorded=[]
    art=types.SimpleNamespace(count=count,max_links=2,link_paths=paths,
        apply_forces_and_torques_at_position=lambda *args:recorded.append(args))
    def view(n):
        return types.SimpleNamespace(_physics_view=types.SimpleNamespace(
            prim_paths=[p[n] for p in paths],get_transforms=lambda:torch.tensor([[0,0,0,0,0,0,1.]]*count)))
    drone=types.SimpleNamespace(_view=types.SimpleNamespace(_physics_view=art,_body_names=names),
        rotors_view=view(1),base_link=view(0))
    batch=NativeWrenchBatch(types.SimpleNamespace(num_envs=1,device='cpu',drone=drone))
    monkeypatch.setitem(sys.modules,'volley_bots.utils.torch',types.SimpleNamespace(quat_rotate=lambda q,v:v))
    forces=torch.arange(count*3,dtype=torch.float32).reshape(count,3)+1
    batch.queue('rotors',forces,None,None,torch.arange(count),True)
    batch.queue('base',None,forces+10,None,torch.arange(count),True)
    batch.submit()
    assert torch.equal(recorded[0][0][:,1],forces)
    assert torch.equal(recorded[0][1][:,0],forces+10)
    assert not recorded[0][0][:,0].any()
    if count==1:
        assert batch.last_submission['native_body_indices_written']==[0,1]
        assert len(batch.last_submission['world_force'])==2
    with pytest.raises(RuntimeError,match='repeated per-link'):
        batch.queue('rotors',forces,None,None,torch.arange(count),True)
    batch.clear();assert not batch.force.any() and not batch.seen

@pytest.mark.parametrize('player',[0,1])
def test_selected_player_action_and_score_routing(player,tmp_path,monkeypatch):
    from environment.benchmarks.volleybots.duel_isaac6 import Simulator
    sim=Simulator.__new__(Simulator);sim.player=player;sim.torch=torch;sim.dt=.02
    sim.steps=0;sim.done=False;sim.return_sum=0.;sim.output=tmp_path;sim._capture=lambda:None
    sim.action_metadata={'dim':4,'lower':[-1.]*4,'upper':[1.]*4}
    sim.current={('agents','observation'):torch.zeros(1,2,37)}
    sim.current[('agents','observation')][0,1,:]=1.
    seen=[]
    def opponent(obs):
        seen.append(obs.clone());return torch.full((1,4),-.2)
    sim.opponent=opponent
    clock=types.SimpleNamespace(current_time=0.)
    next_state={('agents','observation'):torch.zeros(1,2,37),'done':torch.tensor([True]),
        'terminated':torch.tensor([True]),'truncated':torch.tensor([False]),
        ('agents','reward'):torch.tensor([[[1.],[2.]]]),
        'stats':{'actor_0_wins':torch.tensor([[0.]]),'actor_1_wins':torch.tensor([[1.]])},
        ('stats','actor_0_wins'):torch.tensor([[0.]]),('stats','actor_1_wins'):torch.tensor([[1.]])}
    packets=[]
    def step(packet):
        packets.append(packet);clock.current_time+=.02;return {'next':next_state}
    sim.env=types.SimpleNamespace(device='cpu',sim=clock,step=step)
    monkeypatch.setitem(sys.modules,'tensordict',types.SimpleNamespace(TensorDict=lambda data,**kwargs:data))
    sim.step([.3]*4)
    actions=packets[0]['agents']['action']
    assert torch.allclose(actions[0,player],torch.full((4,),.3))
    assert torch.allclose(actions[0,1-player],torch.full((4,),-.2))
    assert (seen[0]==1-player).all()
    assert sim.last_evaluation['native_success']==(player==1)
    assert sim.last_evaluation['reward']==float(player+1)
    sim.seed=7;sim.random_turn=False;sim.frames=0
    sim.opponent_spec={'version':'fixed-rally-v1'}
    result=sim.result()
    assert result['outcome']==('win' if player==1 else 'loss')
    assert result['success']==(player==1)
    sim.last_evaluation.update(actor_0_wins=False,actor_1_wins=False,native_success=False)
    assert sim.result()['outcome']=='draw_or_no_winner'
    assert sim.result()['success'] is False
    sim.done=False;sim.last_evaluation['native_success']=None
    assert sim.result()['outcome']=='unfinished'
    assert sim.result()['success'] is None
    sim.opponent=lambda obs:torch.full((1,4),float('nan'))
    with pytest.raises(RuntimeError,match='invalid raw rotor'):
        sim.step([.3]*4)
    assert len(packets)==1
