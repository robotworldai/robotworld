"""Adapter contracts without changing/resetting a physical simulator."""
from types import SimpleNamespace
import pytest
from environment.service.ai_cps import AICPSAdapter
from environment.service.wheeledlab import WheeledLabAdapter


@pytest.mark.parametrize('adapter,tool,field,action', [
    (AICPSAdapter,'move_joints','arm_action',[0.1]*7),
    (WheeledLabAdapter,'drive','action',[0.1,-0.2]),
])
def test_prepare_preserves_original_actions(adapter,tool,field,action):
    obj=adapter.__new__(adapter)
    result=obj.prepare([{'tool':tool,'arguments':{'note':'test','steps':12,field:action}}])
    assert result == [(action+([0.,0.] if adapter is AICPSAdapter else []),12)]
    for bad in [True,0,51,1.5]:
        with pytest.raises(ValueError):obj.prepare([{'tool':tool,'arguments':{'note':'test','steps':bad,field:action}}])
    with pytest.raises(ValueError):obj.prepare([{'tool':'coding_control','arguments':{'code':'x'}}])
    with pytest.raises(ValueError):obj.prepare([{'tool':tool,'arguments':{'note':'x','steps':1,field:[float('nan')]*len(action)}}])


def test_wheeled_observation_does_not_use_review_camera():
    adapter=WheeledLabAdapter.__new__(WheeledLabAdapter)
    adapter.instruction='drive'
    adapter.sim=SimpleNamespace(observation=lambda:{'encoder':1},camera=SimpleNamespace(image='private'))
    adapter.history=[[{'camera':'front','control_step':i}] for i in range(9)]
    result=adapter.observe()
    assert result['state']=={'encoder':1}
    assert [i['control_step'] for i in result['images']]==[0,2,4,6,8]
    assert 'private' not in str(result)


def test_wheeled_step_preserves_world_success():
    import io
    adapter=WheeledLabAdapter.__new__(WheeledLabAdapter)
    adapter.sim=SimpleNamespace(step=lambda x:None,steps=20,done=True,truncated=False,result=lambda:{'success':True})
    adapter.capture=lambda:None
    adapter.log=io.StringIO()
    assert adapter.step([0,0])==dict(success=True,terminated=False,truncated=False)
    adapter.sim.result=lambda:{'success':False}
    assert adapter.step([0,0])==dict(success=False,terminated=True,truncated=False)
