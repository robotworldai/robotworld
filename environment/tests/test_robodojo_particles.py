import pytest
from environment.benchmarks.robodojo.compat.particles import material_wrapper

def test_only_absent_aerodynamic_parameters_can_be_omitted():
 recorded=[]
 call=material_wrapper(lambda **kw:recorded.append(kw))
 call(stage='s',path='p',drag=None,lift=None,damping=.99,viscosity=.0091)
 assert recorded==[{'stage':'s','path':'p','damping':.99,'viscosity':.0091}]
 for value in [0.,.1,-.1]:
  with pytest.raises(RuntimeError,match='refusing to discard'):
   call(drag=value)
 assert len(recorded)==1
