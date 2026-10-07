import pytest
from environment.benchmarks.omnidrones.project import validate_action, observation_layout, instructions

def test_rotor_validation():
    assert validate_action([-1,0,.5,1])==[-1,0,.5,1]
    for bad in ([0]*3,[True,0,0,0],[float('nan'),0,0,0],[2,0,0,0]):
        with pytest.raises(ValueError): validate_action(bad)

def test_native_layouts_and_no_hover_controller():
    assert sum(t['size'] for t in observation_layout('T16')) == 36
    assert sum(t['size'] for t in observation_layout('T17')) == 51
    assert 'action_transform=null' in instructions('T16')
    assert 'No extra wind' in instructions('T17')
