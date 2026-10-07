from copy import deepcopy
from environment.validation.driving_line_audit import bridge_violations
from third_party.benchmarks.wheeledlab.robotworld.precision_specs import scenario


def state():
    return {'wheel_positions': {f'{end}_{side}_wheel_link': [x, y, .35]
        for end, x in [('front', 1.3), ('back', 1.)]
        for side, y in [('left', .115), ('right', -.115)]}}


def test_bridge_inside_and_outside_support_intervals():
    spec = scenario('rw-twin-beam')
    s = state()
    assert bridge_violations(spec, s) == []
    for direction in [-1., 1.]:
        altered = deepcopy(s)
        altered['wheel_positions']['front_left_wheel_link'][1] += direction * .02
        assert bridge_violations(spec, altered) == ['tire_touched_rail_edge:front_left_wheel_link']


def test_bridge_drop_and_x_region_are_independent():
    spec = scenario('rw-twin-beam')
    s = state()
    s['wheel_positions']['back_right_wheel_link'][2] = .30
    assert bridge_violations(spec, s) == ['wheel_dropped_from_beam:back_right_wheel_link']
    s['wheel_positions']['back_right_wheel_link'][0] = -1.
    assert bridge_violations(spec, s) == []


def test_parking_scene_has_no_bridge_obligation():
    assert bridge_violations(scenario('rw-reverse-bay'), {}) == []
