"""Executable counterpart of the approved v0.3 designs; all quantities are SI.

Changing a threshold requires a new version. Native benchmark scores stay separate.
"""
from math import radians
VERSION = 'world-state-v1'


def condition(field, op, value):
    return (field, op, value)


def rule(name, operator, *conditions, seconds=None, window=None):
    return dict(id=name, operator=operator, conditions=list(conditions), seconds=seconds, window=window)


def hold(name, seconds, *conditions):
    return rule(name, 'hold', *conditions, seconds=seconds)


def final(name, *conditions):
    return rule(name, 'final', *conditions)


def always(name, window, *conditions):
    return rule(name, 'always', *conditions, window=window)


def target(distance, speed, tilt=None):
    values = [('goal_distance', '<=', distance), ('speed', '<=', speed)]
    if tilt is not None: values.append(('tilt', '<=', radians(tilt)))
    return values


PROFILES = {}
def add(bench, task, steps, hz, *checks, event_end=False):
    PROFILES[bench + '/' + task] = dict(benchmark=bench, task=task, steps=steps, hz=hz,
        duration=steps/hz, checks=list(checks), event_end=event_end, version=VERSION)

add('go2_push', 'quadruped_push_recovery', 1000, 50,
    hold('S01', 2, *target(.25, .10, 20)))
add('omniisaacgymenvs', 'anymal_rough_terrain', 800, 40,
    final('S01', ('route_complete', '==', True)), hold('S02', 2, *target(.30, .10, 20)))
add('wheel_legged', 'wheel_legged_rough_terrain', 2000, 100,
    final('S01', ('route_complete', '==', True)),
    hold('S02', 2, *target(.20, .08), ('tilt', '<', .25), ('clearance', '>', .14)))
add('steadytray', 'tray_balancing_walk', 1000, 50,
    hold('S01', 2, *target(.25, .10), ('object_supported', '==', True),
         ('disturbances_complete', '==', True)))
PROFILES['steadytray/tray_balancing_walk'].update(version='steadytray-recovery-v2', early_success=True)
HANDS = [('left_wrist_position_error','<=',.08), ('right_wrist_position_error','<=',.08),
         ('left_wrist_rotation_error','<=',radians(15)), ('right_wrist_rotation_error','<=',radians(15))]
add('digit', 'digit_walk_hand_tracking', 700, 50,
    always('S01', (2, 12), *HANDS), hold('S02', 2, *target(.25, .10), *HANDS))
add('robot_lab', 'a1_front_leg_handstand', 500, 50,
    hold('S01', 3, ('front_feet_supported','==',True), ('rear_feet_clear','==',True),
         ('rear_foot_min_clearance','>=',.40), ('handstand_gravity_error','<=',radians(15))))
add('wheeled_quadruped', 'rear_wheel_upright_balance', 1000, 50,
    hold('S01', 3, ('rear_wheels_supported','==',True), ('front_wheels_clear','==',True),
         ('other_body_support','==',False), ('base_height_error','<=',.08), ('tilt','<=',radians(15))))
PROFILES['wheeled_quadruped/rear_wheel_upright_balance']['version']='rear-wheel-balance-v2'
add('flamingo', 'wheel_legged_jump_balance', 1000, 50,
    final('S01', ('completed_jumps','==',2)),
    hold('S02', 2, ('both_wheels_supported','==',True), ('tilt','<=',radians(15)), ('abs_vertical_speed','<=',.15)))
for task in ['mushr-drift', 'f1tenth-drift']:
    add('wheeledlab', task, 400, 50,
        final('S01', ('lap_complete','==',True)), final('S02', ('drift_corners','>=',2)))
add('wheeledlab', 'visual', 150, 5,
    final('S01', ('route_complete','==',True)),
    hold('S02', 2, ('whole_car_in_goal','==',True), *target(.20,.08), ('heading_error','<=',radians(15))))
add('omnidrones', 'drone_payload_hover', 500, 62,
    hold('S01', 2, ('payload_error','<=',.15), ('payload_speed','<=',.20),
         ('swing','<=',radians(10)), ('drone_tilt','<=',radians(15))))
add('omnidrones', 'drone_inverted_pendulum_tracking', 620, 62,
    always('S01', (.5,8.5), ('tip_error','<=',.20), ('rod_tilt','<=',radians(20))),
    hold('S02', 1, ('tip_error','<=',.10), ('tip_speed','<=',.20), ('rod_tilt','<=',radians(10))))
add('volleybots', 'drone_volleyball_solo_juggle', 800, 50,
    final('S01', ('num_height_hits','>=',4)),
    hold('S02', .6, ('ball_z','>',.15), ('ball_support_contact','==',False)))
add('ttrl', 'humanoid_table_tennis_return', 1500, 50,
    final('S01', ('scored_serves','==',5), ('valid_returns','>=',3)), event_end=True)
add('wheel_legged', 'wheel_legged_upright_recovery', 2000, 100,
    final('S01', ('recovered_attempts','==',4)),
    hold('S02', 1, ('tilt','<',.25), ('clearance','>',.14)))


def get_profile(benchmark, task):
    from environment.evaluation.task_names import canonical
    return PROFILES.get(benchmark + '/' + canonical(benchmark, task))
