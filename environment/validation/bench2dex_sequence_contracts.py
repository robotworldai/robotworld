"""Native soup/desk/puzzle predicates on synthetic state sequences, no physics."""
import copy
import importlib
import json
import math
import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(WORLD / 'third_party/benchmarks/bench2dex/checkout'))
    soup = importlib.import_module('success.custom.task_76_soup_serving')
    desk = importlib.import_module('success.custom.task_80_gaming_desk_setup')
    puzzle = importlib.import_module('success.custom.task_73_jigsaw_puzzle_assembly')
    wine = importlib.import_module('success.custom.task_34_fridge_wine_interhand_pour')
    faucet = importlib.import_module('success.custom.task_67_faucet_cup_water_fill')
    microwave = importlib.import_module('success.custom.task_44_microwave_bowl_loading')
    rows = []

    def obj(x=0., y=0., z=.8, quat=(0., 0., 0., 1.)):
        return {'pose_world': [x, y, z, *quat], 'qpos': {},
                'lin_vel_world': [0., 0., 0.], 'ang_vel_world': [0., 0., 0.]}

    def check(case, label, actual, expected):
        rows.append({'case': case, 'fixture': label, 'actual': bool(actual),
                     'expected': expected, 'passed': bool(actual) == expected})

    def serve(count, returned=True, upright=True, served=True):
        states = {'obj_333_stove_1': obj(), 'obj_330_pot_1': obj(0., .24),
                  'obj_330_bowl_2': obj(-.3, .3), 'obj_330_ladle_3': obj()}
        ctx = {}

        def head(x, y, z):
            states['obj_330_ladle_3']['pose_world'][:3] = [x - .0224, y + .00004, z - .00148]
            return soup.check_success(states, ctx, {})

        for _ in range(count):
            head(0., .24, .85)
            head(-.3, .3, .9)
        if served:
            states['obj_330_bowl_2']['pose_world'][:3] = [-.47, .3, .8]
        if not upright:
            states['obj_330_bowl_2']['pose_world'][3:] = [1., 0., 0., 0.]
        return head(0., .24, .85) if returned else soup.check_success(states, ctx, {})

    check('42', 'two_transfers_return_and_serve', serve(2), True)
    check('42', 'only_one_transfer', serve(1), False)
    check('42', 'zero_transfers', serve(0), False)
    check('42', 'ladle_not_returned', serve(2, returned=False), False)
    check('42', 'bowl_not_served', serve(2, served=False), False)
    check('42', 'bowl_inverted', serve(2, upright=False), False)
    check('42', 'third_extra_transfer_not_rejected', serve(3), True)

    def wine_sequence(pour=True, close=True, upright=True):
        states = {wine.FRIDGE_ID: obj(0., 0., 1.15), wine.BOTTLE_ID: obj(0., 0., .9),
                  wine.GLASS_IDS[0]: obj(.2, 0., .9)}
        ctx = {}
        states[wine.FRIDGE_ID]['qpos']['joint_0'] = 1.5
        wine.check_success(states, ctx, {})
        wine.check_success(states, ctx, {})
        if pour:
            states[wine.BOTTLE_ID]['pose_world'][3:] = [0., math.sqrt(.5), 0., math.sqrt(.5)]
            wine.check_success(states, ctx, {})
        states[wine.BOTTLE_ID] = obj(-.046, -.019, .814,
                                    (0., 0., 0., 1.) if upright else (1., 0., 0., 0.))
        wine.check_success(states, ctx, {})
        if close:
            states[wine.FRIDGE_ID]['qpos']['joint_0'] = 0.
        return wine.check_success(states, ctx, {})

    check('44', 'open_lift_tilt_return_close', wine_sequence(), True)
    check('44', 'no_pour_orientation', wine_sequence(pour=False), False)
    check('44', 'door_left_open', wine_sequence(close=False), False)
    check('44', 'returned_bottle_inverted', wine_sequence(upright=False), False)

    def fill_sequence(leave_open_window=False, cycle=True, spoon_final=True, moving=False):
        states = {faucet.MUG: obj(.01, -.04), faucet.SPOON: obj(.01, -.04, .85, (0., -math.sqrt(.5), 0., math.sqrt(.5))),
                  faucet.FAUCET: obj(), faucet.TRAY: obj(.4, 0., .8)}
        ctx = {}
        faucet.check_success(states, ctx, {})
        if cycle:
            states[faucet.FAUCET]['qpos']['joint_0'] = 1.2
            faucet.check_success(states, ctx, {})
            if leave_open_window:
                states[faucet.MUG]['pose_world'][0] = .5
                faucet.check_success(states, ctx, {})
                states[faucet.MUG]['pose_world'][0] = .01
            states[faucet.FAUCET]['qpos']['joint_0'] = 0.
            faucet.check_success(states, ctx, {})
        states[faucet.MUG]['pose_world'][:3] = [.4, 0., .82]
        states[faucet.SPOON]['pose_world'][:3] = [.4 if spoon_final else 1., 0., .87]
        if moving:
            states[faucet.MUG]['lin_vel_world'] = [.06, 0., 0.]
        return faucet.check_success(states, ctx, {})

    check('45', 'spoon_fill_then_serve', fill_sequence(), True)
    check('45', 'cup_leaves_open_window', fill_sequence(leave_open_window=True), False)
    check('45', 'skip_tap_cycle', fill_sequence(cycle=False), False)
    check('45', 'spoon_missing_from_final_cup', fill_sequence(spoon_final=False), False)
    check('45', 'cup_moving_on_tray', fill_sequence(moving=True), False)

    loaded = {'obj_104_microwave_2': obj(), 'obj_024_bowl_3': obj(),
              'obj_163_baguette_1': obj(-.02225925 - .0000367, -.0655335 - .0225)}
    check('46', 'loaded_and_closed_without_prior_open_history', microwave.check_success(loaded, {}, {}), True)
    for label, mutate in [
        ('door_open', lambda s: s['obj_104_microwave_2']['qpos'].__setitem__('joint_0', .5)),
        ('bread_outside_bowl', lambda s: s['obj_163_baguette_1']['pose_world'].__setitem__(0, .3)),
        ('bowl_inverted', lambda s: s['obj_024_bowl_3']['pose_world'].__setitem__(slice(3, 7), [1., 0., 0., 0.])),
        ('bread_moving', lambda s: s['obj_163_baguette_1'].__setitem__('lin_vel_world', [.06, 0., 0.])),
    ]:
        changed = copy.deepcopy(loaded)
        mutate(changed)
        check('46', label, microwave.check_success(changed, {}, {}), False)

    def desk_sequence(release_esc=True, release_mouse=True, aligned=True, move_after=False):
        states = {'obj_090_display_1': obj(0., .2, .8, (math.sqrt(.5), 0., 0., math.sqrt(.5))),
                  'obj_098_keyboard_2': obj(), 'obj_300_mouse_3': obj(0., -.1)}
        ctx = {}
        if not aligned:
            states['obj_090_display_1']['pose_world'][3:] = [0., 0., 0., 1.]
        desk.check_success(states, ctx, {})
        states['obj_098_keyboard_2']['qpos']['joint_100'] = .006
        desk.check_success(states, ctx, {})
        if release_esc:
            states['obj_098_keyboard_2']['qpos']['joint_100'] = 0.
            desk.check_success(states, ctx, {})
        states['obj_300_mouse_3']['qpos']['joint_0'] = .008
        desk.check_success(states, ctx, {})
        if release_mouse:
            states['obj_300_mouse_3']['qpos']['joint_0'] = 0.
        actual = desk.check_success(states, ctx, {})
        if move_after:
            states['obj_090_display_1']['pose_world'] = [10., 10., .8, 0., 0., 0., 1.]
            actual = desk.check_success(states, ctx, {})
        return actual

    check('47', 'align_esc_cycle_mouse_cycle', desk_sequence(), True)
    check('47', 'esc_held_without_release', desk_sequence(release_esc=False), False)
    check('47', 'mouse_held_without_release', desk_sequence(release_mouse=False), False)
    check('47', 'monitor_never_aligned', desk_sequence(aligned=False), False)
    check('47', 'terminal_latched_after_monitor_moved_away', desk_sequence(move_after=True), True)

    states = {name: obj(x, y) for name, (x, y) in puzzle._PUZZLE_TARGETS.items()}
    states['obj_326_piece_1'] = obj(-.1, .25)
    check('48', 'four_pieces_at_targets', puzzle.check_success(states, {}, {}), True)
    for label, mutate, expected in [
        ('xy_19mm_inside', lambda s: s['obj_326_piece_2']['pose_world'].__setitem__(0, -.065 + .019), True),
        ('xy_21mm_outside', lambda s: s['obj_326_piece_2']['pose_world'].__setitem__(0, -.065 + .021), False),
        ('height_11mm_above_center', lambda s: s['obj_326_piece_2']['pose_world'].__setitem__(2, .811), False),
        ('piece_rotated_180_not_checked', lambda s: s['obj_326_piece_2']['pose_world'].__setitem__(slice(3, 7), [0., 0., 1., 0.]), True),
        ('piece_not_static', lambda s: s['obj_326_piece_2'].__setitem__('lin_vel_world', [.061, 0., 0.]), False),
        ('missing_outer_piece', lambda s: s.pop('obj_326_piece_2'), False),
    ]:
        changed = copy.deepcopy(states)
        mutate(changed)
        check('48', label, puzzle.check_success(changed, {}, {}), expected)
    result = {'scope': 'Exact original custom functions, synthetic positions/joints/velocities; '
                       'not simulator contact or physical task verification.',
              'tests': rows, 'total': len(rows), 'passed': sum(row['passed'] for row in rows)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] == result['total'] else 1)


if __name__ == '__main__':
    main()
