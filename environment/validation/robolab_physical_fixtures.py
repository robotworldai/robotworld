"""Injected simulator-state predicate fixtures, never a robot policy evaluation.

Only fixture instances are repositioned. Original source, predicates, collision
settings and assets remain unchanged. Record every short settling physics step.
"""
import argparse
import json
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--isaac601', action='store_true')
    parser.add_argument('--steps', type=int, default=30)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if args.isaac601:
        import isaacsim
        args.experience = str(Path(isaacsim.__file__).resolve().parent/'apps/isaacsim.exp.full.kit')
    app = AppLauncher(args).app
    env = None
    writers = {}
    out = Path('/runs')
    rows = []
    try:
        if args.isaac601:
            import sys
            sys.path.insert(0, '/compat')
            from isaac601 import install
            install(exact_contact_paths=True)
        import cv2
        import numpy as np
        import torch
        from isaaclab.utils.math import quat_apply
        from robolab.constants import set_output_dir
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        set_output_dir('/runs/official')
        auto_register_droid_envs(task=[args.task])
        env, cfg = create_env(get_envs(task=[args.task])[0], device=args.device, num_envs=1, use_fabric=True)
        env.reset()
        world = get_world(env)
        robot = env.scene['robot']
        robot.set_joint_position_target(robot.data.joint_pos.clone())
        term = cfg.terminations.success
        frames = 0
        snapshots = {}
        placements = {}
        mug_geometry = {}

        def initial(name):
            if name not in snapshots:
                snapshots[name] = world.get_body(name).data.root_state_w.clone()
            return snapshots[name].clone()

        def place(name, state):
            body = world.get_body(name)
            placements[name] = state[:, :7].detach().cpu().tolist()
            body.write_root_pose_to_sim(state[:, :7])
            body.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]))

        def fixture(label, expected):
            nonlocal frames
            world.reset_predicate_state(torch.tensor([0], device=env.device))
            values = []
            for _ in range(args.steps):
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                obs = env.observation_manager.compute()
                actual = bool(term.func(env, **term.params)[0])
                values.append(actual)
                for key, value in obs['image_obs'].items():
                    rgb = value[0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
                    if key not in writers:
                        writers[key] = cv2.VideoWriter(str(out/(key+'-physics-fixtures.mp4')),
                            cv2.VideoWriter_fourcc(*'mp4v'), 1/env.physics_dt, (rgb.shape[1], rgb.shape[0]))
                        if not writers[key].isOpened():raise RuntimeError('Cannot create fixture video')
                    writers[key].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frames += 1
            states = {name: world.get_body(name).data.root_state_w.detach().cpu().tolist() for name in snapshots}
            rows.append({'task':args.task, 'fixture':label, 'expected':expected, 'actual':values[-1],
                         'passed':values[-1] == expected, 'native_values_by_physics_step':values,
                         'last_frame':frames, 'injected_root_poses':dict(placements), 'body_states':states})
            (out/'physical-fixtures.json').write_text(json.dumps({'tests':rows,'complete':False},indent=2))

        if args.task == 'RubiksCubeLeftOfBowlTask':
            cube = initial('rubiks_cube')
            bowl = initial('bowl')
            left = quat_apply(robot.data.root_quat_w, torch.tensor([[0.,1.,0.]], device=env.device))
            for side, expected in [(1.,True),(-1.,False),(1.,True)]:
                pose = cube.clone()
                pose[:, :2] = bowl[:, :2] + .25*side*left[:, :2]
                place('rubiks_cube', pose)
                fixture('robot_relative_left' if side > 0 else 'robot_relative_right', expected)
        elif args.task == 'ReorientWhiteMugsTask':
            names = ['upright_white_mug','sideways_white_mug']
            for name in names:
                pose = initial(name)
                corners = world._get_local_geometry(name)['corners'].clone()
                initial_bottom = (quat_apply(pose[:,3:7].expand(len(corners),-1),corners)[:,2].min()+pose[0,2]).item()
                mug_geometry[name] = (corners,initial_bottom)
            for inverted in [None,names[0],names[1]]:
                for name in names:
                    pose = initial(name)
                    pose[:,3:7] = torch.tensor([1.,0.,0.,0.] if name != inverted else [0.,1.,0.,0.],device=env.device)
                    # Start each orientation just above its original support plane.
                    # Rotating at a fixed root height can inject deep table penetration.
                    corners,initial_bottom = mug_geometry[name]
                    rotated_bottom = quat_apply(pose[:,3:7].expand(len(corners),-1),corners)[:,2].min()
                    pose[:,2] = initial_bottom + .01 - rotated_bottom
                    place(name,pose)
                fixture('all_upright' if inverted is None else 'inverted_'+inverted,inverted is None)
        else:
            raise ValueError('No physical fixture defined for '+args.task)
        report = {'complete':True,'task':args.task,'tests':rows,'passed':all(row['passed'] for row in rows),
                  'scope':'Injected real rigid-body states plus short native physics settling; '
                          'not a manipulation policy or general physical-equivalence proof.',
                  'physics_steps_per_fixture':args.steps,'physics_dt':env.physics_dt,'frames':frames,'model_calls':0}
        (out/'physical-fixtures.json').write_text(json.dumps(report,indent=2))
        (out/'probe.json').write_text(json.dumps({'fixture_only':True,'passed':report['passed']}))
        if not report['passed']:raise RuntimeError('Native physical fixture verdict mismatch; inspect physical-fixtures.json')
    except BaseException:
        import traceback
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        for writer in writers.values():writer.release()
        if env is not None:env.close()
        app.close()


if __name__ == '__main__':
    main()
