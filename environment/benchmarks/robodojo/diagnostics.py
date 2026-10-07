"""Offline scene diagnostics; never included in model observations."""
import json
import numpy as np


def scene_snapshot(env):
    from pxr import Gf, UsdGeom
    from .compat.conveyor import snapshot
    # Lab 3 owns an in-memory stage, not necessarily the UI context stage.
    stage = env.sim.sim.stage
    objects = {}
    for name, obj in env.scene_manager.get_objects(env_ids=[0], object_type='rigid').items():
        pos, ori = obj.get_world_pose()
        objects[name] = {'position': np.asarray(pos).tolist(),
                         'orientation': np.asarray(ori).tolist(),
                         'label': obj.instance_config.get('label')}
    surfaces = []
    for prim in stage.Traverse():
        attr = prim.GetAttribute('physxSurfaceVelocity:surfaceVelocity')
        if '/dynamic/conveyor/' not in str(prim.GetPath()) or not attr:
            continue
        value = attr.Get()
        if value is None:
            continue
        local = prim.GetAttribute('physxSurfaceVelocity:surfaceVelocityLocalSpace').Get()
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        world = matrix.TransformDir(Gf.Vec3d(*value)) if local else value
        surfaces.append({'path': str(prim.GetPath()), 'local_space': local,
                         'authored_vector': list(value), 'world_vector': list(world),
                         'world_rotation': str(matrix.ExtractRotationQuat())})
    return {'env_step': int(env.take_action_cnt[0]), 'objects': objects,
            'surfaces': surfaces, 'graph': snapshot()}


def run_hold_diagnostic(env, adapter, steps, output):
    adapter.observe_content()
    before = scene_snapshot(env)
    raw = env.get_obs()
    action = {key: value for key, value in raw['state'].items()
              if key.endswith(('arm_joint_state', 'ee_joint_state'))}
    records=[]
    original_get_reward=env.reward_manager.get_reward
    latest_reward=None
    def tracked_reward(*args,**kwargs):
        nonlocal latest_reward
        result=original_get_reward(*args,**kwargs)
        latest_reward=np.asarray(result).tolist()
        return result
    env.reward_manager.get_reward=tracked_reward
    try:
        for _ in range(steps):
            ended=env.is_episode_end()
            records.append({'step':int(env.take_action_cnt[0]),'episode_ended':bool(ended),'success':bool(env.success[0]) if ended else None,'native_step_limit':int(env.step_lim),'native_reward':latest_reward})
            if ended:
                break
            env.take_action(action)
            raw = env.get_obs()
            images = {}
            for name, camera in raw['vision'].items():
                color = camera.get('color')
                if color is not None:
                    if not isinstance(color, np.ndarray) or color.ndim != 3:
                        color = adapter.decode_image(color)
                    images[name] = color
            adapter.video.write(images, int(env.take_action_cnt[0]))
        ended=env.is_episode_end()
        records.append({'step':int(env.take_action_cnt[0]),'episode_ended':bool(ended),'success':bool(env.success[0]) if ended else None,'native_step_limit':int(env.step_lim),'native_reward':latest_reward})
        (output/'predicates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        (output/'result.json').write_text(json.dumps({'diagnostic_only':True,'model_calls':0,'control_steps':int(env.take_action_cnt[0]),'native_episode_complete':bool(ended),'success':bool(env.success[0]) if ended else None,'diagnostic_budget_exhausted':not ended,'positive_predicate_fixture_verified':False},indent=2))
        after = scene_snapshot(env)
        (output / 'scene-diagnostic.json').write_text(json.dumps({
            'purpose': 'fixed robot hold, no model; not a task evaluation',
            'before': before, 'after': after}, indent=2) + '\n')
    finally:
        env.reward_manager.get_reward=original_get_reward
        adapter.video.close()
