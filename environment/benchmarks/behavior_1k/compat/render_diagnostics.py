"""Opt-in timing of native render calls; never changes scene or render settings."""
import json
import os
from pathlib import Path
import time

_sequence = 0


def capture_sequence(robot, config, output, frames=60):
    """Render stationary cameras without physics steps, exposure edits or image enhancement."""
    import imageio.v2 as imageio
    import numpy as np
    import omnigibson as og
    from PIL import Image
    output = Path(output) / 'render-probe'
    output.mkdir(exist_ok=True)
    writers, sensors, timings, statistics = {}, {}, [], {}
    try:
        for role, name in config['eval']['camera_sensor_names'].items():
            sensors[role] = robot.sensors[name]
            writers[role] = imageio.get_writer(str(output / f'{role}.mp4'), fps=30, codec='libx264', macro_block_size=1)
        for index in range(frames):
            start = time.monotonic()
            og.sim.render()
            timings.append(time.monotonic() - start)
            for role, sensor in sensors.items():
                rgb = sensor.get_obs()[0]['rgb']
                if hasattr(rgb, 'detach'):
                    rgb = rgb.detach().cpu().numpy()
                rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
                writers[role].append_data(rgb)
                if index in (0, frames - 1):
                    Image.fromarray(rgb).save(output / f'{role}-{index:03d}.png')
                    statistics[f'{role}-{index}'] = {'mean': float(rgb.mean()), 'max': int(rgb.max()),
                                                    'shape': list(rgb.shape)}
    finally:
        for writer in writers.values():
            writer.close()
    (output / 'summary.json').write_text(json.dumps({'frames': frames, 'physics_steps': 0,
        'purpose': 'stationary render stability; not a task episode', 'render_seconds': timings,
        'rgb_statistics': statistics}, indent=2))


def snapshot(output):
    """Read-only evidence of effective settings and authored light/camera attributes."""
    import carb.settings
    import omnigibson as og
    from pxr import UsdGeom, UsdLux
    settings = carb.settings.get_settings()
    values = {key: settings.get(key) for key in (
        '/rtx/rendermode', '/rtx/rtx/modes/rt2/enabled',
        '/persistent/rtx/modes/rt2/enabled', '/rtx/modes/rt2/enabled',
        '/rtx/post/tonemap/op', '/rtx/post/tonemap/cameraShutter',
        '/rtx/post/tonemap/fNumber', '/rtx/post/tonemap/filmIso',
        '/rtx/post/histogram/enabled', '/rtx/indirectDiffuse/enabled',
    )}
    prims = []
    xforms = UsdGeom.XformCache()
    links = {link.prim_path: link for scene in og.sim.scenes for obj in scene.objects for link in obj.links.values()}
    for prim in og.sim.stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdGeom.Camera) or str(prim.GetTypeName()) in ('RenderSettings', 'RenderProduct'):
            record = {'path': str(prim.GetPath()), 'type': str(prim.GetTypeName()),
                      'attributes': {str(a.GetName()): str(a.Get()) for a in prim.GetAttributes()}}
            if prim.HasAPI(UsdLux.LightAPI):
                record['usd_world_position'] = list(xforms.GetLocalToWorldTransform(prim).ExtractTranslation())
                record['effective_visibility'] = str(UsdGeom.Imageable(prim).ComputeVisibility())
                parent = prim.GetParent()
                while parent and str(parent.GetPath()) != '/':
                    if str(parent.GetPath()) in links:
                        link = links[str(parent.GetPath())]
                        pos, quat = link.get_position_orientation()
                        record['parent_link'] = {'path': str(parent.GetPath()), 'position': pos.tolist(), 'quaternion': quat.tolist(),
                                                 'usd_position': list(xforms.GetLocalToWorldTransform(parent).ExtractTranslation())}
                        break
                    parent = parent.GetParent()
            prims.append(record)
    (Path(output) / 'render-state.json').write_text(json.dumps({'settings': values, 'prims': prims}, indent=2))


def render(context):
    global _sequence
    path = Path(os.environ['WORLD_BEHAVIOR_RENDER_TRACE'])
    _sequence += 1
    sequence = _sequence
    start = time.monotonic()
    record = {'event': 'render_start', 'sequence': sequence, 'monotonic': start}
    if sequence == 1:
        import carb.settings
        settings = carb.settings.get_settings()
        record['settings'] = {key: settings.get(key) for key in (
            '/rtx/rendermode', '/rtx/rtx/modes/rt2/enabled',
            '/rtx/post/dlss/execMode', '/rtx/indirectDiffuse/enabled',
            '/app/renderer/skipMaterialLoading',
        )}
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    try:
        return context.render()
    finally:
        with path.open('a') as stream:
            stream.write(json.dumps({'event': 'render_return', 'sequence': sequence,
                                     'elapsed_seconds': time.monotonic() - start}) + '\n')
