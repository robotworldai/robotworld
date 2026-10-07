"""Isolate R1Pro RGB-D initialization from task assets and evaluator/model logic."""
import faulthandler
import signal
import json
from pathlib import Path


def main():
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    from environment.benchmarks.behavior_1k.compat import activate
    activate('/runs')
    from omegaconf import OmegaConf
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    cfg = OmegaConf.to_container(OmegaConf.load(Path(__file__).resolve().parents[1]/'robots/r1pro/behavior_ik.yaml'), resolve=True)
    cfg.pop('eval')
    cfg['obs_modalities'] = ['rgb', 'depth_linear', 'proprio']
    cfg['sensor_config']['VisionSensor']['sensor_kwargs'].update(image_height=480,image_width=480)
    print('EMPTY_R1_CAMERA_START',flush=True)
    env = og.Environment(configs={'scene':{'type':'Scene'},'robots':[cfg]})
    obs, _ = env.reset()
    (Path('/runs')/'probe.json').write_text(json.dumps({'status':'empty_r1_camera_ready','action_dim':env.scene.robots[0].action_dim}))
    print('EMPTY_R1_CAMERA_OK',flush=True)
    og.shutdown()


if __name__ == '__main__':main()
