"""Recording-only overlay; executes the unmodified native EEF worker."""
from pathlib import Path
import json
import os
import runpy
import sys

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy

from continuous_video import ContinuousVideo


class RecordingEnvironment(og.Environment):
    def __init__(self, *args, **kwargs):
        self.recorder = None
        self.recorded_steps = 0
        super().__init__(*args, **kwargs)
        if self.env_config["action_frequency"] != 30 or self.env_config["physics_frequency"] != 120:
            raise RuntimeError("Recording requires the verified 30Hz/120Hz native cadence")
        robot = self.scene.robots[0]
        heads = [s for name, s in robot.sensors.items() if "zed" in name]
        if len(heads) != 1:
            raise RuntimeError("Expected exactly one native head camera")
        # A second render product shares the original camera prim, without changing
        # the model's sensor resolution, pose, observation schema or physics.
        with og.sim.editing_usd():
            self.video_product = lazy.omni.replicator.core.create.render_product(heads[0].prim_path, (640, 480))
            self.video_rgb = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator("rgb")
            self.video_rgb.attach([self.video_product])
        self.recorder = ContinuousVideo("/output/video", 30)
        self.video_physics = []
        self.video_subscription = lazy.omni.physx.get_physx_interface().subscribe_physics_step_events(
            lambda dt: self.video_physics.append(float(dt)))
        self.video_rows = []
        instances.append(self)

    def step(self, *args, **kwargs):
        before = len(self.video_physics) if self.recorder is not None else 0
        result = super().step(*args, **kwargs)
        if self.recorder is not None:
            count = len(self.video_physics) - before
            if count != 4:
                raise RuntimeError(f"Recording observed {count} physics callbacks, expected 4")
            high = np.asarray(self.video_rgb.get_data())[..., :3]
            if high.shape != (480, 640, 3) or high.dtype != np.uint8 or high.std() <= 1:
                raise RuntimeError("Recording camera returned empty or invalid RGB")
            # Native per-step RGB is retained too, independently of sparse model observations.
            images = {"head_640x480": high}
            for name, value in result[0][0][self.scene.robots[0].name].items():
                if not isinstance(value, dict) or "rgb" not in value:
                    continue
                label = "head" if "zed" in name else "left_wrist" if "left_realsense" in name else "right_wrist" if "right_realsense" in name else None
                if label:
                    rgb = value["rgb"]
                    images[label] = rgb.detach().cpu().numpy()[..., :3] if hasattr(rgb, "detach") else np.asarray(rgb)[..., :3]
            self.recorded_steps += 1
            self.recorder.write(images, self.recorded_steps)
            self.video_rows.append(dict(frame=self.recorded_steps-1, native_step=self.recorded_steps,
                                        physics_callbacks=count, simulation_time=self.recorded_steps/30))
        return result

    def close_video(self):
        if self.recorder is not None:
            self.recorder.close()
            Path("/output/video/capture.json").write_text(json.dumps(dict(
                continuous=True, interpolated=False, capture="after every original env.step",
                extra_physics_steps=0, model_observation_unchanged=True,
                resolution=[640, 480], fps=30, initial_frame_in_video=False,
                frames=self.video_rows), indent=2) + "\n")
            self.recorder = None


instances = []
og.Environment = RecordingEnvironment
native_shutdown = og.shutdown


def shutdown(*args, **kwargs):
    # OmniGibson shutdown can exit the process; finalize MP4 before invoking it.
    for env in instances:
        env.close_video()
    return native_shutdown(*args, **kwargs)


og.shutdown = shutdown
sys.path.insert(0, "/work")
try:
    if "BEHAVIOR_FEISHU_ID" in os.environ:
        runpy.run_path("/recording/task_worker.py", run_name="__main__")
    else:
        runpy.run_path("/work/eef_worker.py", run_name="__main__")
finally:
    for env in instances:
        env.close_video()
