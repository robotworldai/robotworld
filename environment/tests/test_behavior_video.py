import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from environment.benchmarks.behavior_1k.diagnostics import verify_video
from environment.benchmarks.robodojo.video import ContinuousVideo


def test_recording_frame_count_and_native_physics_cadence(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video tools unavailable")
    video = ContinuousVideo(tmp_path, 30)
    for step in range(1, 5):
        images = {name: np.full(shape, step*40, np.uint8) for name, shape in (
            ("head_640x480", (480, 640, 3)), ("head", (128, 128, 3)),
            ("left_wrist", (128, 128, 3)), ("right_wrist", (128, 128, 3)))}
        video.write(images, step)
    video.close()
    capture = dict(frames=[dict(native_step=n, physics_callbacks=4) for n in range(1, 5)])
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    assert verify_video(tmp_path, 4)["duration_seconds"] == 4/30
    capture["frames"][1]["physics_callbacks"] = 5
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    with pytest.raises(RuntimeError, match="cadence"):
        verify_video(tmp_path, 4)


def test_recording_overlay_preserves_original_worker():
    root = Path(__file__).resolve().parents[1] / "benchmarks/behavior_1k"
    source = (root / "record_worker.py").read_text()
    assert 'runpy.run_path("/work/eef_worker.py"' in source
    assert source.count("super().step(") == 1
    assert "og.sim.step(" not in source and "og.sim.render(" not in source
    compile(source, "record_worker.py", "exec")
