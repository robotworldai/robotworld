import json
import shutil
import subprocess

import numpy as np
import pytest

from environment.benchmarks.robodojo.video import ContinuousVideo


def test_video_keeps_all_steps_and_deduplicates_observation(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    video = ContinuousVideo(tmp_path, 25)
    for step in range(10):
        frame = np.full((32, 32, 3), step * 20, dtype=np.uint8)
        video.write({'head': frame}, step)
        video.write({'head': frame}, step)
    video.close()
    meta = json.loads((tmp_path / 'manifest.json').read_text())
    assert meta['env_steps'] == list(range(10))
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate', '-of', 'json',
        str(tmp_path / 'head.mp4')], text=True)
    stream = json.loads(result)['streams'][0]
    assert int(stream['nb_frames']) == 10
    assert stream['r_frame_rate'] == '25/1'
