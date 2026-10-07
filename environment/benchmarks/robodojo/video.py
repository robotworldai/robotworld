"""Stream every control-step RGB frame; independent of model image history."""
import json
import subprocess
from pathlib import Path

import numpy as np


class ContinuousVideo:
    def __init__(self, output, fps):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.streams = {}
        self.steps = []

    def write(self, images, step):
        if self.steps and step == self.steps[-1]:
            return
        if self.steps and step != self.steps[-1] + 1:
            raise RuntimeError("Video requires consecutive environment steps")
        if not images:
            raise RuntimeError("No RGB frames available for video")
        if self.streams and set(images) != set(self.streams):
            raise RuntimeError("Video camera set changed")
        for name, image in images.items():
            frame = np.ascontiguousarray(image, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Video expects RGB frames")
            if name not in self.streams:
                safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
                log = (self.output / f'{safe}.ffmpeg.log').open('w')
                height, width = frame.shape[:2]
                process = subprocess.Popen([
                    'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                    '-r', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                    '-threads', '2', '-preset', 'veryfast', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(self.output / f'{safe}.mp4')], stdin=subprocess.PIPE, stderr=log)
                self.streams[name] = (process, log, frame.shape)
            process, _, shape = self.streams[name]
            if frame.shape != shape:
                raise ValueError("Video frame dimensions changed")
            process.stdin.write(frame.tobytes())
        self.steps.append(step)

    def close(self):
        errors = []
        for name, (process, log, _) in self.streams.items():
            try:
                process.stdin.close()
                if process.wait(timeout=30) != 0:
                    errors.append(name)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
                errors.append(name)
            finally:
                log.close()
        (self.output / 'manifest.json').write_text(json.dumps({
            'fps': self.fps, 'frames_per_camera': len(self.steps),
            'env_steps': self.steps, 'cameras': list(self.streams),
            'time_basis': 'simulation control steps, excludes model wall-clock wait',
            'encoding_errors': errors}, indent=2) + '\n')
        self.streams.clear()
        if errors:
            raise RuntimeError(f"Video encoding failed: {errors}")
