## Quick Guide: Recording Simulation Videos

This guide explains how to capture frames from the Isaac Lab simulation and convert them into an MP4 video.

### Capture Frames in `play.py`

Modify `legged_lab/scripts/play.py` to save simulation frames as images.

#### Add Imports
Add these lines to the top of `legged_lab/scripts/play.py`:
```python
from legged_lab.utils.data_recorder.frame_recorder import FrameRecorder, ensure_world_camera
```

#### Add Recorder Code
In the `play()` function, add the following snippet immediately after the environment is created (`env = env_class(...)`):
```python
# -- VIDEO RECORDING SETUP --
# Define and position the recording camera
camera_path = ensure_world_camera("/World/RecorderCamera", pos=Gf.Vec3d(4, 4, 2.5))

# Instantiate and start the recorder
rgb_recorder = FrameRecorder(
    camera_prim_paths=[camera_path], # List of cameras to record
    output_root="_recordings",       # Where to save the frames
    resolution=(1920, 1080)          # Video resolution
)
rgb_recorder.start()
print(f"[INFO] Recording frames to: {rgb_recorder.output_dir}")
# -- END RECORDING SETUP --
```
To setup camera's position: 

```python
env.sim.set_camera_view([-5.0, 5.0, 10.0], [30.0, 15.0, 0.0]) # (Position of camera, Position where it looks at)
```

When you run `play.py`, this will save PNG frames to a timestamped sub-directory inside `_recordings/`.

> **Note**: In `--headless` mode, please add a `--enable_cameras` flag to avoid errors:
> 
```bash
python legged_lab/scripts/preview.py --num_envs 1024 --logger tensorboard --task t1_tt --headless --enable_cameras
```

### Convert Frames to Video

Use the `frames2video.sh` script to combine the saved image frames into a video.

#### Make Script Executable (One-time setup)
```bash
chmod +x legged_lab/utils/data_recorder/frames2video.sh
```

#### Run the Script
Open your terminal and use one of the following commands. Replace `path/to/frames` with the actual directory created in step 1 (e.g., `_recordings/run_20240809_153000/`).

> **Note:** The script requires `ffmpeg`. If not installed, run `sudo apt-get install ffmpeg` on Ubuntu

*   **Create video named `output.mp4` inside the frames folder:**
    ```bash
    ./legged_lab/utils/data_recorder/frames2video.sh path/to/frames
    ```

*   **Specify a custom output path and name:**
    ```bash
    ./legged_lab/utils/data_recorder/frames2video.sh path/to/frames ~/Videos/my_simulation.mp4
    ```

