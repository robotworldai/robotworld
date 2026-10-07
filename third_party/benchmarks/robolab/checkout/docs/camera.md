# Cameras

RoboLab supports two types of cameras: **scene cameras** (fixed in the world) and **robot-attached cameras** (mounted on the robot, e.g., wrist cameras). Both use IsaacLab's `TiledCameraCfg` and are passed into [environment registration](environment_registration.md) as config classes.

## Scene Cameras

Scene cameras are fixed-position cameras defined as standalone `@configclass` objects. They are independent of the robot and observe the scene from a static viewpoint. These configs can live in your own repository.

```python
# my_repo/cameras.py

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class MyExternalCameraCfg:
    over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66),
            rot=(-0.393, -0.195, 0.399, 0.805),
            convention="opengl",
        ),
    )
```

Key fields:

| Field | Description |
|-------|-------------|
| `prim_path` | Scene path; use `{ENV_REGEX_NS}/<camera_name>` for multi-env support |
| `height` / `width` | Image resolution in pixels |
| `data_types` | List of data types to capture (e.g., `["rgb"]`, `["rgb", "depth"]`) |
| `spawn` | Camera model — `PinholeCameraCfg` with focal length, aperture, etc. |
| `offset` | Camera pose: `pos` (x, y, z), `rot` (quaternion w, x, y, z), and `convention` (`"opengl"` or `"ros"`) |

### Built-in Scene Cameras

RoboLab ships several scene camera presets in `robolab/variations/camera.py`:

| Config | Attribute Name | Description |
|--------|---------------|-------------|
| `OverShoulderLeftCameraCfg` | `over_shoulder_left_camera` | Over-the-shoulder view from the left |
| `OverShoulderRightCameraCfg` | `over_shoulder_right_camera` | Over-the-shoulder view from the right |
| `HeadCameraCfg` | `head_camera` | Front overhead view (operator's eye) |
| `EgocentricWideAngleCameraCfg` | `egocentric_wide_angle_camera` | Wide-angle front view |
| `EgocentricMirroredCameraCfg` | `egocentric_mirrored_camera` | Front-facing mirrored view (480×864) |
| `EgocentricMirroredWideAngleCameraCfg` | `egocentric_mirrored_wide_angle_camera` | Wide-angle front mirrored view |
| `EgocentricMirroredWideAngleHighCameraCfg` | `egocentric_mirrored_wide_angle_high_camera` | High-angle front mirrored view |

### Passing Scene Cameras to Registration

Scene cameras are passed as `camera_cfg` (a single config or a list) in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.variations.camera import OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
    # ... other registration kwargs
)
```

## Robot-Attached Cameras

Robot-attached cameras (e.g., wrist cameras, head cameras) move with the robot during execution. These are defined as fields on the **robot config** rather than as standalone config classes.

> **Important:** A robot-attached camera's `prim_path` must be under the robot's prim hierarchy in the USD scene. This ensures the camera moves rigidly with the robot link it is attached to.

For example, the built-in `DroidCfg` defines a wrist camera:

```python
@configclass
class DroidCfg:
    robot = ArticulationCfg(...)

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )
```

When defining your own robot with a wrist camera, ensure the camera's `prim_path` points to a link in your robot's USD. The `offset` is relative to that link. See [Robots](robots.md#adding-a-wrist-camera-to-your-robot) for a full example.

## Wiring Cameras to Observations

Camera names in the observation config must match the attribute names on the camera config classes. For example, if your scene camera config has an attribute `over_shoulder_left_camera` and your robot config has `wrist_cam`, then your observation config references those same names.

The quickest path is to let RoboLab generate the observation group from the same list of camera configs you attach to the scene:

```python
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras
from robolab.registrations.droid.camera_presets import WRIST_LEFT

ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
```

Any camera attached to the scene renders every step, so the preset should list exactly the cameras you want the policy to read. Available presets in `camera_presets.py`: `WRIST`, `WRIST_LEFT`, `WRIST_RIGHT`, `WRIST_LEFT_RIGHT`, `WRIST_LEFT_RIGHT_HEAD`, `LEFT_RIGHT`. Pass your chosen preset (or your own list) to `auto_register_droid_envs(cameras=...)`. Viewport-only cameras like `EgocentricMirroredCameraCfg` are attached separately for video recording and are not listed in the presets.

**Define your own `ImageObsCfg` when you need per-camera customization** — different data types (e.g. depth), noise corruption, normalization flags, or reading the same tiled prim via multiple keys:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class ImageObsCfg(ObsGroup):
    over_shoulder_left_camera = ObsTerm(
        func=mdp.observations.image,
        params={"sensor_cfg": SceneEntityCfg("over_shoulder_left_camera"), "data_type": "rgb", "normalize": False},
    )
    wrist_cam = ObsTerm(
        func=mdp.observations.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
```

The `SceneEntityCfg("over_shoulder_left_camera")` string must match the **attribute name** on the camera config class (e.g., `OverShoulderLeftCameraCfg.over_shoulder_left_camera`).

## Camera Pose Randomization

For robustness testing, you can randomize camera poses at episode reset. See [Running Environments — Initial Condition Randomization](environment_run.md#initial-condition-randomization):

```python
from robolab.core.events.reset_camera import RandomizeCameraPoseUniform

events = RandomizeCameraPoseUniform.from_params(
    cameras=["over_shoulder_left_camera"],
    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
)
env, env_cfg = create_env("BananaInBowlTask", events=events)
```

## See Also

- [Robots](robots.md) — Robot definitions and wrist camera attachment
- [Environment Registration](environment_registration.md) — Wiring cameras into registered environments
- [Running Environments](environment_run.md) — Camera pose variation at runtime
