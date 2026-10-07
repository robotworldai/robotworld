# DROID Dataset Module

A LeRobot format loader for DROID data sets.

## Structure design

DROID All Speciality Envelopes in `DroidLerobotDataset`, processor End Zero Knowledge - Cross emb
The same `GalaxeaCoTProcessor` path as r1lite/r1pro/agibot when mixing batch, etc.

```
DroidLerobotDataset.__getitem__:
  super (image_meta Injected cam2_left For loading)
    └─ Read cam1 / cam2 / wrist (uint8 [T,3,H,W])
  _swap_exterior_images:  Training 50% cam1 ← cam2, drop cam2
  _inject_dummy_images:   Press shape_meta Medium dummy:true Injection zero tensors
  _pick_lang_alternative: 3-choose-1 over (task, lang2, lang3)
  processor.preprocess (Universal GalaxeaCoTProcessor)
```

## File Structure

```
droid/
├── __init__.py              # Module portal
├── droid_lerobot_dataset.py # LeRobot Format data loader (inclusive) Droid-specific behaviour)
├── droid_utils_pytorch.py   # Coordinate Change Tool
└── README.md                # This document
```

## Configure Use

The processor end simply states that the camera that the model actually sees - aux exterior - was injected by dataset itself
`image_meta` is used for loading and is not exposed to processor.

```yaml
# configs/data/droid.yaml
processor:
  # Do Not Write _target_, Come on. task config Default (Usually. GalaxeaCoTProcessor)
  shape_meta:
    images:
      - key: exterior_image     # Lord exterior (Maybe during training. cam2 Replace)
        camera_type: exterior
        lerobot_key: observation.images.exterior_1_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: wrist_image
        camera_type: wrist_left
        lerobot_key: observation.images.wrist_left
        raw_shape: [3, 256, 256]
        shape: [3, 224, 224]
      - key: dummy_wrist_right  # Optional: Jean. pixel_values Keyset and 3-cam emb Alignment
        camera_type: wrist_right
        dummy: true             # dataset Seeing this mark will inject. zeros, Do Not Read lerobot
        raw_shape: [3, 256, 256]
        shape: [3, 256, 160]

Droid_Franka:
  type: g05.data.droid.droid_lerobot_dataset.DroidLerobotDataset
  random_swap_exterior_images: true
  random_select_instruction: true
  filter_failed_trajectories: true
  shape_meta: ${processor.shape_meta}
  ...
```

## DROID-specific Behaviour

| Behaviour | Achieved position |
|---|---|
| Gripper Flip (`1 - x`) | `_slice_meta_feature` |
| 2-choose-1 exterior swap | `__getitem__` → `_swap_exterior_images` |
| 3-choose-1 lang augmentation | `__getitem__` → `_pick_lang_alternative` |
| dummy zero-image Injection | `__getitem__` → `_inject_dummy_images` |
| Failed track filter | `_build_valid_local_indices` (`filter_failed_trajectories`) |
| Idle frame filter | `R1LiteJointActionFilter` (in processor.action_filter configuration) |

## List Map

| sample key | LeRobot Listing |
|----------|--------------|
| `exterior_image` (primary, after the internal dataset swap) | `observation.images.exterior_1_left` |
| `exterior_image_2` (dataset loading itself, swap discarded) | `observation.images.exterior_2_left` |
| `wrist_image` | `observation.images.wrist_left` |
| `joint_position` (state) | `observation.state.joint_position` |
| `gripper` (state) | `observation.state.gripper_position` |

## Coordinate Change Tool

```python
from g05.data.droid import (
    euler_to_rmat,
    rmat_to_euler,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    velocity_act_to_wrist_frame,
)
```
