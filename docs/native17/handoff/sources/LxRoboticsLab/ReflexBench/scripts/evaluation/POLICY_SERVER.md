# Policy server protocol

`eval.py` can evaluate any policy exposed through this HTTP interface. The server may wrap an RL policy, VLA, diffusion policy, ACT model, or another action-chunking policy.

## Endpoints

### `GET /info`

Return model metadata once at client startup:

```json
{
  "action_dim": 8,
  "action_horizon": 16,
  "model_name": "my_policy",
  "control_mode": "joint_pos"
}
```

`action_dim` and `action_horizon` are required. `control_mode` should match the `eval.py --control` argument.

### `POST /predict`

The client sends one batched request per policy update. A state-only request has this form:

```json
{
  "type": "state",
  "num_envs": 2,
  "state": [[0.1, 0.2], [0.3, 0.4]],
  "control_mode": "joint_pos",
  "action_format": "abs_joint",
  "state_format": "joint",
  "orientation_rep": "quat",
  "step_ids": [42, 17]
}
```

A vision-language request has this form:

```json
{
  "type": "vla",
  "num_envs": 2,
  "control_mode": "ik_rel",
  "action_format": "rel_eef_pose",
  "state_format": "eef_pose",
  "orientation_rep": "quat",
  "proprioception": {
    "gripper_state": [[1.0], [0.0]],
    "eef_pos": [[0.3, 0.1, 0.4], [0.4, 0.0, 0.5]],
    "eef_orient": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  },
  "images": {
    "fixed_cam": ["<base64 JPEG>", "<base64 JPEG>"],
    "wrist_cam": ["<base64 JPEG>", "<base64 JPEG>"]
  },
  "task_description": "catch the incoming ball",
  "step_ids": [42, 17]
}
```

When `--image_history` is greater than one, each camera entry is shaped `(N, T)` instead of `(N)`, ordered from oldest to newest. The payload also includes `image_history` and `image_history_stride`.

Images are RGB `uint8`, JPEG-encoded at quality 90, then base64-encoded. A server can decode one image with:

```python
import base64
import io

import numpy as np
from PIL import Image


def decode_image(value: str) -> np.ndarray:
    data = base64.b64decode(value)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
```

The response must contain batched action chunks:

```json
{
  "actions": [
    [[0.1, 0.2], [0.3, 0.4]],
    [[0.5, 0.6], [0.7, 0.8]]
  ],
  "latency_s": 0.015
}
```

- `actions` is required and normally has shape `(N, H, D)`. The client also accepts per-environment ragged chunks and single-step `(N, D)` responses.
- `latency_s` is optional server-side model latency. It is used for logging and asynchronous action alignment.

### `POST /reset` (optional)

The client sends reset notifications so a stateful policy can clear history for selected environments:

```json
{"env_ids": [0, 3]}
```

The client silently ignores connection failures for this optional endpoint.

## Action representations

| Control mode | Dimension | Server output |
| --- | ---: | --- |
| `joint_pos` | 8 | Seven absolute arm-joint targets and one gripper value (`> 0.5` means open) |
| `ik_abs` | 7 | Absolute end-effector position and roll-pitch-yaw, plus gripper |
| `ik_rel` | 7 | Relative end-effector position and roll-pitch-yaw delta, plus gripper |

For IK modes, gripper values use `-1` for closed and `+1` for open. `eval.py` converts absolute joint targets to the relative controller input immediately before each environment step.

## Timing behavior

`--ctrl_freq` controls how frequently the robot consumes actions. `--infer_freq` controls the fixed action-chunk update period and is independent of the physics and control frequencies. `--use_real_latency` replaces the fixed period with measured request latency.

- Synchronous mode advances the world during inference delay with a freeze-arm action, then executes the newly returned chunk.
- Asynchronous mode advances the world using the previous chunk and applies the newly returned chunk on the next cycle.

The HTTP request itself is blocking in both modes; the distinction is how simulation time and action chunks are scheduled.

## Minimal FastAPI server

```python
import time

import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()
ACTION_DIM = 8
ACTION_HORIZON = 16


@app.get("/info")
def info():
    return {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "control_mode": "joint_pos",
    }


@app.post("/predict")
def predict(request: dict):
    start = time.monotonic()
    batch_size = request["num_envs"]
    # Replace this array with model inference.
    actions = np.zeros((batch_size, ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    return {
        "actions": actions.tolist(),
        "latency_s": time.monotonic() - start,
    }


@app.post("/reset")
def reset(request: dict):
    # Clear policy state for request["env_ids"] when needed.
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Install `fastapi` and `uvicorn`, start the server, then pass its URL through `eval.py --server_url`.
