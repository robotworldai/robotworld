# Evaluating a New Policy

This guide walks through how to evaluate your own policy against the RoboLab benchmark. You do **not** need to fork or modify RoboLab — everything can live in your own separate repository that imports `robolab` as a dependency.

RoboLab uses a **server-client architecture**: your model runs as a standalone server (any framework, any GPU), and a lightweight inference client inside the simulator sends observations and receives actions.

**Prerequisites:** You need registered environments before running evaluation. For DROID with joint-position actions, RoboLab ships a built-in registration you can use directly. If you need custom observations, a different robot, or different simulation parameters, first follow the [Environment Registration](environment_registration.md) guide.

## Your Repository Structure

```
my_policy_eval/
  my_policy/
    __init__.py
    inference_client.py        # Your inference client (Step 1)
  run_eval.py                  # Your evaluation script (Step 2)
  requirements.txt             # includes robolab as a dependency
```

## Step 1: Implement an Inference Client

Subclass `robolab.eval.InferenceClient`. The base provides the control loop
(`infer`, `reset`, chunking, multi-env bookkeeping); subclasses implement
four narrow hooks:

```python
# my_policy/inference_client.py

import numpy as np
from robolab.eval import InferenceClient


class MyPolicyClient(InferenceClient):
    open_loop_horizon = 8  # how many actions to consume per server query

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000) -> None:
        super().__init__()
        # Connect to your model server
        ...

    # --- required hooks ---------------------------------------------------

    def _extract_observation(self, raw_obs, *, env_id=0) -> dict:
        # For the default DROID registration, raw_obs contains:
        #   raw_obs["image_obs"]["over_shoulder_left_camera"]    - (N, H, W, 3) torch tensor, uint8
        #   raw_obs["image_obs"]["wrist_cam"]       - (N, H, W, 3) torch tensor, uint8
        #   raw_obs["proprio_obs"]["arm_joint_pos"] - (N, 7) torch tensor, float32
        #   raw_obs["proprio_obs"]["gripper_pos"]   - (N, 1) torch tensor, float32
        return {
            "image":    raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].cpu().numpy(),
            "joint_pos": raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy(),
        }

    def _pack_request(self, extracted_obs, instruction):
        # Whatever wire format your server expects
        return {"image": extracted_obs["image"], "prompt": instruction}

    def _query_server(self, request):
        return self.client.infer(request)

    def _unpack_response(self, response) -> np.ndarray:
        # Must return a (horizon, action_dim) array; base handles the rest.
        return np.asarray(response["actions"])

    # --- optional hooks (defaults are identity / None) -------------------

    def _postprocess_chunk(self, chunk):
        # Binarize gripper, pad 7->8, flip sign, etc.
        return chunk

    def _build_visualization(self, extracted_obs):
        return extracted_obs["image"]
```

**Key contract:**
- `_extract_observation` + `_pack_request` split repo-specific obs munging from backend-specific wire format. The ABC's default `infer()` wires them together: extract → pack → query → unpack → postprocess → cache chunk → step one action.
- Action dict returned by `infer()` has `"action"` (numpy array, typically 8-dim: 7 joints + 1 gripper) and `"viz"` (image for the live display window, or `None`).
- `reset(env_id=...)` clears per-episode state; `begin_episode(episode_idx)` is called before each episode starts. Override either only if your server needs session notification; otherwise the base's defaults are enough.
- There is no central registry or factory — each runner imports its client class directly and constructs it inline:

```python
from my_policy.inference_client import MyPolicyClient

client = MyPolicyClient(remote_host="localhost", remote_port=8000)
```

See the [existing clients](#existing-clients-as-reference) for complete working examples.

## Step 2: Write Your Evaluation Script and Run It

For the full evaluation script template, CLI reference, and run instructions, see [Running Environments](environment_run.md#writing-an-evaluation-script).

In short:

1. **Install robolab** as a dependency:
   ```bash
   cd /path/to/robolab && uv pip install -e .
   ```

2. **Install your package** so its modules are importable:
   ```bash
   cd /path/to/my_policy_eval && uv pip install -e .
   ```

3. **Start your model server** (in a separate terminal):
   ```bash
   python -m my_model.serve --checkpoint /path/to/model --port 8000
   ```

4. **Run evaluation**:
   ```bash
   # Run on all benchmark tasks
   python run_eval.py --headless

   # Run on a specific task
   python run_eval.py --task BananaInBowlTask

   # Run on a tag of tasks
   python run_eval.py --tag pick_place

   # Run multiple runs with parallel envs (total episodes = num_runs * num_envs)
   python run_eval.py --headless --num-runs 5 --num_envs 2

   # Custom server address
   python run_eval.py --remote-host 10.0.0.1 --remote-port 5555
   ```

5. **View results**: Results are saved to `output/<timestamp>_my_policy/`. See [Analysis and Results Parsing](analysis.md) for summarization tools.

## Optional: Ground-Truth State in Your Client

If your client (or the server behind it) wants privileged simulator state — object poses, gripper contacts, live subtask progress — run with `--enable-gt-state`. The episode loop then attaches `obs["gt_state"] = {env_id: state}` (one entry per active env, once per environment step) to the observation your `_extract_observation` receives. The base class provides generic helpers to read it: `self._get_env_gt_state(raw_obs, env_id)` returns your env's entry, `self._find_obs_term(raw_obs, name)` looks up any observation term by name across groups (e.g. depth or camera-calibration terms), and `self._to_numpy(value, env_id)` selects one env's row as numpy. Each client reads only its own env's entry, so multi-env evaluations stay independent.

The state is a *raw snapshot*: object poses and velocities, end-effector pose, gripper closedness, gripper contacts, and the subtask recorder's per-object/per-condition tracking. Frames are uniform — positions in meters in the env-local frame (world minus env origin), quaternions world-frame `(w, x, y, z)`, velocities world-frame — and everything is plain numpy, msgpack-serialisable. Full schema: [Running Environments — Ground-Truth State Export](environment_run.md#ground-truth-state-export).

Derived quantities (e.g. "was this object lifted", "is the gripper grasping something") are deliberately not part of the snapshot — thresholds and heuristics are the consumer's business. Accumulate them client-side from the per-step stream; `_extract_observation` runs every environment step, even on steps where a cached action chunk skips the server round-trip, so client-side tracking never misses a step.

Separately from the inference-time channel, environments can be registered with `object_state_obs=True` (on `auto_register_droid_envs`) to add a per-task `object_state_obs` observation group with `<object>_pos` / `<object>_quat` / `<object>_vel` terms (same frame conventions) generated from the task's `contact_object_list` — ground-truth object state through the standard observation pipeline, for consumers that want it batched, recorded, and replayable like any other observation.

## Existing Clients as Reference

See [Inference Clients](../policies/README.md) for server setup instructions.
