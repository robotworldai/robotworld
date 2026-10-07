# Pretrained policies

Final trained policies for the two tasks, ready to use without retraining. Each was trained on a single RTX 4090 Laptop GPU (16 GB), 2048 environments, headless. For the full design and math, see the [project wiki](../docs/wiki/Home.md).

| Folder | Task | Iterations | Mean reward | Episode length | Tracking |
|---|---|:---:|:---:|:---:|:---:|
| [`balance/`](balance/) | Balance in place (`Wheeled-Quadruped-Balance-v0`) | 1000 | approx 19.5 | 1000 / 1000 | |
| [`velocity/`](velocity/) | Drive while balancing (`Wheeled-Quadruped-Velocity-v0`, yaw-tuned) | 3000 | approx 36.5 | 996 / 1000 | lin approx 0.82, yaw approx 0.85 |
| [`balance_rough/`](balance_rough/) | Balance on rough terrain (`Wheeled-Quadruped-Balance-Rough-v0`) | 1500 | approx 7.5 | approx 948 / 1000 | |
| [`velocity_rough/`](velocity_rough/) | Drive on rough terrain (`Wheeled-Quadruped-Velocity-Rough-v0`) | 3000 | approx 18.4 | approx 878 / 1000 | lin approx 0.55, yaw approx 0.30 |

The `*_rough` folders hold the checkpoints `model_1499.pt` / `model_2999.pt` respectively; everything below (file formats, observation layout, inference) is identical across all four. The rough policies use the same network shape and onboard-only observations, just trained on uneven terrain.

## Files in each folder

- **`policy.onnx`**: the actor network in ONNX. Framework-agnostic; run it with `onnxruntime` in Python, C++, or any ONNX runtime. This is the **deployment** artifact.
- **`policy_torchscript.pt`**: the same actor as a TorchScript module. Load with `torch.jit.load`, no Isaac Lab required.
- **`model_1399.pt`** / **`model_2999.pt`**: the raw rsl-rl training checkpoint (actor + critic + optimizer). Use it to replay or resume inside Isaac Sim.

Both exported forms contain **only the onboard-deployable actor** (plus its observation normalizer). The privileged critic used during training is already stripped, so the policy consumes exactly the observations a real robot can measure.

## Replay in Isaac Sim

With the project installed (see the [top-level README](../README.md)):

```bash
# Velocity policy (all robots drive forward at 0.5 m/s in the Play variant)
python scripts/rsl_rl/play.py --task Wheeled-Quadruped-Velocity-Play-v0 --num_envs 9 \
  --checkpoint pretrained/velocity/model_2999.pt

# Balance policy
python scripts/rsl_rl/play.py --task Wheeled-Quadruped-Balance-Play-v0 --num_envs 9 \
  --checkpoint pretrained/balance/model_1399.pt
```

## Observation layout (what the ONNX/TorchScript actor expects)

The actor's input is the concatenated **`policy`** observation group, onboard-obtainable quantities only. Order and dimensions:

**Balance actor, 16 inputs:**

| Slice | Term | Dim | Symbol |
|---|---|:---:|---|
| 0:3 | base angular velocity (body frame) | 3 | $\omega$ |
| 3:6 | projected gravity (body frame) | 3 | $g_b$ |
| 6:8 | thigh joint positions (relative to default) | 2 | $q_\text{thigh}-q_\text{default}$ |
| 8:12 | joint velocities (all 4 actuated joints) | 4 | $\dot q$ |
| 12:16 | last action | 4 | $a_{t-1}$ |

**Velocity actor, 19 inputs:** the same 16, followed by the commanded velocity $(c_x, c_y, c_z)$ at indices 16:19 (with $c_y \equiv 0$).

**Output, 4 actions**, in `[-1, 1]` (roughly): `[thigh_left_pos, thigh_right_pos, rl_wheel_vel, rr_wheel_vel]`. The environment scales these (thigh × 0.5 rad around the default pose; wheel × 5 rad/s for balance, × 12 for velocity) before applying them through the joint PD actuators. See [wiki: Balance Task](../docs/wiki/05-Balance-Task.md) and [Velocity Task](../docs/wiki/06-Velocity-Task.md).

## Minimal ONNX inference

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("pretrained/velocity/policy.onnx")
inp = sess.get_inputs()[0].name          # single input tensor
obs = np.zeros((1, 19), dtype=np.float32)  # fill with your live observation
action = sess.run(None, {inp: obs})[0]     # -> (1, 4)
```

Feed your robot's real onboard observation (same order as the table above), then map the 4 outputs to thigh position targets and wheel velocity targets with the scales listed. On hardware you also need a state estimator only for quantities the policy does **not** use; it deliberately avoids base linear velocity and absolute height.
