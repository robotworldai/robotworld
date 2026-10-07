# Sim-to-sim validation in MuJoCo

The four policies in this repo were trained in **Isaac Sim (PhysX)**. This folder runs
the exact same exported actors in **MuJoCo**, a completely independent physics engine,
to check that the learned behaviour survives a change of simulator. A policy that only
works in the simulator it was trained in has likely overfit that engine's quirks; one
that also works in MuJoCo is far more likely to survive the next gap, the one to real
hardware. Sim-to-sim is the cheap, honest rehearsal for sim-to-real.

## Results

Twenty episodes per policy (10 for the table below were enough to be stable), 1000
control steps each (a full 20 s), reset with small randomization:

| Policy | MuJoCo episode length | Full-episode survival | Lin-vel MAE | Yaw-rate MAE |
|---|:---:|:---:|:---:|:---:|
| **balance** | 1000 / 1000 | 100% | | |
| **velocity** (cmd 0.5 m/s) | 1000 / 1000 | 100% | 0.26 m/s | 0.06 rad/s |
| **balance-rough** | 1000 / 1000 | 100% | | |
| **velocity-rough** (cmd 0.5 m/s) | ~956 / 1000 | 20% | 0.17 m/s | 0.15 rad/s |

All four policies transfer well to MuJoCo: they balance, and (for the velocity tasks)
drive to the commanded speed, in an engine they never saw during training. The two
balance policies and the flat velocity policy survive every episode. The hardest task,
**driving while balancing on rough terrain**, stays up about 96% of the episode on
average and tracks its command closely, but it is the only policy that ever falls (it
completes the full episode about 20% of the time). That is the honest, useful outcome of
a sim-to-sim check: it flags the one policy closest to the edge of its robustness, the
one that would benefit most from more domain randomization before hardware.

Two caveats worth stating plainly. The MuJoCo model is rebuilt from the robot's URDF and
is **not identical** to the Isaac USD (contact model, solver, and the rough height field
all differ), so this is a genuine cross-engine test, not a replay. And the numbers above
are cross-engine transfer of a fixed policy, not a re-training, so some accuracy loss
versus the Isaac numbers is expected and is itself the signal.

## How it works

- **The policy runs in pure numpy.** The actor is a small ELU MLP with no observation
  normalization, so `export_policies.py` dumps its Linear layers to `policies/*.npz` and
  `sim2sim.py` does the forward pass with numpy. No torch or onnxruntime is needed in the
  MuJoCo environment, exactly as you would run the exported ONNX policy on hardware.
- **The MuJoCo model matches the Isaac articulation.** `make_model.py` imports the ROS
  URDF into MuJoCo, whose fixed-joint handling reproduces the same merged rigid bodies
  Isaac trains on (one base of about 17.9 kg, two front-thigh subtrees, two rear wheels).
  It then floats the base, adds a floor (flat) or a smoothed random height field (rough),
  and adds actuators that match Isaac Lab's implicit PD: thighs are a position servo
  (kp=1000, joint damping 20), wheels are a velocity servo (kv=10).
- **The observation and action conventions are matched exactly** to the Isaac task:
  base-frame angular velocity, projected gravity, front-thigh positions, the joint order
  `[FL_thigh, FR_thigh, rl_wheel, rr_wheel]`, the last action, and (velocity tasks) the
  3-value command; actions map to a thigh position target of `0.5 * a` and a wheel
  velocity target of `scale * a` (scale 5 for balance, 12 for velocity). Control runs at
  50 Hz (decimation 4) over a 200 Hz sim, as in Isaac.

## Running it

MuJoCo (3.x) and numpy are the only requirements for the runner. On this machine that is
the WSL `mybot_mjx` conda env; any `mujoco` + `numpy` Python works.

```bash
# 1. Export the actor weights (run ONCE with the Isaac Lab venv, which has torch):
python export_policies.py

# 2. (Re)build the MuJoCo models (run with the MuJoCo env):
python make_model.py            # writes wheeled_quadruped.xml and wheeled_quadruped_rough.xml

# 3. Validate a policy headless (mean episode length + tracking error):
python sim2sim.py --task balance         --episodes 20
python sim2sim.py --task velocity        --episodes 20 --cmd_x 0.5
python sim2sim.py --task balance_rough   --episodes 20
python sim2sim.py --task velocity_rough  --episodes 20 --cmd_x 0.5

# 4. Watch it live in the MuJoCo viewer (loops, resets on a fall):
python sim2sim.py --task velocity --viewer --cmd_x 0.5
```

## Files

| File | Role |
|---|---|
| `export_policies.py` | (Isaac venv) dump each actor MLP to `policies/<task>.npz` |
| `make_model.py` | (MuJoCo env) build `wheeled_quadruped.xml` (flat) and `..._rough.xml` (height field) from the URDF |
| `sim2sim.py` | (MuJoCo env) run a policy: Isaac-matched observation, action scaling, control loop, headless eval or viewer |
| `wheeled_quadruped.xml`, `..._rough.xml` | the generated MuJoCo models (regenerable with `make_model.py`) |
| `meshes/` | STL collision meshes copied from `src/robot_description/` |
| `policies/` | exported actor weights (numpy `.npz`) |
