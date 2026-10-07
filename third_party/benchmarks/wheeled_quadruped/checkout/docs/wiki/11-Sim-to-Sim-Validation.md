# Sim-to-Sim Validation in MuJoCo

**Abstract.** A policy trained in one simulator can quietly overfit that simulator's quirks: its exact contact solver, its integrator, its numerical friction. This chapter is about the cheapest honest test of whether that has happened, running the *same* trained policy in a *different* physics engine. We take the four Isaac Sim (PhysX) policies and run them unchanged in **MuJoCo**, rebuilt from the robot's URDF, and read off how well the behaviour survives the engine swap. It is the dress rehearsal for the sim-to-real gap, and it lives in [`sim2sim_mujoco/`](https://github.com/MickyasTA/wheeled_quadruped_robot/tree/main/sim2sim_mujoco).

**Prerequisites / see also:** [Asymmetric Actor-Critic and Sim-to-Real](08-Asymmetric-Actor-Critic-and-Sim2Real.md) (why the actor is onboard-only, which is what makes this test meaningful), [The Robot](02-The-Robot.md) (the URDF and joints), [Balance Task](05-Balance-Task.md) and [Velocity Task](06-Velocity-Task.md) (the observation and action conventions we must match), [Training and Reproducing](10-Training-and-Reproducing.md).

<div align="center">

<img src="../images/sim2sim_demo.gif" width="75%" alt="The Isaac-trained policy running in MuJoCo, a different physics engine, driving over rough terrain">

<sub><i>The same trained policy, running in MuJoCo instead of the Isaac Sim it was trained in.</i></sub>

</div>

---

## 1. Why bother running it in a second simulator

Reinforcement learning optimizes a policy against whatever dynamics it is given. If those dynamics are a simulator, the policy is free to exploit anything specific to that simulator: a particular way contacts are resolved, a particular friction cone approximation, a particular integrator error. None of that exists on real hardware, so a policy that leans on it will transfer poorly.

You cannot easily measure that overfitting inside the training simulator, because the training simulator is exactly what the policy was tuned to. You need a *second opinion* from a different engine. MuJoCo and PhysX (which Isaac Sim uses) are independent implementations: different solvers, different contact models, different numerics. If a policy trained in PhysX still works in MuJoCo, it is far less likely to be exploiting engine-specific artifacts, and far more likely to survive the next and largest gap, the one to a physical robot. Sim-to-sim is the cheap, fast, non-destructive rehearsal for sim-to-real.

This test is only meaningful *because* of the design in [chapter 8](08-Asymmetric-Actor-Critic-and-Sim2Real.md): the deployed actor sees only onboard-obtainable signals. If the actor consumed privileged simulator state (base linear velocity, absolute height), a sim-to-sim run would either be impossible or dishonest. Because the actor is already restricted to what a real IMU and joint encoders provide, running it in MuJoCo is exactly the same operation as running it on hardware, just with MuJoCo standing in for the real world.

## 2. How the validation is built

Three pieces, all in `sim2sim_mujoco/`.

**The MuJoCo model is rebuilt from the URDF, not copied from Isaac.** `make_model.py` imports the robot's ROS URDF into MuJoCo. MuJoCo's handling of the URDF fixed joints reproduces exactly the merged articulation Isaac trains on: the welded legs, rear thighs, and shins collapse into one rigid base of about 17.9 kg, leaving the two front-thigh subtrees and the two rear wheels as the only moving bodies. The script then floats the base with a free joint, adds a floor (flat) or a rough height field, and adds actuators that match Isaac Lab's implicit PD:

$$
\tau_{\text{thigh}} = k_p\,(q^{\ast } - q) - k_d\,\dot q, \quad k_p = 1000,\ k_d = 20, \qquad
\tau_{\text{wheel}} = k_v\,(\dot q^{\ast } - \dot q), \quad k_v = 10.
$$

Because the model comes from the URDF rather than the Isaac USD, the contact model, solver, and terrain all genuinely differ. This is a cross-engine test, not a replay of recorded states.

**The policy runs in pure numpy.** The actor is a small ELU MLP with no observation normalization (see [chapter 7](07-PPO-Algorithm.md)), so `export_policies.py` dumps its Linear layers to `policies/*.npz` and `sim2sim.py` does the forward pass with numpy alone. No torch and no onnxruntime are needed in the MuJoCo environment. This is deliberate: it is the same minimal inference path you would run on an embedded computer with the exported ONNX policy.

**The observation and action conventions are matched exactly.** This is the part that makes or breaks a sim-to-sim run. The observation is assembled in the base (root) frame in the Isaac joint order `[FL_thigh, FR_thigh, rl_wheel, rr_wheel]`: base angular velocity, projected gravity $g_b = R_b^{\mathsf T}[0,0,-1]$, front-thigh positions, joint velocities, the last action, and for the velocity tasks the 3-value command. The action maps to targets exactly as in training, a thigh position target of $q^{\ast } = 0.5\,a$ and a wheel velocity target of $\text{scale}\cdot a$ (scale 5 for balance, 12 for velocity). Control runs at 50 Hz over a 200 Hz sim (decimation 4), matching the Isaac timing. Get any of these wrong, the joint order, a sign, the scale, and the policy receives nonsense and falls immediately, which also makes this a strict correctness check on the whole observation-action contract.

## 3. Results

Twenty episodes per policy, 1000 control steps each (a full 20 seconds), with small reset randomization:

| Policy | MuJoCo episode length | Full-episode survival | Lin-vel MAE | Yaw-rate MAE |
|---|:---:|:---:|:---:|:---:|
| Balance | 1000 / 1000 | 100% | | |
| Velocity (cmd 0.5 m/s) | 1000 / 1000 | 100% | 0.26 m/s | 0.06 rad/s |
| Balance-rough | 1000 / 1000 | 100% | | |
| Velocity-rough (cmd 0.5 m/s) | ~956 / 1000 | 20% | 0.17 m/s | 0.15 rad/s |

All four policies transfer well to an engine they never saw during training. The two balance policies and the flat velocity policy survive every episode. The hardest task, driving while balancing on rough terrain, stays up about 96% of the episode on average and tracks its command closely, but it is the only policy that ever falls (it completes the full 20 seconds about one time in five). That is exactly the signal a sim-to-sim check exists to give: not a binary "it works", but a ranking of which policy sits closest to the edge of its robustness, and therefore which one most needs more domain randomization before it meets hardware.

## 4. An honest lesson: the terrain has to be faithful too

The first version of the MuJoCo rough terrain was wrong in an instructive way. I built it as roughly 6 cm of high-frequency bumps, which turned out to be *rougher at the wheel scale* than the Isaac terrain, which is mostly gentle slopes with small roughness on top. On that too-harsh terrain the velocity-rough policy fell far more often and its yaw error was 0.45 rad/s. Rebuilding the height field as rolling hills plus small bumps, faithful to what the policy actually trained on, dropped the yaw error to 0.15 rad/s and roughly tripled the full-episode survival.

The lesson generalizes: a sim-to-sim (or sim-to-real) gap is only as trustworthy as the environment you measure it in. A transfer "failure" can be the test environment being unfaithful rather than the policy being fragile. Matching the validation environment to the training distribution is part of the experiment, not an afterthought.

## 5. What it means for sim-to-real

Three of four policies surviving an independent physics engine unchanged is real evidence that they are not PhysX artifacts, and that the onboard-only observation design is doing its job. The one policy that occasionally falls in MuJoCo is precisely the one to harden (more aggressive domain randomization, or a smaller train-test terrain gap) before spending time on a physical robot. And because the whole inference path is already the deployable one, a numpy or ONNX forward pass over onboard signals, the step from "runs in MuJoCo" to "runs on hardware" is mostly about the sensors and the state estimator, not the policy.

See [`sim2sim_mujoco/README.md`](https://github.com/MickyasTA/wheeled_quadruped_robot/tree/main/sim2sim_mujoco) for the exact commands to reproduce every number here and to watch any policy live in the MuJoCo viewer.
