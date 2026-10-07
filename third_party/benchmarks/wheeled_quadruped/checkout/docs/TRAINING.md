# Training & RL design

This is the design document for the reinforcement-learning tasks in this repo: what the robot observes, what it controls, how it is rewarded, why each choice was made, and how to tune it when training misbehaves. It reflects the code in [`source/wheeled_quadruped/`](../source/wheeled_quadruped) exactly — if the two ever disagree, the code wins.

> Companion docs: [README.md](../README.md) for the overview, [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for install/run.

## Contents

- [The robot as a control problem](#the-robot-as-a-control-problem)
- [Task 1 — Balance](#task-1--balance)
- [Task 2 — Velocity tracking](#task-2--velocity-tracking)
- [Observations & the asymmetric actor-critic](#observations--the-asymmetric-actor-critic)
- [Actions & scaling](#actions--scaling)
- [Rewards](#rewards)
- [The base-height sign bug — a cautionary tale](#the-base-height-sign-bug--a-cautionary-tale)
- [Terminations](#terminations)
- [Domain randomization & events](#domain-randomization--events)
- [Why 20-second episodes](#why-20-second-episodes)
- [Expected convergence](#expected-convergence)
- [Tuning guide](#tuning-guide)
- [Future work](#future-work)

## The robot as a control problem

The robot is pitched up, resting on its **two rear wheels** at a base height of **0.828 m** — a segway on two wheels rather than one. Only four joints are actuated: the two rear wheels (continuous, velocity-controlled) and the two front thighs (revolute, ±0.785 rad, position-controlled). The two front wheels are fixed.

This makes it an **unstable, underactuated inverted pendulum in 3D**. Left alone it falls. Balancing on two coaxial rear wheels also means it behaves like a **differential drive**: it can move forward/backward and yaw, but it *cannot* translate sideways. That constraint shows up directly in the velocity task (lateral command pinned to zero).

## Task 1 — Balance

`Wheeled-Quadruped-Balance-v0` (+ `-Play-v0`, and the legacy alias `Custom-Wheeled-Quadruped-v0`).

**Goal:** stand still and upright at 0.828 m, indefinitely, and recover from disturbances. No commanded motion — the ideal behavior is to hold position. This is the foundation task; the velocity task is built on top of it.

The play variant (`WheeledQuadrupedBalanceEnvCfg_PLAY`) shrinks the scene to 32 envs, disables observation noise, and removes the periodic push — a clean deterministic rollout for eyeballing a trained policy.

## Task 2 — Velocity tracking

`Wheeled-Quadruped-Velocity-v0` (+ `-Play-v0`).

**Goal:** keep balancing while **driving to a commanded velocity** — forward/backward linear velocity and yaw rate. The task extends the balancer:

- Adds a `UniformVelocityCommand` resampled every 10 s, with `lin_vel_x ∈ [−1, 1]`, `ang_vel_z ∈ [−1, 1]`, and **`lin_vel_y ≡ 0`** (the two-wheel stance cannot translate laterally). 10 % of envs get a zero "stand still" command so the policy does not forget how to hold.
- Appends the command vector to **both** observation groups (the policy must know what it is being asked to do).
- Adds two exponential tracking rewards (below) and **softens the balance-hold shaping** so the policy is free to move: `alive` 1.0 → 0.25, `base_height` −20 → −10, `flat_orientation` −5 → −2, and the **wheel-spin penalty is removed entirely** (the wheels are *supposed* to spin now).

The play variant fixes a constant 0.5 m/s forward command for every env for a clean demo.

## Observations & the asymmetric actor-critic

The single most important design decision is that **the deployed policy only sees what a real robot could measure with onboard sensors.** Concretely, the actor never sees its world-frame linear velocity or its absolute height — neither is directly observable on the hardware (there is no motion-capture, no external positioning).

But those signals are *extremely* helpful for learning to balance, and they are free in simulation. So we give them to the **critic only**. PPO's critic (the value function) is used purely to compute advantages during training and is **discarded at deployment**. Feeding it privileged state lowers value-estimate variance and speeds up learning without contaminating the policy with signals it will not have on the real robot. This is a standard **asymmetric actor-critic**.

**Policy group (actor) — 16 dims, onboard-obtainable, corrupted with noise during training:**

| Term | mdp func | Size | Onboard source | Train noise |
|---|---|:---:|---|---|
| Base angular velocity | `base_ang_vel` | 3 | IMU gyro | ±0.2 |
| Projected gravity | `projected_gravity` | 3 | IMU accel → gravity dir. | ±0.05 |
| Front-thigh positions | `joint_pos_rel` (thighs) | 2 | Thigh encoders | ±0.01 |
| Joint velocities | `joint_vel_rel` (all 4) | 4 | Joint encoders | ±1.5 |
| Last action | `last_action` | 4 | Command buffer | — |

**Critic group (value only) — 20 dims, privileged, no noise:**

| Term | mdp func | Size | Privileged? |
|---|---|:---:|:---:|
| Base **linear** velocity | `base_lin_vel` | 3 | **yes** |
| Base angular velocity | `base_ang_vel` | 3 | |
| Projected gravity | `projected_gravity` | 3 | |
| Base **height** (z) | `base_pos_z` | 1 | **yes** |
| Front-thigh positions | `joint_pos_rel` (thighs) | 2 | |
| Joint velocities | `joint_vel_rel` (all 4) | 4 | |
| Last action | `last_action` | 4 | |

The mapping is wired up in the rsl_rl runner cfg: `obs_groups = {"policy": ["policy"], "critic": ["critic"]}`. In rsl-rl ≥ 3.0 this lives in the runner cfg, not the env wrapper. The velocity task appends a 3-dim `velocity_commands` term to both groups (→ 19 / 23 dims).

**Why the specific noise magnitudes?** They approximate real sensor error so the policy does not overfit to simulator-perfect readings. Joint-velocity noise is the largest (±1.5) because raw encoder-differenced velocity is genuinely noisy; gravity direction is trusted most (±0.05) because a filtered IMU gravity estimate is quite stable. `enable_corruption = True` turns this on for the policy group and is switched off in the play/eval variants.

## Actions & scaling

Four continuous actions, produced at **50 Hz**:

| Action | mdp cfg | Joints | Scale | Semantics |
|---|---|---|:---:|---|
| Thigh position | `JointPositionActionCfg` (`use_default_offset=True`) | 2 front thighs | **0.5** | target = default + 0.5·a rad |
| Wheel velocity | `JointVelocityActionCfg` | 2 rear wheels | **5.0** | target = 5.0·a rad/s |

**Scaling rationale.** The thigh scale of **0.5** keeps a unit action within roughly the ±0.785 rad joint limit while centering on the default (upright) pose via `use_default_offset` — so "do nothing" (a ≈ 0) means "hold the balancing pose," which is exactly the behavior we want the network to fall back to. The wheel scale of **5.0** gives the policy enough wheel-speed authority (±5 rad/s at unit action) to make the fore/aft corrections balancing needs, without being so large that small action noise produces violent lurches. The thighs are position-controlled (stiff, stiffness 1000) because they set posture; the wheels are velocity-controlled (zero stiffness, damping 10) because balancing is fundamentally about commanding wheel *speed*.

## Rewards

Balance-task weights, straight from [`balance_env_cfg.py`](../source/wheeled_quadruped/wheeled_quadruped/tasks/balance/balance_env_cfg.py):

| Term | Function | Weight | Sign meaning | Purpose |
|---|---|:---:|---|---|
| `alive` | `is_alive` | **+1.0** | bonus | A per-step "stay up" incentive; the spine of the reward |
| `terminating` | `is_terminated` | **−2.0** | penalty | One-off cost for falling, on top of losing the alive stream |
| `base_height` | `base_height_l2` (target 0.828) | **−20.0** | penalty (squared error) | **The primary objective** — hold the balancing height |
| `flat_orientation` | `flat_orientation_l2` | **−5.0** | penalty | Keep the torso upright (gravity aligned with body z) |
| `lin_vel_z` | `lin_vel_z_l2` | **−2.0** | penalty | Damp vertical bouncing |
| `ang_vel_xy` | `ang_vel_xy_l2` | **−0.05** | penalty | Damp roll/pitch wobble |
| `joint_torques` | `joint_torques_l2` | **−1.0e−5** | penalty | Discourage brute-force effort |
| `joint_acc` | `joint_acc_l2` | **−2.5e−7** | penalty | Smooth out jerk |
| `action_rate` | `action_rate_l2` | **−0.01** | penalty | Penalize twitchy step-to-step command changes |
| `wheel_spin` | `joint_vel_l2` (wheels only) | **−1.0e−3** | penalty | Stop the robot quietly **driving away** while "balancing" |

**The `wheel_spin` penalty deserves a note.** Without it, a lazy but valid solution to "stay upright at 0.828 m" is to keep rolling — a moving inverted pendulum is easy to balance dynamically, but it drifts off across the plane and is *not* the standing behavior we want. Penalizing squared wheel velocity pins the balance task to **balancing in place**. It is deliberately removed in the velocity task, where spinning wheels are the whole point.

The velocity task adds:

| Term | Function | Weight | Purpose |
|---|---|:---:|---|
| `track_lin_vel_xy` | `track_lin_vel_xy_exp` (std = √0.25) | **+1.0** | Reward matching the commanded linear velocity (exp kernel, 0→1) |
| `track_ang_vel_z` | `track_ang_vel_z_exp` (std = √0.25) | **+0.5** | Reward matching the commanded yaw rate |

These use an **exponential kernel** — `exp(−error² / std²)` — which is 1.0 at perfect tracking and decays smoothly, giving dense gradient toward the command without a hard cliff.

## The base-height sign bug — a cautionary tale

`base_height_l2` returns `square(base_z − target_height)` — a **squared error**, i.e. a *cost* that is zero at the target and grows the further away you get. To make the robot *hold* the target height, its weight must be **negative** (penalize the error). This is now `−20.0`.

The earlier prototype gave this term a **positive** weight. A positive weight on a squared *error* means the optimizer is rewarded for **maximizing** deviation from the target — the exact opposite of the goal. It literally paid the robot to leave 0.828 m. This is the single easiest way to silently break a locomotion reward: **any `*_l2` / `*_error` / squared-distance term is a cost and must carry a negative weight.** If a policy trains but converges to something absurd (e.g. it collapses or launches instead of holding), check the signs on your squared-error terms first.

## Terminations

| Term | Function | Condition | Kind |
|---|---|---|---|
| `time_out` | `time_out` | Episode reaches 20 s | truncation (bootstrapped) |
| `bad_orientation` | `bad_orientation` | Torso tilt > **π/3 (60°)** | failure |
| `base_too_low` | `root_height_below_minimum` | Base height < **0.4 m** | failure |

`time_out` is flagged as a time-out (not a failure), so PPO bootstraps the value at the horizon rather than treating the cutoff as a fall. The two failure terms define "has fallen": tipped past 60°, or sunk below 0.4 m (roughly half the standing height). Both end the episode and trigger the `−2.0` `terminating` penalty.

## Domain randomization & events

Randomization is applied through the event manager. Each term protects against a specific brittleness:

| Event | Mode | Range | Protects against |
|---|---|---|---|
| `physics_material` | startup | static friction 0.5–1.25, dynamic 0.4–1.0, restitution 0.0–0.05 (64 buckets, all bodies) | Overfitting to one wheel/ground friction; real surfaces vary |
| `add_base_mass` | startup | trunk body (`robot1_base_footprint`, ~17.9 kg after the USD merged its fixed-joint links) mass **+(−1.0 … +2.0) kg** | Payload / mass-estimate error; keeps balance robust to CoM shift |
| `reset_base` | reset | pose x,y ±0.1 m, yaw ±π; velocity all axes ±0.1 | Memorizing one start pose/heading; forces balance from varied initial states |
| `reset_joints` | reset | joint pos/vel offset ±0.1 | Same, at the joint level |
| `push_robot` | interval (every 10–15 s) | body-frame velocity kick x,y ±0.5 m/s | Un-modeled disturbances; teaches active recovery, not just static holding |

The **periodic push** is what turns "stand still if undisturbed" into "actively recover" — it repeatedly knocks the robot mid-episode so the policy must learn corrective control, not a fragile fixed point. All randomization is disabled in the `_PLAY` variants for clean evaluation.

## Why 20-second episodes

The prototype used 5 s episodes. Balancing is a **long-horizon** stability problem: at 5 s the robot barely experiences one push cycle and can score well by simply not having fallen *yet*, which rewards precarious near-falls. **20 s** (1000 control steps at 50 Hz) is long enough to (a) let the periodic push fire at least once and demand a genuine recovery, and (b) make transient "lucky" balances score worse than truly stable ones, because instability compounds over the extra time. It also gives the velocity task room to actually track a command through at least one resample (every 10 s).

## Expected convergence

| Task | rsl_rl `max_iterations` | Practical convergence |
|---|:---:|---|
| Balance | 1000 | Typically a **few hundred** iterations to a stable stand; 1000 is a generous ceiling |
| Velocity | 3000 | Expect **1000–3000**; tracking-while-balancing is a harder joint objective |

With 4096 envs × 24 steps/iteration, a few hundred iterations is on the order of tens of millions of environment steps — minutes-to-an-hour class on an RTX 4090, headless. Watch `Mean/reward` and the episode length in TensorBoard: rising episode length toward the 20 s cap (1000 steps) is the clearest "it's learning to stay up" signal.

## Tuning guide

If training fails or plateaus, adjust in roughly this order:

1. **Signs first.** Confirm every squared-error term (`base_height_l2`, `flat_orientation_l2`, the `*_l2` penalties) has a **negative** weight. See the [sign-bug section](#the-base-height-sign-bug--a-cautionary-tale). This is the most common and most catastrophic error.
2. **`base_height` vs `flat_orientation` balance.** These are the two shaping forces that define "standing." If the robot **squats or over-corrects height** while staying level, its height penalty (−20) is too strong relative to orientation — lower its magnitude or raise `flat_orientation`. If it **holds height but leans/tips**, do the reverse. Tune them as a *ratio*, not independently.
3. **Entropy coefficient (`entropy_coef`, 0.005).** If the policy collapses early to a rigid, low-return behavior (premature determinism), raise it (e.g. 0.01) to keep exploring. If it stays jittery and never sharpens, lower it.
4. **Push magnitude (`push_robot` velocity range, ±0.5).** If the policy never becomes robust — falls on the first disturbance at eval — the pushes may be too weak to teach recovery; increase them. If it *cannot* get off the ground at all early in training, the pushes may be overwhelming a not-yet-competent policy; soften them or lengthen the interval, then restore once it can stand.
5. **`action_rate` / smoothness penalties.** If the motion is buzzy/vibrating, increase `action_rate` (−0.01) magnitude. If the policy is sluggish and under-reacts, decrease it.
6. **Velocity task only:** if it balances but won't drive, the balance-hold terms are still dominating — the config already softens them (`alive` 0.25, `base_height` −10, `flat_orientation` −2, wheel-spin off); push those further and/or raise the tracking-reward weights (currently +1.0 / +0.5).

Change **one axis at a time** and re-evaluate with the `_PLAY` variant so noise/pushes don't muddy the read.

## Future work

- **Curriculum.** Both tasks currently train at full difficulty from step 0 (full push magnitude, full command range, full DR). A curriculum — easing pushes/commands in as competence grows, à la Isaac Lab's terrain-levels and command-range curricula — would likely speed convergence and improve the final policy, especially for velocity tracking. This is deliberately left out for now to keep the first version simple and legible.
- **Velocity-task tuning.** The reward re-weighting for the velocity task is a first pass and has not been trained to convergence; expect to iterate on the tracking-vs-balance weight ratio.
- **Sim-to-real.** The onboard-only policy design exists specifically to make hardware transfer possible; deploying it is the next milestone after the velocity task lands.
