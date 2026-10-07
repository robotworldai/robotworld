# The Velocity Task: Driving While Balancing

The velocity task teaches the wheeled quadruped to *follow commanded velocities*, drive forward and backward at up to 1 m/s and yaw at up to 1 rad/s, while never losing the two-wheel balance it learned about in the balance task. It is implemented not as a new environment but as a **Python subclass** of the balance environment (`WheeledQuadrupedVelocityEnvCfg(WheeledQuadrupedBalanceEnvCfg)` in `source/wheeled_quadruped/wheeled_quadruped/tasks/velocity/velocity_env_cfg.py`), which adds a command generator, three command observations, and two tracking rewards, then re-weights the balance shaping so that motion is no longer punished.

**Prerequisites / see also:** [Balance Task](05-Balance-Task.md) (everything not mentioned here is inherited from it unchanged), [The Robot](02-The-Robot.md) (actuators, wheel geometry), [RL and MDP Foundations](03-RL-and-MDP-Foundations.md) (rewards, returns), [Isaac Lab Architecture](04-Isaac-Lab-Architecture.md) (managers, command terms), [PPO Algorithm](07-PPO-Algorithm.md) (how it is trained), [Asymmetric Actor-Critic](08-Asymmetric-Actor-Critic-and-Sim2Real.md), [Training and Reproducing](10-Training-and-Reproducing.md).

---

## 1. One task, defined as a *diff* against another

In classical software terms, the velocity task is a **derived class**: it imports the balance task's configuration classes and overrides only what must change. The imports at the top of `velocity_env_cfg.py` (lines 23–27) are literally:

```python
from wheeled_quadruped.tasks.balance.balance_env_cfg import (
    ObservationsCfg, RewardsCfg, WheeledQuadrupedBalanceEnvCfg
)
```

Everything else, the scene (ground plane, lights, robot articulation), the **action space** (2 thigh position targets + 2 wheel velocity targets, 4-dim, see [Balance Task](05-Balance-Task.md)), all five **domain-randomization events** (friction, added base mass, reset pose/joints, periodic pushes), all three **terminations** (timeout at 20 s, tilt beyond $\pi/3$, base below 0.4 m), and the **timing**, is inherited byte-for-byte. In particular, because `__post_init__` calls `super().__post_init__()` *first* (lines 118–132), the velocity task runs with exactly the same clock as balance: physics timestep $dt = 0.005$ s (200 Hz), decimation $D = 4$, control period $\Delta t = D \cdot dt = 0.02$ s ($f_c = 50$ Hz), and `episode_length_s = 20.0`, i.e. **1000 control steps per episode**.

```mermaid
classDiagram
    class WheeledQuadrupedBalanceEnvCfg {
        scene: 4096 envs
        actions: 4-dim (thigh pos, wheel vel)
        observations: policy 16 / critic 20
        rewards: 10 terms
        terminations: 3
        events: 5 (DR)
        dt=0.005, D=4, 20 s episodes
    }
    class WheeledQuadrupedVelocityEnvCfg {
        +commands: base_velocity (new)
        +observations: +3 cmd dims → 19 / 23
        +rewards: +2 tracking, −1 wheel_spin, 3 reweighted
        +wheel_vel scale: 5.0 → 12.0
        (everything else inherited)
    }
    class WheeledQuadrupedVelocityEnvCfg_PLAY {
        32 envs, no noise, no pushes
        fixed command c=(0.5, 0, 0)
    }
    WheeledQuadrupedBalanceEnvCfg <|-- WheeledQuadrupedVelocityEnvCfg
    WheeledQuadrupedVelocityEnvCfg <|-- WheeledQuadrupedVelocityEnvCfg_PLAY
```

Registration is also leaner than for balance: `tasks/velocity/__init__.py` registers only **two** gym IDs, `Wheeled-Quadruped-Velocity-v0` and `Wheeled-Quadruped-Velocity-Play-v0`, and wires **only** an `rsl_rl_cfg_entry_point`. Unlike the balance task, there are no skrl/SB3/rl_games YAML configs anywhere under `tasks/velocity/agents/`; the velocity task can only be trained through the rsl_rl pipeline described in [Training and Reproducing](10-Training-and-Reproducing.md).

---

## 2. The command generator: what "the task" actually asks for

A balance policy needs no external input: "don't fall" is a property of the state. A velocity policy needs to be *told what to do* every moment. In Isaac Lab this is the job of the **command manager** (see [Isaac Lab Architecture](04-Isaac-Lab-Architecture.md)). The velocity task adds one command term (`velocity_env_cfg.py` lines 36–52):

```python
base_velocity = mdp.UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
)
```

The command is the 3-vector from our shared notation,

$$c = (c_x,\; c_y,\; c_z) = (\text{lin-vel-x}^\ast ,\; \text{lin-vel-y}^\ast ,\; \text{ang-vel-z}^\ast ),$$

meaning: desired forward speed $c_x$ (m/s, in the base frame), desired lateral speed $c_y$ (m/s), and desired yaw rate $c_z$ (rad/s). Isaac Lab's `UniformVelocityCommand._resample_command` (v2.3.2, `isaaclab/envs/mdp/commands/velocity_command.py`) draws each component **independently and uniformly** over its configured range:

$$c_x \sim \mathcal{U}(-1,\, 1), \qquad c_y \sim \mathcal{U}(0,\, 0) = 0, \qquad c_z \sim \mathcal{U}(-1,\, 1).$$

Three details of the sampling schedule matter for training:

1. **Resampling every 10 s.** `resampling_time_range=(10.0, 10.0)` is a degenerate interval, the resample period is *exactly* 10 s. Since an episode lasts 20 s, every episode contains **exactly two command windows** of 500 control steps each. The policy must therefore handle command *switches* mid-episode (e.g. full-speed forward → hard reverse), which forces it to learn transient maneuvers, not just steady-state cruising.
2. **Standing environments.** At each resample, every environment flips a biased coin: with probability `rel_standing_envs = 0.1` it becomes a *standing env*, and the command update loop zeroes **all three** components of its command every control step ($c = 0$). So at any time roughly 10% of the 4096 parallel environments are being asked to *stand still while balancing*, this keeps the balance skill from atrophying and teaches a clean stop.
3. **No heading mode.** `heading_command=False` and `rel_heading_envs=0.0` disable Isaac Lab's alternative mode where $c_z$ is computed by a proportional controller on heading error; here $c_z$ is always sampled directly as a raw yaw-rate target.

### Why is $c_y$ pinned to zero?

Because the robot is physically incapable of satisfying any other value. The drivetrain is a **differential drive**: two independently driven rear wheels on a common lateral axis (see [The Robot](02-The-Robot.md)). Each wheel can only produce velocity along its rolling direction, the robot's $x$ axis. There is no actuator that can generate lateral (sideways, $y$) velocity; a wheel would have to *skid* to do so. In robotics language, the platform is subject to a **nonholonomic constraint**: at every instant the body-frame lateral velocity must satisfy

$$v_y \approx 0 \qquad \text{(rolling without lateral slipping)}.$$

Asking a learning agent for $c_y \neq 0$ would be asking for the impossible; the tracking reward would become noise that the policy cannot influence. Pinning `lin_vel_y=(0.0, 0.0)` makes the command distribution match the robot's *feasible* velocity set. (Interestingly, the command term is still 3-dimensional and the tracking reward still includes the $y$ error, with $c_y = 0$ that term becomes a *penalty on lateral skidding*, as we'll see in §5.)

---

## 3. Differential-drive kinematics: from wheel speeds to body velocity

To understand both the $c_y = 0$ pin and the action-scale change, derive how the two rear wheels move the base. Let

- $r$, wheel radius. The code comment in `velocity_env_cfg.py:128–131` states $r \approx 0.1008$ m (comment only; the value lives inside the binary USD asset and is not a separately configured parameter),
- $b$, track width (lateral distance between the two rear wheel contact points), $b \approx 0.44$ m per the same comment,
- $\omega_L, \omega_R$, angular velocities of the left and right rear wheels. In the joint vector $q \in \mathbb{R}^4$ (order $[\text{front-left-thigh}, \text{front-right-thigh}, \text{rl-wheel}, \text{rr-wheel}]$), these are $\dot q_3$ and $\dot q_4$.

Under pure rolling, each wheel's contact point moves forward at its rim speed: $v_L = r\,\omega_L$ and $v_R = r\,\omega_R$. The base sits midway between the wheels, so its forward velocity is the average of the two contact velocities, and the yaw rate comes from their *difference* acting across the track (the two wheels form a lever arm of length $b$ about the midpoint):

$$\boxed{\;v_x = \frac{r\,(\omega_L + \omega_R)}{2}, \qquad \omega_z = \frac{r\,(\omega_R - \omega_L)}{b}\;}$$

In words: spin both wheels equally → drive straight; spin the right wheel faster → the robot pivots left (positive yaw about the up axis, right-hand rule). Inverting the pair gives the wheel speeds required to realize a command $(c_x, c_z)$:

$$\omega_{L} = \frac{c_x}{r} - \frac{b\,c_z}{2r}, \qquad \omega_{R} = \frac{c_x}{r} + \frac{b\,c_z}{2r}.$$

**Worked example, the command extremes.** Plugging in the extreme simultaneous command $c_x = 1.0$ m/s, $c_z = 1.0$ rad/s with $r = 0.1008$ m, $b = 0.44$ m:

$$\frac{c_x}{r} = \frac{1.0}{0.1008} \approx 9.92\ \text{rad/s}, \qquad \frac{b\,c_z}{2r} = \frac{0.44 \times 1.0}{2 \times 0.1008} \approx 2.18\ \text{rad/s},$$

$$\omega_R \approx 9.92 + 2.18 \approx 12.1\ \text{rad/s}.$$

### Why the wheel action scale jumps from 5 to 12

Recall from [Balance Task](05-Balance-Task.md) that the wheel action term is a `JointVelocityActionCfg`: the policy outputs a normalized action $a \in [-1, 1]$ per wheel, and the PD velocity target handed to the actuator is $\dot q^{\ast } = \text{scale} \cdot a$ (no offset, `use_default_offset=False`, so the default wheel velocity 0 contributes nothing). The wheel actuator is a pure velocity servo, $\tau = k_d(\dot q^\ast  - \dot q)$ with $k_d = 10$ and a 100 N·m effort clip (`assets/__init__.py`, lines 55–68).

- Balance used `scale = 5.0`: the wheels can reach at most $\pm 5$ rad/s, i.e. a top speed of $r \cdot 5 \approx 0.50$ m/s, plenty for corrective jiggling in place, far too slow to track $c_x = 1.0$ m/s.
- Velocity sets `self.actions.wheel_vel.scale = 12.0` in `__post_init__` (line ~127): the reachable envelope becomes $\pm 12$ rad/s, which **just covers** the worst-case simultaneous demand of $\approx 12.1$ rad/s computed above (the policy also has closed-loop slack: tracking is rewarded smoothly, not required exactly).

This is a nice example of *reward-feasible action design*: the action scale is derived from the command ranges through the kinematics, so that every command the generator can draw is (approximately) achievable at $|a| \le 1$. The thigh action term is untouched, same `scale = 0.5`, same default-offset targets $q^\ast  = 0.5\,a + q_{\text{default}}$.

---

## 4. Observations: the policy must *see* the command

A command the policy cannot observe is useless. `VelocityObservationsCfg` (lines 56–76) subclasses both observation groups from balance and appends one term to each:

```python
velocity_commands = ObsTerm(func=mdp.generated_commands,
                            params={"command_name": "base_velocity"})
```

`mdp.generated_commands` simply reads the command manager's current buffer, returning $c = (c_x, c_y, c_z) \in \mathbb{R}^3$. Because Isaac Lab concatenates observation terms **in declaration order** and the new term is declared after the five inherited ones, the command occupies the *last three slots* of each vector:

| Group | Balance dims | + `velocity_commands` | Velocity dims |
|---|---|---|---|
| **policy** (actor $\pi_\theta$, noisy) | $\omega$(3) + $g_b$(3) + thigh $q - q_{\text{default}}$(2) + $\dot q$(4) + $a_{t-1}$(4) = **16** | +3 | **19** |
| **critic** ($V_\varphi$, clean, privileged) | $v$(3) + $\omega$(3) + $g_b$(3) + $h$(1) + thigh(2) + $\dot q$(4) + $a_{t-1}$(4) = **20** | +3 | **23** |

The command term is appended to **both** groups: the actor needs it to act, and the critic needs it because the *value of a state depends on what is being commanded* (the same physical state is worth more under an easy command than under a hard one). Note the command itself is not noise-corrupted, it is generated by the environment, not sensed, while the inherited policy terms keep their uniform noise (see [Asymmetric Actor-Critic](08-Asymmetric-Actor-Critic-and-Sim2Real.md) for why the actor is noisy/onboard-only and the critic privileged).

```mermaid
flowchart LR
    CMD["Command manager<br/>resample c ~ U every 10 s<br/>(10% envs: c = 0)"] -->|c appended to obs| PI["Actor π_θ<br/>19-dim obs"]
    CMD -->|c appended to obs| V["Critic V_φ<br/>23-dim obs"]
    PI -->|a ∈ [-1,1]^4| ACT["Action terms<br/>thigh: q* = 0.5a + q_def<br/>wheel: q̇* = 12a"]
    ACT --> SIM["PhysX PD servos<br/>200 Hz"]
    SIM -->|v, ω measured| RWD["Tracking rewards<br/>exp(−err²/0.25)"]
    CMD -->|c| RWD
```

---

## 5. Tracking rewards: the exponential kernel

Two new positive reward terms turn "follow the command" into a learning signal (`VelocityRewardsCfg`, lines 80–100). A naming subtlety worth internalizing when reading logs: the reward **term attributes** are called `track_lin_vel_xy` and `track_ang_vel_z`, but the **functions** they invoke are the stock Isaac Lab kernels `mdp.track_lin_vel_xy_exp` and `mdp.track_ang_vel_z_exp`, that trailing `_exp` names the kernel shape.

**Linear-velocity tracking** (weight $+1.0$, `std` $= \sqrt{0.25} = 0.5$, so $\sigma^2 = 0.25$):

$$f_{\text{lin}} = \exp\!\left(-\frac{\lVert c_{xy} - v_{xy}\rVert^2}{\sigma^2}\right) = \exp\!\left(-\frac{(c_x - v_x)^2 + (c_y - v_y)^2}{0.25}\right),$$

where $v_{xy} = (v_x, v_y)$ is the *measured* base linear velocity in the base frame (the critic-only `base_lin_vel` quantity, the reward manager reads simulator truth, it does not need the actor to observe it). Since $c_y \equiv 0$, the second term reduces to $v_y^2$: any lateral skid directly shrinks the reward, so the kernel doubles as an anti-slip regularizer.

**Yaw-rate tracking** (weight $+0.5$, same $\sigma^2 = 0.25$):

$$f_{\text{ang}} = \exp\!\left(-\frac{(c_z - \omega_z)^2}{0.25}\right),$$

with $\omega_z$ the measured base-frame yaw rate.

### Why an exponential kernel?

Compare with a naive quadratic penalty $-\lVert c - v \rVert^2$. The exponential form $e^{-e/\sigma^2}$ (with $e$ the squared error) has three properties that make training stable:

- **Bounded in $(0, 1]$.** Perfect tracking gives exactly $1$; even a catastrophic error gives $\ge 0$, never an unbounded negative spike. A quadratic penalty explodes when the robot is flung by a domain-randomization push (±0.5 m/s, inherited from balance), producing huge-variance advantages that destabilize [PPO](07-PPO-Algorithm.md). The exponential just quietly saturates near 0.
- **A tunable "good-enough" width.** $\sigma$ sets the error at which reward drops by $1/e$: here an error of $\sigma = 0.5$ m/s yields $e^{-1} \approx 0.368$, an error of $0.25$ m/s yields $e^{-0.25} \approx 0.779$, and an error of $1$ m/s yields $e^{-4} \approx 0.018$. The gradient is steep exactly in the region the robot can act on, and flat where it can't.
- **Always a positive incentive.** Combined with early termination (falling ends the episode, forfeiting all future tracking reward), positive bounded rewards make *staying alive and tracking* strictly better than any alternative, there's no way to "farm" negative-avoidance.

**Per-step magnitudes.** Remember from [Balance Task](05-Balance-Task.md) that Isaac Lab's `RewardManager` multiplies every term by its weight **and** by $\Delta t = 0.02$ s: $r_t = \sum_i w_i f_i(s_t)\,\Delta t$. So perfect tracking contributes $(1.0 \cdot 1 + 0.5 \cdot 1)\cdot 0.02 = 0.03$ per step, i.e. up to $30$ over a full 1000-step episode, compared to at most $5$ from the alive term (next section). The reward budget now decisively favors motion over mere survival.

---

## 6. Reward re-weighting: relaxing the balance shaping

The velocity `__post_init__` (lines 118–132) surgically re-weights the inherited balance rewards:

| Term (function) | Balance $w_i$ | Velocity $w_i$ | Why the change |
|---|---|---|---|
| `alive` (`is_alive`) | $+1.0$ | $+0.25$ | Survival must not dominate. At $+1.0$ the policy could collect $\le 20$/episode by standing perfectly still; at $+0.25$ ($\le 5$/episode) tracking's $\le 30$ is the main prize, so "freeze and survive" is no longer a competitive local optimum. |
| `base_height` (`base_height_l2`, target 0.828 m) | $-20.0$ | $-10.0$ | Accelerating on two wheels requires transient height/pitch excursions (the base dips as the robot leans). Halving the spring constant of this shaping gives the dynamics room to breathe while still forbidding a slow collapse. |
| `flat_orientation` (`flat_orientation_l2`, $g_{b,x}^2 + g_{b,y}^2$) | $-5.0$ | $-2.0$ | A two-wheel balancer *must lean into acceleration*, exactly like a Segway or a person on a hoverboard, the only way to accelerate the base is to pitch so gravity's moment matches the wheel reaction. Punishing tilt at $-5.0$ fights the very mechanism of driving; $-2.0$ keeps uprightness a soft preference. |
| `wheel_spin` (`joint_vel_l2` on the 2 wheels) | $-10^{-3}$ | **removed** (`None`) | This term is $-10^{-3}\sum \dot q_{\text{wheel}}^2$, a direct tax on wheel speed. In balance it suppressed useless spinning; in a driving task it literally penalizes the objective (at $\dot q = 12$ rad/s per wheel it would cost $10^{-3}\cdot 2\cdot 144 \approx 0.29$ per unweighted step, larger than the re-weighted alive bonus ($0.25$) and about $20\%$ of the maximum per-step tracking reward ($1.5$ before $\Delta t$-scaling), i.e. a direct drag on the objective). It is deleted outright: `self.rewards.wheel_spin = None`. |

Unchanged from balance: `terminating` $(-2.0)$, `lin_vel_z` $(-2.0)$, `ang_vel_xy` $(-0.05)$, `joint_torques` $(-10^{-5})$, `joint_acc` $(-2.5\times 10^{-7})$, `action_rate` $(-0.01)$. Tally: 10 balance terms − 1 removed + 2 tracking = **11 active reward terms**, of which 3 carry new weights.

The general lesson: reward terms that *shape* a skill (hold height, stay flat, don't spin) are scaffolding. When the next task needs the robot to exercise the degrees of freedom that scaffolding pinned down, the scaffolding must be loosened, otherwise the old shaping and the new objective pull in opposite directions and the policy converges to a timid compromise.

---

## 7. Training configuration and the Play variant

The velocity PPO runner (`tasks/velocity/agents/rsl_rl_ppo_cfg.py`) keeps **every algorithm hyperparameter identical** to balance ($\gamma = 0.99$, $\lambda = 0.95$, clip $0.2$, entropy coef $0.005$, adaptive LR from $10^{-3}$ at desired KL $0.01$, 24 steps/env, 5 epochs × 4 minibatches, see [PPO Algorithm](07-PPO-Algorithm.md)) and changes only capacity and duration: actor/critic MLPs grow from $[128, 128]$ to $[256, 128, 64]$ (ELU), `max_iterations` grows from 1000 to 3000, and `experiment_name = "wheeled_quadruped_velocity"`. A harder task gets a bigger network and three times the training budget; the optimization recipe itself is unchanged. As in balance, all observation normalization is explicitly off (`empirical_normalization=False`, `actor/critic_obs_normalization=False`).

`WheeledQuadrupedVelocityEnvCfg_PLAY` (lines 136–152) is the demo/evaluation variant used by `scripts/rsl_rl/play.py`. On top of the usual play conveniences inherited from the pattern in balance, 32 environments, observation noise off (`enable_corruption=False`), pushes disabled (`events.push_robot=None`), it makes the command *deterministic*:

- `rel_standing_envs = 0.0`, no environment is randomly zeroed;
- `ranges.lin_vel_x = (0.5, 0.5)`, forward speed fixed at exactly $0.5$ m/s;
- `ranges.ang_vel_z = (0.0, 0.0)`, yaw rate fixed at $0$ (not just $c_x$: the yaw range is zeroed too);
- `lin_vel_y` stays $(0.0, 0.0)$ as always.

Every play robot therefore receives the constant command $c = (0.5,\, 0,\, 0)$: drive straight ahead at half a meter per second, forever, a clean visual check that the two-wheel balancer has genuinely become a vehicle.
