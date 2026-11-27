# Aerial-Balance-Bench

Official implementation of:

**Aerial-Balance-Bench: A Controlled and Reproducible Drone-Ball Balancing Benchmark for Indirect Dynamic Aerial Manipulation**

Aerial-Balance-Bench is an Isaac Lab/Sim based benchmark for studying indirect dynamic aerial manipulation. A vertically constrained tethered drone tilts a beam through a rope, and the beam motion drives a rolling ball to a target position or along a reference trajectory.

The benchmark provides:

- Two task families: target-position balancing and trajectory tracking
- Three high-level command interfaces: thrust, velocity, and position
- A Gym-style Isaac Lab environment with unified observations, actions, rewards, and evaluation metrics
- Robustness tests for mass variation, low-level gain variation, action delay, and external disturbance
- Reference baselines for cascaded PID, nonlinear MPC, and model-free RL

## Contents

- [Overview](#overview)
- [Benchmark Design](#benchmark-design)
- [User Guide](#user-guide)
- [Baselines and Results](#baselines-and-results)
- [Citation](#citation)
- [License](#license)

## Overview

Most aerial manipulation benchmarks focus on direct interaction through robotic arms, grippers, or attached payloads. Aerial-Balance-Bench focuses on **indirect dynamic manipulation**: the drone does not touch the ball directly, but instead regulates the motion of an intermediate beam that moves the ball.

The physical setup contains a drone, rope, beam, and ball. One end of the beam is connected to the drone through a rope, the drone is constrained to move vertically, and the other end of the beam is fixed during nominal operation. This simplified setup isolates the core coupling between aerial actuation, beam inclination, and ball rolling dynamics while keeping the system reproducible in simulation and deployable on a real platform.

The implementation is built on Isaac Lab and exposes a Gym-style interface for controller development, RL training, and evaluation. The benchmark is designed to support controlled comparison between classical feedback control, optimization-based control, learning-based control, robustness methods, and sim-to-real transfer strategies.

<p align="center">
  <img src="docs/figures/benchmark_framework.png" alt="Aerial-Balance-Bench benchmark framework" width="90%">
</p>

<p align="center">
  <sub>Benchmark framework overview. Source figure: <a href="docs/figures/benchmark_framework.png">benchmark_framework.png</a>.</sub>
</p>

## Benchmark Design

### Tasks

| Task | Config value | Goal | Reset/reference design |
| --- | --- | --- | --- |
| Target-position balancing | `task_name: target_position` | Drive the ball to a fixed or sampled target position on the beam. | Initial ball position and target position are sampled from configurable ranges. |
| Trajectory tracking | `task_name: trajectory_tracking` | Track a time-varying reference ball position. | The ball starts from a configurable center position; references can be sine, triangle, trapezoid, or random among these families. |

The ball reference trajectories are visualized below. The constant reference corresponds to the target-position balancing task, while the sine, triangle, and trapezoidal references correspond to trajectory-tracking settings.

<p align="center">
  <img src="docs/figures/reference_trajectories.png" alt="Ball reference trajectories for target-position balancing and trajectory tracking" width="90%">
</p>

### Control interfaces

The selected interface controls how the high-level policy acts on the drone. All interfaces use incremental actions, so the action updates the current command rather than replacing it with an absolute command.

| Interface | Config value | Action meaning | Command executed by low-level controller | Notes |
| --- | --- | --- | --- | --- |
| Thrust command | `interface_name: thrust` | `delta_Frz`, an increment of vertical thrust | SE(3) attitude controller | Most direct actuation, strongest coupling with low-level flight dynamics. |
| Velocity command | `interface_name: velocity` | `delta_vrz`, an increment of drone vertical velocity | SE(3) velocity controller | Main interface used by the reference baselines; practical separation between balancing and flight control, but delay-sensitive. |
| Position command | `interface_name: position` | `delta_drz`, an increment of drone vertical position | SE(3) position controller | Highest-level abstraction, simpler high-level command semantics, usually more lag-prone. |

The command update is:

```text
u_k = u_{k-1} + clip(a_k)
```

where `a_k` is the high-level action and the clip bound is determined by the selected interface configuration.

For the detailed derivation of the dynamic model associated with each control interface, please refer to [dynamic_model.pdf](docs/dynamic_model.pdf).

### Observation space

The policy observation is an 11-D tensor returned under `observations["policy"]`:

```text
[pb, vb, ab, theta, omega, alpha, drz, vrz, arz, pg, a_prev]
```

| Field | Meaning |
| --- | --- |
| `pb` | Ball position along the beam |
| `vb` | Ball velocity along the beam |
| `ab` | Ball acceleration along the beam |
| `theta` | Beam angle |
| `omega` | Beam angular velocity |
| `alpha` | Beam angular acceleration |
| `drz` | Drone vertical displacement |
| `vrz` | Drone vertical velocity |
| `arz` | Drone vertical acceleration |
| `pg` | Desired ball position, fixed for target-position balancing and time-varying for tracking |
| `a_prev` | Previous high-level action |

The same observation layout is used by both benchmark tasks. The index constants are available in `baselines/base_policy.py` as `ObservationIndex`.

### Rewards and termination

Target-position balancing uses a regulation reward with object terms, control effort terms, a failure penalty, and a near-goal bonus:

```text
r = r_object + r_control + r_failure + r_goal
```

Trajectory tracking uses position and velocity tracking terms, control effort, a failure penalty, and progress toward the reference:

```text
r = r_object + r_control + r_failure + r_progress
```

For the exact reward parameters and the mapping between paper notation and code configuration fields, see the [reward design](docs/reward_design.md) document.

Episodes terminate when the episode horizon is reached, the beam angle exceeds the configured limit, the ball leaves the feasible beam range, or the task-specific error exceeds its configured failure threshold.

### Evaluation metrics

For target-position balancing, the evaluator reports:

| Metric | Meaning |
| --- | --- |
| `SR` / `success_rate` | Fraction of episodes that enter the target zone and remain there until the end |
| `SE` / `steady_state_error` | Mean absolute final-window position error |
| `CONT` / `convergence_time` | First time the ball enters the tolerance band and stays there |
| `CLIT` / `climbing_time` | First time the ball enters the tolerance band |
| `COMT` | Controller computation time, reported by policy runners |

For trajectory tracking, the evaluator reports:

| Metric | Meaning |
| --- | --- |
| `MAE` / `mean_absolute_error` | Mean absolute position tracking error |
| `RMSE` / `root_mean_square_error` | Root mean square position tracking error |
| `MAXE` / `maximum_absolute_error` | Worst position tracking error in an episode |
| `COMT` | Controller computation time, reported by policy runners |

### Robustness tests

Robustness settings are configured under `robustness` in the environment YAML files.

| Test | Config fields | Purpose |
| --- | --- | --- |
| Ball-mass variation | `ball_mass_variation_enabled`, `ball_mass_range` | Tests generalization to object parameter changes. |
| Low-level gain variation | `controller_gain_variation_enabled`, `controller_gain_range` | Tests sensitivity to imperfect command tracking by the drone controller. |
| Action delay | `action_delay_enabled`, `delay_step` | Inserts a fixed-step command delay between high-level output and executed command. |
| External disturbance | `external_disturbance_enabled`, OU process parameters | Applies temporally correlated vertical motion at the otherwise fixed beam endpoint. |

## User Guide

### Installation

This repository is meant to be run inside an Isaac Sim / Isaac Lab Python environment.

Recommended one-command setup for the project-side Python dependencies:

```bash
conda env create -f conda_env.yml
conda activate aerial_balance_bench
```

The `conda_env.yml` file is a curated environment specification, not a full export of a developer machine. It includes the packages required by the core benchmark, rollout utilities, NMPC baseline, and RL/RPO baseline.

Isaac Sim and Isaac Lab are still required as platform dependencies. After creating the environment, install or register Isaac Lab for this conda environment following your Isaac Lab installation, then verify:

```bash
python -c "import torch, gymnasium, yaml; import omni.isaac.lab"
```

Required core dependencies:

- Isaac Sim 4.2.0
- Isaac Lab 1.4.0
- Python 3.10
- PyTorch
- Gymnasium
- NumPy
- PyYAML
- tqdm and Matplotlib for rollout utilities

Optional baseline dependencies:

- NMPC: `do-mpc` and `casadi`
- RL/RPO: `skrl`; `wandb` is optional for experiment logging

From the project root, expose the package parent on `PYTHONPATH` when using interactive scripts or notebooks:

```bash
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
```

Run all commands below from the repository root:

```bash
cd /path/to/aerial_balance_bench
```

### Smoke test

Run a short zero-action rollout with the existing smoke-test config:

```bash
# target-position balancing
python3 scripts/zero_action_policy_eval.py \
  --config environments/configs/target_position_balancing.yaml \
  --episodes 10 \
  --num_envs 10 \
  --headless

# trajectory tracking
python3 scripts/zero_action_policy_eval.py \
  --config environments/configs/trajectory_tracking.yaml \
  --episodes 10 \
  --num_envs 10 \
  --headless
```

Logs are written under `logs/zero_action/` unless overridden by the YAML file or `--run_name`.

### Evaluate baselines

CPID:

```bash
python3 scripts/cpid_policy_eval.py \
  --config baselines/configs/cpid_target_position_eval_delay_free.yaml \
  --episodes 10 \
  --num_envs 10 \
  --headless
```

NMPC:

```bash
python3 scripts/nmpc_policy_eval.py \
  --config baselines/configs/nmpc_target_position_eval_delay_free.yaml \
  --n_horizon 25 \
  --episodes 10 \
  --num_envs 2 \
  --headless
```

RL evaluation requires a trained checkpoint:

```bash
python3 scripts/rl_policy_eval.py \
  --config baselines/configs/rl_target_position_rpo_eval_delay_free.yaml \
  --checkpoint /path/to/best_agent.pt \
  --episodes 10 \
  --num_envs 10 \
  --headless
```

### Train an RL policy using RPO

```bash
python3 scripts/rl_train.py \
  --config baselines/configs/rl_target_position_rpo_train.yaml \
  --num_envs 1024 \
  --max_iterations 300 \
  --headless
```

The default RPO training configuration uses the velocity interface and the `legacy8` observation adapter.


### Configuration files

Environment configs live in `environments/configs/`; baseline configs live in `baselines/configs/`.

Common environment fields:

| Field | Meaning |
| --- | --- |
| `task_name` | Selects `target_position` or `trajectory_tracking`. |
| `interface_name` | Selects `velocity`, `position`, or `thrust`. |
| `env` | Sets seed, number of parallel environments, episode length, device, and selected physical constants. |
| `target_position_task` | Target-position reset ranges, goal sampling, reward weights, and failure threshold. |
| `trajectory_tracking_task` | Reference type, amplitude/period settings, randomization, reward weights, and failure threshold. |
| `velocity_interface`, `position_interface`, `thrust_interface` | Interface-specific action limits and low-level controller settings. |
| `robustness` | Enables mass, gain, delay, and disturbance tests. |
| `target_position_evaluator`, `trajectory_tracking_evaluator` | Evaluation episode count, tolerance, and final-window settings. |
| `runner` | Evaluation episode target, maximum rollout steps, rendering, and rollout saving. |
| `logging` | Output root and run name. |

Template configs:

- `environments/configs/target_position_balancing.yaml`
- `environments/configs/trajectory_tracking.yaml`


### Implement a custom policy

Policies should implement the minimal high-level interface in `baselines/base_policy.py`:

```python
import torch

from aerial_balance_bench.baselines.base_policy import BasePolicy, BasePolicyCfg, ObservationIndex


class MyPolicy(BasePolicy):
    def __init__(self, cfg: BasePolicyCfg, num_envs: int, device: str | torch.device, step_dt: float):
        super().__init__(cfg, num_envs, device)
        self.step_dt = float(step_dt)

    def reset(self, env_ids=None):
        # Reset policy-local state for all envs or selected env ids.
        return None

    def act(self, observations, extras=None) -> torch.Tensor:
        obs = self._extract_policy_observation(observations)
        error = obs[:, ObservationIndex.PB] - obs[:, ObservationIndex.PG]
        action = -0.01 * error.unsqueeze(-1)
        return action
```

The returned action must be a torch tensor with shape `(num_envs, 1)` on the environment device. The physical meaning and valid range depend on the selected control interface. For a practical custom runner, copy the structure of `scripts/cpid_policy_eval.py`, replace `CPIDPolicy` with your policy class, and keep the same logging/evaluator flow.


## Baselines and Results

### Reference control framework

The reference baselines use the velocity-command interface. This gives the high-level controller a practical command abstraction, but also introduces latency because the low-level drone controller must track the commanded velocity.

To compensate for action delay, the repository includes a model-based state predictor in `baselines/model_state_predictor.py`. Predictor-enabled configs, such as `cpid_predictor_15.yaml`, `nmpc_predictor_15.yaml`, and `rl_rpo_predictor_15.yaml`, use a velocity-interface model to predict the future observation after the configured delay horizon.

### Baselines

| Baseline | Main files | Idea |
| --- | --- | --- |
| Cascaded PID | `baselines/cpid_policy.py`, `baselines/configs/cpid.yaml` | Outer-loop incremental PID generates a beam-angle reference from ball-position error; inner-loop incremental PID generates a vertical-velocity increment. |
| NMPC | `baselines/nmpc_policy.py`, `baselines/nmpc_core.py`, `baselines/configs/nmpc.yaml` | Solves a nonlinear optimal control problem over velocity-interface dynamics using do-mpc/CasADi. |
| RL/RPO | `baselines/rl_policy.py`, `baselines/rl_models.py`, `baselines/configs/rl_rpo.yaml` | Uses skrl RPO/PPO-compatible MLP actor-critic models; the default policy uses an 8-D adapted observation. |

### Simulation results

The paper evaluates target-position balancing at 60 Hz with maximum vertical acceleration `0.5 m/s^2`, nominal ball mass `0.0005 kg`, low-level velocity-controller gain `10`, target tolerance `0.01 m`, and steady-state window `1 s`.

Key takeaways:

- In delay-free simulation, CPID, NMPC with longer horizons, and RL all achieve strong balancing performance.
- Fixed action delay without compensation strongly degrades all controllers.
- With model-predictive delay compensation, CPID remains the strongest practical baseline in the reported delayed setting.
- Mass variation, low-level gain variation, external disturbance, and longer delays remain challenging for all methods.

The six policies trained with the RPO algorithm from six random seeds are saved in `baselines/rpo_models/`.

For the detailed simulation result tables, robustness protocols, and long-delay evaluations, see the [simulation result details](docs/simulation_results.md).

### Real-world results

The real-world experiments directly transfer the simulation-designed or simulation-trained controllers without real-world fine-tuning. The paper evaluates three target positions, `pg = 0.15`, `0.35`, and `0.55`, with target tolerance `0.03 m`, steady-state window `5 s`, 9 sampled initial ball positions per target, and 20 s episodes.

For the detailed result table, real-world trajectory plots, and experiment video, see the [real-world experiment details](docs/real_world_results.md).

CPID achieves relatively good balancing performance at the center target, while off-center targets remain difficult due to unmodeled or non-ideal dynamics such as beam bending.

## Citation

If you use this benchmark in your research, please cite the project paper. Replace the placeholder fields below with the official metadata once available:

```bibtex
@article{aerial_balance_bench_2026,
  title   = {Aerial-Balance-Bench: A Controlled and Reproducible Drone-Ball Balancing Benchmark for Indirect Dynamic Aerial Manipulation},
  author  = {Author names to be added},
  journal = {Venue to be added},
  year    = {2026}
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
