# Reward Design

This document lists the reward parameters used by Aerial-Balance-Bench and maps the notation in the paper to the implementation fields in the code. The task-specific weights are defined in `TargetPositionTaskCfg` and `TrajectoryTrackingTaskCfg`; the shared beam-angle limit `theta_max` is defined by `AerialBalanceEnvCfg.max_theta`.

## Target-Position Balancing

The target-position reward uses the ball-position error

```text
e_k = p_b(t_k) - p_g(t_k)
```

and is implemented as:

```text
r = r_object + r_control + r_failure + r_goal

r_object = -k_1 e_k^2 - k_2 v_b^2
r_control = -k_3 u_k^2 - k_4 a_k^2
r_failure = -k_5, if |theta| > theta_max or |e_k| > e_max; otherwise 0
r_goal = (c - k_7 |e_k|) exp(-k_6 |v_b|), if |e_k| < e_goal; otherwise 0
```

| Paper parameter | Code field | Default value | Source |
| --- | --- | ---: | --- |
| `k_1` | `position_weight` | `5.0` | `TargetPositionTaskCfg` |
| `k_2` | `ball_velocity_weight` | `0.5` | `TargetPositionTaskCfg` |
| `k_3` | `command_weight` | `0.5` | `TargetPositionTaskCfg` |
| `k_4` | `action_weight` | `1.0` | `TargetPositionTaskCfg` |
| `k_5` | `failure_penalty` | `500.0` | `TargetPositionTaskCfg` |
| `k_6` | `goal_velocity_decay` | `1.0` | `TargetPositionTaskCfg` |
| `k_7` | `goal_position_decay` | `60.0` | `TargetPositionTaskCfg` |
| `theta_max` | `max_theta` | `50 deg` (`0.8727 rad`) | `AerialBalanceEnvCfg` |
| `e_max` | `max_error_for_failure` | `0.7` | `TargetPositionTaskCfg` |
| `e_goal` | `goal_radius` | `0.05` | `TargetPositionTaskCfg` |
| `c` | `goal_bonus` | `5.0` | `TargetPositionTaskCfg` |

In code, `u_k` is the interface-specific `command_z`, and `a_k` is the high-level action increment. For the velocity interface, these correspond to the commanded vertical velocity and its increment.

## Trajectory Tracking

The trajectory-tracking reward uses the ball-position error and reference velocity:

```text
e_k = p_b(t_k) - p_g(t_k)
v_g,k = reference velocity at t_k
```

and is implemented as:

```text
r = r_object + r_control + r_failure + r_progress

r_object = -hat{k}_1 e_k^2 - hat{k}_2 (v_b - v_g,k)^2
r_control = -hat{k}_3 u_k^2 - hat{k}_4 a_k^2
r_failure = -hat{k}_5, if |theta| > theta_max or |e_k| > e_max; otherwise 0
r_progress = hat{k}_6 (|e_{k-1}| - |e_k|)
```

| Paper parameter | Code field | Default value | Source |
| --- | --- | ---: | --- |
| `hat{k}_1` | `position_weight` | `5.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_2` | `velocity_weight` | `0.5` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_3` | `command_weight` | `0.5` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_4` | `action_weight` | `1.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_5` | `failure_penalty` | `500.0` | `TrajectoryTrackingTaskCfg` |
| `hat{k}_6` | `progress_weight` | `1.0` | `TrajectoryTrackingTaskCfg` |
| `theta_max` | `max_theta` | `50 deg` (`0.8727 rad`) | `AerialBalanceEnvCfg` |
| `e_max` | `max_error_for_failure` | `0.5` | `TrajectoryTrackingTaskCfg` |


## Custom Configuration

All reward weights can be customized through the configuration file.
