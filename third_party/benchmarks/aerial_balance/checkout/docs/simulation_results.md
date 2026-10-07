# Simulation Results

This page contains the detailed simulation result tables referenced from the main README. The paper evaluates target-position balancing at 60 Hz with maximum vertical acceleration `0.5 m/s^2`, nominal ball mass `0.0005 kg`, low-level velocity-controller gain `10`, target tolerance `0.01 m`, and steady-state window `1 s`.

## Standard Target-Position Simulation Results

| Setting | Controller | SR (%) up | SE (mm) down | CONT (s) down | CLIT (s) down | COMT (ms) down |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Delay-free | Cascaded PID | 100 | 0.209 +/- 0.122 | 3.278 +/- 1.345 | 2.352 +/- 0.138 | 0.444 |
| Delay-free | NMPC, horizon=15 | 0 | 66.419 +/- 35.714 | 10.017 +/- 0.000 | 8.451 +/- 1.849 | 13.239 |
| Delay-free | NMPC, horizon=20 | 100 | 27.729 +/- 14.943 | 9.751 +/- 0.148 | 6.755 +/- 2.671 | 14.968 |
| Delay-free | NMPC, horizon=25 | 99.600 | 14.536 +/- 7.838 | 9.369 +/- 0.727 | 5.644 +/- 2.106 | 17.510 |
| Delay-free | NMPC, horizon=30 | 41.400 | 18.284 +/- 9.742 | 9.148 +/- 1.605 | 5.380 +/- 1.609 | 21.066 |
| Delay-free | NMPC, horizon=35 | 100 | 1.458 +/- 0.793 | 6.615 +/- 1.644 | 5.083 +/- 1.193 | 22.999 |
| Delay-free | NMPC, horizon=40 | 100 | 0.162 +/- 0.085 | 4.804 +/- 0.896 | 4.727 +/- 0.991 | 27.156 |
| Delay-free | RL | 100 +/- 0 | 4.597 +/- 2.641 | 4.158 +/- 1.501 | 2.783 +/- 0.898 | 0.767 |
| Delay without compensation | Cascaded PID | 4.400 | 144.326 +/- 83.208 | 10.013 +/- 0.023 | 1.558 +/- 0.109 | 0.468 |
| Delay without compensation | NMPC, horizon=15 | 2.600 | 160.492 +/- 113.996 | 10.012 +/- 0.039 | 5.373 +/- 2.237 | 16.897 |
| Delay without compensation | NMPC, horizon=20 | 2.000 | 159.592 +/- 70.467 | 10.015 +/- 0.011 | 4.314 +/- 1.474 | 21.795 |
| Delay without compensation | NMPC, horizon=25 | 3.400 | 101.514 +/- 102.729 | 10.005 +/- 0.083 | 5.276 +/- 1.879 | 26.042 |
| Delay without compensation | NMPC, horizon=30 | 0 | 216.719 +/- 106.113 | 10.017 +/- 0.014 | 3.355 +/- 1.119 | 39.391 |
| Delay without compensation | NMPC, horizon=35 | 0 | 223.654 +/- 81.406 | 10.017 +/- 0.013 | 1.823 +/- 0.798 | 53.574 |
| Delay without compensation | NMPC, horizon=40 | 0 | 231.804 +/- 81.578 | 10.017 +/- 0.013 | 1.571 +/- 0.105 | 73.862 |
| Delay without compensation | RL | 0 +/- 0 | 216.298 +/- 69.126 | 10.017 +/- 0.003 | 1.518 +/- 0.176 | 0.840 |
| Delay with compensation | Cascaded PID | 100 | 0.795 +/- 0.410 | 3.899 +/- 1.285 | 2.443 +/- 0.108 | 14.059 |
| Delay with compensation | NMPC, horizon=15 | 9.000 | 58.718 +/- 39.991 | 9.975 +/- 0.154 | 7.695 +/- 2.318 | 11.827 |
| Delay with compensation | NMPC, horizon=20 | 37.400 | 52.955 +/- 31.472 | 9.950 +/- 0.114 | 7.083 +/- 2.884 | 15.081 |
| Delay with compensation | NMPC, horizon=25 | 17.800 | 56.832 +/- 45.293 | 9.952 +/- 0.244 | 3.736 +/- 0.287 | 17.482 |
| Delay with compensation | NMPC, horizon=30 | 1.400 | 57.085 +/- 45.624 | 10.015 +/- 0.013 | 3.376 +/- 0.836 | 21.053 |
| Delay with compensation | NMPC, horizon=35 | 10.200 | 15.251 +/- 8.413 | 9.908 +/- 0.323 | 3.419 +/- 1.202 | 23.495 |
| Delay with compensation | NMPC, horizon=40 | 100 | 9.627 +/- 5.601 | 8.639 +/- 1.624 | 3.388 +/- 1.195 | 27.001 |
| Delay with compensation | RL | 82.800 +/- 32.828 | 6.198 +/- 5.274 | 7.056 +/- 2.172 | 2.508 +/- 0.496 | 11.717 |

## Robustness Results With 15-Step Delay Compensation

The robustness tests use per-episode randomized parameters. Mass variation samples the ball mass from `[0.005 kg, 0.03 kg]`; controller-gain variation samples the low-level velocity-controller gain from `[4.0, 8.0]`; external disturbance uses an Ornstein-Uhlenbeck process,

```text
x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * epsilon_t,
epsilon_t ~ N(0, 1),
v_disturbance = clip(x_{t+1}, -clip, clip).
```

For the external-disturbance test, `mu = 0`, `clip = 0.1 m/s`, `theta` is sampled from `[0.1, 0.3]`, and `sigma` is sampled from `[0.01, 0.05]`.

| Test | Controller | SR (%) up | SE (mm) down | CONT (s) down | CLIT (s) down |
| --- | --- | ---: | ---: | ---: | ---: |
| Mass | Cascaded PID | 28.200 | 16.341 +/- 8.812 | 8.751 +/- 2.227 | 3.719 +/- 2.823 |
| Mass | NMPC, horizon=15 | 0.400 | 258.485 +/- 129.922 | 10.016 +/- 0.000 | 4.533 +/- 2.028 |
| Mass | NMPC, horizon=25 | 1.800 | 209.995 +/- 116.766 | 10.014 +/- 0.020 | 6.474 +/- 3.541 |
| Mass | NMPC, horizon=35 | 7.600 | 83.680 +/- 73.668 | 9.995 +/- 0.088 | 5.962 +/- 3.623 |
| Mass | RL | 5.033 +/- 5.662 | 32.897 +/- 18.612 | 9.957 +/- 0.380 | 5.275 +/- 3.629 |
| Gain | Cascaded PID | 92.200 | 5.648 +/- 11.465 | 5.778 +/- 2.466 | 2.320 +/- 0.091 |
| Gain | NMPC, horizon=15 | 8.600 | 72.754 +/- 53.077 | 9.992 +/- 0.093 | 7.544 +/- 2.535 |
| Gain | NMPC, horizon=25 | 7.800 | 105.193 +/- 77.616 | 9.991 +/- 0.158 | 3.846 +/- 1.059 |
| Gain | NMPC, horizon=35 | 0.400 | 75.333 +/- 95.653 | 10.013 +/- 0.065 | 2.734 +/- 0.141 |
| Gain | RL | 26.033 +/- 13.980 | 45.202 +/- 65.999 | 9.683 +/- 0.971 | 2.358 +/- 0.195 |
| Disturbance | Cascaded PID | 14.000 | 53.425 +/- 40.160 | 9.905 +/- 0.429 | 3.773 +/- 2.531 |
| Disturbance | NMPC, horizon=15 | 0.600 | 248.262 +/- 121.605 | 10.016 +/- 0.016 | 4.355 +/- 2.360 |
| Disturbance | NMPC, horizon=25 | 0.200 | 226.220 +/- 87.548 | 10.016 +/- 0.014 | 4.154 +/- 2.178 |
| Disturbance | NMPC, horizon=35 | 2.000 | 196.699 +/- 109.675 | 10.014 +/- 0.030 | 4.087 +/- 2.651 |
| Disturbance | RL | 6.100 +/- 1.050 | 125.050 +/- 100.728 | 9.991 +/- 0.148 | 3.795 +/- 2.700 |

## Long-Delay Robustness Results

| Delay | Controller | SR (%) up | SE (mm) down | CONT (s) down | CLIT (s) down |
| --- | --- | ---: | ---: | ---: | ---: |
| 30 steps, 0.5 s | Cascaded PID | 84.000 | 7.958 +/- 4.838 | 7.955 +/- 2.047 | 2.637 +/- 0.083 |
| 30 steps, 0.5 s | NMPC, horizon=15 | 11.000 | 52.121 +/- 36.245 | 9.967 +/- 0.152 | 7.779 +/- 2.243 |
| 30 steps, 0.5 s | NMPC, horizon=25 | 4.200 | 68.035 +/- 38.087 | 9.995 +/- 0.171 | 4.006 +/- 0.473 |
| 30 steps, 0.5 s | NMPC, horizon=35 | 52.000 | 31.296 +/- 33.580 | 9.879 +/- 0.209 | 3.108 +/- 0.162 |
| 30 steps, 0.5 s | RL | 37.900 +/- 15.109 | 27.074 +/- 18.950 | 9.861 +/- 0.457 | 2.652 +/- 0.262 |
| 45 steps, 0.75 s | Cascaded PID | 33.800 | 18.103 +/- 19.096 | 9.759 +/- 0.513 | 2.925 +/- 0.086 |
| 45 steps, 0.75 s | NMPC, horizon=15 | 15.200 | 47.756 +/- 32.725 | 9.967 +/- 0.132 | 7.856 +/- 2.175 |
| 45 steps, 0.75 s | NMPC, horizon=25 | 1.400 | 64.664 +/- 34.857 | 10.004 +/- 0.114 | 4.414 +/- 0.964 |
| 45 steps, 0.75 s | NMPC, horizon=35 | 2.200 | 67.687 +/- 67.625 | 10.014 +/- 0.022 | 3.373 +/- 0.172 |
| 45 steps, 0.75 s | RL | 0.767 +/- 0.559 | 100.575 +/- 61.400 | 10.016 +/- 0.027 | 2.900 +/- 0.179 |
| 60 steps, 1.0 s | Cascaded PID | 2.600 | 53.045 +/- 76.236 | 9.993 +/- 0.177 | 3.228 +/- 0.101 |
| 60 steps, 1.0 s | NMPC, horizon=15 | 15.000 | 47.354 +/- 29.214 | 9.982 +/- 0.095 | 7.928 +/- 2.114 |
| 60 steps, 1.0 s | NMPC, horizon=25 | 5.600 | 47.965 +/- 27.404 | 9.983 +/- 0.141 | 5.025 +/- 1.574 |
| 60 steps, 1.0 s | NMPC, horizon=35 | 0.800 | 42.323 +/- 74.530 | 10.010 +/- 0.078 | 3.698 +/- 0.200 |
| 60 steps, 1.0 s | RL | 1.500 +/- 1.399 | 248.859 +/- 102.127 | 10.016 +/- 0.004 | 3.198 +/- 0.204 |
