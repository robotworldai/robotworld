# Published checkpoint

## `wheel_legged_moving_jump_model_844.pt`

- Task: `Wheel-Legged-Jump-Moving-Curriculum-Flat-v0`
- Training backend: RSL-RL PPO
- Final moving-jump curriculum level: `4`
- Evaluation command range: `1.0 m/s`
- Evaluation yaw range: `1.2 rad/s`
- Jump-height argument used by the included demos: `0.10 m`
- SHA-256:
  `d1d271f0c323fa13538b80119839b29eb267942af63a6e0d6c3dbf5ef319deb0`

Run from the repository root:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint checkpoints/wheel_legged_moving_jump_model_844.pt \
  --num_envs 1 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

This checkpoint is a research and learning artifact. It has not been validated
on physical hardware.
