# GR00T N1.5 TacMap Cross-Attention

New training uses `post_dit + grid=2 + dropout=0.3` by default to evaluate the default forecast 16 step, and reprogramme the 4 step before execution. Optimizing analysis, dissertation and digestion commands see [Touch programme analysis](TACTILE_DESIGN_REVIEW.md).

A post-touch training branch independent of `GR00T_n15_Tactile`. It loads pre-training GR00T N1.5, freezes Eagle visual towers and language models, and combines training for the former Action Head, DiT branches and TacMap residual correction branches.

## Model

Enter original depth: `[B, N, 1, H, W]`, each site from HDF5:

```text
robot/tactile/distance_along_normal_m/<site>
```

Depth is `float32` in millimetres. Unified calculation using HDF5 `robot/tactile/meta/max_distance_m` (default `0.015`) for reading:

```python
depth = clip(depth_m, 0, d_max_m) / d_max_m
```

A shared CNN produces four tokens in a 2×2 grid per site, with sensor and spatial embeddings. By default, fusion occurs after DiT and before the action decoder. DiT action tokens serve as queries, TacMap tokens as keys/values, and Q/K use LayerNorm:

```text
action_features += CrossAttention(LN_Q(action_features), LN_K(tactile_tokens), tactile_tokens)
```

Touch token does not collide with Eagle VLM token. `out_proj` zero of Cross-Attention is initialized and therefore untrained to match the original GR00T output. 0.3 integer tactile dropout during training; A sample of drop controls the exact door of tactile residual to zero. The loss is still based on the original Flow Matching loss.

Freezing: Eagle visual tower and language model.

Trainable components: VLLN, VL self-attention, state/action encoders, action decoder, future/position embeddings, DiT, TacMap CNN, projections, sensor/spatial embeddings, and one eight-head cross-attention layer.

Training and reasoning default on `240×240`. Checkpoint save site order, image size, `resolution_step`, depth unit and `d_max`; The reasoning end reads and validates these parameters from checkpoint `config.json`.

## Training

Retain the old script as Cross directory:

```bash
bash policy/GR00T_n15_Tactile_Cross/train.sh 03 --tag tactile_cross --gpu 6
```

The training output is:

```text
../policy_ckpt/03/<robot_key>/gr00t_n15_trunc_tactile_cross_tactile_cross
```

The default batch size is 64, which trains 20000 steps without additional touch parameters. This directory is automatically discovered by `eval_double_env.sh`; You can also enter the active model directory.

The new default is effective only when the tactile branch is created from the non-touch base. (a) The structure that is kept while loading Cross checkpoint or `--resume`; checkpoint loads the old `pre_dit` when the integration field is missing. The use of the new structure requires retraining, and the old weight cannot be converted by changing the default value. tag should be changed to avoid mixing old and new results.

## Evaluation

```bash
bash policy/GR00T_n15_Tactile_Cross/eval_double_env.sh 03 all --sii --headless \
  --model-path ../policy_ckpt/03/multi_xarm7_with_ability/gr00t_n15_trunc_tactile_cross_tactile_cross
```

There is no need to import integration parameters and reason to read structure from checkpoint. `execution_horizon: 4` control prefix for `deploy_policy.yml`, `action_horizon: 16` maintains the server 's projected length; Add `--chunk-size 16` to restore the old execution length. The 4 step increases the frequency of strategic queries and does not guarantee real-time ingestion.

`TacMapRig.capture()` has provided `distance_along_normal_m` and Cross will use it directly; The old quantitative field `tacmap` will not enter the policy.

Old VLM-fused tactile checkpoint is incompatible with this branch. Retrain from the GR00T N1.5 base or load a Cross checkpoint produced by this branch.
