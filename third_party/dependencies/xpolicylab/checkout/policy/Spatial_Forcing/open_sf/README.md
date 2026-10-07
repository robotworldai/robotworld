# OpenPI-SF for Spatial_Forcing

## Training Configuration

Default training configuration name:

```text
pi05sf_jax_robodojo_v21_offcache
```

Main fields:

| Fields | Current value [p] |
|------|--------|
| `repo_id` | `RoboDojo_lerobot_v21_video` |
| `model` | `Pi0Config(pi05=True)` |
| `align_enabled` | `True` |
| `align_target_model` | `vggt` |
| `vla_layers_align` | `12` |
| `vggt_layers_align` | `-1` |
| `pooling_func` | `bilinear` |
| `use_vggt_pe` | `True` |
| `use_vlm_norm` | `True` |
| `align_loss_coeff` | `0.2` |
| `sf_cache_enable` | `True` |
| `sf_cache_mode` | `readonly` |
| `sf_cache_miss_policy` | `error` |
| `sf_cache_save_dtype` | `bf16` |
| `sf_cache_chunk_size` | `128` |
| `sf_dataset_uid` | `0` |
| `batch_size` | `256` |
| `num_workers` | `8` |
| `num_train_steps` | `60000` |
| `save_interval` | `5000` |

Weights and paths are covered by environmental variables:

| Variables | Default value | Annotations |
|------|--------|------|
| `PI05_BASE_PATH` | `./checkpoints/pi05_base` | Pi05 JAX base checkpoint root directory must contain `params/` and `assets/` |
| `VGGT_WEIGHT_PATH` | `./checkpoints/VGGT-1B` | VGGT weight directory, must contain `model.pt` |
| `SF_CACHE_DIR` | `./results/sf_cache` | OpenPI-SF chunked offline feature cache |
| `PI05SF_ASSETS_DIR` | `./assets/pi05sf_robodojo_v21` | norm stats / assets Directory |
| `HF_LEROBOT_HOME` | `${XPL_DATA_ROOT}` or `../data` | LeRobot Data Root Directory |


## Policy Server

Start OpenPI policy server from checkpoint:

```bash
cd /path/to/openpi05-sf/openpi

uv run --no-sync scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05sf_jax_robodojo_v21_offcache \
  --policy.dir=/path/to/checkpoint_step_dir \
  --port=8000
```