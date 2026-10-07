# Tactile Policy

This directory provides a separate set of Tactile-Flat + ACT training, offline validation and Isaac Sim
Online assessment process. It reads.
Four-way RGB in Dex2Bench HDF5, robotic joint status, future action block, and
`robot/tactile/tacmap` One channel TacMap defined by `site_names`.

The existing `policy/ACT` and `run_policy.py` have not been modified. Independent in online assessment of use strategy catalogue
`policy/ACT-Tactile/tactile_run_policy.py`, local loading of tactile checkpoint and reset,
action chunk, success rate, metric and result output logic.

## Model input

- RGB: the default four-way ACT camera with a `[B, 4, 3, 480, 640]` shape;
- qpos: Defaultly enable the current robot key corresponding to active DOF;
- TacMap: Sort `robot/tactile/meta/site_names` by HDF5 in shape
  `[B, N, 1, 240, 240]`。 Sites `N` from HDF5 metadata such as Allegro/LEAP
  Currently 8, other configured hand models are 10;
- action: Future 30 step target, end and `action_valid=False` position used
  `is_pad` shield.

RGB uses shared ResNet-18 to generate space token. All TacMap Merge to batch Later Pass
Shared single channel ResNet-18, each touchpoint is digitized globally and adds site
identity embedding。 A learning query by combining all of attention pooling
Sites, which ultimately generate only one touch token regardless of the number of stations in the fixed configuration; The token
Add tactile modality embedding. Next, visual, single touch token, qpos and
cVAE latent token entered ACT Transformer and predicted action chunk.

Single token attention pooling starts with checkpoint format version 4.version 3
checkpoint is incompatible with the current structure and requires retraining.

## Environment

Relying on the existing PyTorch, torchvision, h5py, NumPy and OpenCV environments of the project. It's available.
IsaacLab Python for example:

```powershell
$python = ".\.venv\Scripts\python.exe"
```

All commands are executed from the `dex2scene` root directory.

## Minimum smoke test

The following command only checks data, forward, reverse and checkpoint for pass. It uses small models and
A reduced image does not represent a formal training configuration:

```powershell
& $python -m policy.tactile_policy.train `
  --dataset-dir "..\tactile_leverage" `
  --ckpt-dir ".\outputs\logs\tactile_policy\smoke" `
  --epochs 1 `
  --batch-size 1 `
  --num-workers 0 `
  --val-ratio 0 `
  --max-steps-per-epoch 1 `
  --backbone tiny `
  --hidden-dim 128 `
  --dim-feedforward 256 `
  --nheads 4 `
  --enc-layers 2 `
  --dec-layers 2 `
  --latent-dim 16 `
  --image-height 120 `
  --image-width 160 `
  --tactile-height 60 `
  --tactile-width 60 `
  --device cpu
```

## Formal Configuration Example

```powershell
& $python -m policy.tactile_policy.train `
  --dataset-dir "..\tactile_leverage" `
  --ckpt-dir ".\outputs\logs\tactile_policy\vision_tactile" `
  --epochs 100 `
  --batch-size 2 `
  --num-workers 4 `
  --chunk-size 30 `
  --backbone resnet18 `
  --robot-key multi_panda_with_allegro `
  --device cuda
```

`--robot-key` is an optional parameter. If omitted, training detects the robot key from HDF5; When provided
Strictly validates all episode metadata. Inconsistencies or errors are reported and values in HDF5 are not overridden. The same.
The catalogue still contains only one hand type.

The policy only receives visual input with TacMap and does not provide `vision_only` training switches. Pure visual baseline use unmodified `policy/ACT` directly.

Linux may also use packaged scripts:

```bash
bash policy/tactile_policy/train.sh \
  ../tactile_leverage \
  outputs/logs/tactile_policy/vision_tactile \
  --epochs 100 --batch-size 2
```

## Offline touch.

```powershell
& $python -m policy.tactile_policy.offline_eval `
  --checkpoint ".\outputs\logs\tactile_policy\vision_tactile\policy_best.ckpt" `
  --dataset-dir "..\tactile_leverage" `
  --batch-size 2 `
  --num-workers 0 `
  --output ".\outputs\logs\tactile_policy\vision_tactile\ablation.json"
```

The report includes:

- `normal`: Normal TacMap;
- `zero`: all TacMap zeros;
- `shuffle`: Disruption of TacMap in the same episode for certainty time.

Compare three sets of L1 loss to check whether the model relies on touch. It is not a substitute for experimenting with mission success.

## Isaac Sim Online Assessment

Online input is ** 4 RGB + checkpoint-defined TacMaps**: Left and right wrist camera, right and right double-eye camera, right and right.
and the dynamic number of touch stations saved in checkpoint `site_names`. Camera, Touch site, robot key,
active indices and state dimension will verify before rollout. Model with prior-only
Arguments that the action chunk integration of the output returns to the actual joint value and is executed item by item with 20 Hz.

TacMap ray casting requires Isaac Sim to use CUDA equipment and cannot use CPU simulation.
The current script only supports local checkpoint without passing REMOTE server/client.

Example of direct command:

```bash
python policy/ACT-Tactile/tactile_run_policy.py \
  --policy-type TACTILE \
  --task scenes/06_fruit_bowl_loading.yaml \
  --ckpt-dir outputs/logs/tactile_policy/vision_tactile \
  --ckpt-name policy_best.ckpt \
  --robot-key multi_panda_with_allegro \
  --enable-rgb --active-dof --device cuda:0 \
  --temporal-agg --temporal-agg-k 0.1 \
  --num-episodes 10 --episode-steps 400 \
  --enable-generalization --generalization-profile none \
  --anchor-hdf5 /path/to/episode_000000.hdf5 \
  --headless
```

temporal aggregation enabled by direct command visible above. Once enabled, the policy will be in each 20 Hz control step
Re-engineered and indexed the overlap action chunk to cover the current moment instead of one-time execution complete
chunk。 `--temporal-agg-k` controls the degree of decay of integration weights; Use even average for setting `0`.

You can also use packing scripts:

Online wrapper forwards with `mapfile` and array parameters, requesting ** Bash 4 + **; Target operating environment is
Isaac/Linux。 The Bash 3 bundled with macOS does not support this wrapper; run it in the Isaac/Linux environment.

```bash
TACTILE_TASK=06_fruit_bowl_loading \
TACTILE_CKPT_DIR=outputs/logs/tactile_policy/vision_tactile \
TACTILE_ROBOT_KEY=multi_panda_with_allegro \
TACTILE_ANCHOR_HDF5=/path/to/episode_000000.hdf5 \
bash policy/ACT-Tactile/eval_direct.sh
```

`eval_direct.sh` Default enabled temporal aggregation with a default decay factor of `0.1`. Yes.
`TACTILE_TEMPORAL_AGG_K` adjustments, e. g.
`TACTILE_TEMPORAL_AGG_K=0.05 bash policy/ACT-Tactile/eval_direct.sh`。

Selected environment variables include `TACTILE_CKPT_NAME`, `TACTILE_GPU_ID`,
`TACTILE_NUM_EPISODES`、`TACTILE_EPISODE_STEPS`、`TACTILE_SEED`、
`TACTILE_GEN_PROFILE`, `TACTILE_OUTPUT_DIR`, `TACTILE_ROBOT_KEY` and
`ISAAC_PYTHON`。 wrapper forwards `--robot-key` only when `TACTILE_ROBOT_KEY` is not empty.
(a) For visible hand-checking; If omitted, the checkpoint determines the robot key.

## Output File

- `resolved_config.json`: This model, data and dimension configuration;
- `dataset_stats.pkl`: qpos/action integration statistics based on training frames only;
- `policy_last.ckpt`: Last round checkpoint;
- `policy_best.ckpt`: checkpoint selected by pressing validation loss.

checkpoint also save camera/site order, robot key, active joint indices in semantic order
active joint names, TacMap resolution/max-distance Encoding Parameters, Model Configuration, Normalization Statistics and
optimizer status. Online assessments reject the absence of these run-time metadata or checkpoint in old formats.
It is not a strategy to silence the touch or misalign the joint order. This type of old checkpoint needs retraining/storage.
checkpoint is embodiment-specific, which can only be used for the training of handstyles and cannot be reused across hands.

## Current Data Limit

`../tactile_leverage/episode_000000.hdf5` has only one track and is marked as
`success=False`、`demo_eligible=False`。 Code will allow it to perform smoke test, but will print
Clear warning. It validates the TacMap reading, integration and gradient links, and does not train a reliable strategy alone.
Nor can it prove that touch has increased mission success.
